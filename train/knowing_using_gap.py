#!/usr/bin/env python3
"""Knowing-using gap of per-fact (n=1) chaining runs: T_mem / T_gen / dT and A_mem / A_gen / dA.

  T_mem         first epoch (> 0) with eval/memorization == 1.0
  T_gen         first record with eval/chaining == 1.0 whose next 9 records are also 1.0 (the
                window is cut at the end of the log); a run whose first such record is epoch 0
                gets no T_gen
  A_mem, A_gen  mean of the records at the last epoch, over runs
Runs with memorization > 0 at epoch 0 are dropped; T means are over the runs that reach them;
dT and dA are differences of the displayed (half-up rounded) values. A run directory holds
config.yaml + training_logs.json; only chaining_tasks runs with data_size 1 and the chosen
training type are used, and runs whose log stops before sft.num_train_epochs are counted and skipped.

Usage (from the repo root):
    python train/knowing_using_gap.py checkpoints/per_fact/lora
    python train/knowing_using_gap.py --method fft checkpoints/per_fact/fft
    python train/knowing_using_gap.py --multi checkpoints/multi_fact_checkpoints/chaining
"""
import argparse
import json
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import yaml

MEM, GEN, WINDOW = "eval/memorization", "eval/chaining", 10


def find_runs(paths):
    """Yield (run_dir, config, records); records is None when training_logs.json is missing."""
    for path in paths:
        for root, dirs, files in os.walk(path):
            dirs.sort()
            if "config.yaml" not in files or not {"training_logs.json", "experiment_data.json"} & set(files):
                continue
            dirs[:] = []  # checkpoints inside a run are not runs
            with open(os.path.join(root, "config.yaml")) as f:
                cfg = yaml.safe_load(f)
            recs = None
            if "training_logs.json" in files:
                with open(os.path.join(root, "training_logs.json")) as f:
                    recs = [r for r in json.load(f) if r and "train/epoch" in r and MEM in r]
            yield root, cfg, recs


def stable_gen_epoch(logs):
    for i, l in enumerate(logs):
        if l.get(GEN) == 1.0 and all(logs[j].get(GEN, 1.0) == 1.0 for j in range(i + 1, min(i + WINDOW, len(logs)))):
            return l["train/epoch"]
    return None


def aggregate(runs):
    mem_by_ep, gen_by_ep, t_mem, t_gen, n_used = {}, {}, [], [], 0
    for logs in runs:
        if logs[0][MEM] > 0:  # the fact is already known before training
            continue
        n_used += 1
        for l in logs:
            mem_by_ep.setdefault(int(l["train/epoch"]), []).append(l[MEM])
            if GEN in l:
                gen_by_ep.setdefault(int(l["train/epoch"]), []).append(l[GEN])
        m = next((l["train/epoch"] for l in logs if l[MEM] == 1.0), None)
        g = stable_gen_epoch(logs)
        if m:
            t_mem.append(m)
        if g:
            t_gen.append(g)
    last = max(mem_by_ep) if mem_by_ep else None
    return dict(n_used=n_used, t_mem=t_mem, t_gen=t_gen, a_mem=mem_by_ep.get(last, []), a_gen=gen_by_ep.get(last, []))


def q(values, nd):
    """Displayed value: the mean rounded half-up to nd decimals."""
    if not values:
        return None
    return Decimal(str(float(np.mean(values)))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def summarize(model, method, runs, n_incomplete):
    s = aggregate(runs)
    tm, tg, am, ag = q(s["t_mem"], 1), q(s["t_gen"], 1), q(s["a_mem"], 2), q(s["a_gen"], 2)
    dt = tg - tm if tm is not None and tg is not None else None
    da = am - ag if am is not None and ag is not None else None
    row = " / ".join("--" if v is None else str(v) for v in (tm, tg, dt, am, ag, da))
    msd = lambda v, nd: f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1) if len(v) > 1 else 0.0:.{nd}f}" if v else "--"
    print(f"{model} {method} chaining: {len(runs)} runs ({n_incomplete} incomplete skipped), {s['n_used']} used, "
          f"n(T_mem)={len(s['t_mem'])}, n(T_gen)={len(s['t_gen'])}")
    print(f"  T_mem {msd(s['t_mem'], 2)}  T_gen {msd(s['t_gen'], 2)}  A_mem {msd(s['a_mem'], 3)}  A_gen {msd(s['a_gen'], 3)}")
    print(f"  T_mem / T_gen / dT / A_mem / A_gen / dA: {row}")
    return {"method": method, "n_runs": len(runs), "n_incomplete": n_incomplete, "n_used": s["n_used"],
            "n_t_mem": len(s["t_mem"]), "n_t_gen": len(s["t_gen"]), "row": row,
            **{k: float(np.mean(s[k])) if s[k] else None for k in ("t_mem", "t_gen", "a_mem", "a_gen")}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run directories or parent directories (searched recursively)")
    p.add_argument("--method", choices=["lora", "fft"], default="lora", help="per-fact runs of this training type")
    p.add_argument("--multi", action="store_true", help="multi-fact runs instead: final-epoch scores of each run")
    p.add_argument("--json", default=None, metavar="OUT", help="also write the numbers to this JSON file")
    args = p.parse_args()

    out, groups, incomplete, duplicates = {}, {}, {}, {}
    for run, cfg, recs in find_runs(args.paths):
        d, m = cfg["dataset"], cfg["model"]
        per_fact = d.get("data_size") == 1
        if d.get("task_type") != "chaining_tasks" or per_fact == args.multi:
            continue
        if per_fact and m.get("training_type", "lora") != args.method:
            continue
        model = m.get("short_name", m["name"])
        if not recs or int(recs[-1]["train/epoch"]) != int(cfg["sft"]["num_train_epochs"]):
            incomplete[model] = incomplete.get(model, 0) + 1
        elif args.multi:
            last = recs[-1]
            out[run] = {"model": model, "n": d["data_size"], "epoch": last["train/epoch"],
                        "memorization": last[MEM], "chaining": last.get(GEN)}
            print(f"{run}  {model}  n={d['data_size']}  epoch {last['train/epoch']:g}  "
                  f"memorization {last[MEM]:.3f}  chaining {last.get(GEN, float('nan')):.3f}")
        else:  # one run per fact: a re-run of the same fact replaces the older run
            key = tuple(d.get("specific_samples") or [run])
            t = os.path.getmtime(os.path.join(run, "training_logs.json"))
            runs = groups.setdefault(model, {})
            if key in runs:
                duplicates[model] = duplicates.get(model, 0) + 1
            if key not in runs or t > runs[key][0]:
                runs[key] = (t, recs)

    for model, n in duplicates.items():
        print(f"{model}: {n} older run(s) of an already seen fact ignored (the latest complete run is used)")
    groups = {model: [recs for _, recs in runs.values()] for model, runs in groups.items()}
    if not out and not groups and not incomplete:
        print(f"no {'multi-fact' if args.multi else 'per-fact ' + args.method} chaining runs under {args.paths}")
    elif args.multi:
        print(f"({sum(incomplete.values())} incomplete run(s) skipped; training-callback scores, the paper's "
              f"1-to-N memorization comes from validation/evaluate_checkpoints.py)")
    else:
        for model in sorted(set(groups) | set(incomplete)):
            out[model] = summarize(model, args.method, groups.get(model, []), incomplete.get(model, 0))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
