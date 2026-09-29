"""Summary of the in-training LRSD runs (run_inject_scale.sh): 4 models x {B, lam 0.1, lam 1} x seeds {848, 1, 2}
on chaining, layer pair 0.75L -> 0.5L.

Per run (from log.jsonl, no-patch greedy EM on all unique two-hop questions, 1toN gold):
late = mean EM over the --late window (epochs 30-50), final = EM at the last epoch, mem = memorization 1toN at the
last epoch, mem_min = lowest memorization seen from epoch 10 on (collapse check).  Runs without a log or that have not
reached their last epoch yet are reported and skipped.
Per setting: mean and sd over the seeds, difference to B and relative gain (arm / B - 1) on the seeds where both
runs exist, number of those seeds where arm > B, exact unpaired permutation p (one-sided; with 3 vs 3 the smallest
possible p is 0.05).
Pooled per task over the models: exact sign-flip test on the seed-paired differences (arm - B), two-sided
(sign flips within a setting are exchangeable under H0).

Usage (from the repo root): python distill/analyze_scale.py [--out results/lrsd/scale_analysis.json]
"""
import argparse
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "experiment_logs", "distill", "inject"))
PAIR = {"qwen2.5-1.5b": "21-14", "qwen2.5-3b": "27-18", "llama3.2-1b": "12-8", "llama3.2-3b": "21-14"}
MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-1b", "llama3.2-3b"]
SEEDS = ["848", "1", "2"]
LATE = (30, 50)
ARMS = [("L01", "lam 0.1"), ("L1", "lam 1")]


def run_dir(arm, model, task, seed, runs=RUNS):
    return os.path.join(runs, f"inject_{arm}__{model}{'_inter' if task == 'intersection' else ''}_s{seed}")


def load(d, late=None):
    """Summary of one run, or None if it has no log or has not reached its last epoch (--epochs in its args.json)
    or the end of the `late` window (default LATE)."""
    late = late or LATE
    p = os.path.join(d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = [json.loads(x) for x in open(p) if x.strip()]
    ev = {r["epoch"]: r for r in L if r.get("epoch", -1) >= 0}
    a = os.path.join(d, "args.json")
    epochs = json.load(open(a))["epochs"] if os.path.exists(a) else late[1]
    if not ev or max(ev) < max(epochs, late[1]):
        return None
    last = max(ev)
    return dict(late=float(np.mean([ev[e]["chain_uq_em"] for e in range(late[0], late[1] + 1)])),
                final=ev[last]["chain_uq_em"], mem=ev[last]["mem_em"],
                mem_min=min((ev[e]["mem_em"] for e in ev if "mem_em" in ev[e] and e >= 10), default=float("nan")),
                ce_max=max((ev[e]["train_ce"] for e in ev if e >= 10), default=float("nan")), n_q=len(L[0]["questions"]))


def perm_p(a, b):
    allv = np.array(a + b)
    obs = np.mean(a) - np.mean(b)
    null = [allv[list(c)].mean() - np.delete(allv, list(c)).mean() for c in itertools.combinations(range(len(allv)), len(a))]
    return float(np.mean([x >= obs - 1e-12 for x in null]))


def signflip_p(d):
    d = np.asarray(d, float)
    obs = d.mean()
    signs = ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1) * 2 - 1
    return float(np.mean(np.abs(signs @ d / len(d)) >= abs(obs) - 1e-12))


def summarize(runs):
    """{key: mean, sd, per_seed} over the seeds of one arm ({seed: load(...)})."""
    seeds = list(runs)
    return {k: dict(mean=float(np.mean([runs[s][k] for s in seeds])),
                    sd=float(np.std([runs[s][k] for s in seeds], ddof=1)) if len(seeds) > 1 else None,
                    per_seed=[round(runs[s][k], 4) for s in seeds])
            for k in ("late", "final", "mem", "mem_min", "ce_max")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=RUNS, help="directory with one inject_<arm>__<model>_s<seed>/ per run")
    ap.add_argument("--out", default=None, help="write the summary to this .json (and the table to .txt)")
    ap.add_argument("--tasks", default="chaining")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(SEEDS))
    ap.add_argument("--late", default=f"{LATE[0]}-{LATE[1]}", help="epoch window averaged into 'late'")
    args = ap.parse_args()
    tasks, models, seeds = args.tasks.split(","), args.models.split(","), args.seeds.split(",")
    late = tuple(int(x) for x in args.late.split("-"))

    res, pooled = {}, {}
    for task in tasks:
        for model in models:
            names = {"B": "B", **{a: f"P{PAIR[model]}{a}" for a, _ in ARMS}}
            runs = {}
            for a, n in names.items():
                runs[a] = {}
                for s in seeds:
                    d = run_dir(n, model, task, s, args.runs_dir)
                    r = load(d, late)
                    if r is None:
                        print(f"skip {os.path.basename(d)}: "
                              f"{'no log' if not os.path.exists(os.path.join(d, 'log.jsonl')) else 'unfinished'}")
                    else:
                        runs[a][s] = r
            if not runs["B"]:
                continue
            st = dict(pair=PAIR[model], n_q=next(iter(runs["B"].values()))["n_q"])
            for a in names:
                if runs[a]:
                    st[a] = dict(seeds=list(runs[a]), **summarize(runs[a]))
            for a, _ in ARMS:
                paired = [s for s in seeds if s in runs[a] and s in runs["B"]]
                if not paired:
                    continue
                d = [runs[a][s]["late"] - runs["B"][s]["late"] for s in paired]
                arm_late, b_late = [runs[a][s]["late"] for s in paired], [runs["B"][s]["late"] for s in paired]
                st[a].update(paired_seeds=paired, d_late=float(np.mean(d)),
                             rel_late=float(np.mean(arm_late) / np.mean(b_late) - 1),
                             n_pos=int(sum(x > 0 for x in d)), p_perm=perm_p(arm_late, b_late),
                             d_mem=float(np.mean([runs[a][s]["mem"] - runs["B"][s]["mem"] for s in paired])))
                pooled.setdefault(f"{task}|{a}|late", []).extend(d)
            res[f"{model}|{task}"] = st
    out = dict(settings=res, pooled={a: dict(n=len(d), mean=float(np.mean(d)), n_pos=int(sum(x > 0 for x in d)),
                                              p_signflip=signflip_p(d)) for a, d in pooled.items()})

    def pm(x):
        return f"{x['mean']:.3f}" + (f"±{x['sd']:.3f}" if x["sd"] is not None else "")

    lines = []
    for task in tasks:
        lines.append(f"\n=== {task}  (late = mean EM epochs {late[0]}-{late[1]}, mean ± sd over seeds; Δ and relative "
                     "gain vs B on the seeds of both arms, seeds > B, perm p; mem = memorization at the last epoch)")
        for model in models:
            st = res.get(f"{model}|{task}")
            if st is None:
                continue
            b = st["B"]
            lines.append(f"{model}  pair {st['pair']}  n_q {st['n_q']}")
            n_b = len(b["seeds"])
            lines.append(f"   B        late {pm(b['late'])} ({n_b} seed{'s' * (n_b > 1)}){'':22s}mem {b['mem']['mean']:.3f}")
            for a, lab in ARMS:
                x = st.get(a)
                if x is None or "d_late" not in x:
                    continue
                lines.append(f"   {lab:8s} late {pm(x['late'])} (Δ {x['d_late']:+.3f}, {x['rel_late']:+4.0%}, "
                             f"{x['n_pos']}/{len(x['paired_seeds'])}, p={x['p_perm']:.2f})  mem {x['mem']['mean']:.3f} "
                             f"(Δ {x['d_mem']:+.3f}, min {min(x['mem_min']['per_seed']):.2f})")
    lines.append("\npooled over the models (seed-paired differences, two-sided exact sign-flip p):")
    for k, p in out["pooled"].items():
        lines.append(f"  {k:24s} {p['n_pos']:2d}/{p['n']} positive, mean Δ {p['mean']:+.3f}, p = {p['p_signflip']:.2g}")
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        open(os.path.splitext(args.out)[0] + ".txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
