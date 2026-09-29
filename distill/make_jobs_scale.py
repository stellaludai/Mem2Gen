"""Job list for the in-training LRSD runs (run by run_inject_scale.sh through run_queue.py).

4 models x {B (lam 0), lam 1, lam 0.1} x seeds {848, 1, 2} = 36 chaining runs.
Layer pair 0.75L -> 0.5L: Qwen2.5-3B (36 layers) 27->18, Qwen2.5-1.5B / LLaMA-3.2-3B (28) 21->14, LLaMA-3.2-1B (16)
12->8.  Names: inject_<arm>__<model>_s<seed> with arm = B or P<s>-<t>L1 / L01 (finished runs are skipped by name).
Order: seed-major (all settings of seed 848 first), longest runs first within a seed.

Usage (from the repo root): python distill/make_jobs_scale.py  -> distill/jobs_inject_scale.json
"""
import argparse
import json
import os

LAYERS = {"llama3.2-3b": 28, "qwen2.5-3b": 36, "qwen2.5-1.5b": 28, "llama3.2-1b": 16}
# minutes per run at 2 runs per GPU (probe / finished runs), used only to order the queue
MINUTES = {("llama3.2-3b", "chaining"): 65, ("qwen2.5-3b", "chaining"): 55, ("llama3.2-3b", "intersection"): 53,
           ("qwen2.5-3b", "intersection"): 46, ("qwen2.5-1.5b", "chaining"): 38, ("qwen2.5-1.5b", "intersection"): 32,
           ("llama3.2-1b", "chaining"): 31, ("llama3.2-1b", "intersection"): 26}
SEEDS = [848, 1, 2]
LAMS = [("0", "B"), ("1", "L1"), ("0.1", "L01")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="chaining")
    ap.add_argument("--models", default=",".join(LAYERS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs_inject_scale.json"))
    args = ap.parse_args()
    tasks, models = args.tasks.split(","), args.models.split(",")
    jobs = []
    for seed in (int(s) for s in args.seeds.split(",")):
        for (model, task), _ in sorted(MINUTES.items(), key=lambda kv: -kv[1]):
            if model not in models or task not in tasks:
                continue
            L = LAYERS[model]
            s, t = round(0.75 * L), round(0.5 * L)
            for lam, tag in LAMS:
                arm = "B" if tag == "B" else f"P{s}-{t}{tag}"
                name = f"inject_{arm}__{model}{'_inter' if task == 'intersection' else ''}_s{seed}"
                jobs.append(dict(name=name, args=["--model", model, "--task", task, "--lam", lam, "--pair", f"{s}-{t}",
                                                  "--seed", str(seed), "--epochs", "50"]))
    json.dump(jobs, open(args.out, "w"), indent=1)
    print(len(jobs), "jobs ->", args.out)


if __name__ == "__main__":
    main()
