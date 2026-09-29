"""Write the injected-fact samples for train_inject_lrsd.py with the training pipeline's own sampler:
random.seed(seed) then train/training_utils.prepare_multi_dataset(data/multi_fact_data_template, 1000, task), exactly
as train/train_multi_fact.py does for the paper configs (multi_chaining_<model>_n1000.yaml).  The sample depends only
on the family's seed (chaining: Qwen 848, LLaMA 1340), so the 1B-class and 3B models of a family share it.

Each sample goes to common.DATA[(family, task)] as experiment_data.json + original_fact.json
(train/training_utils.save_experiment_data).  If the files already exist they are not rewritten: the regenerated
sample must equal them (training_data, eval_data and original_fact), otherwise the script stops with an error.

Usage (from the repo root): python distill/make_inject_data.py
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from training_utils import prepare_multi_dataset, save_experiment_data  # noqa: E402

from common import DATA, gen_items  # noqa: E402

SEEDS = {("qwen", "chaining"): 848, ("llama", "chaining"): 1340,
         ("qwen", "intersection"): 8057, ("llama", "intersection"): 8694}
SRC = os.path.normpath(os.path.join(HERE, "..", "data", "multi_fact_data_template"))


def sample(task, seed):
    random.seed(seed)
    return prepare_multi_dataset(SRC, 1000, f"{task}_tasks")


def verify(ed, out, name):
    """Stop with an error unless the sample `ed` equals the one saved in `out`."""
    ref = json.load(open(os.path.join(out, "experiment_data.json")))
    ref["original_fact"] = json.load(open(os.path.join(out, "original_fact.json")))
    diff = [k for k in ("training_data", "original_fact") if list(ed[k]) != ref[k]]
    diff += [f"eval_data[{k}]" for k, ds in ed["eval_data"].items() if list(ds) != ref["eval_data"].get(k)]
    if diff:
        sys.exit(f"{name}: the regenerated sample differs from {out} in {', '.join(diff)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chaining", choices=["chaining", "intersection"])
    args = ap.parse_args()
    for fam in ("qwen", "llama"):
        seed, out = SEEDS[(fam, args.task)], DATA[(fam, args.task)]
        name = f"{fam} {args.task} (seed {seed})"
        ed = sample(args.task, seed)
        of = ed["original_fact"]
        n_facts, n_q = len(ed["training_data"]), len({it["question"] for it in gen_items(of, args.task)})
        if os.path.exists(os.path.join(out, "experiment_data.json")):
            verify(ed, out, name)
            status = "verified"
        else:
            os.makedirs(out, exist_ok=True)
            save_experiment_data(ed, os.path.join(out, "experiment_data.json"))
            status = "written"
        rows = "two-hop" if args.task == "chaining" else args.task
        print(f"{name}: {n_facts} facts, {len(of)} {rows} rows, {n_q} unique questions -> {out} ({status})",
              flush=True)


if __name__ == "__main__":
    main()
