#!/usr/bin/env bash
# Smoke test of the Mem2Gen release (chaining on STaRK-Prime): every entry point once, at toy scale, on one GPU.
#   bash tests/smoke_test.sh [GPU]        (default GPU 0; works from any directory)
# PYTHON picks the interpreter (default: python). Outputs go to SMOKE_OUT (default: a new mktemp directory);
# nothing is written inside the repository except data/inject_lrsd/, which step 2 writes (or verifies).
# Models (Qwen/Qwen2.5-1.5B-Instruct) come from the HF hub or cache; HF_HUB_OFFLINE=1 uses the cache only.
#   0  every release .py compiles
#   1  seeded sampling: seed 848 -> 1000 facts / 500 two-hop rows / 409 unique questions, seed 1340 -> 427
#   2  distill/make_inject_data.py
#   3  multi-fact training, 4 facts, 1 epoch (train/train_multi_fact.py)
#   4  per-fact runs, 2 facts, 2 epochs (train/run_specific_samples.py) + train/knowing_using_gap.py
#   5  train/knowing_using_gap.py --multi on the run of step 3
#   6  self-patching grid on 2 questions of that run + patch/compute_oracle.py
#   7  validation/evaluate_checkpoints.py --run_dir on that run
#   8  LRSD, 1 epoch per arm (baseline, lam 1) + distill/analyze_scale.py
#   9  no release file was modified
# A failed step is reported and the next steps still run; the exit code is 1 if any step failed.
# Takes about 10 min on one 80 GB GPU once the model is cached.
set -euo pipefail

GPU=${1:-0}
PY=${PYTHON:-python}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=$(realpath -m "${SMOKE_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/mem2gen_smoke.XXXXXX")}")
case "$OUT/" in "$REPO"/*) echo "SMOKE_OUT must be outside the repository: $OUT" >&2; exit 2 ;; esac
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT")" ] || { echo "SMOKE_OUT must be a new or empty directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT/logs" "$OUT/configs"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"

shopt -s nullglob
RELEASE_PY=(train/{train_multi_fact,dataloader,training_utils,eval_callback,run_specific_samples,knowing_using_gap}.py
            patch/{multi_fact_experiment_nsamples,layer_patching,utils,compute_oracle}.py
            distill/{common,train_inject_lrsd,make_inject_data,analyze_scale,plot_scale,make_jobs_scale,run_queue}.py
            validation/evaluate_checkpoints.py)
RELEASE_FILES=(README.md LICENSE .gitignore assets/* environment/requirements.txt tests/smoke_test.sh
               data/multi_fact_data_template/{chaining_tasks,summary}.json distill/run_inject_scale.sh "${RELEASE_PY[@]}"
               train/configs/experiment_configs_specific_samples/multi_specific_sample_template{,_fft}.yaml
               train/configs/multi_experiment_configs_model/multi_chaining_*_n1000.yaml)
shopt -u nullglob

snapshot() {  # checksums of the release files and git status of the release paths
    md5sum "${RELEASE_FILES[@]}" 2>&1 || true
    git status --porcelain --untracked-files=all -- README.md LICENSE .gitignore assets environment tests train \
        patch distill validation data/multi_fact_data_template ':(exclude)data/inject_lrsd' 2>&1 || true
}
ok() { echo "OK $*"; }

# latest run directory of step 3
toy_run() {
    local run
    run=$(find "$OUT/multi/chaining/qwen2.5-1.5b/n4" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
    [ -n "$run" ] || { echo "no run of step 3 under $OUT/multi" >&2; return 1; }
    echo "$run"
}

step0() {
    "$PY" - "$OUT/pyc" "${RELEASE_PY[@]}" <<'EOF'
import os, py_compile, sys
out, files = sys.argv[1], sys.argv[2:]
for f in files:  # .pyc files go to SMOKE_OUT, not to __pycache__/ in the repo
    py_compile.compile(f, cfile=os.path.join(out, f.replace("/", ".") + "c"), doraise=True)
print(f"OK {len(files)} release .py files compile")
EOF
    bash -n distill/run_inject_scale.sh
    ok "distill/run_inject_scale.sh parses"
}

step1() {
    "$PY" - <<'EOF'
import random, sys
sys.path.insert(0, "train")
from training_utils import prepare_multi_dataset
for seed, n_unique in ((848, 409), (1340, 427)):
    random.seed(seed)
    of = prepare_multi_dataset("data/multi_fact_data_template", 1000, "chaining_tasks")
    got = (len(of["training_data"]), len(of["original_fact"]),
           len({o["generalization_tasks"][0]["prompt"] for o in of["original_fact"]}))
    assert got == (1000, 500, n_unique), (seed, got)
    print(f"OK seed {seed}: {got[0]} facts, {got[1]} two-hop rows, {got[2]} unique questions")
EOF
}

step2() {
    local out
    out=$("$PY" distill/make_inject_data.py)
    echo "$out"
    grep -q "^qwen chaining (seed 848): 1000 facts, 500 two-hop rows, 409 unique questions" <<<"$out"
    grep -q "^llama chaining (seed 1340): 1000 facts, 500 two-hop rows, 427 unique questions" <<<"$out"
    grep "^[a-z]* chaining (seed" <<<"$out" | sed 's/ -> .* (/ (/; s/^/OK /'
}

step3() {
    "$PY" - train/configs/multi_experiment_configs_model/multi_chaining_qwen2.5-1.5b_n1000.yaml \
        "$OUT/configs/toy_multi.yaml" "$OUT/multi" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["experiment"]["name"] = "smoke_chaining_qwen2.5-1.5b_n4"
c["dataset"]["data_size"] = 4
c["sft"].update(num_train_epochs=1, per_device_train_batch_size=4)
c["evaluation"]["eval_batch_size"] = 4
c["checkpoint"]["base_dir"] = sys.argv[3]
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
EOF
    "$PY" train/train_multi_fact.py --config "$OUT/configs/toy_multi.yaml"
    "$PY" - "$(toy_run)" <<'EOF'
import json, os, sys
run = sys.argv[1]
for f in ("config.yaml", "experiment_data.json", "original_fact.json", "training_logs.json",
          "checkpoint-last-epoch1/adapter_model.safetensors"):
    assert os.path.exists(os.path.join(run, f)), f"missing {f}"
logs = json.load(open(os.path.join(run, "training_logs.json")))
keys = {"train/epoch", "eval/memorization", "eval/chaining"}
assert logs and all(set(r) == keys for r in logs), logs
assert logs[0]["train/epoch"] == 0 and logs[-1]["train/epoch"] == 1, logs
print(f"OK {os.path.basename(run)}: training_logs.json has {sorted(keys)}; epoch 1: "
      f"memorization {logs[-1]['eval/memorization']:.2f}, chaining {logs[-1]['eval/chaining']:.2f}")
EOF
}

step4() {
    "$PY" - train/configs/experiment_configs_specific_samples/multi_specific_sample_template.yaml \
        "$OUT/configs/per_fact.yaml" "$OUT/per_fact/lora" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["sft"]["num_train_epochs"] = 2
c["checkpoint"]["base_dir"] = sys.argv[3]
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
EOF
    # the paper's 100 facts (seed 42), then the first two of them for 2 epochs
    "$PY" train/run_specific_samples.py --config "$OUT/configs/per_fact.yaml" --gpus "$GPU" --dry_run \
        --output "$OUT/per_fact_ids/batch_results.json" --log_dir "$OUT/per_fact_ids"
    "$PY" train/run_specific_samples.py --config "$OUT/configs/per_fact.yaml" --gpus "$GPU" --sample_ids 106,409 \
        --output "$OUT/per_fact_logs/batch_results.json" --log_dir "$OUT/per_fact_logs"
    "$PY" train/knowing_using_gap.py "$OUT/per_fact/lora" --json "$OUT/per_fact/gap.json"
    "$PY" - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
ids = json.load(open(f"{out}/per_fact_ids/batch_results_sample_ids.json"))["sample_ids"]
assert len(ids) == 100 and ids[:5] == [106, 409, 434, 488, 520], ids[:5]
print(f"OK seed 42 picks 100 facts: {ids[:5]} ...")
batch = json.load(open(f"{out}/per_fact_logs/batch_results.json"))
assert batch["successful"] == 2 and batch["failed"] == 0, batch
gap = json.load(open(f"{out}/per_fact/gap.json"))["qwen2.5-1.5b"]
assert gap["n_runs"] == 2 and gap["n_incomplete"] == 0, gap
print(f"OK facts 106, 409 trained; knowing_using_gap.py: {gap['n_runs']} runs, {gap['n_used']} used, "
      f"T_mem / T_gen / dT / A_mem / A_gen / dA: {gap['row']}")
EOF
}

step5() {
    "$PY" train/knowing_using_gap.py --multi "$OUT/multi" --json "$OUT/gap_multi.json"
    "$PY" - "$OUT/gap_multi.json" "$(toy_run)" <<'EOF'
import json, sys
res = json.load(open(sys.argv[1]))
r = res[sys.argv[2]]
assert r["model"] == "qwen2.5-1.5b" and r["n"] == 4 and r["epoch"] == 1, r
print(f"OK {len(res)} run(s); final epoch: memorization {r['memorization']:.2f}, chaining {r['chaining']:.2f}")
EOF
}

step6() {
    local grid=$OUT/patch/patching_results_entity_offset0.npy
    "$PY" patch/multi_fact_experiment_nsamples.py --ckpt_dir "$(toy_run)" --max_instances 2 --out "$grid"
    "$PY" patch/compute_oracle.py "$grid" --topk 3 --json "$OUT/patch/oracle.json"
    "$PY" - "$grid" "$OUT/patch/oracle.json" <<'EOF'
import json, sys
import numpy as np
a = np.load(sys.argv[1])
assert a.shape == (2, 2, 28, 28), a.shape  # Qwen2.5-1.5B has 28 layers
mrr = a[:, 0]
diag = np.stack([np.diagonal(g) for g in mrr])
assert np.allclose(diag, mrr[:, :1, 0], atol=1e-6), "patching a layer onto itself changed the MRR"
r = json.load(open(sys.argv[2]))[sys.argv[1]]
assert r["N"] == 2 and r["L"] == 28 and not r["diag_mismatch"], r
assert 0 <= r["em"]["no_patch"] <= r["em"]["oracle"] <= 1, r
print(f"OK grid {a.shape}, diagonal (no-op patch) == unpatched; "
      f"exact match no_patch {r['em']['no_patch']:.3f} -> oracle {r['em']['oracle']:.3f}")
EOF
}

step7() {
    local run
    run=$(toy_run)
    "$PY" validation/evaluate_checkpoints.py --run_dir "$run" --eval_batch_size 4 --output "$OUT/eval/eval_toy.json"
    "$PY" - "$OUT/eval/eval_toy.json" <<'EOF'
import json, sys
res = json.load(open(sys.argv[1]))
assert len(res) == 1 and list(res[0]["checkpoints"]) == ["checkpoint-last-epoch1"], res
c = res[0]["checkpoints"]["checkpoint-last-epoch1"]
assert all(c[t]["num_samples"] > 0 and 0 <= c[t]["accuracy"] <= 1 for t in ("memorization", "chaining")), c
print("OK checkpoint-last-epoch1: " + ", ".join(
    f"{t} {c[t]['accuracy']:.2f} ({c[t]['num_samples']} samples)" for t in ("memorization", "chaining")))
EOF
}

step8() {
    local arm
    for arm in B:0 P21-14L1:1; do
        "$PY" distill/train_inject_lrsd.py --model qwen2.5-1.5b --lam "${arm#*:}" --seed 848 --epochs 1 \
            --save_epochs '' --out_dir "$OUT/lrsd/inject_${arm%%:*}__qwen2.5-1.5b_s848"
    done
    "$PY" distill/analyze_scale.py --runs_dir "$OUT/lrsd" --models qwen2.5-1.5b --seeds 848 --late 0-1 \
        --out "$OUT/lrsd/scale.json"
    "$PY" - "$OUT/lrsd" <<'EOF'
import json, sys
d = sys.argv[1]
run = f"{d}/inject_P21-14L1__qwen2.5-1.5b_s848"
assert json.load(open(f"{run}/args.json"))["pair"] == "21-14"
ep1 = [json.loads(x) for x in open(f"{run}/log.jsonl") if x.strip()][-1]
assert ep1["epoch"] == 1 and ep1["train_lrsd"] > 0, ep1
print(f"OK lam 1, epoch 1: chain_uq_em {ep1['chain_uq_em']:.4f}, train_ce {ep1['train_ce']:.4f}, "
      f"train_lrsd {ep1['train_lrsd']:.4f} (paper run: 0.0196, 1.6506, 0.3242)")
st = json.load(open(f"{d}/scale.json"))["settings"]["qwen2.5-1.5b|chaining"]
assert st["n_q"] == 409 and st["L1"]["paired_seeds"] == ["848"], st
print(f"OK analyze_scale.py: late EM (epochs 0-1) B {st['B']['late']['mean']:.3f}, "
      f"lam 1 {st['L1']['late']['mean']:.3f} on {st['n_q']} unique questions")
EOF
}

step9() {
    snapshot > "$OUT/logs/snapshot_after.txt"
    diff "$OUT/logs/snapshot_before.txt" "$OUT/logs/snapshot_after.txt"
    ok "${#RELEASE_FILES[@]} release files unchanged, git status of the release paths unchanged"
}

SUMMARY=()
N_FAIL=0
run_step() {  # run_step <n> <title>: runs step<n> in a subshell with errexit; its output goes to logs/<n>.log
    local n=$1 title=$2 log=$OUT/logs/$1.log t0=$SECONDS rc=0 line
    echo "[$n] $title"
    set +e
    (set -euo pipefail; "step$n") > "$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        line="PASS [$n] $title ($((SECONDS - t0)) s)"
        echo "$line"
        sed -n 's/^OK /       /p' "$log"
    else
        line="FAIL [$n] $title ($((SECONDS - t0)) s, exit $rc, log: $log)"
        N_FAIL=$((N_FAIL + 1))
        echo "$line"
        tail -n 20 "$log" | sed 's/^/       | /'
    fi
    SUMMARY+=("$line")
}

echo "repo: $REPO"
echo "outputs: $OUT"
echo "python: $(command -v "$PY"), GPU $GPU"
snapshot > "$OUT/logs/snapshot_before.txt"
T0=$SECONDS
run_step 0 "release .py files compile"
run_step 1 "seeded chaining samples (seeds 848, 1340)"
run_step 2 "distill/make_inject_data.py"
run_step 3 "multi-fact training, 4 facts, 1 epoch"
run_step 4 "per-fact runs, 2 facts, 2 epochs + knowing_using_gap.py"
run_step 5 "knowing_using_gap.py --multi"
run_step 6 "self-patching grid, 2 questions + compute_oracle.py"
run_step 7 "evaluate_checkpoints.py --run_dir"
run_step 8 "LRSD, 1 epoch per arm + analyze_scale.py"
run_step 9 "no release file modified"

echo
echo "== summary: $((${#SUMMARY[@]} - N_FAIL)) passed, $N_FAIL failed, $((SECONDS - T0)) s (outputs: $OUT)"
printf '%s\n' "${SUMMARY[@]}"
[ "$N_FAIL" -eq 0 ]
