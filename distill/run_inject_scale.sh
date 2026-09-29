#!/bin/bash
# In-training LRSD on chaining: 4 models (Qwen2.5-1.5B/3B, LLaMA-3.2-1B/3B) x {baseline lam 0, lam 1, lam 0.1}
# x seeds {848, 1, 2}; layer pair 0.75L -> 0.5L (Qwen2.5-3B 27->18, Qwen2.5-1.5B / LLaMA-3.2-3B 21->14,
# LLaMA-3.2-1B 12->8).  36 jobs, see make_jobs_scale.py; 30-65 min per run on an A800.
# Memory: up to ~19 GB per 1B/1.5B run and ~31 GB per 3B run, so 2 runs share an 80 GB GPU; a job starts only on a
# GPU with >= 30 GB free and each GPU takes the next job as soon as one of its runs finishes.  Finished runs (log at
# epoch 50) are skipped, so running this script again resumes after an interruption (a killed run restarts from
# scratch).
#
# Usage (with the Python environment activated): [GPUS=0,1,2,3] bash distill/run_inject_scale.sh
# Environment variables (default):
#   GPUS           comma-separated GPU ids (0)
#   MAX_PARALLEL   concurrent runs per GPU (2)
#   RUNS_DIR       run directories, relative to distill/ (../experiment_logs/distill/inject)
#   MODELS, SEEDS  subsets of qwen2.5-1.5b,qwen2.5-3b,llama3.2-1b,llama3.2-3b and 848,1,2 (all)
#   MIN_FREE_GB    free memory a GPU needs before a job starts on it (30; ~20 is enough for 1B/1.5B models only)
# Runs: $RUNS_DIR/<job name>/ (log.jsonl, stdout.log, adapter_epoch50/); summary: python distill/analyze_scale.py
set -e
cd "$(dirname "$0")"
GPUS=${GPUS:-0}
MAX_PARALLEL=${MAX_PARALLEL:-2}
RUNS_DIR=${RUNS_DIR:-../experiment_logs/distill/inject}
MODELS=${MODELS:-qwen2.5-1.5b,qwen2.5-3b,llama3.2-1b,llama3.2-3b}
SEEDS=${SEEDS:-848,1,2}
MIN_FREE_GB=${MIN_FREE_GB:-30}

# the injected 1000-fact samples (Qwen seed 848, LLaMA seed 1340), drawn from data/multi_fact_data_template;
# existing samples are only verified
python make_inject_data.py
python make_jobs_scale.py --models "$MODELS" --seeds "$SEEDS" --out jobs_lrsd_chaining.json
python run_queue.py --script train_inject_lrsd.py --jobs jobs_lrsd_chaining.json \
  --runs_dir "$RUNS_DIR" --gpus "$GPUS" --max_parallel "$MAX_PARALLEL" --min_free_gb "$MIN_FREE_GB"
