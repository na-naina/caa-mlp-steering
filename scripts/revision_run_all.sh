#!/usr/bin/env bash
# End-to-end driver for the October 2026 revision runs on a 2x32GB box.
#   phase A: training jobs, one at a time, sharded over all GPUs
#   phase B: generation jobs, 2 per GPU
#   phase C: open (AllenAI) judge scoring of every generation file
# Re-running resumes: finished jobs are listed in jobs_*.txt.done.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PWD/cache/transformers}
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
[ -f "$HF_HOME/token" ] && export HF_TOKEN=$(cat "$HF_HOME/token")
mkdir -p logs
.venv/bin/python scripts/revision_queue.py --out-prefix jobs
echo "[$(date)] phase A (train)"; bash scripts/jobqueue.sh jobs_train.txt 0
echo "[$(date)] phase B (generate)"; bash scripts/jobqueue.sh jobs_gen.txt 2
echo "[$(date)] phase C (open judges)"
.venv/bin/python scripts/judge_open.py data/outputs/rcv_* data/outputs/rg4b_* data/outputs/rcathold_* > logs/judge_open.log 2>&1
echo "[$(date)] ALL PHASES DONE rc=$?"
