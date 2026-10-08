#!/usr/bin/env bash
# Minimal multi-GPU job queue. Each line of JOBS is "NAME<TAB>COMMAND".
# Runs SLOTS concurrent jobs per visible GPU (SLOTS=0: one job at a time that
# sees every GPU, for sharded training); finished/failed names are
# appended to JOBS.done / JOBS.failed, so re-running skips completed jobs.
#   bash scripts/jobqueue.sh jobs.txt 3
set -u
JOBS=$1; SLOTS=${2:-2}
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p logs/queue; touch "$JOBS.done" "$JOBS.failed"
LOCK="$JOBS.lock"; CURSOR="$JOBS.cursor"; echo 0 > "$CURSOR"

next_job() {  # atomically advance the cursor; print the job line or nothing
  exec 9>"$LOCK"; flock 9
  local n; n=$(cat "$CURSOR")
  while :; do
    n=$((n+1))
    local line; line=$(sed -n "${n}p" "$JOBS")
    [ -z "$line" ] && { echo "$n" > "$CURSOR"; flock -u 9; return; }
    local name=${line%%$'\t'*}
    grep -qxF "$name" "$JOBS.done" && continue
    echo "$n" > "$CURSOR"; flock -u 9; echo "$line"; return
  done
}

worker() {
  local gpu=$1   # "all" = leave CUDA_VISIBLE_DEVICES unset
  while :; do
    local line; line=$(next_job)
    [ -z "$line" ] && break
    local name=${line%%$'\t'*} cmd=${line#*$'\t'}
    echo "[$(date +%T)] gpu$gpu START $name"
    local vis=""; [ "$gpu" != all ] && vis="CUDA_VISIBLE_DEVICES=$gpu"
    # GEN_BATCH_SIZE: batched TruthfulQA generation (src/evaluation/truthfulqa.py), ~10x faster;
    # greedy outputs match the per-item path on 42/48 (LLaMA) and 28/32 (Qwen3.5) questions.
    env $vis PYTORCH_ALLOC_CONF=expandable_segments:True GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-8} \
      bash -c "$cmd" > "logs/queue/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "$name" >> "$JOBS.done"; else echo "$name rc=$rc" >> "$JOBS.failed"; fi
    echo "[$(date +%T)] gpu$gpu END $name rc=$rc"
  done
}

if [ "$SLOTS" -eq 0 ]; then
  worker all &
else
  for g in $(seq 0 $((NGPU-1))); do
    for s in $(seq 1 "$SLOTS"); do worker "$g" & sleep 20; done
  done
fi
wait
echo "[$(date +%T)] QUEUE DRAINED: $(wc -l < "$JOBS.done") done, $(wc -l < "$JOBS.failed") failed"
