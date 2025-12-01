#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <user@host> <remote_path>" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_PATH="$2"

SYNC_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=venv/"
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=outputs/"
  "--exclude=.mypy_cache/"
)

rsync -avz "${SYNC_EXCLUDES[@]}" ./ "$REMOTE:${REMOTE_PATH}/"

ssh "$REMOTE" "REMOTE_PATH='$REMOTE_PATH' bash -s" <<'REMOTE_CMDS'
set -euo pipefail
cd "$REMOTE_PATH"
module purge
module load Python/3.10.4-GCCcore-11.3.0 || module load python/3.10
module load CUDA/12.1.1 || true
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs outputs cache/datasets
REMOTE_CMDS

cat <<EOF_MSG
Remote environment prepared.
Next steps:
  1. ssh $REMOTE
  2. source $REMOTE_PATH/venv/bin/activate
  3. export PYTHONPATH=$REMOTE_PATH
  4. Run: python -m src.jobs.run_experiment --config configs/gemma2-2b.yaml
EOF_MSG
