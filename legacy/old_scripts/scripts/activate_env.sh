#!/bin/bash
# Helper to prepare the experiment environment on Blythe
set -euo pipefail

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

PROJECT_ROOT=${PROJECT_ROOT:-/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma}
cd "$PROJECT_ROOT"

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

CODex_BIN_DIR="/springbrook/share/dcsresearch/u5584851/.npm/bin"
if [ -d "$CODex_BIN_DIR" ]; then
  export PATH="$CODex_BIN_DIR:$PATH"
fi

if [ -f "$PROJECT_ROOT/setup_env.sh" ]; then
  source "$PROJECT_ROOT/setup_env.sh"
fi

echo "Environment ready. PYTHONPATH=$PYTHONPATH"
