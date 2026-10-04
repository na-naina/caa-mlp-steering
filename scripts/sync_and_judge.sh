#!/usr/bin/env bash
# Pull revision outputs from the GPU box, judge the core runs with the fine-tuned
# GPT-4o-mini judges, and regenerate every paper/thesis number.
#   BOX="root@HOST" PORT=NNNN bash scripts/sync_and_judge.sh [--no-judge]
# GPT judges are used only for the core set (Table 1, lr curves, Gemma, one-shot);
# the remaining runs are scored on the box with the open AllenAI judges
# (scripts/judge_open.py), whose open_judge_results.json files are synced too.
set -u
cd "$(dirname "$0")/.."
BOX=${BOX:-root@220.135.0.171}; PORT=${PORT:-59232}; KEY=${KEY:-$HOME/.ssh/vastai_key}
REMOTE=/workspace/caa-mlp-steering

rsync -az -e "ssh -p $PORT -i $KEY -o BatchMode=yes" \
  --include='*/' --include='*.json' --include='*.yaml' --include='**/vectors/*.pt' --exclude='lora_adapter/**' --exclude='*' \
  --prune-empty-dirs --info=stats1 "$BOX:$REMOTE/data/outputs/" data/outputs/
# open-judge scores of earlier (local) runs, re-scored on the box for calibration / category hold-out
rsync -az -e "ssh -p $PORT -i $KEY -o BatchMode=yes" --include='*/' --include='open_judge_results.json' --exclude='*' \
  --prune-empty-dirs "$BOX:/workspace/calib/" data/outputs/ 2>/dev/null
rsync -az -e "ssh -p $PORT -i $KEY -o BatchMode=yes" "$BOX:$REMOTE/logs/" logs/box/ 2>/dev/null

if [ "${1:-}" != "--no-judge" ]; then
  for f in rcv_main rcv_dvzero rcv_dvcaa rcv_loradpo rcv_mast_lr rcv_g4bmain rg4b_ roneshot_; do
    .venv/bin/python scripts/evaluate_with_gpt_judge.py evaluate --model "$f" -w 8
  done
fi
.venv/bin/python scripts/make_paper_numbers.py
D=paper/drafts/revision_oct2026
cp $D/paper/{numbers.tex,lr_table.tex,cathold_table.tex,gemma_table.tex} $D/thesis/msc/ 2>/dev/null
.venv/bin/python scripts/aggregate_revision.py --judge gpt --out paper/figures/revision_oct2026 > /dev/null
.venv/bin/python scripts/aggregate_revision.py --judge open --out paper/figures/revision_oct2026 > /dev/null
echo "done: see paper/figures/revision_oct2026/revision_results_{gpt,open}.md"
