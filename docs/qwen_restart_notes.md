# Qwen3-8B restart notes (July 2026)

Qwen3-8B training OOM'd repeatedly on 2×RTX 4090 (24GB each) and was deferred.
Use a single ≥48GB card (A6000/L40S/A100) next time — it will run first-try.

## Why it OOM'd on 2×24GB

- `device_map="auto"` balances by weights; the 152k-vocab lm_head + upper
  layers stack on GPU 1, which sat at 23.45/23.52 GiB and failed on ~96MB.
- Gradient checkpointing (already wired via `mc_training.gradient_checkpointing: true`)
  got it close but not reliably under the limit at batch 2.
- The fragmentation mitigation was silently ignored: torch ≥2.9 renamed the env
  var — use `PYTORCH_ALLOC_CONF=expandable_segments:True`
  (NOT the deprecated `PYTORCH_CUDA_ALLOC_CONF`).

## Ready-to-run setup

Configs already in repo: `configs/models/qwen3_8b_bn8_L{7,9,14,16}.yaml`
(paper recipe: bn=8, no bank, lr 5e-4; batch 2 + grad-accum 4 +
steps_per_epoch 200 to keep the effective schedule at 100 optimizer steps of
effective batch 8; gradient_checkpointing on).

On a 48GB+ card, simplify: batch_size 8, no accumulation, steps_per_epoch 50,
checkpointing optional.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
# layer sweep (train-only, pick by train-signal + Info sanity check; see
# paper/reviews/arr_jul2026_response_drafts.md on why train-signal alone
# can select Info-eroding configs)
for L in 7 9 14 16; do
  python run.py --model qwen3_8b_bn8_L$L --stage train-only --seed 42 \
    --output-dir data/outputs/q8b_sweep/L$L
done
# then full run at the picked layer:
python run.py --model qwen3_8b_bn8_L<PICK> --stage train-only --seed 42 --output-dir data/outputs/q8b_bn8_full
python run.py --model qwen3_8b_bn8_L<PICK> --stage generate  --seed 42 --run-dir    data/outputs/q8b_bn8_full
# judge locally:
python scripts/evaluate_with_gpt_judge.py evaluate --model q8b_bn8_full --workers 4
```

Context for interpreting results: gemma3-4b with this recipe went
baseline 53.7 → raw CAA 54.7 → MAST 88.2 (n=408, seed 42, L13 of 34).
Qwen baselines are penalized by the raw QA prompt format (they self-QA loop;
~94% truncation-cleaning on small Qwen) — consider a chat-template variant as
a secondary condition if Qwen numbers look weak.
