# TruthfulQA Results: Comparison with RaLFiT (ACL 2025)

All results on Llama2-7B-chat. Generation metrics evaluated by fine-tuned GPT-4o-mini (same setup as ours).

## Main Results Table

| Method | Type | MC1 (%) | MC2 (%) | AVG (%) | True (%) | Info (%) | T*I (%) |
|--------|------|--------:|--------:|--------:|---------:|---------:|--------:|
| **Llama2-chat** | Baseline | 33.66 | 51.29 | 42.48 | 64.14 | 85.07 | 54.56 |
| | | | | | | | |
| *Contrastive Decoding* | | | | | | | |
| SH2 | CD | 33.90 | 57.07 | 45.49 | 64.38† | 65.59† | 42.23† |
| ICD | CD | 46.32 | 69.08 | 57.70 | - | - | - |
| | | | | | | | |
| *Representation Editing* | | | | | | | |
| ITI | RepEdit | 34.64 | 51.55 | 43.10 | 65.73 | 83.47 | 54.86 |
| TrFr | RepEdit | 36.70 | - | - | 67.44† | 80.91† | 54.56† |
| TruthX | RepEdit | 54.22 | 73.90 | 64.06 | 67.81 | 92.66 | 62.83 |
| | | | | | | | |
| *Fine-tuning Methods* | | | | | | | |
| RED | PEFT | 48.60 | 66.98 | 57.79 | 80.29 | 88.24 | 70.85 |
| LoFiT | PEFT | 54.50 | 74.90 | 64.70 | - | - | - |
| LoRA (SFT) | PEFT | 41.01 | 58.74 | 49.88 | 69.52 | 83.85 | 58.29 |
| LoRA (DPO) | PEFT | 57.78 | 75.24 | 66.51 | 81.64 | 93.76 | 76.54 |
| GRATH | PEFT | 54.71 | 69.10 | 61.91 | - | - | - |
| AdaLoRA | PEFT | 45.96 | 65.84 | 55.90 | 79.80 | 87.88 | 70.13 |
| Sora | PEFT | 56.80 | 74.31 | 65.56 | 81.76 | 92.90 | 75.95 |
| **RaLFiT** | PEFT | **60.22** | **77.76** | **68.99** | **82.98** | 93.27 | **77.40** |
| | | | | | | | |
| *Our Methods (Inference-Time)* | | | | | | | |
| CAA Steering | RepEdit | - | - | - | - | - | - |
| MLP-Gen | RepEdit | - | - | - | - | - | - |
| MLP-MC | RepEdit | - | - | - | - | - | - |

† Evaluated with fine-tuned GPT-3 (deprecated), not directly comparable.

## Notes

- **RaLFiT** (Rank-adaptive LoRA Fine-tuning): Their SOTA, requires training
- **Our methods**: Inference-time interventions, no training on target model
- All our methods use the same GPT-4o-mini judges as RaLFiT

## Method Categories

| Category | Training Required | Model Modification |
|----------|-------------------|-------------------|
| Contrastive Decoding | No | No |
| Representation Editing | No (probe only) | No |
| Fine-tuning (PEFT) | Yes | Yes (adapters) |
| **Ours** | No* | No |

*Our MLP is trained on steering vectors, not the target model itself.

## Key Comparison Points

1. **Fair comparison**: Same evaluation setup (GPT-4o-mini judges)
2. **Same model**: Llama2-7B-chat
3. **Different approach**: They fine-tune the model, we steer at inference time
4. **Advantage if competitive**: No model modification, works on any model

---

*Source: Li et al., "Alleviating Hallucinations in Large Language Models via Truthfulness-driven Rank-adaptive LoRA" (ACL 2025 Findings)*
