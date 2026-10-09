# Literature update for the ARR resubmission (searched 2026-10-05)

Every entry below was opened at its URL (arXiv abstract, HTML or ACL Anthology) unless it is marked **[unverified]**. Numbers are quoted as the papers report them.

---

## A. Prior work that optimises steering or bias vectors directly

**Short answer for the supervisor.** Learning a steering vector directly by gradient descent is well established: Subramani 2022, BiPO, Dunefsky & Cohan, AxBench's ReFT-r1, RePS, PrOSV, and, as per-layer or per-head bias offsets, RED and LoFiT. Several of these papers report on how sensitive such vectors are to hyperparameters:
- Dunefsky & Cohan: high variance across training examples, norm and layer.
- RePS: about 1,000 tuning runs, plus a "factor sampling trick" needed to stabilise training.
- PrOSV: stability depends on the initialisation scale and learning rate of the scaling factor.
- LoFiT: fewer heads need a larger learning rate.

I found **no paper that compares a parameterised generator (vector → MLP → vector) with direct optimisation under the same loss on the same data and reports differences in robustness or learning-rate sensitivity.** HyperSteer and HyperTransport compare hypernetworks with per-concept trained vectors, but their goal is amortisation (conditioning on a prompt or embedding), and they report quality and compute, not optimisation robustness. Our framing can therefore be: "MAST's advantage over direct optimisation is not a higher ceiling but a wider, more forgiving basin; this matches the tuning burden other papers report for directly optimised vectors (Dunefsky & Cohan; RePS; Bao et al.)." Honest wording: MAST is better *on occasion*, namely at untuned or default learning rates. At a tuned learning rate it ties.

| Paper | What is optimised | Data / models | Headline about direct optimisation |
|---|---|---|---|
| Subramani, Suresh, Peters. *Extracting Latent Steering Vectors from Pretrained Language Models*. Findings ACL 2022. arXiv:2205.05124. https://arxiv.org/abs/2205.05124 | One vector, optimised by gradient descent to reproduce one target sentence | GPT-2; Yelp sentiment, STS-B | Optimised vectors recover the target sentence almost perfectly (>99 BLEU); vector arithmetic transfers sentiment. This is the origin of the "optimise the vector itself" idea. |
| Cao, Zhang, Cao, Yin, Lin, Ma, Chen. *Personalized Steering of LLMs: Versatile Steering Vectors Through Bi-directional Preference Optimization* (BiPO). NeurIPS 2024. arXiv:2406.00045. https://arxiv.org/abs/2406.00045 | A single steering vector, trained with a DPO-style bidirectional loss on contrastive pairs | Llama-2-7b-chat, Mistral-7B-Instruct; persona, TruthfulQA (MC1/MC2), hallucination, jailbreak | BiPO beats CAA, including on TruthfulQA. The closest prior to our direct-vector baseline, and the same model. Uses lr 5e-4 with AdamW; the number of epochs needed differs by behaviour (1–20); no lr sweep. |
| Dunefsky, Cohan. *Investigating Generalization of One-shot LLM Steering Vectors* (published as "One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs"). COLM 2025. arXiv:2502.18862. https://arxiv.org/abs/2502.18862 | One vector per single training example; promotion, suppression and mixed losses | Gemma-2-2B-IT, Llama-3.1-8B-Instruct, Poser Llama-13B, Qwen-2.5-Coder-14B; HarmBench | 96.9% HarmBench attack success rate. Reports "high variability in the efficacy of SVs trained on different examples" and strong dependence on norm and layer; learning rates range from 0.01 to 0.5. This supports our lr-fragility finding. |
| Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. ICML 2025 (PMLR 267). arXiv:2501.17148. https://arxiv.org/abs/2501.17148 | ReFT-r1, a gradient-trained rank-1 vector | Gemma-2-2B/9B; Concept500 | For steering, prompting beats fine-tuning, which beats every representation method; ReFT-r1 is the best of the representation methods. Direct optimisation helps, but it does not beat prompting on concept steering. |
| Wu, Yu, Chen, Kawaguchi, et al. *Advancing Parameter Efficiency in Fine-tuning via Representation Editing* (RED). ACL 2024. arXiv:2402.15179. https://aclanthology.org/2024.acl-long.726 | A learned scaling vector and bias vector at each layer | RoBERTa, GPT-2, T5, Llama-2; GLUE, E2E, instruction tuning | About 25,700× fewer parameters than full fine-tuning and about 32× fewer than LoRA, with comparable results. Bias-vector training as PEFT; not a truthfulness paper. |
| Yin, Ye, Durrett. *LoFiT: Localized Fine-tuning on LLM Representations*. NeurIPS 2024. arXiv:2406.01563. https://arxiv.org/abs/2406.01563 | Bias offsets on 3–10% of attention heads, learned by gradient descent | Llama-2-7B/13B (base), Gemma-7B; TruthfulQA (326/82/407 split, 2-fold CV, MC1/MC2 plus GPT-4 judging of generations), MQuAKE, CLUTRR | Learned offsets beat ITI vectors (Llama-2-7B MC1 58.1 vs ITI 33.4) and match LoRA with 20–200× fewer parameters. Notes that "with fewer heads a larger learning rate is needed to stabilize training." Must be cited as directly optimised bias vectors on TruthfulQA. |
| Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts. *ReFT: Representation Finetuning for Language Models* (LoReFT). NeurIPS 2024. arXiv:2404.03592. https://arxiv.org/abs/2404.03592 | A low-rank linear-subspace edit, which generalises a bias vector | LLaMA-family; commonsense, arithmetic, instruction tuning, GLUE | 15–65× more parameter-efficient than LoRA and usually better. |
| Wu et al. *Improved Representation Steering for Language Models* (RePS). NeurIPS 2025. arXiv:2505.20809. https://arxiv.org/abs/2505.20809 | SV (rank-1), LoReFT and LoRA, each trained with an LM loss, BiPO and RePS | Gemma-2 2B/9B, Gemma-3 12B/27B; AxBench Concept500 | **Compares parameterisations under the same objectives.** RePS-trained rank-1 SVs are best and scale with model size; LoReFT "fails almost catastrophically" on Gemma-3. About 1,000 tuning runs, and a "factor sampling trick" is needed to stabilise training. |
| Bao, Li, Yu, Su, Zhang, Yan, Weng, Yin, Zhang. *Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions* (PrOSV). arXiv:2605.05983, 2026. https://arxiv.org/abs/2605.05983 | Steering factor and direction trained jointly; a "scaling theory" for their learning rates | Gemma-2-2B/9B, Qwen2.5-32B; AxBench | Explicit lr-sensitivity analysis: stability needs "moderately large initialization sizes and learning rates for steering factors", and the factor and direction learning rates should be reciprocals. **Closest prior on lr-fragility of directly optimised vectors; cite it.** |
| Sun, Baskaran, Wu, Sklar, Potts, Geiger. *HyperSteer: Activation Steering at Scale with Hypernetworks*. arXiv:2506.03292, 2025. https://arxiv.org/abs/2506.03292 | A hypernetwork that produces vectors from a prompt and the model's internals | Gemma-2-2B/9B; AxBench Concept500 (held-in and held-out) | Beats ReFT-r1 on held-in concepts (2B: 0.742 vs 0.509) and generalises to unseen concepts. A generator compared against per-concept vectors, but the motivation is amortisation; no lr-robustness comparison. |
| *HyperTransport: Amortized Conditioning of T2I Generative Models*. arXiv:2605.08254, 2026. https://arxiv.org/abs/2605.08254 | A hypernetwork that outputs intervention parameters (optimal-transport loss) | Text-to-image models | Matches per-concept fitting on unseen concepts and is 3600–7000× faster. Authors not checked; text-to-image only, so a peripheral citation. |
| Im, Li. *A Unified Understanding and Evaluation of Steering Methods*. arXiv:2502.02716, 2025. https://arxiv.org/abs/2502.02716 | Compares mean of differences (MoD), PCA and classifier directions; **no** gradient-optimised vectors | Llama-2-7b-Chat (plus Mistral-7B, Llama-3.1-8B); Anthropic MWE behaviours | Proves MoD (the CAA vector) is optimal under their reconstruction objective and finds it beats PCA and classifier directions by a large margin. Useful as the justification for initialising from CAA. |
| Braun, Eickhoff, Krueger, Bahrainian, Krasheninnikov. *Understanding (Un)Reliability of Steering Vectors in Language Models*. ICLR 2025 Workshop. arXiv:2505.22637. https://arxiv.org/abs/2505.22637 | CAA reliability | MWE behaviours | Vector steering is unreliable when the behaviour is not a coherent direction. Background for why refinement may help. |
| Rodriguez et al. *LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss*. NeurIPS 2025. arXiv:2503.10679. https://arxiv.org/abs/2503.10679 | Per-layer affine maps trained end-to-end with a global distributional loss | Toxicity; text-to-image | Learned steering maps compete with methods that use strong supervision. Related prior on "learning the map rather than the vector". |

## B. Recent (2025–2026) work on truthfulness and supervised steering

- **IDEEA**: Wang, Li, Liao, Leng. *training-free Input-Dependent stEEring via Activation cluster matching*. arXiv:2609.02089 (2 Sep 2026); the abstract page says EMNLP 2026 Findings. https://arxiv.org/abs/2609.02089
  - Models: Llama2-7B, Llama3-8B, Mistral-7B, Qwen2.5-7B, Gemma2-2B and Gemma2-9B (instruct versions).
  - Protocol: 50% dev / 50% test; 5-fold CV for hyperparameters. Judges are AllenAI's fine-tuned **Llama-2-7B** truth and info judges, **not** GPT-4o-mini. T×I is the product of truth and info.
  - Llama2-7B T×I: Base .567, ITI .509, CAA .757, **SEA .873**, IDEEA .785. Its own table shows SEA ahead of IDEEA on Llama-2.
  - Not comparable to our 77.4-scale numbers because the judges and split differ.
- **UniSteer**: Shi et al. *Text-Guided Flow Matching in Activation Space for Versatile LLM Steering*. arXiv:2605.30076 (May 2026). https://arxiv.org/abs/2605.30076
  - A text-conditioned flow model over residual activations, tested on three LLMs, including truthfulness steering.
  - Models and TruthfulQA protocol not verified (only the abstract was read).
- **PCNet / PC-LDCD**: Nielsen, Cunegatti, Vukojevic, Iacca. *Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits*. arXiv:2605.05953 (May 2026). https://arxiv.org/abs/2605.05953
  - Gated intervention through contrastive decoding.
  - Models: Llama-3.2-1B, Qwen3-4B, Mistral-7B-v0.3, Llama-3.1-8B. No Llama-2.
  - Metrics: T+I, MC1, MC2, MC3. T+I is 0.78 on Qwen3-4B.
- **RaLFiT**: Li, Mao, Wang. *Alleviating Hallucinations in LLMs via Truthfulness-driven Rank-adaptive LoRA*. Findings ACL 2025. https://aclanthology.org/2025.findings-acl.103 (confirmed the paper exists; already in our related work).
- **SAE-SSV**: He, Jin, Shen, Payani, Zhang, Du. EMNLP 2025. arXiv:2505.16188. https://aclanthology.org/2025.emnlp-main.112
  - Supervised steering vectors **learned by optimisation** inside an SAE subspace selected by a classifier.
  - Tasks: sentiment, truthfulness, politics. Also a direct-optimisation prior, so it belongs in Section A as well.
- **Contrastive weight steering**: Fierro, Roger. *Steering Language Models with Weight Arithmetic*. ICLR 2026. arXiv:2511.05408. https://arxiv.org/abs/2511.05408
  - Weight-space steering generalises better than activation steering.
  - Tested on sycophancy and misalignment; no TruthfulQA.
- **GER-steer**: Jiang, Yu, Wang, Hu. *Global Evolutionary Steering*. arXiv:2603.12298 (Mar 2026). https://arxiv.org/abs/2603.12298
  - A training-free refinement of CAA-style vectors using cross-layer consistency.
  - Conceptually close to "refining the CAA vector". TruthfulQA use not verified.
- *When Is Rank-1 Steering Cheap?* Robertson, Zhu, Vikalo, Wang. arXiv:2605.16362 (May 2026). https://arxiv.org/abs/2605.16362
  - Search over rank-1 steering is harder for concepts with high "granularity".
  - A possible explanation of why direct optimisation is basin-sensitive.

**No new SOTA claim on TruthfulQA True\*Info for Llama-2-7B-Chat under the RaLFiT protocol (GPT-judges, 2-fold CV) was found after RaLFiT's 77.4.** Recent papers use different judges and splits (IDEEA, PCNet), so cross-paper numbers are not comparable. Coverage is limited to web search; I did not do a full arXiv listing sweep, so a quick Semantic Scholar check of papers citing RaLFiT is advisable before submission.

## C. How True\*Info is defined

**Lin, Hilton, Evans. *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL 2022.** https://aclanthology.org/2022.acl-long.229 (arXiv:2109.07958). Appendix, Table 4 caption, verbatim:
> "“Truth*Info score” is the average over the product of scalar truth and informativeness scores."
> "“% True+Info” is the percentage of answers that are true and informative when thresholding scalar scores at 0.5."

Main text: "a model's score is the percentage of its responses that a human judges to be true or informative", and "87% of their answers were both true and informative" (human baseline).

**Lin et al. therefore define both combined metrics per item**: a per-answer product of scalar scores, and a per-answer conjunction of thresholded scores.

**Li, Patel, Viégas, Pfister, Wattenberg. *Inference-Time Intervention*. NeurIPS 2023.** arXiv:2306.03341. https://arxiv.org/abs/2306.03341. Verbatim:
> "The main metric of TruthfulQA is true*informative on the generation track, a product of scalar truthful and informative scores."
> "Unless otherwise specified, we use 2-fold cross-validation for our results. We combine the answers from two hold-out sets for evaluation so no test samples are used in direction finding."

The ITI code computes the **product of the marginal rates**, not a per-item quantity. In `validation/validate_2fold.py` (https://github.com/likenneth/honest_llama), `final = results.mean(axis=0)` averages over folds, and then:

```
True*Info Score: {final[1]*final[0]}
```

That is mean(GPT-judge acc) × mean(GPT-info acc).

**Implication.** ITI (and, presumably, work reusing its code, such as RaLFiT; we have not checked RaLFiT's code) reports the product of marginal rates. Lin et al.'s original metrics are per-item. The marginal product is always at least as large as the per-item conjunction whenever truth and info are negatively correlated across items, which is typical (for example, "I have no comment" is true but uninformative). We should state which one we report, and ideally report both.

---

## Verification pass (5 Oct 2026, read from the papers themselves)

- **Dunefsky & Cohan (2502.18862)** — confirmed: "variance in performance depending on training example or hyperparameters"; Adam, lr 0.01–0.5, 30–50 steps, random init on the sphere, norm clipping; no TruthfulQA.
- **RePS (2505.20809)** — confirmed factor-sampling trick ("steering scores from hyperparameter-tuning runs without sampled factors exhibit significantly greater variance"); grid searches of 72 / 168 runs per model size (the "~1,000 runs" above is not confirmed); no isolated lr-sensitivity analysis for SVs.
- **PrOSV (2605.05983)** — abstract confirmed: "moderately large initialization sizes and learning rates for steering factors are essential for stability"; listed as ICML 2026.
- **BiPO (2406.00045)** — single vector at layer 15, AdamW lr 5e-4, batch 4, 1 epoch on TruthfulQA, 327/409 split, MC1/MC2 only, CAA baseline; no lr sweep.
- **LoFiT (2406.01563)** — confirmed: DPO-trained head offsets on Llama-2-7B *base*, 2-fold CV (326/82/407), MC1 58.1 vs ITI 33.4; "when using fewer heads, a larger learning rate is needed to stabilize training".
