# ARR July 2026 — Reviewer Response Material (Submission 16990)

Working material for the Google doc. Tags per point:

- **[RESTATE]** — already in the paper; response mostly points to it.
- **[WRITING]** — needs a text/table change, no new compute.
- **[ANALYSIS]** — new analysis of *existing* artifacts (no GPU training).
- **[EXPERIMENT]** — needs new runs (GPU: 40GB+ for training; 24GB fine for generation).

---

## Part 0 — Facts verified against the codebase (basis for everything below)

1. **T×I metric — reviewer upJj is factually correct, and the error is conservative.**
   The pipeline reports the per-item conjunction: fraction of answers judged *both* truthful and informative (`evaluate_with_gpt_judge.py:369-377`, stored as `truth_and_info_accuracy`; seed 42 = 0.8088 = 330/408). The prior-work rows reproduced from RaLFiT are product-of-means (their baseline row satisfies 64.14 × 85.07 = 54.56 exactly; RaLFiT's paper defines "True\*Info" without a formula but their numbers are products). Recomputed from the stored per-seed judgments: **per-item 77.9 ± 2.7; product-of-means 78.7 ± 2.4**. So the metric mixing *understates* MAST. Bonus bug: `bootstrap_ci.py` mixes estimands (point estimate = conjunction at line 53, bootstrap replicates = product at line 39) — fix before quoting CIs.

2. **Test split is per-seed.** The split shuffle is seeded by the run seed (`src/data/truthfulqa.py:101-105`); "reserved first" only stabilizes the test set *within* a seed (across pool/train-size sweeps). Each of the 10 seeds evaluates on a different 408-question split, so ±2.7 includes test-sampling variation as well as training variance. The paper's "conditional on the reserved test split" (§4) is misleading — fix. (This also means there is no 10-seed baseline/CAA distribution on matched splits — only mlp_mc was re-run per seed.)

3. **The headline runs really do feed the single fixed v_CAA to the MLP** (verified: `multiseed/seed_*/config.yaml` has `vector_bank.num_vectors: 0`; empty bank ⇒ `sample_interpolated` returns `base_vector`). So the reviewers' "f_θ only ever sees one fixed input" premise is accurate for the submitted system. At inference the MLP output is a constant vector (`run.py:319-323`). During training, dropout (p=0.1) over the k=8 hidden units makes the *optimization* stochastic — the one functional difference from directly optimizing a vector, besides the parameterization itself.

4. **The direct-vector baseline was already run** (`scripts/train_direct_vector.py`, same hinge margin loss, same splits, seed 42, judged on n=408):
   - init at v_CAA → Truth **92.4** / Info **48.3** / T×I **41.9**
   - random init (norm-matched) → 84.3 / 48.8 / **36.8**
   Direct GD on the vector maximizes Truth but collapses informativeness (over-steering) — strong preliminary evidence the bottleneck parameterization matters. **BUT it is not yet a fair comparison**: lr 0.01 (vs MAST's 5e-4), batch 4 (vs 8), and **no MSE anchor** (MAST has λ=0.01 ‖f(v)−v‖²). A reviewer would call this a strawman. The rebuttal-grade version: identical lr/batch/steps + λ‖v−v_CAA‖² anchor, ≥3 seeds (ideally 10). Whichever way it lands, it's reportable (see strategy).

5. **Noise ablation** (already in paper): 60.5 T×I. **LoRA-DPO** (repo's own): 65.7; **LoRA-DPO + MAST**: 79.7 (single seed). Raw CAA α-sweep peaks 58.3 (α=2).

6. **Geometry, measured across all 10 stored seeds (July 2026, from saved vectors):** the learned correction g = v_MAST − v_CAA is consistently orthogonal to v_CAA (cos(g, v_CAA) = 0.02 ± 0.02) but **large**: ‖g‖/‖v_CAA‖ = 3.5 ± 3.1, cos(v_CAA, v_MAST) = 0.40 ± 0.18. So the §3.2 "small correction / refinement" framing is geometrically wrong — training largely replaces the CAA direction. Meanwhile the per-seed corrections are strongly aligned with *each other* (pairwise cos 0.62 ± 0.03 in d=4096) while the per-seed CAA vectors are unstable (norms 0.35–3.03, pairwise cos wildly scattered). Narrative: supervision converges to a consistent truthful direction from a noisy CAA anchor; CAA matters as *initialization/anchor* (noise ablation: 60.5), not as the surviving direction. This tension must be addressed head-on in the revision — it also strengthens the case for the direct-vector baseline.

7. **Per-category breakdown is cheap.** Stored judge outputs lack a category field, but `metadata/splits.json` stores TruthfulQA question indices per split — join by index against the dataset's `category` column. No new generation or judging needed.

8a. **TRANSFER DEMONSTRATED (July 11): Gemma3-4B-IT, paper recipe (bn=8, no bank, lr 5e-4), layer L13/34 selected by train-signal only, n=408, seed 42:** baseline 53.7 T&I → raw CAA 54.7 (inert) → **MAST 88.2** (Truth 89.2 / Info 99.0; generations spot-checked clean). This is the recipe-level transfer result Z6VF's scope objection asked for — single seed and TruthfulQA-tuned judges, so before it goes in the paper: ~30-sample hand audit + 2 more seeds + ideally the zero-init-vector control on Gemma (does the parameterization matter there, or does supervision alone transfer too?). Artifacts: `data/outputs/g4b_bn8_full/`. Qwen3-8B attempt pending (OOM-fixed configs ready, box died).

8b. **Cross-model "failure" evidence is almost entirely invalid — the paper method was never properly run cross-model (forensics, July 2026).** Every judged Gemma/Qwen number came from the Jan-2026 sweep, which used the **fat 3-layer MLP** (e.g. 52.4M params on Qwen3-4B — verified from saved state dicts), i.e. the exact architecture already identified as mode-collapse-prone and replaced by bn=8 for the paper. The observed collapses (qwen3-0.6b Info 2%, gemma3-1b refusal-hacking at Info 16%) are the documented fat-MLP failure signature. Additional misconfigurations: intervention layer copied at ~50% depth (Llama uses 25%; the one per-model probe shows layer choice swings T×I by 20 pp), scale=1.0 despite base-vector norms spanning 2.5 (Llama) to 829 (Gemma3-27B), raw QA prompts (94% of qwen3-0.6b baseline generations needed truncation-cleaning), n=200, and the big-Gemma runs effectively undertrained (14 optimizer steps; bf16 NaNs documented in the thesis appendix). The only bn=8 cross-model run ever (Gemma3-12B L24, −1.4 T×I) has no training-health artifacts. **Positive signal exists even under the wrong recipe: gemma3-4b +21.0 T×I (53.0→74.0, Info intact), qwen3-8b +11.0, qwen3-14b +9.0.** A proper revival (bn=8 recipe on gemma3-4b/qwen3-8b/gemma3-12b, per-model layer + α selection, chat templates, n=408) ≈ 2 weeks on one 48GB GPU; the Gemma3-12B+ bf16-NaN issue is the main risk. Do NOT concede "does not transfer" in responses — say transfer experiments with the final recipe are ongoing/future work.

9. **Table 2 (capability cost) — RE-RUN AND NOW ARTIFACT-BACKED (July 2026).** The submitted numbers were bn=16-era hardcodes with lost JSONs. Fresh run with the exact stored k=8 seed-42 vector (`data/outputs/coherence_bn8_seed42/coherence_results.json`, lm-eval, both variants same protocol): ARC-E −6.4 (72.9→66.5), ARC-C −6.0 acc_norm (43.5→37.5), HellaSwag −2.0 acc_norm (76.0→74.1), MMLU −3.0 (46.5→43.4). Baselines match the paper to the decimal on HS/MMLU/ARC-C; deltas shift 1–2 pp from the submitted table. **Use these numbers in the revision** (qualitative claim unchanged: few-pp cost, HS nearly preserved). (With-LoRA mitigation numbers — ARC-E −3.6, ARC-C −1.1, HS −0.3, MMLU −1.2 — remain single-seed bn=16-era; re-run if cited.)

10. **Paper-internal inconsistency neither reviewer caught (fix quietly):** §3.1 says activations are "pooled by mean over the answer-token positions (not over the prompt)", but the code (`src/steering/extract.py:157-166`) and §4 ("mean over non-padding token positions") pool over *all* non-pad tokens including the question. §3.1 is wrong about our own implementation.

11. **Reproduction verdict (July 2026, 2×4090 vast.ai box).** `run.py` never seeded torch, so MLP init/dropout/decoding were uncontrolled in every historical run. An unseeded re-run of the seed-42 config on identical splits landed at T&I 68.4 (vs stored 80.9), with visibly worse convergence (last-10 loss 1.215 vs 0.892) — under torch 2.13. After pinning torch 2.9.1 and fixing seeding (`--seed` + new `--torch-seed` flag), **five same-split reruns give T&I 78.9 / 78.7 / 78.7 / 77.5 / 79.7 → 78.7 ± 0.8**, recovering the paper's 10-seed mean (77.9 ± 2.7); the stored seed-42 point (80.9) was a mildly favorable draw. Consequences for the revision: (a) report same-split variance (±0.8, seeded) alongside cross-split variance (±2.7); (b) the ±2.7 headline stands; (c) the unseeded-outlier episode argues for reporting the seeded protocol and releasing seeds.

12. **MC1/MC2 are now artifact-backed** (`scripts/eval_mc_harness.py`, lm-eval 0.4.12, per-sample records saved to `multiseed/seed_42/mc_harness_results.json`). Test-split (n=408, no chat template, harness default protocol): baseline MC1 30.15 / MC2 45.44; MAST (stored seed-42 vector) MC1 49.02 / MC2 69.93 → gains of +18.9 / +24.5. The paper's starred cells (26.96/45.12 baseline; 47.06/68.84 MAST; gains +20.1/+23.7) are 1–3 pp off in absolute terms — consistent with harness-version/protocol drift, the exact thing the ⋆ caveat warns about — but the within-protocol gains reproduce. For the revision: regenerate the starred cells from this pinned script so the numbers trace to a committed artifact, and cite the harness version.

---

## Part 1 — Global response strategy

> **⚠ FINAL BASELINE OUTCOME (July 11, complete — overrides earlier strategy bullets where they conflict).** The reviewers' baseline, fully explored: a zero-init direct vector reaches **80.2 ± 0.4** at lr 2e-3 (Info 96) — above MAST's 78.7 ± 0.8. **BUT the zero-init lr curve is a knife-edge**: 5e-4→69.4, 1e-3→73.8, 2e-3→80.2, 5e-3→72.5 with Info collapsing to 82.8 — and train-signal is monotone in lr (it would select 5e-3, i.e. 72.5 with degraded Info; the 80.2 peak is only locatable by consulting test scores). CAA-init behaves the same (peak 76.9 at 2e-3).
>
> **Final synthesis for the revision:** the contribution is *behaviorally supervised steering vectors* + a robustness finding: direct vector optimization can match/beat MAST but only at an oracle-tuned lr that no legitimate (train-signal) selection procedure can find; MAST at its default lr gets within ~1.5 pp of the oracle ceiling, unturned, Info-safe, across every run — i.e. **the bottleneck parameterization converts a knife-edge hyperparameter landscape into a usable one**. Report the 80.2 openly as the oracle-tuned ceiling in Table 1. Supporting item to add in the revision (cheap, 2 runs): a MAST lr mini-sweep (1e-3, 2e-3) to substantiate "flat basin" empirically, not just at one point. This resolves the (a)/(b) decision into a coherent (a′): reframe around supervision + selection-robustness, keep MAST as the recommended instantiation, disclose everything.

All three reviewers accept the within-protocol raw-CAA → MAST gap (~19–22 pp); upJj explicitly calls it "convincing." The two negative reviews rest on (i) the missing direct-vector baseline and (ii) statistics/metric hygiene around the SOTA-parity claim. Both are addressable within one cycle:

1. **Concede the direct-vector point; run the matched baseline; report it regardless of outcome.** We already have preliminary evidence (T×I 41.9) that naive direct optimization over-steers, but at mismatched hyperparameters. Two honest outcomes:
   - *Matched direct vector also collapses or lags* → the bottleneck+residual parameterization is doing real work (implicit regularization / optimization geometry); the paper's story stands and gets its missing pillar.
   - *Matched direct vector ≈ MAST* → reframe as "behaviorally supervised steering-vector optimization"; the deliverable becomes a single 4096-dim vector (even stronger parameter story), and every headline number survives. The MLP becomes "one convenient parameterization".
   Supporting argument either way: the bottleneck sweep (§5.3) already shows the parameterization is not inert — under a pure-reparameterization view, k should not matter, yet k=64 collapses Info% (62.3) while k=8 peaks (80.4/80.9).
2. **Fix the metric story**: state both formulas, report both aggregations (77.9 ± 2.7 conjunction / 78.7 ± 2.4 product-of-means), note ours is the stricter one.
3. **Adopt Z6VF's second horn on RaLFiT**: headline claim = matched-protocol improvement over tuned raw CAA; RaLFiT becomes context, "comparable under a different protocol", with a TOST in place of the inverted "cannot reject equivalence".
4. **Bank the cheap wins for 74Zo**: per-column bolding, category breakdown, geometry subsection.
5. **Release code + judge outputs + vectors** — all three reviewers scored Datasets/Software at 1 (one gave software 2). This is free credibility.

---

## Part 2 — Point-by-point drafts

### Reviewer 74Zo (Overall 3.0, Soundness 4.0 — keep them here, cheap to satisfy)

**(1) Bold best-per-column in Table 1** [WRITING]

> Agreed — we will bold the best score per column in Table 1 (the current bolding was row-emphasis for our method, which we agree is confusing).

**(2) "Is the 70K middle ground interesting when smaller do ok / larger do better?"** [WRITING]

> We take the point and will reframe: the interesting axis is not parameter count but *where the intervention lives*. MAST keeps the base model frozen and its inference-time footprint is a single 4096-dim vector added at one layer — removable per-query by setting α=0 — yet it recovers weight-space-level truthfulness. Among activation-space methods (base model untouched), the best published T×I on this model is TruthX's 62.8; MAST reaches 77.9 (78.7 under the product-of-means aggregation used by prior rows). The 70K MLP is training-time scaffolding, not the deliverable. We will revise §1/§2 to carry the comparison on the weight-space vs activation-space axis rather than raw parameter counts.

**(3) "ITI/CAA tell us about geometry; MAST does not"** [ANALYSIS — measured, July 2026]

> The refined vector remains a single interpretable direction in activation space, and we will add a geometry subsection with three measured findings (10 seeds, saved vectors): (i) the learned correction is nearly orthogonal to v_CAA (cos = 0.02 ± 0.02) — supervision adds a *new* direction rather than rescaling CAA; (ii) the correction dominates the final vector (‖g‖/‖v_CAA‖ ≈ 3.5; cos(v_CAA, v_MAST) = 0.40 ± 0.18); and (iii) the corrections found from different seeds are strongly mutually aligned (pairwise cos 0.62 ± 0.03 in d = 4096, where random directions would give ≈ 0) even though the CAA extractions themselves vary substantially across splits. Together these say the optimization converges to a consistent "truthful" direction that CAA initialization helps find (cf. the noise ablation) but does not itself contain. [Optional additions: overlap with top PCs of layer-8 activations; logit-lens of v_CAA vs v_MAST.]

Caution for us: this contradicts the current §3.2 "small correction" framing — the revision must reword "refines rather than replaces" (the residual form and λ-anchor shape the *optimization path*, not the final geometry). It also honestly strengthens the reviewers' demand for the direct-vector baseline, since the final vector is mostly learned, not extracted.

**(4) Per-category TruthfulQA breakdown (à la ITI)** [ANALYSIS — done July 2026, `scripts/category_breakdown.py`]

> We agree TruthfulQA is heterogeneous and will add the per-category breakdown (following Li et al.'s ITI analysis), computed from the same stored generations and judge decisions as Table 1. Pooled over the 10 seeds (4,080 judged answers), the pattern is informative: MAST's gains concentrate in "common falsehood" categories — Superstitions (91.4 T&I), Paranormal (92.7), Nutrition (98.7), Conspiracies (88.4) — and in categories where the unsteered model is confidently wrong (baseline→MAST: Economics 29.0→67.6, Sociology 52.7→77.6, Misconceptions 58.0→76.8, Health 67.3→85.6). It remains weak where answering requires precise recall of specific facts: Distraction (41.4), Logical Falsehood (49.3), Misquotations (50.7), Confusion: People/Places (57–61). This is consistent with a single global "decline the common falsehood" direction rather than added knowledge, and we will say so.

---

### Reviewer upJj (Overall 2.0, Soundness 2.5 — most precise review; concede facts, deliver experiments)

**(1) "MAST = reparameterized direct vector optimization; that baseline is absent; §5.3 rank claim is single-input sparsity"** [EXPERIMENT — DONE July 2026. Full result: reviewers are more right than the original framing, and the honest reframe is stronger.]

Complete measured picture (fixed seed-42 splits, identical loss/anchor/optimizer/steps/generation/judges; a bare vector trains deterministically, so per-config spread = replicate generations):

| Intervention | T&I | Info | cos(v_final, v_CAA) / ‖Δ‖ |
|---|---|---|---|
| Raw CAA, tuned α | 58.3 | 98.3 | 1.0 / 0 |
| Direct vector, lr 5e-4 (MAST-matched) | 68.1 ± 1.8 (×5) | 96.2 | 0.94 / 0.9 |
| Direct vector, zero init, lr 5e-4 | 69.4 ± 0.6 (×3) | 96.0 | 0.04 / 0.9 |
| Direct vector, no anchor, lr 5e-4 | 67.8 (×2) | 96 | 0.94 / 0.9 (anchor is inert) |
| Direct vector, lr 1e-3 | 74.3 | 95.3 | 0.84 / 1.6 |
| **Direct vector, lr 2e-3 (tuned)** | **76.9 ± 0.6 (×3)** | 95.5 | 0.68 / 2.7 |
| Direct vector, lr 5e-3 | 76.0 | 88.2 (eroding) | 0.44 / 5.2 |
| Direct vector, lr 1e-2 | 41.9 | 48.3 (collapsed) | — |
| **MAST (k=8), default lr** | **78.7 ± 0.8 (×5)** | 95.7 | 0.40 / ~3.3 |

Key facts: (i) a tuned-lr direct vector gets within ~1.8 pp of MAST (probably a real but small gap: every MAST replicate ≥ 77.5, every tuned-direct replicate ≤ 77.2); (ii) performance tracks displacement from v_CAA, and the direct vector's lr sits on a knife's edge (undertrained → peak → Info-erosion → collapse across 20×), while the MLP reaches the right displacement at its default lr with Info intact; (iii) supervision, not the MLP, contributes most of the raw-CAA gap.

Draft response:

> We thank the reviewer for insisting on this baseline — we ran it comprehensively and it sharpens the paper's claims. Directly optimizing the vector with the identical loss, anchor, and schedule reaches T×I 68.1 ± 1.8 at MAST's learning rate; sweeping the direct vector's learning rate recovers most of the gap, peaking at 76.9 ± 0.6 (lr 2e-3) vs MAST's 78.7 ± 0.8 — while at 2.5× that lr informativeness erodes (88.2) and at 5× it collapses (T×I 41.9). We will revise the paper's framing accordingly: the central finding is that *behavioral supervision of the steering vector* closes the gap from training-free CAA to weight-space methods; the bottleneck MLP is a parameterization that reaches the best result without per-run learning-rate tuning and with no informativeness-collapse mode anywhere in its observed range (its default lr transfers across all 15 runs we report), plus a small residual advantage (+1.8 pp, replicate-separated) over the best tuned direct vector. Both baselines requested by the reviewers (direct-at-CAA and learned-bias-from-scratch, which reaches 69.4 ± 0.6 at matched lr [zero-init at tuned lr pending — slot in]) will appear in Table 1, with the lr sweep in the appendix.
> On §5.3's rank claim: correct — the unit-sparsity observation characterizes the learned solution at its single input, not generalization; we will reword accordingly.

*Internal note:* this supersedes the earlier "MLP wins decisively (+10.6)" version — that gap was an artifact of matching lr. Abstract/intro must adopt the reframe (Part 1 outcome 2): lead with supervised steering-vector optimization; sell the MLP as the robust default parameterization, not as irreplaceable.

**Selection-protocol addendum (July 10, late) — the strongest form of the MLP defense.** Train-signal (final loss / margin accuracy) improves MONOTONICALLY with the direct vector's lr: 5e-4 → 1.083/0.600, 1e-3 → 0.914/0.650, 2e-3 → 0.698/0.762, 5e-3 → 0.368/0.875. So the paper's legitimate hyperparameter-selection protocol (train-time signals only, as used for layer and k selection) would select lr 5e-3 for the direct vector → test T&I 76.0 with Info visibly eroded to 88.2. The 76.9 peak at 2e-3 is only identifiable by consulting test T×I, which MAST never got to do. Meanwhile MAST at its default lr has WORSE train metrics (0.960/0.600) than every tuned direct vector yet the best test T&I (78.7 ± 0.8, Info ≥ 95). Framing for the revision: the direct vector optimizes the margin objective harder and converts the surplus into truth-hacking (Truth ↑, Info ↓ — same signature as k=64 and the lr 1e-2 collapse); the bottleneck acts as a regularizer against gaming the margin loss, not as a better optimizer. This "train-harder, test-worse" generalization-gap analysis should become a subsection — it unifies the k-sweep, the lr sweep, and the Info-collapse modes, and it answers upJj-1/Z6VF-2 on the reviewers' own terms.

**(2) "Statistical parity overstated; no RaLFiT variance; equivalence claim inverts NHST; no TOST"** [WRITING + ANALYSIS]

> We agree and will make three changes. (i) The abstract/intro claim becomes the within-protocol one: raw CAA (tuned α) 58.3 → MAST 77.9 under identical splits, judges, and decoding; the RaLFiT comparison is presented as context with the protocol asymmetry stated. (ii) We remove the "cannot reject equivalence" sentence — the reviewer is right that it inverts the burden of proof. (iii) We add a TOST against RaLFiT's (single-run, variance-unavailable) point estimate: the 90% CI of MAST's mean T×I over 10 seeds is 77.9 ± 1.6 = [76.3, 79.5]; under ±3 pp equivalence bounds TOST rejects both one-sided nulls; under ±2 pp it does not. We will report exactly this, so the strongest supportable statement is "within 3 pp of RaLFiT's reported number under a different protocol", not parity. [Recompute after finalizing the metric aggregation; under product-of-means the CI is 78.7 ± 1.4 ≈ [77.3, 80.1].]

**(3) "Undisclosed/inconsistent T×I: prior rows Truth×Info, MAST per-item mean (Tables 4–5)"** [WRITING + ANALYSIS — concede; it's conservative]

> Correct, and we thank the reviewer for catching it. Our pipeline computes the per-item conjunction (fraction of answers judged both truthful and informative), while rows reproduced from Li et al. (2025) are the product of the two marginal rates; we failed to state either formula. In the revision we define both and report MAST under both: per-item 77.9 ± 2.7, product-of-means 78.7 ± 2.4. Because the judgments are negatively correlated on our outputs, the per-item conjunction is the stricter aggregation — the inconsistency worked against our method. All MAST-side tables will switch to [choose one; recommend product-of-means for the RaLFiT-comparable table, conjunction in the appendix] with the formula in §4.

**(4) "Table 2 reports capability drops only vs baseline, never vs RaLFiT/LoRA"** [WRITING (+ internal re-run, see Part 0 #9)]

> We will add the corresponding numbers from Li et al. (2025, Table 2): under their (few-shot leaderboard) protocol, RaLFiT and LoRA-DPO are approximately capability-neutral (e.g., ARC 53.7→58.1, HellaSwag 78.6→79.8, MMLU 47.3→46.8), whereas MAST costs 2–8 pp under our 0-shot protocol. The protocols differ, so we will present signs and magnitudes side-by-side with that caveat rather than a merged table. The honest summary — which we will state explicitly — is that a strong fixed activation intervention has a real capability cost that DPO-trained adapters do not; mitigations (α=0 per-query switch-off, pairing with a LoRA adapter — our preliminary combined run recovers most of the loss [ARC-E −3.6, ARC-C −1.1, HellaSwag −0.3, MMLU −1.2, single seed]) are discussed in Limitations.

**(5) Single-seed ablations, no CIs** [ANALYSIS + EXPERIMENT]

> All single-seed cells will gain per-item bootstrap 95% CIs over the 408-question test split (judgments are stored). Additionally we will run 3 seeds for the two ablations that carry headline claims: the noise-input ablation (§5.2) and the bottleneck sweep (§5.3).

**(6) Typo** [WRITING] — "can be summaries as follows" → "summarized" (L119). Also fixed elsewhere per other reviews.

---

### Reviewer Z6VF (Overall 1.5, Soundness 1.5 — same core objection + scope; the baseline + reframe answers most of the soundness score)

**(1) "Narrow contribution, broad framing"** [WRITING]

> We will narrow the framing to match the evidence: the contribution is a supervised refinement procedure for CAA steering, demonstrated on truthfulness with LLaMA-2-7B-Chat; the title already scopes to truthful QA and the abstract will drop "matching the published state-of-the-art" in favor of the matched-protocol gain over tuned CAA. Transfer to other models/behaviors is future work and will be labeled as such.

(Internal: this was already the supervisor-agreed direction in April — the reframe is aligned, just wasn't executed hard enough in the submitted abstract/positioning.)

**(2) Direct-vector baseline + learned-bias baseline; "the trainable output bias makes this comparison even more important"** [EXPERIMENT]

> Both requested baselines are now run (see response to upJj-1 for the full table): (a) direct vector at v_CAA init with identical loss/anchor/schedule — 68.1 ± 1.8 at MAST's lr, 76.9 ± 0.6 with its own lr tuned, vs MAST 78.7 ± 0.8; (b) learned vector from scratch (zero init, the reviewer's "output bias alone" case) — 69.4 ± 0.6 at matched lr, and notably it converges to a direction nearly orthogonal to v_CAA (cos 0.04) with tiny norm yet matches the CAA-initialized vector, confirming that supervision rather than the CAA direction itself carries most of the effect for bare vectors. Together with the noise-input ablation (§5.2, 60.5) these decompose supervision, initialization, and parameterization; the revision will present all three and reframe the contribution accordingly (behaviorally supervised steering vectors; the bottleneck MLP as the stable default parameterization with a small residual advantage).

**(3) "RaLFiT comparison not decisive: reproduce it under your protocol, or claim only the raw-CAA improvement"** [WRITING]

> We adopt the reviewer's second option: the primary claim becomes the within-protocol improvement over properly-tuned raw CAA (58.3 → 77.9; +18.4 MC1 / +22.7 MC2 under the matched lm-eval protocol), with RaLFiT presented as cross-protocol context. Reproducing RaLFiT end-to-end is beyond one cycle [check: their code availability], and we prefer not to re-report a method under conditions its authors did not endorse; the reframed claim plus the TOST (response to upJj-2) is, we believe, the honest fix.

**(4) "Benchmark-specific supervision rather than general truthfulness"** [RESTATE + ANALYSIS + optional EXPERIMENT]

> We agree and the paper's Limitations states this scope; the revised framing (point 1) moves it to the front. Within the cycle we will add: (i) the per-category breakdown (with 74Zo) — TruthfulQA's 38 categories are diverse, and train/test questions never overlap, so category-level uniformity vs concentration is informative about what was learned; (ii) [decide: zero-shot evaluation of the trained vector on PopQA or NQ-open with the same frozen model — generation fits on local 24GB cards; judging via API]. We will not claim general truthfulness improvement absent (ii).

*Internal decision needed:* commit to (ii) only if you're willing to publish a possibly-null transfer result; a null there is survivable in a Findings-scoped paper but Z6VF will read silence as evasion. The Gemma/Qwen mess should stay out of the response.

**(5) Software=2/Datasets=1** [RELEASE]

> The full pipeline (training, evaluation, judging), configs, per-seed judge outputs, and trained vectors will be released under [MIT/Apache-2.0] with the revision. [Do the release-readiness list in Part 3 first — the current README reproduction path is broken.]

---

## Part 3 — Revision work plan (priority order)

| # | Item | Type | Addresses |
|---|------|------|-----------|
| 1 | **Matched direct-vector baseline** (lr 5e-4, batch 8, λ-anchor, 10 seeds) + bias-only variant | EXPERIMENT | upJj-1, Z6VF-2 |
| 2 | **Re-run Table 2 capability evals for the exact k=8 seed-42 vector** (current numbers are bn=16-era, hardcoded, source JSONs lost) | EXPERIMENT (internal integrity) | upJj-4 |
| 3 | Metric: define both T×I aggregations, recompute all MAST cells, fix `bootstrap_ci.py:39` | ANALYSIS | upJj-3 |
| 4 | Reframe abstract/intro (within-protocol headline, drop parity + "cannot reject equivalence", add TOST) | WRITING | upJj-2, Z6VF-1/3 |
| 5 | Per-category breakdown (join split indices ↔ dataset categories) | ANALYSIS | 74Zo-4, Z6VF-4 |
| 6 | Geometry subsection (cos(g,v)≈0.03 orthogonal correction, norms, rank; optional logit-lens) | ANALYSIS | 74Zo-3 |
| 7 | Bootstrap CIs on all single-seed cells; 3-seed noise + bottleneck ablations | ANALYSIS + EXPERIMENT | upJj-5 |
| 8 | Capability table: add RaLFiT/LoRA-DPO deltas + protocol caveat + LoRA-pairing mitigation numbers | WRITING | upJj-4 |
| 9 | Table 1 per-column bolding; typo L119; fix "conditional on reserved test split" wording; fix §3.1 vs §4 pooling contradiction (code pools over all non-pad tokens incl. prompt) | WRITING | 74Zo-1, upJj-6, integrity |
| 10 | Optional: PopQA/NQ-open zero-shot transfer of the trained vector | EXPERIMENT | Z6VF-4 |
| 11 | Code/data release (see checklist below) | RELEASE | all repro scores |

**GPU budget note:** items 1, 2, 7 need the 40GB+ class for training (item 2 is inference-only → 24GB OK; item 1 training is ~2 min/run on A6000 + generation + judging, so the cost is dominated by generation/judging, not training).

## Part 4 — Release-readiness checklist (from the code audit)

Highest-impact first:

1. **README reproduction path is broken**: the documented `python -m src.stages.*` flow never passes `bottleneck_dim` (trains the ~134M fat MLP, then crashes on state-dict load in `src/stages/generate.py:62-66`); `src/jobs/run_experiment.py` additionally hardcodes `normalize=True`. Root `run.py` is the only driver that reproduces the paper. → Make `run.py` the single entry point; delete/repair the other two; update README.
2. `run.py` never seeds torch (`set_random_seeds` exists but isn't called) — dropout and sampled decoding are unseeded even with `--seed`. One-line fix.
3. Commit the untracked scripts that produced paper numbers (`bootstrap_ci.py`, `judge_alpha_sweep.py`, `truthfulqa_judge_finetune.py`, `plot_*`, etc. — currently ~12 scripts untracked) and the per-example `gpt_judge_results.json` files so tables regenerate offline (judges are private fine-tunes; without stored judgments nobody can verify).
4. Replace hardcoded figure numbers in `plot_final.py` with reads from result JSONs.
5. Add LICENSE; pin `requirements.txt` (currently all `>=`, missing `openai`, `peft`, `trl`, `python-dotenv`).
6. Misc correctness to fix or caveat: KL-regularizer indexes pad positions (`src/utils/scoring.py:37-40` — affects only the KL ablation runs, not the paper); `evaluate_multiple_choice` is a nonstandard binary A/B forced choice (paper's MC1/MC2 correctly come from lm-eval-harness / `compute_mc_metrics.py` — never quote the internal one); direct-vector script saves under a misleading `mlp_mc/` path; judge parse failures silently count as "no" (report an error rate); duplicate dirs/zips inside `data/outputs/` (`baseline (2)`, stray 164MB zip).
7. Document seed semantics and the per-seed-test-split design; state 40GB+ GPU requirement for training.

## Part 5 — Additional self-critique (things reviewers did NOT raise; decide whether to fix silently)

1. §3.1 vs §4 vs code contradiction on activation pooling (Part 0 #10) — fix silently.
2. Single-seed sweep cells are interpreted with more confidence than ±2.7 seed-noise warrants (e.g., "k=8 peaks", pool 50→100 "+5pp", α-sweep shape). The CIs of item 7 will either support or soften these sentences — audit each claim after computing CIs. Note pool100 sweep cell (79.7) ≠ the seed-42 headline (80.9) quoted for pool=100 in §5.6 — the text mixes two different runs of the same nominal config; pick one and label it.
3. Judge circularity is stated only implicitly: the truth/info judges are fine-tuned on TruthfulQA reference annotations, and the training loss also uses those annotations. A short "judge validity" paragraph (judge agreement with human labels from Lin et al.; spot-audit N=50 generations by hand) would preempt the obvious next-cycle review.
4. Refusal metric is substring matching ("I don't know"/"I cannot") — fine for a trend, but say so in the caption.
5. Decoding is sampled (T=0.3) with one generation per question and (currently) unseeded torch — the same config re-run would not exactly reproduce Table 1 cells. Fix seeding (Part 4 #2) and mention sampling in §4.
6. The ±2.7 conflates split-resampling with training noise (Part 0 #2) — after the fix in wording, consider adding a same-split multi-seed column (cheap: 3 seeds, fixed split) so both variance sources are quantified.
7. MC1/MC2 for MAST were measured without the chat template while the steering vector was trained on chat-formatted prompts — mention, since the +18.4 MC1 within-protocol claim leans on that protocol.
8. Abstract still says "matching the published state-of-the-art … within one standard deviation" — this is the sentence both negative reviewers quote back; it must go in the revision regardless of anything else.
