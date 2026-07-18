# ARR Response Draft (Submission 16990) — for Google Doc

*Draft for prof review. Tone: factual, concise, grateful; new results reported as revision commitments, not as re-argument of the paper's thesis. All new numbers are per-item Truth∧Info on the fixed seed-42 test split (n=408) with the same fine-tuned judges as the paper unless stated.*

---

## General response (post once, addressed to all reviewers)

We thank all three reviewers for unusually constructive reviews. Since submission we have run every requested experiment; we summarize the shared ones here and address individual points below.

**1. Directly-optimized steering vector baselines (upJj, Z6VF).** We ran the requested baselines under the paper's exact protocol (identical margin loss, anchor, optimizer, schedule, splits, generation, and judges):

| Intervention | T×I (%) |
|---|---|
| Raw CAA, tuned α (paper) | 58.3 |
| Direct vector, v_CAA init, MAST's lr (5e-4) | 68.1 ± 1.8 |
| Direct vector, zero init ("learned bias"), same lr | 69.4 ± 0.6 |
| Direct vector, v_CAA init, own lr tuned (2e-3) | 76.9 ± 0.6 |
| Direct vector, zero init, own lr tuned (2e-3) | 80.2 ± 0.4 |
| MAST (k=8), default lr, no tuning | 78.7 ± 0.8 |

Directly-optimized vectors are therefore strong — with an important caveat we will analyze in the revision: their performance is sharply peaked in learning rate (e.g., zero-init: 69.4 → 73.8 → 80.2 → 72.5 across lr 5e-4→5e-3, with informativeness collapsing to 82.8% at the high end), and training-time signals improve monotonically past the peak, so train-signal-based hyperparameter selection (the only selection our protocol permits; the test split is never consulted) lands at 72.5% with degraded informativeness. MAST reaches 78.7 ± 0.8 at its default configuration in all runs with informativeness ≥95%. The revision will add all of these baselines to Table 1 and an analysis of this selection-robustness property.

**2. T×I metric definition (upJj).** The reviewer is correct that our pipeline computes the per-item conjunction (fraction judged both truthful and informative) while rows reproduced from Li et al. (2025) are the product of marginal rates. The conjunction is the stricter aggregation on our outputs: MAST's headline is 77.9 ± 2.7 (conjunction) and 78.7 ± 2.4 (product-of-means). The revision defines both explicitly and reports both.

**3. Reproducibility and release.** All experiments above, plus a full re-verification of the paper's main numbers (T×I, MC1/MC2 via a pinned lm-eval-harness script with per-sample records, and the Table 2 capability benchmarks), are now backed by committed artifacts. The complete codebase, configs, per-seed generations, judge decisions, and trained vectors will be released under MIT with the revision.

---

## Reviewer 74Zo

**"Bold the top scores in each column of Table 1."**
Agreed — the revision bolds best-per-column (the current bolding was row-emphasis for our method, which we agree is confusing).

**"Is the 70K middle ground interesting, given smaller models do ok and larger seem better?"**
We take the point and will reframe the comparison axis: the claim we intend is not that 70K is a sweet spot on the parameter axis, but that a frozen-model, activation-space intervention — whose inference-time footprint is a single d=4096 vector, removable per-query by setting α=0 — closes essentially the entire gap from training-free steering (TruthX 62.8, tuned CAA 58.3) to the weight-space state of the art. The 70K MLP is training-time machinery; the revision's positioning section will carry the comparison on the weight-space vs. activation-space axis rather than parameter counts.

**"ITI and CAA tell us something about the geometry of activation space; MAST does not."**
We agree this was missing and have added a geometry analysis (all 10 seeds, from saved vectors): (i) the learned correction is consistently near-orthogonal to v_CAA (cos = 0.02 ± 0.02) — supervision adds a new direction rather than rescaling CAA; (ii) the correction is large (‖Δ‖ ≈ 3.5× ‖v_CAA‖; cos(v_CAA, v_MAST) = 0.40 ± 0.18), so we will also correct §3.2's "small correction" phrasing, which this measurement contradicts; (iii) the corrections found from different seeds are strongly mutually aligned (mean pairwise cos 0.62 in d=4096, vs ≈0 for random directions), i.e., training converges to a consistent "truthfulness" direction. This will appear as a new analysis subsection with a figure.

**"Break results down by TruthfulQA category, as in the ITI paper."**
Done — thank you for the suggestion; the pattern is informative. Pooling the 10 seeds (4,080 judged answers): MAST's gains concentrate where the unsteered model repeats common falsehoods — e.g., baseline→MAST T×I: Economics 29.0→67.6, Sociology 52.7→77.6, Misconceptions 58.0→76.8, Health 67.3→85.6; and it is strongest in Superstitions (91.4), Paranormal (92.7), Nutrition (98.7). It remains weak where answers require precise recall of specific facts: Distraction (41.4), Logical Falsehood (49.3), Misquotations (50.7). This is consistent with the intervention steering the model away from common falsehoods rather than adding knowledge, and the revision will include the full per-category table and say so explicitly.

**Typo** ("summaries" → "summarized") — fixed, thank you.

---

## Reviewer upJj

**W1: "f_θ is only ever applied to the fixed v_CAA … MAST = directly optimizing a steering vector. That direct-vector baseline is absent."**
We agree the baseline was required and have run it comprehensively — see the table in the general response. In brief: at MAST's own hyperparameters the direct vector reaches 68.1 ± 1.8; with its learning rate tuned it reaches 76.9 ± 0.6 (v_CAA init) / 80.2 ± 0.4 (zero init); however the tuned configurations are only identifiable by consulting test scores, because training-time signals improve monotonically into the over-trained regime where informativeness degrades (the same failure signature as §5.3's k=64 and App. B's α=5). Under the paper's selection protocol (train signals only), the direct vector attains 72.5 with Info 82.8, vs MAST's 78.7 ± 0.8 with Info ≥95 at default settings. The revision adds all baselines to Table 1 plus this analysis.

**W1b: "5.3's 'rank' claim is single-input sparsity, not generalization."**
Correct — we will reword §5.3 to describe the sparsity of the learned solution at its input, and remove the generalization implication.

**W2: "Statistical parity is overstated … 'cannot reject equivalence' inverts NHST; no TOST."**
We agree on all three subpoints and will make these changes: (i) the abstract's headline claim becomes the within-protocol result (tuned raw CAA 58.3 → MAST 77.9 under identical splits, judges, and decoding); (ii) the "cannot reject equivalence" sentence is removed; (iii) we add a TOST against RaLFiT's (single-run, variance-unavailable) point estimate: the 90% CI of MAST's mean is [76.3, 79.5] (n=10 seeds), so equivalence holds under ±3 pp bounds and does not under ±2 pp — we will state exactly this and no more, alongside the protocol asymmetry already discussed in §4.

**W3: "Undisclosed/inconsistent T×I."**
Correct — see general response item 2. Both formulas will be defined and reported; the inconsistency was conservative against our method (product-of-means gives MAST 78.7 ± 2.4 vs the reported 77.9 ± 2.7).

**W4: "Table 2 reports MAST's drops only vs baseline, never vs RaLFiT/LoRA."**
We will add the corresponding numbers from Li et al. (2025, Table 2), with the protocol caveat (their few-shot leaderboard configuration vs our 0-shot): under their protocol RaLFiT and LoRA-DPO are approximately capability-neutral (e.g., ARC 53.7→58.1), whereas MAST costs 2–6 pp under ours. We re-ran our Table 2 with the final k=8 vector so the revision's numbers are artifact-backed (ARC-E −6.4, ARC-C −6.0, HellaSwag −2.0, MMLU −3.0). The honest summary — which the revision states explicitly — is that DPO-trained adapters do not show this cost; mitigations (per-query α=0 switch-off; pairing with a LoRA adapter) are discussed in Limitations.

**W5: "Single-seed ablations, no CIs."**
The revision adds per-item bootstrap 95% CIs to every single-seed cell (all judgments are stored), and two calibration measurements we made during re-verification: same-split multi-seed variance for the main configuration (±0.8 over 5 seeded reruns) and pure generation+judging noise (±1.6, from replicate generations of deterministic vectors) — the latter bounds how much any single-seed sweep cell can be over-read, and we will temper the sweep-discussion language accordingly.

**Typo (L119)** — fixed, thank you.

---

## Reviewer Z6VF

**W1: "The main contribution is narrow and the framing broad" (one model, one layer, one benchmark).**
We will narrow the framing as suggested: the abstract's parity language is replaced by the within-protocol claim (see response to upJj W2), and scope statements move to the front. On the "one model" point specifically, we have since run the identical recipe (bn=8, default hyperparameters, layer selected by train-time signals only) on Gemma-3-4B-IT: baseline 53.7 → raw CAA 54.7 → MAST 88.2 T×I (n=408, single seed; we are adding seeds and a manual audit of judge decisions for the revision, and note the judges are TruthfulQA-fine-tuned). We report this as evidence the recipe is not LLaMA-specific, with its caveats stated.

**W2: "A crucial baseline is missing: directly optimizing a steering vector … also a learned residual-bias baseline, since the MLP input is fixed and there is a trainable output bias."**
Both requested baselines are run — see the general response table. The learned-bias-from-scratch case (zero init) is particularly informative: at matched hyperparameters it reaches 69.4 ± 0.6 and converges to a direction nearly orthogonal to v_CAA (cos 0.04), and with an oracle-tuned learning rate it reaches 80.2 ± 0.4 — but that configuration is not discoverable under train-signal selection (general response item 1). The revision reports all of this in Table 1 with the selection analysis, and rewords the contribution statements accordingly.

**W3: "The RaLFiT comparison is not fully fair or decisive … either reproduce RaLFiT under the same split and judges, or make the claim primarily about improvement over raw CAA."**
We adopt the reviewer's second option, exactly as phrased: the primary claim becomes the matched-protocol improvement over properly-tuned raw CAA (58.3 → 77.9, same splits, judges, and decoding; +18.4 MC1 / +22.7 MC2 under the matched MC protocol), with RaLFiT presented as cross-protocol context accompanied by the TOST analysis (response to upJj W2) rather than any parity claim.

**W4: "The result may be benchmark-specific supervision rather than general truthfulness steering."**
We agree this boundary must be explicit and the revision states it up front. Two additions probe it within the cycle: (i) the per-category breakdown (with 74Zo) — train and test questions never overlap, and the gains/weaknesses pattern across 38 categories (strong on common-falsehood categories, weak on precise-recall ones) characterizes what is actually learned; (ii) the Gemma-3 result above shows the training recipe is not model-specific. Transfer to out-of-benchmark factuality datasets (e.g., PopQA/NQ-open) is the right further test; we will report it in the revision if complete, and will not claim general truthfulness improvement without it.

**Suggestions (baselines, transfer)** — addressed above. **Typo** — fixed.

---

*Note for us (delete before posting): all numbers trace to artifacts in the repo; the deeper reframe of the paper's central claim (supervision + selection-robustness as the core contribution) is deliberately NOT argued here — it lands in the next-cycle revision after supervisor discussion.*
