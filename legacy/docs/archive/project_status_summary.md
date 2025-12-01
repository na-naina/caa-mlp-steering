# Project Status Summary – TruthfulQA Steering Pipeline

**Date:** 2025-11-05 17:20 UTC
**Prepared by:** Codex assistant (current session)

## 1. Research Goal
- Improve TruthfulQA open-ended accuracy using Contrastive Activation Addition (CAA) vectors with a non-linear MLP transform before reinjecting at the residual stream.
- Evaluate families of Gemma models (Gemma-2 & Gemma-3 initially) across baseline, raw steering, and MLP-steered variants.
- Judge correctness via strict semantic equivalence to reference answers (LLM judge) with optional embedding-based similarity scores.

## 2. Local Pipeline State
### Codebase
- `src/jobs/run_experiment.py`: end-to-end driver now performing:
  - TruthfulQA split into Steering Pool (default 100), Train (250), Val (117), Test (200).
  - Steering vector bank generation (30–50 sample subsets per vector).
  - Optional MC & Generation MLP training loops (hinge loss + log-prob maximisation).
  - Evaluation of baseline/steered/MLP variants with semantic + LLM judgers.
- New utilities: `src/steering/vector_bank.py`, `src/steering/training.py`, `src/utils/batching.py`, `src/utils/scoring.py`.
- Dataset manager (`src/data/truthfulqa.py`) loads generation split, attempts to attach MC data from multiple-choice split when available.

### Configuration
- `configs/base.yaml` sets shared HF cache to `/springbrook/share/dcsresearch/u5584851/hf_cache`, steering vector bank defaults, and judge defaults (Gemma-3-9B IT + sentence-transformers MiniLM).
- Smoke overrides (used in SLURM script) shrink splits and training steps for quick validation.

## 3. Remote Environment (Blythe cluster)
- Project path: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma`.
- Virtualenv rebuilt (`python3 -m venv venv`; `pip install -r requirements.txt`).
- HF token written to `~/.cache/huggingface/token`.
- SLURM smoke script `slurm/smoke_gemma2.slurm` requests 2 × GPUs, 16 CPUs, 80 GB RAM, 40 min limit.
- Cache directory (`cache/`) currently stores HuggingFace artifacts; capacity located on shared volume.

## 4. Attempts & Outcomes
1. **Initial smoke job (Job 1060096)** – Failed while downloading Gemma-2-2B: `OSError: [Errno 122] Disk quota exceeded`.
2. **Retry (Job 1060097)** – Same failure; dataset MC lookup still empty (multiple-choice split unavailable offline).
3. **Third attempt (Job 1060098)** – Same quota failure despite HF cache redirection.
- Notebook-style quick diagnostics via remote Python confirmed MC split download blocked by offline cache; no network access to fetch new data bundles.

## 5. Identified Issues
1. **Disk Quota:** Transformer cache contains ~61 GB (Gemma-3-27B + Gemma-2-2B). Downloading new checkpoints fails because `hf_hub_download` writes temporary files in run directory before moving to shared cache, hitting quota limits.
2. **Dataset Access:** HPC environment appears to be offline for HF datasets; only cached `truthful_qa/generation` is available. The required `multiple_choice` config is missing, preventing MC evaluation and MLP training for MC branch. Pipeline now tolerates missing MC data but logs warning.
3. **Model Availability:** Smoke job cannot proceed without Gemma-2-2B weights; need to stage them into shared cache (either by freeing space on HPC or manually syncing from local machine).
4. **Judge Model Size:** Default judge (Gemma-3-9B IT) will require additional cache storage; ensure space before scaling.

## 6. Recommended Next Steps
1. **Free / Reallocate Cache Space**
   - Option A: Remove existing 27B checkpoints in `/cache/transformers/models--google--gemma-3-27b-it` (~51 GB) if not immediately needed.
   - Option B: Point `HF_HOME` & temp dirs to a higher quota location (e.g., project-level scratch space) before downloads.
2. **Stage Required Models Offline**
   - Download Gemma-2-2B (and future Gemma-3-9B judge) locally, then `rsync` to `/springbrook/share/dcsresearch/u5584851/hf_cache/transformers/models--google--gemma-2-2b`.
   - Verify file permissions (user-writable) on shared cache.
3. **Handle Dataset MC Split**
   - Attempt manual download of TruthfulQA multiple_choice split locally; copy into `cache/datasets/truthful_qa/multiple_choice/...` on Blythe using `datasets-cli` or tarball.
   - Until then, expect MC branch to skip training/evaluation (pipeline warns but continues).
4. **Re-run Smoke Job**
   - After caches populated, resubmit `slurm/smoke_gemma2.slurm`; confirm runtime and assess accuracy outputs in `outputs/run_*/`.
5. **Parallelisation**
   - Once smoke succeeds, create SLURM arrays for multiple scales/models, splitting generation and judge workloads across GPUs (judge-only nodes running smaller models or CPU if using embeddings).
6. **Result Analysis**
   - Implement result parsing + plotting (existing `analysis/` scripts may need updates to new metrics `accuracy` & `semantic_mean`).

## 7. Outstanding Questions for Future Agent
- Confirm HPC policy on external downloads; can we use `srun --pty` with network or must we stage offline via login nodes?
- Preferred strategy for MC dataset absence—should we adjust research scope to generation-only until caches resolved?
- Desired storage cleanup policy (which older checkpoints can be removed?).
- Should the judge default downgrade to embedding-only for smoke runs to reduce dependencies?

## 8. Artifacts to Review
- SLURM logs: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/smoke_gemma2_<jobid>.{out,err}`.
- Partial outputs (none yet—jobs failed before producing run directories).
- Project configs & scripts synced as of this summary (`git status` recommended to ensure no local uncommitted changes before handoff).

---
Use this document as a handoff snapshot. Update sections 4–6 after the next successful run.
