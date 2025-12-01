# Experiment Submission Log

## 2025-11-11: Gemma3-12B Generation-Only (Fix for NaN Activations)

### Gemma3-12B: Generation Only - No Judge Evaluation

**Job ID**: 1066915
**Submission Time**: 2025-11-11 00:14 UTC
**Status**: ✅ RUNNING on node sea002
**Resources**: 1 GPU (lovelace_l40), 128GB RAM, 5 hours max
**Expected Runtime**: ~2-3 hours

**Critical Issue Discovered**: Multi-GPU model splitting causes NaN activations
- Job 1061010 (multi-GPU): 55/100 steering pairs had NaN activations → MLP training loss = NaN → TIMEOUT
- Job 1060500 (single GPU): No NaN issues, only failed during generation due to OOM (before batch size reduction)
- **Root Cause**: `max_memory: {0: "40GiB", 1: "40GiB"}` forces model splitting across GPUs, breaking activation extraction

**Solution Applied**:
```yaml
model:
  device_map: cuda:0  # Single GPU - prevents NaN activations
  # Removed max_memory to allow normal loading

slurm:
  gpus: 1  # Only 1 GPU needed for generation

# No judge evaluation configured - will score separately
```

**Submission Command**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/gemma3_12b_genonly.sh"
# Result: Submitted batch job 1066915 (started immediately)
```

**After Completion**:
Will run separate scoring job using `scripts/score_generated_responses.py` (same as gemma3-4b)

---

## 2025-11-10 (Evening): Gemma3-12B Failed Attempts and Gemma3-4B Re-scoring

### 1. ❌ FAILED: Gemma3-12B Full Run (Multi-GPU NaN Issue)

**Job ID**: 1061010
**Previous Job IDs**: 1060935 (18h - cancelled for scheduling), 1061009 (cancelled - config not synced)
**Submission Time**: 2025-11-10 19:03 UTC
**Status**: ❌ TIMEOUT (ran full 5 hours, then killed)
**Runtime**: 5:00:20
**Resources**: 2 GPUs, 128GB RAM

**Failure Cause**:
Multi-GPU model splitting with `max_memory: {0: "40GiB", 1: "40GiB"}` caused catastrophic NaN activations:
- 55 out of 100 steering pairs produced NaN activations during extraction
- MLP training loss = NaN for both epochs
- Job spent 5 hours slowly processing through batches with errors before timeout
- Example errors: `Invalid activations in batch example 0: NaN count=142080, Inf count=0`

**Earlier OOM Failures** (Jobs 1060500, 1060504, etc.):
- These used single GPU (`device_map: cuda:0`) - NO NaN issues
- Failed during generation due to insufficient memory for gradient accumulation
- Error: `torch.OutOfMemoryError: Tried to allocate 30.00 MiB. GPU 0 has 18.12 MiB free`

**Attempted Fix (Didn't Work)**:
Updated `configs/gemma3_12b_full.yaml`:
```yaml
model:
  max_memory: {0: "40GiB", 1: "40GiB"}  # Reserve memory for gradients

training:
  gen_mlp:
    batch_size: 2  # Reduced from 4
  mc_mlp:
    batch_size: 4  # Reduced from 8
```

**Time Limit Update**:
- Initial submission (1060935): 18 hours - job stuck waiting for resources
- Updated to 5 hours for faster scheduling (scheduler can fit shorter jobs more easily)
- Actual expected runtime: 3-4 hours based on 4B model taking 19 mins

**Submission Command**:
```bash
# Updated both config and SLURM script to 5 hours
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/gemma3_12b_full.sh"
# Result: Submitted batch job 1061010
```

### 2. Gemma3-4B: Re-scoring with Judge Model

**Job ID**: 1060936
**Submission Time**: 2025-11-10 14:00 UTC
**Status**: ✅ COMPLETED
**Completion Time**: 2025-11-10 14:38 UTC
**Runtime**: 15 minutes
**Resources**: 1 GPU, 64GB RAM

**Issue Found**:
- Job 1060499 completed successfully (responses generated)
- Judge model loaded but scoring failed silently
- All responses have `null` accuracy/semantic scores

**Solution**:
Created separate scoring job using `scripts/score_generated_responses.py`:
```bash
# Created slurm/score_gemma3_4b.sh
python scripts/score_generated_responses.py \
  outputs/run_20251107_172642 \
  --judge-model google/gemma-3-12b-it \
  --judge-device auto
```

**Submission Command**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/score_gemma3_4b.sh"
# Result: Submitted batch job 1060936
```

**Results After Fix**:
- Scoring job generated truncated responses (same `max_new_tokens=32` issue)
- Applied local regex-based fix to extract correct match values
- **Baseline**: 1% → 48.0% (corrected)
- **Steered**: 1% → 46.0% (corrected)
- **MLP-Gen**: 0.5% → 49.5% (corrected)
- Changed 94/200 baseline, 90/200 steered, 98/200 mlp_gen match values
- Results now included in cross-model comparison (6 models total)

---

## 2025-11-10 (Morning): Gemma3-12B Investigation

### Expected Output:
- Run ID: `gemma3_12b_full_YYYYMMDD_HHMMSS`
- Location: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/`
- Estimated time: ~4-6 hours (with 2 GPUs)

---

## 2025-11-07: Gemma3-4B Completed Successfully

### Status: ✅ Completed
- **Job**: 1060499
- **Run ID**: `run_20251107_172642`
- **Duration**: 19 minutes (17:26:34 - 17:45:53)
- **Results**: Successfully fetched to `analysis_results/gemma3_4b_full_20251107_172642/`

---

## Previous Successful Runs

### 2025-11-06: 27B Models
- **Gemma2-27B** (Job 1060281): Generated responses, scored separately (Job 1060502)
  - Run: `run_20251106_091841`
  - Issue: OOM when loading judge, fixed with separate scoring job

- **Gemma3-27B** (Job 1060282): Generated responses, scored separately (Job 1060503)
  - Run: `run_20251106_101727`
  - Issue: OOM when loading judge, fixed with separate scoring job

### 2025-11-06: Small Models (2B, 9B, 1B)
- **Gemma2-2B** (Job 1060201): ✅ Completed
- **Gemma2-9B** (Job 1060202): ✅ Completed
- **Gemma3-1B** (Job 1060212): ✅ Completed

---

## To Submit Gemma3-12B:

1. Ensure configs are synced:
```bash
rsync -avz configs/gemma3_12b_full.yaml csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/configs/
```

2. Submit job:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/gemma3_12b_full.sh"
```

3. Monitor:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -u csuqqj"
```

4. After completion, fetch results:
```bash
rsync -avz csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/gemma3_12b_full_YYYYMMDD_HHMMSS/ analysis_results/gemma3_12b_full_YYYYMMDD_HHMMSS/
```
