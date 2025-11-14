# Investigation: Missing Gemma3-4B and Gemma3-12B Models

**Date**: 2025-11-10
**Investigator**: Claude

---

## Summary

Investigated why Gemma3-4B and Gemma3-12B experiments were missing from the analysis. Found that:
- ✅ **Gemma3-4B**: Responses generated but scoring failed → Re-scored and fixed (Job 1060936 completed)
- ⏳ **Gemma3-12B**: Failed repeatedly due to CUDA OOM → Fixed config submitted (Job 1060935 pending)

---

## Gemma3-4B Investigation

### Status: ✅ COMPLETED & FIXED

**Original Job (Generation):**
- Job ID: 1060499
- Status: COMPLETED (0:0) - but judge evaluation failed silently
- Runtime: 19 minutes (2025-11-07 17:26:34 - 17:45:53)
- Output: `run_20251107_172642`

**Problem Found:**
- Responses generated successfully for all 3 conditions
- Judge model loaded but scoring failed silently
- All accuracy/semantic scores were `null`

**Re-scoring Job:**
- Job ID: 1060936
- Runtime: 15 minutes (2025-11-10 14:23 - 14:38)
- Status: COMPLETED successfully

**Fix Applied:**
- Scoring generated truncated JSON (same `max_new_tokens=32` issue)
- Used regex-based parser to extract correct match values from truncated responses
- Corrected 94/200 baseline, 90/200 steered, 98/200 mlp_gen responses

**Final Results:**
- **Baseline**: 48.0% accuracy (52.7% semantic)
- **Steered**: 46.0% accuracy (49.5% semantic)
- **MLP-Gen**: 49.5% accuracy (55.1% semantic)
- **Improvement**: +1.5% with MLP-Gen steering

---

## Gemma3-12B Investigation

### Status: ❌ FAILED (Multiple Attempts)

**Failed Job Examples:**
- Job 1060500 (2025-11-07 17:27): FAILED (ExitCode 2:0)
- Job 1060504, 1060273, 1060277, 1060264, 1060257 (2025-11-06): All FAILED

### Root Cause Analysis

**Error Message:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB.
GPU 0 has a total capacity of 44.42 GiB of which 18.12 MiB is free.
```

**Failure Point:**
Training Generation MLP (after model loaded, steering vectors extracted, MC MLP skipped)
```
File: src/steering/training.py, line 282
Function: train_gen_mlp -> loss.backward()
```

**Problem:**
1. Config specifies `device_map: auto` and `gpus: 2`
2. 12B model (google/gemma-3-12b-pt) loads successfully
3. Model fits comfortably on 1 GPU (~43.7 GiB allocated)
4. When MLP training starts with batch_size=4, gradient computation requires additional memory
5. No room left for gradients → OOM crash

**Why Auto Device Map Didn't Help:**
Even with `device_map: auto`, if a model fits on one GPU, PyTorch won't split it. The 12B model uses ~43.7 GiB, leaving only ~700 MiB. Gradient accumulation needs more.

---

## Solution Implemented

### Config Changes: `configs/gemma3_12b_full.yaml`

```yaml
model:
  name: google/gemma-3-12b-pt
  family: gemma3
  layer: 24
  dtype: bfloat16
  device_map: auto
  max_memory: {0: "40GiB", 1: "40GiB"}  # NEW: Force model splitting

# NEW: Reduced batch sizes
training:
  gen_mlp:
    batch_size: 2  # Reduced from 4
    epochs: 2
    steps_per_epoch: 40
  mc_mlp:
    batch_size: 4  # Reduced from 8
    epochs: 2
    steps_per_epoch: 50
```

### Why This Works:

1. **`max_memory` constraint**: Forces model to split across both GPUs even though it could fit on one
2. **Reduced batch sizes**: Cuts gradient memory requirements in half
3. **Combined effect**: Model parts on GPU0 and GPU1, plus room for gradients during training

---

## Submission Instructions

### Configuration Already Synced:
```bash
rsync -avz configs/gemma3_12b_full.yaml \
  csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/configs/
```

### Submit Job:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk \
  "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && \
   sbatch slurm/gemma3_12b_full.sh"
```

### Monitor Progress:
```bash
# Check queue
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -u csuqqj"

# Watch log (replace JOBID)
ssh csuqqj@blythe.scrtp.warwick.ac.uk \
  "tail -f /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/gemma3_12b_full_JOBID.err"
```

### After Completion:
```bash
# Find the run directory
ssh csuqqj@blythe.scrtp.warwick.ac.uk \
  "ls -lt /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/ | grep gemma3_12b | head -1"

# Fetch results (replace with actual timestamp)
rsync -avz csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/gemma3_12b_full_YYYYMMDD_HHMMSS/ \
  analysis_results/gemma3_12b_full_YYYYMMDD_HHMMSS/
```

---

## Expected Timeline

- **Estimated Runtime**: 4-6 hours
- **Checkpoints**:
  - ~5 min: Model loaded
  - ~10 min: Steering vectors extracted
  - ~15 min: MLP training complete ← **Previous failure point**
  - ~3-4 hrs: Response generation (200 test samples × 3 conditions)
  - ~1 hr: Judge evaluation

---

## Impact on Analysis

### Current Model Coverage:

| Family | 1B | 2B | 4B | 9B | 12B | 27B |
|--------|----|----|----|----|-----|-----|
| Gemma2 | - | ✅ | - | ✅ | - | ✅ |
| Gemma3 | ✅ | - | ✅ | - | ⏳ | ✅ |

**Legend:** ✅ Complete | ⏳ Pending

**Current Status**: 6 models completed, 1 pending (Gemma3-12B)

**After Gemma3-12B completes:**
- Complete Gemma3 series: 1B, 4B, 12B, 27B
- Complete Gemma2 series: 2B, 9B, 27B
- Total: 7 models across 2 families

### What This Enables:
- More complete scaling curves for Gemma3 family
- Better comparison between Gemma2 and Gemma3 at different scales
- Validation of diminishing returns hypothesis for MLP steering on larger models

---

## Files Modified

1. `configs/gemma3_12b_full.yaml` - Added max_memory and reduced batch sizes
2. `SUBMISSION_LOG.md` - Created to track all submissions
3. `FINDINGS_MISSING_MODELS.md` - This document

---

## Lessons Learned

1. **Device Map Auto ≠ Always Split**: PyTorch prefers to keep models on single GPU when possible
2. **Gradient Memory**: Can be significant during training, especially with larger batch sizes
3. **max_memory Parameter**: Useful for forcing model distribution even when not strictly necessary
4. **Batch Size Trade-offs**: Smaller batches = more memory headroom but longer training time
5. **Check Completed Jobs**: Always verify remote outputs directory, job may have succeeded but not been fetched

---

## Next Steps

1. ✅ Gemma3-4B results fetched, re-scored, and fixed
2. ✅ Gemma3-12B config fixed and submitted (Job 1060935)
3. ✅ Gemma3-4B scoring completed and corrected
4. ✅ Cross-model analysis regenerated with 6 models
5. ⏳ Wait for Gemma3-12B to complete (Job 1060935, ~4-6 hours)
6. ⏳ Fetch Gemma3-12B results when complete
7. ⏳ Fix any truncated judge responses in Gemma3-12B
8. ⏳ Regenerate final cross-model analysis with all 7 models

---

**Notes for Future Runs:**
- For models 10B+, consider starting with reduced batch sizes
- Always set `max_memory` when using 2+ GPUs to ensure distribution
- Monitor first 15-20 minutes of large model jobs to catch OOM early
- Keep SUBMISSION_LOG.md updated for reproducibility
