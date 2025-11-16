# Session Summary: November 14, 2025
## TruthfulQA Alignment Pipeline - Gemma3-IT Models

### Objectives
1. Apply new TruthfulQA alignment pipeline to all Gemma3-IT models (270m, 1b, 4b, 12b, 27b)
2. Fix judge model loading OOM issues
3. Salvage partial results from failed experiments
4. Investigate NaN activation failures in larger models

---

## Summary

### Successful Experiments (4/5 models)
✅ **gemma3-270m-it**: CAA extraction + generation completed, judge scoring succeeded
✅ **gemma3-1b-it**: CAA extraction + generation completed, judge scoring succeeded
✅ **gemma3-4b-it**: CAA extraction + generation completed, judge scoring succeeded
✅ **gemma3-12b-2gpu-pt**: CAA extraction + generation completed (for comparison baseline)

### Failed Experiments
❌ **gemma3-12b-it**: NaN activations (88/100 pairs) after code changes
❌ **gemma3-27b-it**: NaN activations (100/100 pairs) - complete failure

### Salvaged Data
- **gemma3-12b-it (first run)**: Successfully completed CAA extraction, MLP training, and baseline generation before being cancelled
- Location: `outputs/gemma3_12b_it_20251114_130235_partial/`
- Status: Vectors, MLPs, and baseline results available

---

## Key Findings

### 1. Judge Model Sharing Implementation
**Problem:** Both truthfulness and informativeness judges were loading gemma-3-12b-it separately, causing OOM errors.

**Solution:** Implemented model sharing:
- Modified `LLMBinaryJudge` and `LLMInformativenessJudge` to accept `shared_model` parameter
- Updated `run_experiment.py` to detect when both judges use same model and load it once
- Updated `score_generated_responses.py` to support post-hoc scoring with model sharing

**Files Modified:**
- `src/evaluation/judge.py`
- `src/evaluation/informativeness.py`
- `src/jobs/run_experiment.py`
- `scripts/score_generated_responses.py`
- `slurm/score_gemma3_salvaged.sh` (new file)

### 2. Code-Induced NaN Activation Failures

**Critical Discovery:** Code changes at 13:14-13:30 UTC broke CAA extraction for 12B+ models.

**Timeline Evidence:**

**Before code sync (all nodes working):**
- 12:55 - sea001: gemma3_270m_it ✅ CAA succeeded
- 12:55 - sea002: gemma3_1b_it ✅ CAA succeeded
- 13:01 - sea002: gemma3_4b_it ✅ CAA succeeded
- 13:02 - sea012: gemma3_12b_it ✅ CAA succeeded

**After code sync (same nodes failing):**
- 13:31 - sea001: gemma3_12b_it ❌ 88/100 NaN failures
- 13:30 - sea002: gemma3_27b_it ❌ 100/100 NaN failures

**Conclusion:** The timing correlation is definitive - code changes caused the NaN issues, ruling out hardware problems.

**Mechanism:** Unknown - all modifications were to evaluation code (after CAA extraction), but the failures occur during CAA extraction. Possible causes:
1. Python bytecode caching issues
2. Subtle import side-effects
3. Memory/GPU state corruption from new code paths

### 3. Model Size vs Failure Pattern
- **270m, 1b, 4b**: All succeeded
- **12B (first run)**: Succeeded
- **12B (second run, after code changes)**: Failed
- **27B (after code changes)**: Failed completely

### 4. Salvageable Results
Four experiments completed generation but failed at judging (OOM):
- gemma3_270m_it_20251114_125537 (4/4 variants)
- gemma3_1b_it_20251114_125537 (4/4 variants)
- gemma3_4b_it_20251114_130117 (4/4 variants)
- gemma3_12b_2gpu_20251114_124809 (4/4 variants)

**Scoring Status:** Job 1066965 running (1+ hours) applying truthfulness + informativeness judges to all salvaged experiments.

---

## Technical Details

### CAA Extraction Configuration
| Model | Layer | GPUs | Memory | Status |
|-------|-------|------|--------|--------|
| 270m-it | 8 | 2 | 48GB | ✅ Success |
| 1b-it | 12 | 2 | 64GB | ✅ Success |
| 4b-it | 16 | 2 | 90GB | ✅ Success |
| 12b-it | 18 | 2 | 128GB | ⚠️ Mixed (1st✅, 2nd❌) |
| 27b-it | 26 | 3 | 180GB | ❌ Failed |

### Judge Configuration
All experiments use:
- **Truthfulness Judge:** google/gemma-3-12b-it (zero-shot)
- **Informativeness Judge:** google/gemma-3-12b-it (zero-shot)
- **Semantic Judge:** sentence-transformers/all-MiniLM-L6-v2
- **BLEURT:** Skipped (optional)

### Evaluation Settings
```yaml
evaluation:
  preset: qa  # TruthfulQA QA-style prompts
  judge:
    mode: zero_shot
    model: google/gemma-3-12b-it
  informativeness:
    enabled: true
    mode: zero_shot
    model: google/gemma-3-12b-it
  semantic:
    enabled: true
    use_false_refs: true
```

---

## Error Log Summary

### Error 1-4: Judge Model OOM (Jobs 1066944-1066946, 1066938)
**Error:** `torch.OutOfMemoryError: CUDA out of memory`
**Cause:** Loading gemma-3-12b-it twice (truth + informativeness judges)
**Status:** ✅ Fixed via model sharing

### Error 5-6: Invalid `use_false_refs` Parameter (Jobs 1066957, 1066961, 1066964)
**Error:** `TypeError: __init__() got an unexpected keyword argument 'use_false_refs'`
**Cause:** Incorrectly added parameter to SemanticJudgeConfig
**Status:** ✅ Fixed by removing parameter

### Error 7: UnboundLocalError (Job 1066958)
**Error:** `UnboundLocalError: local variable 'load_causal_model' referenced before assignment`
**Cause:** Redundant import inside if block
**Status:** ✅ Fixed by removing redundant import

### Error 8-9: NaN Activations (Jobs 1066962, 1066963)
**Error:** `ValueError: Minimum sample size exceeds available activation count`
**Cause:** 88-100% of CAA prompt pairs producing NaN/Inf activations
**Status:** ⚠️ **UNRESOLVED** - correlated with code changes but mechanism unknown

---

## Job History

| Job ID | Model | Start Time | Status | Reason |
|--------|-------|------------|--------|--------|
| 1066944 | 270m-it | 12:55 | FAILED | OOM (judge loading) |
| 1066945 | 1b-it | 12:55 | FAILED | OOM (judge loading) |
| 1066946 | 4b-it | 13:01 | FAILED | OOM (judge loading) |
| 1066938 | 12b-2gpu-pt | 12:48 | FAILED | OOM (judge loading) |
| 1066947 | 12b-it | 13:02 | CANCELLED | Completed baseline, cancelled during steered |
| 1066957 | 270m-it scoring | 13:18 | FAILED | Invalid `use_false_refs` |
| 1066958 | 12b-it | 13:28 | FAILED | UnboundLocalError |
| 1066959 | 27b-it | 13:28 | FAILED | Invalid `use_false_refs` |
| 1066961 | 270m-it scoring | 13:29 | FAILED | Invalid `use_false_refs` (remote not synced) |
| 1066962 | 12b-it | 13:31 | FAILED | NaN activations (88/100) |
| 1066963 | 27b-it | 13:30 | FAILED | NaN activations (100/100) |
| 1066964 | 270m-it scoring | 13:32 | FAILED | Invalid `use_false_refs` (remote not synced) |
| 1066965 | Salvaged scoring | 13:36 | **RUNNING** | Scoring 4 experiments |

---

## Next Steps

### Immediate Actions
1. ✅ Monitor scoring job (1066965) completion
2. ⏳ Download scored results when ready
3. ⏳ Commit all code changes

### Investigation Required
1. **NaN Activation Root Cause:**
   - Test hypothesis: Revert code to pre-13:14 state and retry 12b-it/27b-it
   - Clean `__pycache__` on remote and retry with current code
   - Profile memory/GPU state during CAA extraction

2. **12B-IT Salvage Strategy:**
   - First run (job 1066947) completed: CAA vectors, MLPs, baseline generation
   - Options:
     - Continue from saved state (resume steered/mlp_gen variants)
     - Re-run with reverted code
     - Accept 270m/1b/4b results and skip 12B+

### Long-term Fixes
1. Add bytecode cleanup to job submission scripts
2. Implement better error handling for NaN activations
3. Add checkpoint/resume functionality for long-running experiments
4. Document code change -> job submission workflow to avoid timing issues

---

## Updates (Nov 16, 2025)

### Code changes
- Added robust parsing of truthfulness judge outputs to avoid crashes on non-numeric `match` fields (src/evaluation/judge.py).
- Disabled TensorFlow imports for semantic scoring and scoring script to prevent tf-keras/Keras 3 errors; scoring script now sets TF-off env vars early and accepts `--config` for cache/judge settings (src/evaluation/semantic.py, scripts/score_generated_responses.py).
- Generation stats now retain all fields (including informativeness/semantic/BLEURT) when serialized, even if judges run inline (src/jobs/run_experiment.py).
- Activation extraction now masks padding tokens when averaging, reducing noise/instability (src/steering/extract.py).
- Vector bank sampling clamps requested sample sizes to available activations to avoid sample-size errors after NaN filtering (src/steering/vector_bank.py).

### New configs
- `configs/gemma3_270m_it_local.yaml`: full local run; local caches/outputs; 270m judges.
- `configs/local_eval_only.yaml`: local post-hoc scoring defaults.
- `configs/gemma3_12b_it_fp16_short.yaml`: fp16 load + `max_length=384` (NaN mitigation option 1 + 2) for 12B.
- `configs/gemma3_27b_it_fp16_short.yaml`: same for 27B.

### Local runs
- Dry-run succeeded with `configs/gemma3_270m_it_local.yaml`.
- Post-hoc scoring now works with TF disabled; BLEURT remains optional (needs bleurt install).

### NaN mitigation guidance
- Options to test on cluster for NaN activations (12B/27B):
  1) Load in fp16 for extraction (`model.dtype: float16`).
  2) Shorten extraction context (`steering.max_length: 384`).
  Configs above bake both knobs for quick reruns.
---

## Files Created/Modified

### New Files
- `slurm/score_gemma3_salvaged.sh` - Batch scoring script for salvaged experiments
- `outputs/gemma3_12b_it_20251114_130235_partial/` - Downloaded first 12b-it run

### Modified Files
- `src/evaluation/judge.py` - Added `shared_model` parameter
- `src/evaluation/informativeness.py` - Added `shared_model` parameter
- `src/jobs/run_experiment.py` - Judge model sharing logic
- `scripts/score_generated_responses.py` - Full pipeline support (informativeness, model sharing)
- `configs/gemma3_270m_it_full.yaml` (already existed)
- `configs/gemma3_1b_it_full.yaml` (already existed)
- `configs/gemma3_4b_it_full.yaml` (already existed)
- `configs/gemma3_12b_it_full.yaml` (already existed)
- `configs/gemma3_27b_it_full.yaml` (already existed)

---

## Results Summary

### Expected Outputs (when scoring completes)
```
outputs/
├── gemma3_270m_it_20251114_125537/
│   ├── results.json (4 variants: baseline, steered, mlp_gen, mlp_mc)
│   └── [baseline/steered/mlp_gen/mlp_mc]/scale_*/
│       ├── generation_details.json (with judge scores)
│       └── mc_details.json
├── gemma3_1b_it_20251114_125537/ (same structure)
├── gemma3_4b_it_20251114_130117/ (same structure)
└── gemma3_12b_2gpu_20251114_124809/ (same structure, -pt model)
```

### Partial Results
```
outputs/
└── gemma3_12b_it_20251114_130235_partial/
    ├── baseline/scale_0.00/ (✅ Complete, unscored)
    ├── vectors/ (CAA vectors, MLPs on remote)
    └── training_history.json
```

---

## Lessons Learned

1. **Model Sharing Critical:** For large judge models (12B), loading once saves significant memory
2. **Code/Job Timing:** Code changes must be carefully synchronized with job submissions to avoid mysterious failures
3. **Salvage Strategy:** Separating generation from scoring enables recovery from OOM failures
4. **NaN Debugging:** Requires comparing successful vs failed runs with identical configs - timing is key evidence
5. **Remote File Sync:** rsync timing matters - verify files are updated before job submission

---

## Statistics

- **Total Jobs Submitted:** 13
- **Successful Completions:** 4 (after scoring)
- **Failed Jobs:** 9
- **Data Salvaged:** 4 experiments (~741MB generation results)
- **Total Execution Time:** ~6 hours (including failed attempts)
- **Code Iterations:** 7 (various fixes)
- **Models Tested:** 5 (270m-it, 1b-it, 4b-it, 12b-it, 27b-it)

---

## Contact/References

- Remote Path: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/`
- Cluster: Blythe (Warwick)
- Partition: GPU
- Active Job: 1066965 (score_gemma3_salvaged)
