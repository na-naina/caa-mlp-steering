# NaN Issues in Gemma3-IT Large Models

## Quick Summary

**Problem**: Gemma3-IT models ≥12B experience severe NaN (Not a Number) activations during CAA steering vector extraction, causing 38% (12B) and 19% (27B) data loss.

**Threshold**: Works fine for 270M, 1B, 4B. Breaks at 12B+.

**Mitigations Tried**: Float16 dtype + reduced context (384) → Still fails

**Key Finding**: 100% overlap in failed prompts - all 19 prompts that fail in 27B also fail in 12B, suggesting specific prompt characteristics trigger NaNs (not random numerical noise)

---

## Files in This Folder

### 1. Job Logs
- `job_1067566_gemma3_12b_it_fp16.err` - **12B model** (FAILED with NaNs)
- `job_1067567_gemma3_27b_it_fp16.err` - **27B model** (FAILED with NaNs)
- `job_1067485_gemma3_4b_it_SUCCESS.err` - **4B model** (SUCCESS, for comparison)

### 2. Skipped Prompts
- `12b_skipped_prompts.txt` - 38 questions that caused NaN in 12B
- `27b_skipped_prompts.txt` - 19 questions that caused NaN in 27B

### 3. Configurations
- `gemma3_12b_it_fp16_short.yaml` - Config used for 12B (with mitigations)
- `gemma3_27b_it_fp16_short.yaml` - Config used for 27B (with mitigations)
- `gemma3_12b_it_full.yaml` - Base config for 12B
- `gemma3_27b_it_full.yaml` - Base config for 27B
- `config_gemma3_4b_it_full_SUCCESS.yaml` - Working 4B config

### 4. Analysis
- `OVERLAP_ANALYSIS.md` - **Critical finding: 100% overlap analysis and prompt patterns**
- `README.md` - This file

---

## The Problem

### What Happens
During activation extraction from model layers, certain prompts trigger NaN values:

```
2025-11-17 10:54:09,297 | ERROR | Invalid activations in batch example 0: NaN count=130560
2025-11-17 10:54:09,297 | ERROR | Problematic text: Question: Who is the richest person...
2025-11-17 10:54:09,298 | WARNING | Skipping index 0 due to NaN/Inf
```

This happens for **38/100 prompts** in 12B and **19/100 prompts** in 27B.

### Data Loss Impact

| Model | Total Prompts | Skipped | Success Rate | Usable Data |
|-------|---------------|---------|--------------|-------------|
| 4B | 100 | 0 | 100% | ✅ Full dataset |
| 12B | 100 | 38 | 62% | ⚠️ Reduced dataset |
| 27B | 100 | 19 | 81% | ⚠️ Reduced dataset |

---

## What We Tried

### ✅ Fixed Issues
1. **Config Bug**: Fixed base.yaml override (was loading wrong model)
2. **Correct Models**: Now loading gemma-3-12b-it and gemma-3-27b-it

### ❌ Ineffective Mitigations
1. **Float16 dtype**: Changed from bfloat16 → Still NaNs
2. **Reduced Context**: max_length 512 → 384 → Still NaNs

### Mitigation Comparison
```yaml
# 4B (WORKS)
model:
  dtype: bfloat16
steering:
  max_length: 512

# 12B/27B (FAILS)
model:
  dtype: float16    # ← Changed, didn't help
steering:
  max_length: 384   # ← Changed, didn't help
```

---

## Technical Details

### NaN Characteristics

**12B Model**:
- NaN counts: 80,640 - 157,440 per failed prompt
- Varies by prompt length/complexity
- No clear content pattern

**27B Model**:
- NaN counts: 166,656 - 198,912 per failed prompt
- Higher counts but fewer total failures
- Suggests different failure mode than 12B

### Extraction Process Flow
```
1. Load 100 CAA prompt pairs          ✓
2. Extract activations from layer 18
   ├─ Batch processing
   └─ [NaN Detection] → Skip bad prompts
3. Build vector bank (62-81 samples)  ⚠️ Reduced
4. Train MLP on partial data          ⚠️ Biased
5. Evaluate                            ⚠️ Questionable
```

---

## Why 4B Works But 12B Doesn't

### Hypothesis 1: Numerical Precision
- **4B**: Smaller depth, activations stay in fp16 range
- **12B+**: Deeper network, cumulative errors → overflow/underflow

### Hypothesis 2: Architecture Changes
- Gemma3-IT uses instruction tuning
- Possible use of operations unstable in reduced precision
- Layer norm or attention patterns differ at scale

### Hypothesis 3: Activation Magnitudes
- Larger models produce larger intermediate activations
- Layer 18 (12B) vs Layer 16 (4B) - deeper = more unstable
- Some prompts trigger extreme values

---

## Quick Start for Investigation

### 1. Examine Skip Patterns
```bash
# See what questions fail
cat 12b_skipped_prompts.txt
cat 27b_skipped_prompts.txt

# Compare overlap
comm -12 <(sort 12b_skipped_prompts.txt) <(sort 27b_skipped_prompts.txt)
```

### 2. Check NaN Locations in Logs
```bash
# Count NaN occurrences
grep -c "NaN count" job_1067566_gemma3_12b_it_fp16.err

# See NaN values distribution
grep "NaN count=" job_1067566_gemma3_12b_it_fp16.err | sed 's/.*NaN count=//' | sed 's/,.*//' | sort -n | uniq -c
```

### 3. Compare with Working Model
```bash
# 4B should have no NaN errors
grep "NaN\|ERROR" job_1067485_gemma3_4b_it_SUCCESS.err
```

---

## Potential Solutions (Untested)

### Option 1: Full Precision
```yaml
model:
  dtype: float32  # Higher precision, more memory
```

### Option 2: Earlier Layer Extraction
```yaml
model:
  layer: 12  # Instead of 18 (12B) or 26 (27B)
```

### Option 3: Gradient Checkpointing
Modify extraction code to use gradient checkpointing

### Option 4: Accept Data Loss
Continue with reduced samples, document in results

---

## Questions to Investigate

1. **✅ ANSWERED: Is there overlap in failed prompts between 12B and 27B?**
   - **YES - 100% overlap**: All 19 prompts failing in 27B also fail in 12B
   - **Conclusion**: Specific prompt characteristics trigger NaNs (see OVERLAP_ANALYSIS.md)
   - 12B additionally fails on 19 more prompts that 27B handles successfully

2. **What layer do NaNs originate from?**
   - Need per-layer activation inspection
   - May be specific layer types (attention vs MLP)

3. **Does fp32 solve it?**
   - Memory intensive but definitive test
   - Would isolate precision vs architectural issue

4. **Why does 27B have fewer failures than 12B?**
   - Counterintuitive (larger = more stable?)
   - Suggests non-linear relationship

---

## Contact Context

**Date**: November 17, 2025
**Jobs**: 1067566 (12B), 1067567 (27B) - Both still running with data loss
**Previous Success**: 270M, 1B, 4B all completed with 100% success rate
**Git Repo**: caa-mlp-steering
**Remote**: blythe.scrtp.warwick.ac.uk

---

## Summary for Next Instance

"The CAA steering pipeline works perfectly for Gemma3-IT models up to 4B parameters (bfloat16, full context). At 12B+, severe NaN activations appear during extraction, skipping 19-38% of prompts despite fp16 dtype and reduced context mitigations. **Critical finding: 100% overlap in failed prompts** - all 19 prompts that fail in 27B also fail in 12B, indicating specific prompt characteristics trigger NaNs (not random numerical noise). The 4B vs 12B threshold suggests a critical numerical stability limit. See OVERLAP_ANALYSIS.md for detailed prompt patterns and investigation recommendations."
