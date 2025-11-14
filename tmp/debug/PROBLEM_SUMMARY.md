# SLURM Job Failure Debug Report

## Overview
Three SLURM jobs (1060263, 1060264, 1060265) for running CAA experiments on 27B and 12B Gemma models all failed. This debug folder contains all relevant files for analysis.

## Job Failures

### Job 1060263: Gemma-2 27B (FAILED)
- **Error**: `torch.OutOfMemoryError: CUDA out of memory` on GPU 2
- **Details**:
  - GPU 2 has 44.42 GiB total (48GB GPU)
  - GPU 2 had 44.40 GiB in use when judge model tried to load
  - Main model (google/gemma-2-27b) was supposed to use only GPUs 0-1
  - Judge model (google/gemma-3-12b-it) was supposed to use GPU 2
  - **Root cause**: Main model with `device_map: auto` spread across ALL 3 GPUs (0, 1, 2) despite `max_memory` constraints

### Job 1060264: Gemma-3 12B (FAILED)
- **Error**: `ValueError: Unsupported model architecture for steering` in `src/steering/apply.py:15`
- **Root cause**: Remote `apply.py` is missing the multimodal architecture detection fix
  - Local version has the fix (checks for `model.model.language_model.layers`)
  - Remote version only checks `model.model.layers` and `model.transformer.h`
  - Gemma-3 12B uses multimodal architecture

### Job 1060265: Gemma-3 27B (CANCELLED)
- Status: CANCELLED by SLURM (likely dependency on failed jobs)

## Technical Context

### GPU Allocation Issue (Job 1060263)
The Gemma-2 27B model:
- Size: ~27B parameters in bfloat16 = ~54GB
- **Should fit** in 2×48GB GPUs (96GB total)
- **Actually used**: ~132GB spread across 3 GPUs (~44GB each)
- This suggests either:
  1. `max_memory` parameter is being ignored
  2. PyTorch's `device_map: auto` doesn't respect `max_memory` correctly
  3. Model has significant activation overhead during loading

### Code Sync Issue (Job 1060264)
The multimodal architecture fix was applied locally but not synced to remote server.

## Configuration Files

### gemma2_27b_full.yaml
```yaml
model:
  name: google/gemma-2-27b
  family: gemma2
  layer: 24
  dtype: bfloat16
  device_map: auto
  max_memory:
    0: "40GiB"  # Main model uses GPUs 0 and 1
    1: "40GiB"
    2: "0GiB"   # Reserve GPU 2 for judge

slurm:
  gpus: 3  # 2 GPUs for 27b model + 1 for judge
  cpus: 16
  mem_gb: 180
  time: "24:00:00"

evaluation:
  judge:
    model: google/gemma-3-12b-it
    device_map: cuda:2  # Judge on third GPU
```

### gemma3_27b_full.yaml
Same structure as gemma2_27b_full.yaml but for:
- `model.name: google/gemma-3-27b-pt`
- `model.family: gemma3`
- `model.layer: 26`

## Questions for Investigation

1. **Why does `max_memory` not prevent the main model from using GPU 2?**
   - Is `max_memory` being passed correctly to `AutoModelForCausalLM.from_pretrained()`?
   - Does HuggingFace `device_map: auto` respect `max_memory` constraints?
   - Are there PyTorch/CUDA environment variables interfering?

2. **Why does the 27B model use ~132GB instead of ~54GB?**
   - Is this activation overhead during loading?
   - Does it compact after loading is complete?
   - Should we use different loading strategies?

3. **What's the correct fix for GPU isolation?**
   - Option A: Request 4 GPUs (simple, wasteful)
   - Option B: Use `CUDA_VISIBLE_DEVICES` to hide GPU 2 during main model loading
   - Option C: Fix the `max_memory` implementation
   - Option D: Load main model to specific devices (cuda:0, cuda:1) instead of auto

## Proposed Solutions

### Solution 1: Request 4 GPUs (Immediate Fix)
**Pros**: Guaranteed to work, simple config change
**Cons**: Wastes 1 GPU, longer queue times

Change `slurm.gpus: 4` in both 27B configs

### Solution 2: CUDA_VISIBLE_DEVICES Control
**Pros**: Efficient GPU usage, should work reliably
**Cons**: Requires code changes to `run_experiment.py`

```python
import os

# Before loading main model
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Hide GPU 2
loaded = load_causal_model(...)

# Before loading judge
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # Restore GPU 2
judge = _maybe_build_judge(...)
```

### Solution 3: Explicit Device Mapping
**Pros**: Most control over allocation
**Cons**: Complex, model-specific tuning

Instead of `device_map: auto`, manually specify which layers go to which GPU.

## Files in This Debug Folder

1. **Config files** (local versions):
   - `gemma2_27b_full.yaml`
   - `gemma3_27b_full.yaml`
   - `gemma3_12b_full.yaml`

2. **Source code** (local versions):
   - `loader.py` - Model loading logic with `max_memory` support
   - `apply.py` - Steering application (HAS multimodal fix locally)
   - `apply.py.remote` - Remote version (MISSING multimodal fix)
   - `run_experiment_excerpt.py` - Relevant section showing model loading

3. **Error logs**:
   - `job_1060263_gemma2_27b_error.log` - OOM error on GPU 2
   - `job_1060264_gemma3_12b_error.log` - Architecture error

## Next Steps

1. Fix code sync issue: Ensure `apply.py` multimodal fix is synced to remote
2. Choose GPU allocation strategy (recommend Solution 1 for immediate fix)
3. Investigate why `max_memory` doesn't work as expected
4. Submit new jobs with fixes applied
