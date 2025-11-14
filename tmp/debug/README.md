# Debug Folder Contents

This folder contains all relevant files and context for debugging the SLURM job failures (jobs 1060263, 1060264, 1060265).

## Files

### Documentation
- **PROBLEM_SUMMARY.md** - Comprehensive analysis of all failures, root causes, and proposed solutions

### Config Files (Local Versions)
- **gemma2_27b_full.yaml** - Config for Gemma-2 27B experiment
- **gemma3_27b_full.yaml** - Config for Gemma-3 27B experiment
- **gemma3_12b_full.yaml** - Config for Gemma-3 12B experiment

### Source Code
- **loader.py** - Model loading logic with max_memory support (local version)
- **apply.py** - Steering application with multimodal fix (local version)
- **apply.py.remote** - Remote version MISSING multimodal fix (shows the sync issue)
- **run_experiment_excerpt.py** - Relevant section showing model/judge loading logic

### Error Logs
- **job_1060263_gemma2_27b_error.log** - OOM error on GPU 2
- **job_1060264_gemma3_12b_error.log** - Architecture detection error

## Quick Summary

**Job 1060263 (Gemma-2 27B)**: OOM because main model used all 3 GPUs despite max_memory constraints

**Job 1060264 (Gemma-3 12B)**: Missing multimodal architecture detection in remote apply.py

## Key Question

Why does `max_memory: {0: "40GiB", 1: "40GiB", 2: "0GiB"}` not prevent the model from using GPU 2?

Start with **PROBLEM_SUMMARY.md** for full details.
