#!/usr/bin/env python3
"""
Submit only the 27B model experiments
"""

import subprocess
import time
import sys
from pathlib import Path

# Import the functions from submit_core_experiments
from submit_core_experiments import create_job_script, submit_job

# 27B model experiments
EXPERIMENTS_27B = [
    ("gemma2-27b", "google/gemma-2-27b", 24, False, "coarse", "quick"),
    ("gemma2-27b-mlp", "google/gemma-2-27b", 24, True, "coarse", "quick"),
    ("gemma3-27b", "google/gemma-3-27b", 26, False, "coarse", "quick"),
    ("gemma3-27b-mlp", "google/gemma-3-27b", 26, True, "coarse", "quick"),
]

def main():
    """Submit 27B model experiments"""

    print("=" * 60)
    print("Submitting 27B Model Experiments")
    print("=" * 60)
    print()

    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)

    print("Experiments to submit:")
    for i, exp in enumerate(EXPERIMENTS_27B, 1):
        model_key, _, layer, use_mlp, _, _ = exp
        mlp_str = " (+MLP)" if use_mlp else ""
        print(f"  {i}. {model_key} (Layer {layer}){mlp_str}")
    print()

    job_ids = []
    for exp in EXPERIMENTS_27B:
        job_id = submit_job(exp, dry_run=False)
        if job_id:
            job_ids.append(job_id)
        time.sleep(1)  # Small delay between submissions

    print()
    print("=" * 60)
    print(f"Successfully submitted {len(job_ids)}/{len(EXPERIMENTS_27B)} jobs")

    if job_ids:
        print("\nTo check status:")
        print(f"  squeue -j {','.join(job_ids)}")
        print("\nTo cancel all:")
        print(f"  scancel {' '.join(job_ids)}")

    return 0 if len(job_ids) == len(EXPERIMENTS_27B) else 1

if __name__ == "__main__":
    sys.exit(main())