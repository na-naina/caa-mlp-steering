#!/bin/bash

# Submit a single experiment for testing
echo "Submitting single gemma2-2b experiment (no MLP)..."

# Run the submission with a specific filter
python -c "
import sys
sys.path.insert(0, '.')
from submit_core_experiments import submit_job

# Single experiment: gemma2-2b without MLP
exp = ('gemma2-2b', 'google/gemma-2-2b', 12, False, 'standard', 'standard')
job_id = submit_job(exp, dry_run=False)
if job_id:
    print(f'Successfully submitted job: {job_id}')
    print(f'Check status with: squeue -j {job_id}')
    print(f'Watch logs with: tail -f logs/gemma2-2b_L12_{job_id}.*')
"