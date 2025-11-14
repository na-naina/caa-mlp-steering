#!/bin/bash

# Submit experiments without judge model for faster testing
echo "Submitting gemma2-2b experiment WITHOUT judge model..."
echo "(This will only do MC evaluation, not generation evaluation)"

python -c "
import sys
sys.path.insert(0, '.')
from submit_core_experiments import submit_job

# Modified experiment with no judge model
exp = ('gemma2-2b-no-judge', 'google/gemma-2-2b', 12, False, 'standard', 'standard')

# Create a custom job script
import subprocess
import time
from pathlib import Path

def create_custom_job():
    script_content = '''#!/bin/bash
#SBATCH --job-name=caa_gemma2-2b_nojudge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/gemma2-2b_nojudge_%j.out
#SBATCH --error=logs/gemma2-2b_nojudge_%j.err

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Setup shared storage
SHARED_DIR=\"/springbrook/share/dcsresearch/u5584851\"
PROJECT_DIR=\"\$SHARED_DIR/caa_experiments\"

if [ -d \"\$SHARED_DIR\" ]; then
    echo \"Using shared storage at \$SHARED_DIR\"
    mkdir -p \"\$PROJECT_DIR/cache/huggingface\"
    mkdir -p \"\$PROJECT_DIR/cache/transformers\"
    mkdir -p \"\$PROJECT_DIR/results\"

    export HF_HOME=\"\$PROJECT_DIR/cache/huggingface\"
    export TRANSFORMERS_CACHE=\"\$PROJECT_DIR/cache/transformers\"
    export HF_DATASETS_CACHE=\"\$PROJECT_DIR/cache/datasets\"

    if [ -f ~/.cache/huggingface/token ] && [ ! -f \"\$HF_HOME/token\" ]; then
        cp ~/.cache/huggingface/token \"\$HF_HOME/token\"
    fi

    if [ -f \"\$HF_HOME/token\" ]; then
        export HF_TOKEN=\$(cat \"\$HF_HOME/token\")
    fi

    OUTPUT_BASE=\"\$PROJECT_DIR/results\"
else
    OUTPUT_BASE=\"\$PWD/results\"
fi

echo \"Running experiment WITHOUT judge model (MC evaluation only)...\"

# Run with empty judge model to skip generation evaluation
python caa_truthfulqa.py \\
    --model_name google/gemma-2-2b \\
    --layer 12 \\
    --scales 0 0.5 1.0 2.0 5.0 10.0 \\
    --caa_samples 100 \\
    --num_mc_samples 200 \\
    --num_gen_samples 0 \\
    --judge_model \"\" \\
    --output_dir \$OUTPUT_BASE/gemma2-2b-nojudge

echo \"Experiment completed\"
'''

    script_path = 'temp_nojudge.slurm'
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Submit the job
    result = subprocess.run(
        ['sbatch', script_path],
        capture_output=True,
        text=True,
        check=True
    )

    job_id = result.stdout.strip().split()[-1]
    print(f'Successfully submitted job: {job_id}')
    print(f'Check status with: squeue -j {job_id}')
    print(f'Watch logs with: tail -f logs/gemma2-2b_nojudge_{job_id}.*')

    # Clean up
    Path(script_path).unlink()
    return job_id

create_custom_job()
"