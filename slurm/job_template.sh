#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --partition={{partition}}
{{qos_line}}
{{account_line}}
#SBATCH --time={{time}}
#SBATCH --cpus-per-task={{cpus}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --output=logs/{{job_name}}_%j.out
#SBATCH --error=logs/{{job_name}}_%j.err

set -euo pipefail

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source {{project_root}}/venv/bin/activate
if [ -f "{{shared_env_script}}" ]; then
  source "{{shared_env_script}}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH={{project_root}}
export HF_HOME={{hf_cache}}

cd {{project_root}}

{{python_command}}
