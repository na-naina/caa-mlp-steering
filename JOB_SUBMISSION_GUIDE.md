# Complete Job Submission Guide for Gemma CAA Experiments

## Overview
This guide contains all steps to run CAA (Contrastive Activation Addition) experiments on TruthfulQA dataset with Gemma-2 and Gemma-3 models on Blythe HPC cluster.

## Problem Context & Solutions

### Issue 1: NaN/Inf Numerical Instability
**Problem**: Some TruthfulQA examples (e.g., "What happens if you see a single magpie?") cause NaN values in model activations at certain layers due to bfloat16 numerical overflow/underflow.

**Solution**: Implemented paired filtering that maintains 1:1 correspondence between positive and negative activations while skipping problematic examples.

### Issue 2: GPU Device Allocation
**Problem**: Using `device_map="auto"` for main model causes it to use all available GPUs, leaving none for the judge model → "CUDA error: invalid device ordinal".

**Solution**: Explicit device mapping in configs:
- Small models (2b, 9b, 1b): `device_map: cuda:0` restricts main to GPU 0, judge uses GPU 1
- Large model (27b): `device_map: auto` uses GPU 0,1 for main, judge uses GPU 2

---

## Environment & Cache Setup

### HuggingFace Authentication
The pipeline requires HuggingFace authentication for accessing Gemma models.

**Token location** (on remote):
```
/springbrook/home/c/csuqqj/.cache/huggingface/token
```

**Verify authentication**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cat ~/.cache/huggingface/token"
```

If missing, create token:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
```

### Cache Directory Setup
**CRITICAL**: Set `HF_HOME` to prevent cache conflicts and ensure consistent model loading.

**Remote cache location**:
```
/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers
```

**Environment variables** (set in all SLURM scripts):
```bash
export HF_HOME=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Verify cache directory exists**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -la /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/"
```

**Create if missing**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "mkdir -p /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers"
```

### Python Virtual Environment
**Location**:
```
/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv
```

**Activation** (in SLURM scripts):
```bash
source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate
```

**Verify environment**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate && python --version && pip list | grep -E 'torch|transformers'"
```

### Module Loading
**Required modules** (on Blythe):
```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
```

**Check available modules**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "module avail Python"
ssh csuqqj@blythe.scrtp.warwick.ac.uk "module avail CUDA"
```

---

## Code Changes Required

### 1. Activation Extraction with Paired Filtering
**File**: `src/steering/extract.py`

**Add NaN/Inf detection in `_run_batch()` method** (around line 82):
```python
def _run_batch(self, texts: List[str]) -> torch.Tensor:
    activations: List[torch.Tensor] = []

    def collect(hidden: torch.Tensor) -> None:
        # Upcast to fp32 to avoid bfloat16 numerical issues
        activations.append(hidden.float())

    with _activation_hook(self.layer, collect):
        encoded = self.loaded.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.loaded.primary_device) for k, v in encoded.items()}
        # Disable autocast during extraction to prevent precision issues
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            _ = self.loaded.model(**encoded)

    if not activations:
        raise RuntimeError("Failed to capture activations for batch")

    hidden = activations[0]

    # Check for NaN/Inf in raw activations
    if torch.isnan(hidden).any() or torch.isinf(hidden).any():
        # Find which examples in the batch have issues
        for i, text in enumerate(texts):
            example_acts = hidden[i]
            if torch.isnan(example_acts).any() or torch.isinf(example_acts).any():
                nan_count = torch.isnan(example_acts).sum().item()
                inf_count = torch.isinf(example_acts).sum().item()
                logger.error(
                    f"Invalid activations in batch example {i}: "
                    f"NaN count={nan_count}, Inf count={inf_count}"
                )
                logger.error(f"Problematic text (first 200 chars): {text[:200]}")
        raise ValueError("Activations contain NaN or Inf values")

    mean_hidden = hidden.mean(dim=1)  # batch, hidden_dim
    return mean_hidden.cpu()
```

**Replace `collect_mean_activations()` method** (line 102-149):
```python
def collect_mean_activations(self, texts: Iterable[str]) -> tuple[torch.Tensor, list[int]]:
    """
    Collect activations and return tuple of (activations, valid_indices).

    Returns:
        activations: Tensor of shape (num_valid, hidden_dim)
        valid_indices: List of original indices that produced valid activations
    """
    text_list = list(texts)
    batches = [
        text_list[i : i + self.batch_size]
        for i in range(0, len(text_list), self.batch_size)
    ]

    all_activations = []
    valid_indices = []
    current_idx = 0

    for batch in batches:
        try:
            batch_acts = self._run_batch(batch)
            all_activations.append(batch_acts)
            # All examples in batch were valid
            valid_indices.extend(range(current_idx, current_idx + len(batch)))
            current_idx += len(batch)
        except ValueError as e:
            if "NaN or Inf" in str(e):
                logger.warning(f"NaN/Inf in batch starting at index {current_idx}, processing individually")
                # Process examples one at a time to identify valid ones
                for i, text in enumerate(batch):
                    try:
                        result = self._run_batch([text])
                        all_activations.append(result)
                        valid_indices.append(current_idx + i)
                    except ValueError as e_single:
                        if "NaN or Inf" in str(e_single):
                            logger.warning(f"Skipping index {current_idx + i} due to NaN/Inf: {text[:100]}...")
                        else:
                            raise
                current_idx += len(batch)
            else:
                raise

    if not all_activations:
        raise RuntimeError("All examples produced NaN/Inf - no valid activations collected")

    activations = torch.cat(all_activations, dim=0)
    return activations, valid_indices
```

### 2. Paired Filtering in Experiment Runner
**File**: `src/jobs/run_experiment.py`

**Modify `_build_vector_bank()` function** (around line 200):

Find the section that starts with:
```python
LOGGER.info("Collecting steering activations (%d prompts)", len(pool_prompts_pos))
```

Replace the entire activation collection and CAA vector computation section with:
```python
LOGGER.info("Collecting steering activations (%d prompts)", len(pool_prompts_pos))
pos_acts, pos_valid_indices = extractor.collect_mean_activations(pool_prompts_pos)
neg_acts, neg_valid_indices = extractor.collect_mean_activations(pool_prompts_neg)

# Keep only pairs where both positive and negative are valid
valid_pair_indices = sorted(set(pos_valid_indices) & set(neg_valid_indices))

if len(valid_pair_indices) < len(pool_prompts_pos):
    num_skipped = len(pool_prompts_pos) - len(valid_pair_indices)
    LOGGER.warning(
        f"Skipped {num_skipped} pairs due to NaN/Inf activations, "
        f"using {len(valid_pair_indices)} valid pairs"
    )

# Filter to keep only valid pairs
pos_mask = torch.tensor([i in valid_pair_indices for i in pos_valid_indices])
neg_mask = torch.tensor([i in valid_pair_indices for i in neg_valid_indices])
pos_acts = pos_acts[pos_mask]
neg_acts = neg_acts[neg_mask]

if len(pos_acts) == 0:
    raise RuntimeError("No valid activation pairs remaining after NaN/Inf filtering")

vector_dir = run_dir / "vectors"
vector_dir.mkdir(exist_ok=True)

base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=True)
```

---

## Configuration Files

### Base Configuration
**File**: `configs/base.yaml` (should already exist)

Verify it contains:
```yaml
data:
  dataset: truthful_qa
  steering_pool: 100
  train: 250
  val: 117
  test: 200

mlp:
  steps_per_epoch: 50
  samples_per_step: 25
  epochs: 2
  learning_rate: 0.0001
  margin: 1.0
  mse_reg: 0.01
```

### Gemma-2 Full Dataset Configs

**File**: `configs/gemma2_2b_full.yaml`
```yaml
model:
  name: google/gemma-2-2b
  family: gemma2
  layer: 12
  dtype: bfloat16
  device_map: cuda:0  # Restrict main model to GPU 0, leave GPU 1 for judge

slurm:
  gpus: 2  # Main model + judge
  cpus: 10
  mem_gb: 80
  time: "12:00:00"

# Use full dataset from base.yaml:
# steering_pool: 100, train: 250, val: 117, test: 200

evaluation:
  judge:
    model: google/gemma-3-12b-it
    device_map: cuda:1  # Judge on second GPU
```

**File**: `configs/gemma2_9b_full.yaml`
```yaml
model:
  name: google/gemma-2-9b
  family: gemma2
  layer: 21
  dtype: bfloat16
  device_map: cuda:0  # Restrict main model to GPU 0, leave GPU 1 for judge

slurm:
  gpus: 2  # Main model + judge
  cpus: 10
  mem_gb: 128
  time: "12:00:00"

# Use full dataset from base.yaml:
# steering_pool: 100, train: 250, val: 117, test: 200

evaluation:
  judge:
    model: google/gemma-3-12b-it
    device_map: cuda:1  # Judge on second GPU
```

**File**: `configs/gemma2_27b_full.yaml`
```yaml
model:
  name: google/gemma-2-27b
  family: gemma2
  layer: 24
  dtype: bfloat16
  device_map: auto  # 27b needs 2 GPUs, auto will use cuda:0,1, judge on cuda:2

slurm:
  gpus: 3  # 2 GPUs for 27b model + 1 for judge
  cpus: 16
  mem_gb: 180
  time: "24:00:00"

# Use full dataset from base.yaml:
# steering_pool: 100, train: 250, val: 117, test: 200

evaluation:
  judge:
    model: google/gemma-3-12b-it
    device_map: cuda:2  # Judge on third GPU (model uses cuda:0,1)
```

### Gemma-3 Smoke Test Config

**File**: `configs/smoke_gemma3_1b.yaml`
```yaml
model:
  name: google/gemma-3-1b-pt
  family: gemma3
  layer: 12
  dtype: bfloat16
  device_map: cuda:0  # Restrict main model to GPU 0, leave GPU 1 for judge

slurm:
  gpus: 2  # Main model + judge
  cpus: 8
  mem_gb: 48
  time: "01:00:00"

# Smoke test: smaller splits
data:
  steering_pool: 20
  train: 30
  val: 20
  test: 20

mlp:
  steps_per_epoch: 10
  samples_per_step: 5

evaluation:
  judge:
    model: google/gemma-3-12b-it
    device_map: cuda:1  # Judge on second GPU
```

---

## SLURM Scripts

### Important Notes
- **ALWAYS match GPU count** between config `slurm.gpus` and SLURM `--gres=gpu` directive
- **Module loading order matters**: purge first, then load Python, then CUDA
- **Environment variables** must be set before running Python

### Gemma-2 2B Full Dataset

**File**: `slurm/gemma2_2b_full.sh`
```bash
#!/bin/bash
#SBATCH --job-name=gemma2_2b_full
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:lovelace_l40:2
#SBATCH --output=logs/gemma2_2b_full_%j.out
#SBATCH --error=logs/gemma2_2b_full_%j.err

set -euo pipefail

# Module loading and environment setup
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
export HF_HOME=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers

cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma

python -m src.jobs.run_experiment --base-config configs/base.yaml --config configs/gemma2_2b_full.yaml --run-id gemma2_2b_full_$(date +%Y%m%d_%H%M%S)
```

### Gemma-2 9B Full Dataset

**File**: `slurm/gemma2_9b_full.sh`
```bash
#!/bin/bash
#SBATCH --job-name=gemma2_9b_full
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:lovelace_l40:2
#SBATCH --output=logs/gemma2_9b_full_%j.out
#SBATCH --error=logs/gemma2_9b_full_%j.err

set -euo pipefail

# Module loading and environment setup
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
export HF_HOME=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers

cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma

python -m src.jobs.run_experiment --base-config configs/base.yaml --config configs/gemma2_9b_full.yaml --run-id gemma2_9b_full_$(date +%Y%m%d_%H%M%S)
```

### Gemma-2 27B Full Dataset

**File**: `slurm/gemma2_27b_full.sh`
```bash
#!/bin/bash
#SBATCH --job-name=gemma2_27b_full
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:lovelace_l40:3
#SBATCH --output=logs/gemma2_27b_full_%j.out
#SBATCH --error=logs/gemma2_27b_full_%j.err

set -euo pipefail

# Module loading and environment setup
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
export HF_HOME=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers

cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma

python -m src.jobs.run_experiment --base-config configs/base.yaml --config configs/gemma2_27b_full.yaml --run-id gemma2_27b_full_$(date +%Y%m%d_%H%M%S)
```

### Gemma-3 1B Smoke Test

**File**: `slurm/smoke_gemma3_1b.sh`
```bash
#!/bin/bash
#SBATCH --job-name=smoke_gemma3_1b
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:lovelace_l40:2
#SBATCH --output=logs/smoke_gemma3_1b_%j.out
#SBATCH --error=logs/smoke_gemma3_1b_%j.err

set -euo pipefail

# Module loading and environment setup
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
export HF_HOME=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers

cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma

python -m src.jobs.run_experiment --base-config configs/base.yaml --config configs/smoke_gemma3_1b.yaml --run-id smoke_gemma3_1b_$(date +%Y%m%d_%H%M%S)
```

---

## Deployment Steps

### 1. Pre-flight Checks

**Verify cache directory exists**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -la /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers"
```

**Verify HuggingFace token**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "test -f ~/.cache/huggingface/token && echo 'Token exists' || echo 'Token missing!'"
```

**Verify logs directory**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "mkdir -p /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs"
```

### 2. Upload Files to Remote

**From local directory**:
```bash
cd "/home/tarantulala/Dev/Uni/mi_reasoning_research/SLURM CAA experiments"

# Upload modified Python code
rsync -avz src/steering/extract.py csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/src/steering/
rsync -avz src/jobs/run_experiment.py csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/src/jobs/

# Upload all config files
rsync -avz configs/*.yaml csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/configs/

# Upload all SLURM scripts
rsync -avz slurm/*.sh csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/slurm/
```

### 3. Verify Upload

**Check files on remote**:
```bash
# Check Python files
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -lh /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/src/steering/extract.py"
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -lh /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/src/jobs/run_experiment.py"

# Check configs
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -lh /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/configs/*.yaml"

# Check SLURM scripts
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -lh /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/slurm/*.sh"

# Verify GPU allocation in SLURM scripts
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'gres=gpu' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/slurm/gemma2_*_full.sh"
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'gres=gpu' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/slurm/smoke_gemma3_1b.sh"
```

Expected output:
```
gemma2_2b_full.sh:#SBATCH --gres=gpu:lovelace_l40:2
gemma2_9b_full.sh:#SBATCH --gres=gpu:lovelace_l40:2
gemma2_27b_full.sh:#SBATCH --gres=gpu:lovelace_l40:3
smoke_gemma3_1b.sh:#SBATCH --gres=gpu:lovelace_l40:2
```

### 4. Submit Jobs

**Submit all Gemma-2 full dataset jobs**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/gemma2_2b_full.sh"
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/gemma2_9b_full.sh"
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/gemma2_27b_full.sh"
```

**Submit Gemma-3 smoke test**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma && sbatch slurm/smoke_gemma3_1b.sh"
```

### 5. Verify Submission

```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -u csuqqj --format='%.10i %.20j %.8T %.6D %.8C %.10m'"
```

Expected output:
```
JOBID        NAME              STATE  NODES  CPUS  MIN_MEMORY
<id>     gemma2_2b_full       PENDING   1     10      80G
<id>     gemma2_9b_full       PENDING   1     10     128G
<id>     gemma2_27b_full      PENDING   1     16     180G
<id>     smoke_gemma3_1b      PENDING   1      8      48G
```

---

## Monitoring Jobs

### Check Queue Status

**Basic queue check**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -u csuqqj"
```

**Detailed queue status**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -u csuqqj --format='%.10i %.20j %.8T %.10r %.12P %.10Q %.10M %.10l'"
```

**Check job priority**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "sprio -j <JOB_ID>"
```

### Monitor Job Logs

**Watch live error log**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "tail -f /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"
```

**Check for errors**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "tail -100 /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err | grep -E 'ERROR|NaN|Inf|CUDA|Traceback'"
```

**Check for success indicators**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "tail -100 /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err | grep -E 'INFO|Complete|accuracy|Evaluating'"
```

**Monitor specific stages**:
```bash
# Check if model loaded successfully
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'Model loaded' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"

# Check if steering vectors computed
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'Built vector bank' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"

# Check if judge loaded
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'Loading model.*gemma-3-12b' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"

# Check MLP training progress
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'MLP epoch' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"

# Check evaluation progress
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'Evaluating' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"
```

### Check GPU Partition Status

**Overall GPU availability**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "sinfo -p gpu -o '%P %a %l %D %T %N'"
```

**Running jobs on GPU partition**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -p gpu -t RUNNING --format='%.10i %.12u %.20j %.10M %.10l %.6D %.15R'"
```

**Your position in queue**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -p gpu -t PENDING --format='%.10i %.12u %.20j %.8T %.10r %.12P' | head -20"
```

### Job Control Commands

**Cancel a job**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "scancel <JOB_ID>"
```

**Cancel all your jobs**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "scancel -u csuqqj"
```

**Cancel specific job types**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "scancel -n gemma2_2b_full"
```

**Hold a job** (prevent from starting):
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "scontrol hold <JOB_ID>"
```

**Release a held job**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "scontrol release <JOB_ID>"
```

---

## Retrieving Results

### Check for Completed Jobs

**List output directories**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "ls -lhd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/*/"
```

**Check for results files**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "find /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/ -name 'results.json' -exec ls -lh {} \;"
```

### Download Results

**Download specific run**:
```bash
cd "/home/tarantulala/Dev/Uni/mi_reasoning_research/SLURM CAA experiments"
mkdir -p analysis_results/<run_id>
rsync -avz --progress csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/<run_id>/ analysis_results/<run_id>/
```

**Download all results**:
```bash
cd "/home/tarantulala/Dev/Uni/mi_reasoning_research/SLURM CAA experiments"
rsync -avz --progress csuqqj@blythe.scrtp.warwick.ac.uk:/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/ analysis_results/
```

**Quick check results**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "cat /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs/<run_id>/results.json | python3 -m json.tool | head -50"
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA error: invalid device ordinal"
**Cause**: Mismatch between config `device_map` and actual GPUs allocated by SLURM.

**Check**:
```bash
# Verify SLURM script GPU allocation
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'gres=gpu' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/slurm/<script>.sh"

# Verify config device_map
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'device_map' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/configs/<config>.yaml"
```

**Fix**: Ensure both match:
- 2 GPUs: SLURM `--gres=gpu:lovelace_l40:2` + config `device_map: cuda:0`
- 3 GPUs: SLURM `--gres=gpu:lovelace_l40:3` + config `device_map: auto` (for 27b)

#### Issue: "No valid activation pairs remaining"
**Cause**: All steering examples produced NaN/Inf activations.

**Check**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep 'Skipping index' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err | wc -l"
```

**Fix**:
- Try different layer (`layer: 12` → `layer: 10`)
- Try different model variant
- Check data quality

#### Issue: "HuggingFace token not found"
**Check**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "test -f ~/.cache/huggingface/token && cat ~/.cache/huggingface/token || echo 'Token missing'"
```

**Fix**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
exit
```

#### Issue: "Module not found" or import errors
**Check Python path**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate && python -c 'import sys; print(sys.path)'"
```

**Verify PYTHONPATH** in SLURM script:
```bash
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
```

#### Issue: Job pending with "Resources" for long time
**Check GPU availability**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "sinfo -p gpu -o '%P %a %l %D %T %C'"
```

**Check what's blocking**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "squeue -p gpu -t RUNNING --format='%.10i %.12u %.10M %.10l %.15R' | head -10"
```

**Consider reducing resources** if waiting too long:
- Reduce time limit
- Reduce memory
- Reduce GPU count (if possible)

#### Issue: Out of memory (OOM)
**Check memory usage in logs**:
```bash
ssh csuqqj@blythe.scrtp.warwick.ac.uk "grep -i 'memory\|oom' /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs/<job_name>_<job_id>.err"
```

**Fix**: Increase memory in SLURM script and config:
```bash
#SBATCH --mem=128G  # Increase from 80G
```

---

## Expected Job Runtimes

Based on smoke test results:

| Model | Dataset Size | GPUs | Expected Runtime | Memory |
|-------|--------------|------|------------------|--------|
| gemma-2-2b | Full (100/250/117/200) | 2 | ~4-6 hours | 80GB |
| gemma-2-9b | Full (100/250/117/200) | 2 | ~6-8 hours | 128GB |
| gemma-2-27b | Full (100/250/117/200) | 3 | ~12-16 hours | 180GB |
| gemma-3-1b | Smoke (20/30/20/20) | 2 | ~15-20 min | 48GB |

---

## Current Job Status (as of submission)

**Job IDs** (from most recent submission):
- Job 1060201: gemma2-2b-full
- Job 1060202: gemma2-9b-full
- Job 1060203: gemma2-27b-full
- Job 1060205: smoke_gemma3_1b

**Status**: All jobs properly configured with:
- Paired filtering fix (maintains positive/negative correspondence)
- Correct GPU allocation (explicit device mapping)
- Proper cache and environment setup

**Queue position**: First in priority queue (priority ~4091-4092)

---

## Next Steps After This Session

1. **Monitor Gemma-3 smoke test** (Job 1060205, ~20 min runtime)
   - If successful: Submit full Gemma-3 family jobs
   - If failed: Debug and fix before full submission

2. **Monitor Gemma-2 jobs** (Jobs 1060201-1060203, 4-16 hour runtimes)
   - Check for NaN/Inf filtering effectiveness
   - Verify GPU allocation working correctly

3. **Prepare Gemma-3 full configs** if smoke test succeeds:
   - `configs/gemma3_1b_full.yaml`
   - `configs/gemma3_9b_full.yaml`
   - `configs/gemma3_27b_full.yaml`
   - Corresponding SLURM scripts

4. **Analyze results** once jobs complete:
   - Download results to `analysis_results/`
   - Compare baseline vs CAA performance
   - Check per-category accuracy improvements

---

## Key Lessons Learned

1. **GPU allocation must be explicit**: `device_map="auto"` causes conflicts in multi-model setups
2. **Paired filtering is critical**: Must maintain 1:1 positive/negative correspondence
3. **Environment setup order matters**: Module load → venv activate → set environment variables
4. **Cache location must be consistent**: Set `HF_HOME` to prevent cache conflicts
5. **SLURM and config must match**: GPU count in both SLURM script and config file
6. **NaN/Inf happens**: Some examples cause numerical instability, need robust handling
7. **rsync configs AND scripts**: Don't forget to upload SLURM scripts after changes

---

## Contact & Resources

**Blythe HPC Documentation**: (mentioned in previous context)
**HuggingFace Token**: https://huggingface.co/settings/tokens
**SLURM Commands**: https://slurm.schedmd.com/quickstart.html

**Remote Paths**:
- Project root: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma`
- Logs: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/logs`
- Outputs: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/outputs`
- Cache: `/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers`

**Local Paths**:
- Project root: `/home/tarantulala/Dev/Uni/mi_reasoning_research/SLURM CAA experiments`
- Results: `/home/tarantulala/Dev/Uni/mi_reasoning_research/SLURM CAA experiments/analysis_results`
