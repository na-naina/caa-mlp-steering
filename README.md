# CAA MLP Steering

Contrastive Activation Addition with learned MLP transformations for steering LLM behavior on TruthfulQA.

## Overview

1. **Extract** activation differences between truthful/untruthful responses
2. **Train** two MLPs to transform the steering vector (supervised, no RL)
3. **Evaluate** steered outputs using LLM judges

See [src/README.md](src/README.md) for detailed pipeline architecture.

## Structure

```
├── src/
│   ├── steering/      # mlp.py, apply.py, extract.py, training.py
│   ├── evaluation/    # TruthfulQA eval, LLM judges
│   ├── models/        # Model loading
│   └── data/          # Dataset management
├── configs/
│   ├── base.yaml      # Default settings
│   └── models/        # Model-specific overrides
├── outputs/           # Experiment results
└── legacy/            # Archived old code/configs
```

## Usage

### Local

```bash
source venv/bin/activate
python -m src.jobs.run_experiment --config configs/models/gemma3_4b_it.yaml
```

### Remote (HPC)

**First-time setup:**
```bash
ssh user@blythe.scrtp.warwick.ac.uk
cd /springbrook/share/dcsresearch/$USER

# Clone/sync project
rsync -avz --exclude '.venv' --exclude 'outputs' local/caa-mlp-steering/ caa_steering/

# Create venv
cd caa_steering
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ensure cache directories exist (avoid home quota issues)
mkdir -p /springbrook/share/dcsresearch/$USER/hf_cache
```

**Submit jobs:**
```bash
cd /springbrook/share/dcsresearch/$USER/caa_steering/slurm
./submit.sh gemma3_12b_it           # full pipeline
./submit.sh gemma3_12b_it train     # extract + train + generate
./submit.sh gemma3_12b_it eval      # judge existing outputs
```

**Monitor:**
```bash
squeue -u $USER                     # check queue
sacct -u $USER --format=JobID,JobName,State,Elapsed  # job history
cat logs/caa_gemma3_12b_it_*.err    # check errors
```

**Important:** HF cache is set to `$SHARE/hf_cache` in submit.sh to avoid home directory quota limits.

## Model Configs

Model configs only override what differs from base:

```yaml
model:
  name: google/gemma-3-12b-it
  layer: 24

slurm:
  gpus: 1
  mem_gb: 40
```

## Results

Outputs saved to `outputs/<model>_<timestamp>/`:

- `vectors/` - steering vectors + MLP weights
- `training_history.json` - loss curves
- `results.json` - evaluation metrics
- `<variant>/scale_X.XX/` - generation details
