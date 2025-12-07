# CAA MLP Steering

Contrastive Activation Addition with learned MLP transformations for steering LLM truthfulness.

## Pipeline

4 independent stages, each with separate resource requirements:

```
EXTRACT ──> TRAIN ──> GENERATE ──> SCORE
 1 GPU      2 GPU      1 GPU       1 GPU
```

1. **Extract**: Get steering vectors from activation differences (inference)
2. **Train**: Learn MLP transformations (needs gradients)
3. **Generate**: Produce steered responses (inference)
4. **Score**: Evaluate with 12B LLM judge (inference)

See [src/README.md](src/README.md) for architecture details.

## Quick Start

### Local
```bash
source venv/bin/activate

# Run stages individually
python -m src.stages.extract --model gemma3_4b_it
python -m src.stages.train --model gemma3_4b_it --run-id <from_extract>
python -m src.stages.generate --model gemma3_4b_it --run-id <run_id>
python -m src.stages.score --model gemma3_4b_it --run-id <run_id>

# Analyze results
python scripts/analyze_run.py --model gemma3_4b_it --latest
```

### Remote (SLURM)
```bash
cd /springbrook/share/dcsresearch/$USER/caa_steering

# Full pipeline with job dependencies
./slurm/submit_full.sh gemma3_4b_it

# Or submit individual stages
./slurm/submit_pipeline.sh gemma3_4b_it --only extract
./slurm/submit_pipeline.sh gemma3_4b_it --from train --run-id <id>

# Monitor
squeue -u $USER
tail -f logs/caa_gemma3_4b_it_*.err
```

## Structure

```
├── src/
│   ├── stages/        # Pipeline stages (extract, train, generate, score)
│   ├── steering/      # Core steering (CAA extraction, MLP, hooks)
│   ├── evaluation/    # LLM judges, semantic similarity
│   ├── models/        # Model loading
│   └── data/          # TruthfulQA dataset
├── configs/
│   ├── base.yaml      # Defaults
│   └── models/        # Model-specific (layer, batch sizes, SLURM resources)
├── slurm/
│   ├── submit_full.sh     # Full pipeline with dependencies
│   └── submit_pipeline.sh # Flexible stage submission
├── scripts/
│   ├── analyze_run.py     # Results analysis
│   └── probe_layers.py    # Debug layer selection
└── outputs/               # Organized by family/model/timestamp
    └── gemma3/
        └── gemma3_12b_it/
            └── 20251201_120000/
                ├── vectors/      # Steering vectors, MLP weights
                ├── responses/    # Generated text per variant
                ├── scores/       # Judge evaluations
                ├── metadata/     # Stage completion info
                ├── checkpoints/  # Resumable progress
                └── logs/         # Per-stage logs
```

## Config

Model configs override base defaults:

```yaml
# configs/models/gemma3_12b_it.yaml
model:
  name: google/gemma-3-12b-it
  layer: 24
  family: gemma3

mlp:
  mc_training:
    batch_size: 4  # Reduce for large models
  gen_training:
    batch_size: 2

slurm:
  gpus: 2  # For training stage
  mem_gb: 80
```

## Checkpointing & Resume

Each stage saves progress incrementally, allowing interrupted jobs to resume:

```bash
# Resume an interrupted run (automatically skips completed work)
python -m src.stages.generate --model gemma3_4b_it --run-id 20251201_120000

# Force re-run (clears checkpoints)
python -m src.stages.generate --model gemma3_4b_it --run-id 20251201_120000 --force
```

Checkpoint granularity:
- **train**: After each MLP (MC, Gen) completes
- **generate**: After each variant (baseline, steered, mlp_mc, mlp_gen)
- **score**: After each variant is scored

Progress is saved to `checkpoints/{stage}_progress.json`.

## Debugging

Each stage saves diagnostics:
- `metadata/extract.json`: Vector norms, activation stats
- `metadata/train.json`: Final losses, NaN detection
- `logs/*.log`: Per-stage logs

Check for zero-norm issues:
```bash
python scripts/analyze_run.py outputs/gemma3/gemma3_12b_it/20251201_*
# Look for "diff_norm_pre_normalize" - if < 1e-6, pos/neg activations are identical
```

Probe layers to find best one:
```bash
python scripts/probe_layers.py --model google/gemma-3-12b-it --num-samples 30
```

Re-run failed stage:
```bash
python -m src.stages.train --model gemma3_4b_it --run-id <id> --force
```
