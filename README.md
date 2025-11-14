# Gemma TruthfulQA Steering Pipeline

New pipeline for running Contrastive Activation Addition (CAA) experiments with optional MLP transformations on Gemma-2 and Gemma-3 model families.

## Layout

- `configs/` – base + per-model YAML configs (Gemma 2/3).
- `src/` – modular Python package powering the current pipeline:
  - `data.truthfulqa` – dataset sampling & split management.
  - `steering.extract | steering.vector_bank | steering.training` – CAA extraction, vector bank, and MLP training.
  - `evaluation.*` – MC + open-ended evaluation with semantic/LLM judges.
  - `jobs.run_experiment` – single entrypoint for experiment orchestration.
- `scripts/` – lightweight helper scripts (remote prep, submission helpers).
- `slurm/` – curated SLURM job files (e.g., `slurm/smoke_gemma2.slurm`).
- `analysis/` – placeholder for downstream analysis utilities.
- `legacy/old_pipeline/` – archived scripts/configs from previous experiments. Keep for reference only; not invoked by the modern pipeline.

## Local Dry Run (CPU)

```bash
python -m src.jobs.run_experiment \
  --config configs/gemma3-1b.yaml \
  --run-id local_test \
  --no-mlp \
  --dry-run  # remove this flag for a miniature execution
```

Remove `--dry-run` to execute; add `--scales 0 0.5` and override evaluation counts via `--override evaluation.mc_samples=10` etc for lightweight tests.

## Remote Setup (Blythe)

```bash
./scripts/prepare_remote.sh csuqqj@blythe.scrtp.warwick.ac.uk /springbrook/share/dcsresearch/u5584851/caa_gemma
```

After syncing:

```bash
ssh -i blythe csuqqj@blythe.scrtp.warwick.ac.uk
cd /springbrook/share/dcsresearch/u5584851/caa_gemma
source venv/bin/activate
export PYTHONPATH=$(pwd)
```

Optional shared cache setup (once):

```bash
sbatch scripts/slurm/setup_cache.sh
```

## Submitting Experiments

1. Ensure `slurm/job_template.sh` references the correct remote project root.
2. Submit a family sweep (dry run first):

```bash
python scripts/submit_family.py --family gemma2 --dry-run
python scripts/submit_family.py --family gemma2 --project-root /springbrook/share/dcsresearch/u5584851/caa_gemma
```

Add `--scales` or `--no-mlp` for overrides. The script renders `slurm/tmp_*.sbatch`, submits via `sbatch`, and deletes temporary files.

Each job runs `python -m src.jobs.run_experiment`, storing outputs in `outputs/<run-id>/` with vectors, metadata, and per-scale results.

## Inspecting Results

Results per run:

- `outputs/<run-id>/config.yaml` – resolved configuration.
- `outputs/<run-id>/vectors/` – base & MLP steering vectors.
- `outputs/<run-id>/scale_<scale>/` – detailed MC + generation JSON logs.
- `outputs/<run-id>/results.json` – aggregated metrics.

Future work: add aggregation scripts under `analysis/` for cross-run comparisons.

## Notes

- Gemma-27B loads with `device_map="auto"` and assumes two GPUs; adjust `slurm/gemma*-27b.yaml` if cluster topology differs.
- Judge defaults to `google/gemma-3-27b-it`; swap in `gemma-3-9b-it` on smaller nodes via `--override evaluation.judge_model=google/gemma-3-9b-it`.
- `HF_TOKEN` is auto-populated when `~/.cache/huggingface/token` exists; ensure the token is present on Blythe.
