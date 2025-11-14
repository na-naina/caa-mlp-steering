#!/usr/bin/env python3
"""
Submit CAA experiments to SLURM cluster
"""

import subprocess
import argparse
from pathlib import Path
import sys
import time
from model_configs_updated import (
    MODEL_CONFIGS,
    get_model_slurm_config,
    get_gemma2_models,
    get_gemma3_models,
    get_all_model_keys
)

def submit_job(model_key, layer, scale_set="standard", eval_set="standard",
               use_mlp=False, judge_model=None, dry_run=False):
    """Submit a single job to SLURM"""

    # Get SLURM configuration for this model
    slurm_cfg = get_model_slurm_config(model_key)

    # Build sbatch command
    cmd = [
        "sbatch",
        f"--job-name=caa_{model_key}_L{layer}",
        f"--nodes={slurm_cfg['nodes']}",
        f"--ntasks={slurm_cfg['ntasks']}",
        f"--cpus-per-task={slurm_cfg['cpus_per_task']}",
        f"--mem-per-cpu={slurm_cfg['mem_per_cpu']}",
        f"--time={slurm_cfg['time']}",
        f"--gres={slurm_cfg['gres']}",
        f"--partition={slurm_cfg['partition']}",
        f"--output=logs/{model_key}_L{layer}_%j.out",
        f"--error=logs/{model_key}_L{layer}_%j.err",
        "run_caa_experiment.slurm",
        model_key,
        str(layer),
        scale_set,
        eval_set,
        "true" if use_mlp else "false",
        judge_model if judge_model else ""
    ]

    if dry_run:
        print("Would run:", " ".join(cmd))
        return None

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id} for {model_key} layer {layer}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job for {model_key} layer {layer}: {e.stderr}")
        return None

def submit_model_sweep(model_key, layers=None, scale_set="standard",
                      eval_set="standard", use_mlp=False, judge_model=None,
                      dry_run=False):
    """Submit jobs for all layers of a model"""

    if layers is None:
        layers = MODEL_CONFIGS[model_key]["layers"]

    job_ids = []
    for layer in layers:
        job_id = submit_job(model_key, layer, scale_set, eval_set,
                          use_mlp, judge_model, dry_run)
        if job_id:
            job_ids.append(job_id)
        time.sleep(0.5)  # Small delay between submissions

    return job_ids

def submit_family_experiments(family, **kwargs):
    """Submit experiments for an entire model family"""

    if family == "gemma2":
        models = get_gemma2_models()
    elif family == "gemma3":
        models = get_gemma3_models()
    elif family == "all":
        models = get_all_model_keys()
    else:
        print(f"Unknown family: {family}")
        return []

    all_job_ids = []
    for model_key in models:
        print(f"\nSubmitting jobs for {model_key}...")
        job_ids = submit_model_sweep(model_key, **kwargs)
        all_job_ids.extend(job_ids)

    return all_job_ids

def check_job_status(job_ids):
    """Check status of submitted jobs"""

    if not job_ids:
        print("No job IDs to check")
        return

    cmd = ["squeue", "-j", ",".join(job_ids), "--format=%i,%T,%r"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\nJob Status:")
        print("ID\tStatus\tReason")
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split(",")
            if len(parts) >= 2:
                print(f"{parts[0]}\t{parts[1]}\t{parts[2] if len(parts) > 2 else ''}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to check job status: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Submit CAA experiments to SLURM")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Submit single experiment
    single_parser = subparsers.add_parser("single", help="Submit single experiment")
    single_parser.add_argument("model_key", choices=get_all_model_keys(),
                             help="Model to test")
    single_parser.add_argument("--layer", type=int, help="Layer to test (default: optimal)")
    single_parser.add_argument("--scale-set", default="standard",
                             choices=["fine", "standard", "coarse", "extended"],
                             help="Set of scaling factors")
    single_parser.add_argument("--eval-set", default="standard",
                             choices=["quick", "standard", "full", "thorough"],
                             help="Evaluation configuration")
    single_parser.add_argument("--use-mlp", action="store_true",
                             help="Apply MLP processor to CAA vectors")
    single_parser.add_argument("--judge-model", default="google/gemma-2-27b",
                             help="Model to use as judge for open-ended eval")
    single_parser.add_argument("--dry-run", action="store_true",
                             help="Print commands without executing")

    # Submit model sweep
    sweep_parser = subparsers.add_parser("sweep", help="Submit sweep for one model")
    sweep_parser.add_argument("model_key", choices=get_all_model_keys(),
                            help="Model to test")
    sweep_parser.add_argument("--scale-set", default="standard",
                            choices=["fine", "standard", "coarse", "extended"],
                            help="Set of scaling factors")
    sweep_parser.add_argument("--eval-set", default="standard",
                            choices=["quick", "standard", "full", "thorough"],
                            help="Evaluation configuration")
    sweep_parser.add_argument("--use-mlp", action="store_true",
                            help="Apply MLP processor to CAA vectors")
    sweep_parser.add_argument("--judge-model", default="google/gemma-2-27b",
                            help="Model to use as judge for open-ended eval")
    sweep_parser.add_argument("--dry-run", action="store_true",
                            help="Print commands without executing")

    # Submit family experiments
    family_parser = subparsers.add_parser("family", help="Submit experiments for model family")
    family_parser.add_argument("family", choices=["gemma2", "gemma3", "all"],
                              help="Model family to test")
    family_parser.add_argument("--scale-set", default="standard",
                              choices=["fine", "standard", "coarse", "extended"],
                              help="Set of scaling factors")
    family_parser.add_argument("--eval-set", default="standard",
                              choices=["quick", "standard", "full", "thorough"],
                              help="Evaluation configuration")
    family_parser.add_argument("--use-mlp", action="store_true",
                              help="Apply MLP processor to CAA vectors")
    family_parser.add_argument("--judge-model", default="google/gemma-2-27b",
                              help="Model to use as judge for open-ended eval")
    family_parser.add_argument("--dry-run", action="store_true",
                              help="Print commands without executing")

    # Check job status
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_ids", nargs="+", help="Job IDs to check")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    if args.command == "single":
        layer = args.layer
        if layer is None:
            layer = MODEL_CONFIGS[args.model_key]["optimal_layer"]

        job_id = submit_job(
            args.model_key, layer, args.scale_set, args.eval_set,
            args.use_mlp, args.judge_model, args.dry_run
        )

        if job_id and not args.dry_run:
            print(f"\nSubmitted job {job_id}")

    elif args.command == "sweep":
        job_ids = submit_model_sweep(
            args.model_key, None, args.scale_set, args.eval_set,
            args.use_mlp, args.judge_model, args.dry_run
        )

        if job_ids and not args.dry_run:
            print(f"\nSubmitted {len(job_ids)} jobs")
            check_job_status(job_ids)

    elif args.command == "family":
        job_ids = submit_family_experiments(
            args.family, scale_set=args.scale_set, eval_set=args.eval_set,
            use_mlp=args.use_mlp, judge_model=args.judge_model,
            dry_run=args.dry_run
        )

        if job_ids and not args.dry_run:
            print(f"\nSubmitted {len(job_ids)} jobs")
            check_job_status(job_ids)

    elif args.command == "status":
        check_job_status(args.job_ids)

if __name__ == "__main__":
    main()