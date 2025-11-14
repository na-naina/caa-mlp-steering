#!/usr/bin/env python3
"""
Render and submit SLURM jobs for Gemma CAA experiments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Gemma family experiments")
    parser.add_argument(
        "--family",
        choices=["gemma2", "gemma3", "all"],
        default="gemma2",
        help="Model family to submit",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Base YAML configuration",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing per-model configs",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("slurm/job_template.sh"),
        help="SLURM job template file",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project on remote cluster",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Prefix for run identifiers (default: <family>_<timestamp>)",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help="Optional override for steering scales",
    )
    parser.add_argument(
        "--no-mlp",
        action="store_true",
        help="Disable MLP for submitted jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs without submitting",
    )
    return parser.parse_args()


def load_template(path: Path) -> str:
    return path.read_text()


def select_configs(config_dir: Path, family: str) -> Dict[str, Path]:
    configs: Dict[str, Path] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        with path.open("r") as f:
            content = yaml.safe_load(f) or {}
        model_cfg = content.get("model") or {}
        family_name = model_cfg.get("family") or ""
        model_name = model_cfg.get("name")
        if family != "all" and family_name != family:
            continue
        model_key = path.stem
        if not model_name:
            continue
        configs[model_key] = path
    return configs


def render_job_script(template: str, context: Dict[str, str]) -> str:
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def main() -> int:
    args = parse_args()
    template = load_template(args.template)
    configs = select_configs(args.config_dir, args.family)
    if not configs:
        raise SystemExit("No matching configs were found.")

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix or f"{args.family}_{timestamp}"

    for model_key, config_path in configs.items():
        run_id = f"{run_prefix}_{model_key}"
        job_name = f"caa_{model_key}"

        config = load_config(
            args.base_config,
            overrides=[config_path],
        )
        slurm_cfg = config.get("slurm", {})
        partition = slurm_cfg.get("partition", "gpu")
        cpus = int(slurm_cfg.get("cpus", 8))
        mem_gb = int(slurm_cfg.get("mem_gb", 64))
        walltime = slurm_cfg.get("time", "12:00:00")
        gpus = int(slurm_cfg.get("gpus", 1))
        gpu_type = slurm_cfg.get("gpu_type")
        qos = slurm_cfg.get("qos")
        account = slurm_cfg.get("account")

        gres = slurm_cfg.get(
            "gres",
            f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}",
        )

        command: List[str] = [
            "python",
            "-m",
            "src.jobs.run_experiment",
            "--base-config",
            str(args.base_config),
            "--config",
            str(config_path),
            "--run-id",
            run_id,
        ]
        if args.scales:
            command.append("--scales")
            command.extend(str(scale) for scale in args.scales)
        if args.no_mlp:
            command.append("--no-mlp")

        python_command = " ".join(shlex.quote(part) for part in command)
        shared_env = config.get("paths", {}).get("shared_env_script", "")
        hf_cache = config.get("paths", {}).get("hf_cache", "")

        context = {
            "job_name": job_name,
            "partition": partition,
            "cpus": str(cpus),
            "mem": f"{mem_gb}G",
            "gpus": str(gpus),
            "gres": gres,
            "time": walltime,
            "project_root": str(args.project_root),
            "shared_env_script": shared_env,
            "hf_cache": hf_cache,
            "python_command": python_command,
        }
        if qos:
            context["qos_line"] = f"#SBATCH --qos={qos}"
        else:
            context["qos_line"] = ""
        if account:
            context["account_line"] = f"#SBATCH --account={account}"
        else:
            context["account_line"] = ""

        job_script = render_job_script(template, context)
        if args.dry_run:
            print(f"--- {model_key} ---")
            print(job_script)
            continue

        script_path = Path(f"slurm/tmp_{job_name}_{timestamp}.sbatch")
        script_path.write_text(job_script)
        try:
            subprocess.run(["sbatch", str(script_path)], check=True)
        finally:
            script_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

