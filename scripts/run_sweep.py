#!/usr/bin/env python3
"""Hyperparameter sweep for CAA + MLP steering.

Phase 1 (fast):  Train + MC eval for all grid points.
Phase 2 (expensive): Full generation + scoring for top-K configs.

Usage examples:
    # Full sweep on one machine
    python scripts/run_sweep.py --model llama2_7b_chat

    # Split across two machines
    python scripts/run_sweep.py --model llama2_7b_chat --layers 8 12 16
    python scripts/run_sweep.py --model llama2_7b_chat --layers 20 24 28

    # Phase 1 only (fast proxy)
    python scripts/run_sweep.py --model llama2_7b_chat --phase1-only

    # Phase 2 on existing Phase 1 results
    python scripts/run_sweep.py --model llama2_7b_chat --phase2-only \\
        --sweep-dir data/outputs/llama2/llama2_7b_chat/sweep_20260216_120000

    # Resume interrupted sweep
    python scripts/run_sweep.py --model llama2_7b_chat --sweep-dir <path>
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.stages.common import RunContext, load_config, setup_environment
from src.stages.generate import run_generation
from src.sweep.aggregate import (
    load_phase1_results,
    print_results_table,
    rank_configs,
    select_top_k,
)
from src.sweep.config import SweepConfig
from src.sweep.runner import run_phase1

LOG = logging.getLogger(__name__)

# Default sweep grid
DEFAULT_LAYERS = [8, 12, 16, 20, 24, 28]
DEFAULT_LRS = [5e-5, 1e-4, 5e-4, 1e-3]
DEFAULT_REGS = [0.001, 0.01, 0.1]


def _run_phase2_config(
    config_result: dict,
    base_config: dict,
    sweep_dir: Path,
) -> None:
    """Run full generation for a single promoted config."""
    rank = config_result.get("rank", 0)
    layer = config_result["layer"]
    lr = config_result["lr"]
    mse_reg = config_result["mse_reg"]
    combo_id = config_result["combo_id"]

    phase2_dir = sweep_dir / "phase2" / f"rank_{rank:02d}"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # Build config with this combo's HPs
    bn_dim = config_result.get("bottleneck_dim")
    cfg = copy.deepcopy(base_config)
    cfg["model"]["layer"] = layer
    cfg["mlp"]["mc_training"]["lr"] = lr
    cfg["mlp"]["mc_training"]["mse_reg"] = mse_reg
    cfg["mlp"]["gen_training"]["lr"] = lr
    cfg["mlp"]["gen_training"]["mse_reg"] = mse_reg
    if bn_dim is not None:
        cfg.setdefault("mlp", {}).setdefault("architecture", {})[
            "bottleneck_dim"
        ] = bn_dim

    # Create RunContext-compatible directory layout
    for subdir in ("vectors", "responses", "scores", "metadata", "checkpoints"):
        (phase2_dir / subdir).mkdir(exist_ok=True)

    # Copy vectors + MLPs from sweep artifacts
    layer_vectors = sweep_dir / "vectors" / f"layer_{layer:02d}"
    combo_results = sweep_dir / "results" / f"layer_{layer:02d}" / combo_id

    shutil.copy2(
        layer_vectors / "base_vector.pt",
        phase2_dir / "vectors" / "base_vector.pt",
    )
    shutil.copy2(
        layer_vectors / "vector_bank.pt",
        phase2_dir / "vectors" / "vector_bank.pt",
    )
    for mlp_file in ("mlp_mc_state_dict.pt", "mlp_gen_state_dict.pt"):
        src = combo_results / mlp_file
        if src.exists():
            shutil.copy2(src, phase2_dir / "vectors" / mlp_file)

    # Copy splits
    shutil.copy2(
        sweep_dir / "metadata" / "splits.json",
        phase2_dir / "metadata" / "splits.json",
    )

    # Save config
    with open(phase2_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    # Save which sweep config this came from
    with open(phase2_dir / "sweep_origin.json", "w") as f:
        json.dump(config_result, f, indent=2)

    ctx = RunContext(
        model_name=cfg["model"]["name"],
        run_id=f"sweep_rank_{rank:02d}",
        run_dir=phase2_dir,
        config=cfg,
    )

    LOG.info(
        "Phase 2 [rank %d]: layer=%d, lr=%s, reg=%s → %s",
        rank, layer, lr, mse_reg, phase2_dir,
    )
    run_generation(ctx, force=True)
    LOG.info("Phase 2 [rank %d] complete", rank)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HP sweep for CAA+MLP steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layers to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=None,
        help="Learning rates to sweep",
    )
    parser.add_argument(
        "--regs", type=float, nargs="+", default=None,
        help="MSE reg values to sweep",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Configs to promote to Phase 2 (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help="Resume from / write to existing sweep directory",
    )
    parser.add_argument(
        "--bottleneck", type=int, default=None,
        help="Use low-rank bottleneck MLP instead of fat MLP (e.g. 16, 32, 64)",
    )
    parser.add_argument(
        "--bottlenecks", type=str, nargs="+", default=None,
        help="Sweep bottleneck dims (e.g. 4 8 16 32 64 fat). Use 'fat' or '0' for full-size MLP.",
    )
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--phase2-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    base_config = load_config(args.model)
    base_config.setdefault("run", {})["seed"] = args.seed

    # Apply bottleneck override (single value — applies to all configs)
    if args.bottleneck is not None:
        base_config.setdefault("mlp", {}).setdefault("architecture", {})[
            "bottleneck_dim"
        ] = args.bottleneck

    # Parse bottleneck dims for sweep axis
    bottleneck_dims = None
    if args.bottlenecks is not None:
        bottleneck_dims = []
        for v in args.bottlenecks:
            if v.lower() in ("fat", "none", "0"):
                bottleneck_dims.append(None)
            else:
                bottleneck_dims.append(int(v))

    sweep_config = SweepConfig(
        layers=args.layers or DEFAULT_LAYERS,
        learning_rates=args.lrs or DEFAULT_LRS,
        mse_regs=args.regs or DEFAULT_REGS,
        top_k=args.top_k,
        **({"bottleneck_dims": bottleneck_dims} if bottleneck_dims is not None else {}),
    )

    LOG.info(
        "Sweep grid: %d layers x %d LRs x %d regs x %d archs = %d configs",
        len(sweep_config.layers),
        len(sweep_config.learning_rates),
        len(sweep_config.mse_regs),
        len(sweep_config.bottleneck_dims),
        sweep_config.total_configs,
    )

    # Create or resume sweep directory
    if args.sweep_dir:
        sweep_dir = args.sweep_dir
    else:
        output_root = Path(
            base_config.get("paths", {}).get("output_root", "data/outputs")
        )
        family = base_config.get("model", {}).get("family", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = output_root / family / args.model / f"sweep_{timestamp}"

    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep config
    bn_display = [
        b if b is not None else "fat" for b in sweep_config.bottleneck_dims
    ]
    with open(sweep_dir / "sweep_config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "sweep": {
                    "layers": sweep_config.layers,
                    "learning_rates": sweep_config.learning_rates,
                    "mse_regs": sweep_config.mse_regs,
                    "bottleneck_dims": bn_display,
                    "top_k": sweep_config.top_k,
                },
                "base_model": args.model,
                "seed": args.seed,
            },
            f,
            default_flow_style=False,
        )

    LOG.info("Sweep directory: %s", sweep_dir)

    # ===== Phase 1 =====
    if not args.phase2_only:
        all_results = run_phase1(
            sweep_config,
            base_config,
            sweep_dir,
            target_layers=args.layers,
        )

        print("\n=== PHASE 1 RESULTS ===")
        ranked = rank_configs(all_results)
        print_results_table(ranked)

    if args.phase1_only:
        LOG.info("Phase 1 complete. Results saved to: %s", sweep_dir)
        return 0

    # ===== Phase 2 =====
    all_results = load_phase1_results(sweep_dir)
    top = select_top_k(all_results, top_k=args.top_k)

    print(f"\n=== TOP {len(top)} CONFIGS FOR PHASE 2 ===")
    print_results_table(top)

    for cfg_result in top:
        _run_phase2_config(cfg_result, base_config, sweep_dir)

    LOG.info("Sweep complete. Results in: %s", sweep_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
