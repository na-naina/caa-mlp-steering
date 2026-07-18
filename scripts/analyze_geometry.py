"""Geometry analysis of the trained MAST correction.

Computes, for a given run directory:
  - cosine similarity between g_theta(v_CAA) and v_CAA
  - relative norm |delta| / |v_CAA|
  - bottleneck activation pattern (how many of the k neurons fire on v_CAA)

Usage:
    python scripts/analyze_geometry.py --run data/outputs/mseed_42 [--run ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def build_mlp(input_dim: int, bottleneck_dim: int, hidden_multiplier: float, dropout: float) -> nn.Module:
    if bottleneck_dim is not None:
        return nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, input_dim),
        )
    hidden_dim = max(int(input_dim * hidden_multiplier), input_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, input_dim),
    )


def analyze_run(run_dir: Path, active_threshold: float = 1e-3) -> dict:
    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)

    bottleneck_dim = cfg["mlp"]["architecture"].get("bottleneck_dim")
    hidden_multiplier = cfg["mlp"]["architecture"].get("hidden_multiplier", 2.0)
    dropout = cfg["mlp"]["architecture"].get("dropout", 0.1)

    v_caa = torch.load(run_dir / "vectors" / "base_vector.pt", map_location="cpu")
    if v_caa.ndim > 1:
        v_caa = v_caa.squeeze()

    state = torch.load(run_dir / "vectors" / "mlp_mc_state_dict.pt", map_location="cpu")

    input_dim = v_caa.shape[0]
    net = build_mlp(input_dim, bottleneck_dim, hidden_multiplier, dropout)
    stripped = {k.removeprefix("net."): v for k, v in state.items() if k.startswith("net.")}
    net.load_state_dict(stripped)
    net.eval()

    with torch.no_grad():
        pre_relu = net[0](v_caa)
        post_relu = torch.relu(pre_relu)
        delta = net(v_caa)

    v_norm = v_caa.norm().item()
    delta_norm = delta.norm().item()
    cosine = torch.nn.functional.cosine_similarity(delta.unsqueeze(0), v_caa.unsqueeze(0), dim=1).item()

    # "Active" = post-ReLU activation exceeds threshold × max.
    active_mask = post_relu.abs() > active_threshold * post_relu.abs().max().clamp(min=1e-12)
    num_active = int(active_mask.sum().item())

    return {
        "run": str(run_dir),
        "seed": cfg["run"]["seed"],
        "bottleneck_dim": bottleneck_dim,
        "cosine_delta_vcaa": cosine,
        "norm_delta": delta_norm,
        "norm_vcaa": v_norm,
        "relative_norm": delta_norm / v_norm,
        "num_active_neurons": num_active,
        "total_neurons": bottleneck_dim if bottleneck_dim else None,
        "post_relu_values": post_relu.tolist() if bottleneck_dim and bottleneck_dim <= 64 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--active-threshold", type=float, default=1e-3,
                        help="Neuron is 'active' if post-ReLU value > threshold x max.")
    args = parser.parse_args()

    results = []
    for run_dir in args.run:
        r = analyze_run(run_dir, active_threshold=args.active_threshold)
        results.append(r)
        print(f"[{r['run']}]  seed={r['seed']}  k={r['bottleneck_dim']}")
        print(f"  cos(delta, v_CAA)    = {r['cosine_delta_vcaa']:.4f}")
        print(f"  |delta| / |v_CAA|    = {r['relative_norm']:.4f}")
        print(f"  active neurons       = {r['num_active_neurons']} / {r['total_neurons']}")
        if r["post_relu_values"] is not None:
            vals = [f"{v:.3f}" for v in r["post_relu_values"]]
            print(f"  post-ReLU            = [{', '.join(vals)}]")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
