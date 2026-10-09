#!/usr/bin/env python3
"""Write <run>/vectors/v_mlp_mc.pt (the applied MAST vector f(v_CAA)) for runs that predate its export.

CPU only. Rebuilds the bottleneck MLP from <run>/config.yaml + mlp_mc_state_dict.pt and applies it to
base_vector.pt in float32 (run.py computes it in the model dtype, bf16; the two agree to ~1e-3 relative).
Existing files are left untouched unless --force.

    python scripts/export_mast_vector.py data/outputs/rcv_g4bmain_s42/fold1 [...]
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.steering.mlp import SteeringMLP  # noqa: E402


def export(run: Path, force: bool = False) -> Path:
    dest = run / "vectors" / "v_mlp_mc.pt"
    if dest.exists() and not force:
        return dest
    arch = yaml.safe_load((run / "config.yaml").read_text()).get("mlp", {}).get("architecture", {})
    base = torch.load(run / "vectors" / "base_vector.pt", map_location="cpu").float().flatten()
    mlp = SteeringMLP(input_dim=base.shape[0], bottleneck_dim=arch.get("bottleneck_dim"),
                      hidden_multiplier=arch.get("hidden_multiplier", 2.0), dropout=arch.get("dropout", 0.1))
    mlp.load_state_dict(torch.load(run / "vectors" / "mlp_mc_state_dict.pt", map_location="cpu"))
    with torch.no_grad():
        v = mlp.float().eval()(base.unsqueeze(0)).squeeze(0)
    torch.save(v, dest)
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for r in a.runs:
        print(export(r, a.force))
