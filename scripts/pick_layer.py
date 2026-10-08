#!/usr/bin/env python3
"""Pick the steering layer from a train-only sweep using training signals only.

Score = mean MC margin accuracy over the last 20 logged steps (ties: lower mean loss).
Also prints ||base_vector|| at the chosen layer and the bare-vector lr predicted by the
activation-scale rule lr ~= 2e-3 * ||v_CAA|| / 2.5 (LLaMA-2: ||v_CAA|| ~= 2.5, best lr 2e-3).

    python scripts/pick_layer.py data/outputs/rg4e_sweep   # -> "<layer> <norm> <pred_lr>"
"""
import json
import sys
from pathlib import Path

import torch

rows = []
for d in sorted(Path(sys.argv[1]).glob("L*")):
    h = d / "training_history.json"
    if not h.exists():
        continue
    mc = json.loads(h.read_text())["mc"]
    acc, loss = mc["accuracy"][-20:], mc["loss"][-20:]
    if any(x != x for x in loss):  # NaN
        continue
    rows.append((sum(acc) / len(acc), -sum(loss) / len(loss), int(d.name[1:]), d))
for a, l, L, _ in sorted(rows, reverse=True):
    print(f"# L{L}: acc {a:.3f} loss {-l:.3f}", file=sys.stderr)
a, l, L, d = max(rows)
v = torch.load(d / "vectors" / "base_vector.pt", map_location="cpu")
v = v if torch.is_tensor(v) else next(iter(v.values()))
n = float(v.float().norm())
print(f"{L} {n:.2f} {2e-3 * n / 2.5:.3g}")
