#!/usr/bin/env python3
"""Check model weights using our loader."""
import torch
from src.models.loader import load_causal_model
import logging

logging.basicConfig(level=logging.INFO)

print("Loading model via our loader...")
loaded = load_causal_model(
    "google/gemma-3-12b-it",
    dtype="bfloat16",
    device_map="cpu"
)

print("\nChecking model weights for NaN/Inf...")
nan_params = []
inf_params = []

for name, param in loaded.model.named_parameters():
    if torch.isnan(param).any():
        nan_count = torch.isnan(param).sum().item()
        nan_params.append((name, nan_count))
        print(f"⚠️ NaN in {name}: {nan_count} values")

    if torch.isinf(param).any():
        inf_count = torch.isinf(param).sum().item()
        inf_params.append((name, inf_count))
        print(f"⚠️ Inf in {name}: {inf_count} values")

print("\n" + "="*80)
if nan_params or inf_params:
    print(f"❌ FOUND CORRUPTION!")
    print(f"   Parameters with NaN: {len(nan_params)}")
    print(f"   Parameters with Inf: {len(inf_params)}")
else:
    print("✅ All weights are clean!")
