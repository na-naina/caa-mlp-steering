#!/usr/bin/env python3
"""
Check Gemma3 model weights for NaN and Inf values.

Usage:
    python check_model_weights.py --model google/gemma-3-12b-it
    python check_model_weights.py --model google/gemma-3-27b-it --dtype bfloat16
"""

import argparse
import torch
from transformers import AutoModelForCausalLM
import sys


def check_model_weights(model_name: str, dtype_str: str = "bfloat16"):
    """Check all model parameters for NaN and Inf values."""

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"Loading model: {model_name}")
    print(f"Using dtype: {dtype}")
    print("Device: CPU (for safety)")
    print("-" * 80)

    # Load model on CPU to avoid GPU memory issues
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 80)
    print("\nChecking for NaN and Inf values...\n")

    nan_params = []
    inf_params = []
    total_params_checked = 0

    for name, param in model.named_parameters():
        total_params_checked += 1

        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()

        if has_nan:
            nan_count = torch.isnan(param).sum().item()
            nan_params.append((name, nan_count, param.numel()))
            print(f"⚠️  NaN found in: {name}")
            print(f"   Count: {nan_count:,} / {param.numel():,} ({100*nan_count/param.numel():.2f}%)")
            print(f"   Shape: {param.shape}")
            print()

        if has_inf:
            inf_count = torch.isinf(param).sum().item()
            inf_params.append((name, inf_count, param.numel()))
            print(f"⚠️  Inf found in: {name}")
            print(f"   Count: {inf_count:,} / {param.numel():,} ({100*inf_count/param.numel():.2f}%)")
            print(f"   Shape: {param.shape}")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total parameters checked: {total_params_checked:,}")
    print(f"Parameters with NaN: {len(nan_params)}")
    print(f"Parameters with Inf: {len(inf_params)}")
    print()

    if nan_params:
        print("⚠️  CRITICAL: Model weights contain NaN values!")
        print("\nAffected parameters:")
        for name, count, total in nan_params:
            print(f"  - {name}: {count:,} NaN values ({100*count/total:.2f}%)")
        print()
        return False

    if inf_params:
        print("⚠️  WARNING: Model weights contain Inf values!")
        print("\nAffected parameters:")
        for name, count, total in inf_params:
            print(f"  - {name}: {count:,} Inf values ({100*count/total:.2f}%)")
        print()
        return False

    print("✅ All model weights are clean (no NaN or Inf values)")
    print()

    # Additional checks: look for extreme values
    print("-" * 80)
    print("ADDITIONAL CHECKS: Extreme Values")
    print("-" * 80)

    extreme_params = []
    for name, param in model.named_parameters():
        param_max = param.abs().max().item()
        param_min = param.abs().min().item()
        param_mean = param.abs().mean().item()

        # Check if max value is suspiciously large for the dtype
        if dtype == torch.float16 and param_max > 50000:
            extreme_params.append((name, param_max, "close to fp16 limit"))
        elif dtype == torch.bfloat16 and param_max > 1e10:
            extreme_params.append((name, param_max, "very large for bf16"))
        elif param_max > 1e6:
            extreme_params.append((name, param_max, "extremely large"))

    if extreme_params:
        print("\n⚠️  Found parameters with extreme values:")
        for name, max_val, reason in extreme_params:
            print(f"  - {name}: max={max_val:.2e} ({reason})")
        print()
    else:
        print("\n✅ No extreme values detected")
        print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check Gemma3 model weights for NaN/Inf corruption"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., google/gemma-3-12b-it)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to load model in",
    )

    args = parser.parse_args()

    try:
        is_clean = check_model_weights(args.model, args.dtype)
        sys.exit(0 if is_clean else 1)
    except Exception as e:
        print(f"\n❌ Error checking model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
