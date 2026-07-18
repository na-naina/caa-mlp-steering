#!/usr/bin/env python3
"""Quick sample generation to eyeball quality of a trained MLP."""
import argparse
import json
import torch
from pathlib import Path

from src.models.loader import load_causal_model
from src.data.truthfulqa import TruthfulQADatasetManager
from src.steering.mlp import SteeringMLP
from src.steering.apply import steering_hook
from src.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model config name")
    p.add_argument("--run-dir", required=True, type=Path, help="Directory with trained MLP weights")
    p.add_argument("--n", type=int, default=5, help="Number of examples to generate")
    p.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args()

    base_config = Path("configs/base.yaml")
    model_config = Path(f"configs/models/{args.model}.yaml")
    config = load_config(base_config, overrides=[model_config])
    model_cfg = config["model"]

    loaded = load_causal_model(model_cfg["name"], dtype=model_cfg.get("dtype", "bfloat16"), device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()

    # Load base vector and MLP
    base_vector = torch.load(args.run_dir / "vectors" / "base_vector.pt", map_location="cpu")
    cfg_obj = model.config
    hidden_dim = getattr(cfg_obj, "hidden_size", None) or getattr(cfg_obj.text_config, "hidden_size", None)
    param_dtype = next(model.parameters()).dtype

    arch_cfg = config.get("mlp", {}).get("architecture", {})
    mlp = SteeringMLP(
        input_dim=hidden_dim,
        bottleneck_dim=arch_cfg.get("bottleneck_dim"),
        hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
        dropout=arch_cfg.get("dropout", 0.1),
    ).to(device, dtype=param_dtype)
    mlp.load_state_dict(torch.load(args.run_dir / "vectors" / "mlp_mc_state_dict.pt", map_location="cpu"))
    mlp.eval()

    base_vector = base_vector.to(device, dtype=param_dtype)
    mlp_vector = mlp(base_vector.unsqueeze(0)).squeeze(0).detach()

    # Load dataset and get a few test items
    dataset = TruthfulQADatasetManager()
    splits_file = args.run_dir / "metadata" / "splits.json"
    if splits_file.exists():
        with splits_file.open() as f:
            splits = json.load(f)
        test_indices = splits["test"][:args.n]
    else:
        test_indices = list(range(args.n))

    test_items = dataset.get_items(test_indices)
    layer_index = model_cfg["layer"]

    for item in test_items:
        question = item["question"]
        prompt = f"Q: {question}\nA:"

        def generate_one(vec=None, scale=0.0):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with steering_hook(model, layer_index, vec, scale=scale):
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64, temperature=0.3, top_p=0.9, do_sample=True)
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        baseline = generate_one(None, 0.0)
        caa = generate_one(base_vector, args.scale)
        mlp_out = generate_one(mlp_vector, args.scale)

        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"  Baseline: {baseline}")
        print(f"  CAA:      {caa}")
        print(f"  MLP:      {mlp_out}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
