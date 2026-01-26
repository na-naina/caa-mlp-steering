#!/usr/bin/env python3
"""Generate responses for a single variant (for resuming crashed runs)."""

import argparse
import json
import sys
from pathlib import Path

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=["baseline", "steered", "mlp_mc", "mlp_gen"])
    args = parser.parse_args()

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.models.loader import load_causal_model
    from src.data.truthfulqa import TruthfulQADatasetManager
    from src.evaluation.truthfulqa import evaluate_generation, evaluate_multiple_choice
    from src.steering.mlp import SteeringMLP
    from src.utils.config import load_config

    # Load config
    base_config = Path("configs/base.yaml")
    model_config = Path(f"configs/models/{args.model}.yaml")
    config = load_config(base_config, overrides=[model_config])

    # Load model
    print(f"Loading model: {config['model']['name']}")
    loaded = load_causal_model(
        config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
    )
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()

    # Load dataset and splits
    tqa_cfg = config.get("truthfulqa", {})
    dataset = TruthfulQADatasetManager(
        dataset_name=tqa_cfg.get("dataset_name", "truthful_qa"),
        dataset_config=tqa_cfg.get("dataset_config", "generation"),
        cache_dir=tqa_cfg.get("cache_dir"),
        seed=42,
    )

    splits_file = args.run_dir / "metadata" / "splits.json"
    with open(splits_file) as f:
        splits = json.load(f)

    test_indices = splits["test"]
    test_items = dataset.get_items(test_indices)
    mc_indices = [i for i in test_indices if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)

    # Load vectors
    vectors_dir = args.run_dir / "vectors"
    param_dtype = next(model.parameters()).dtype
    layer_index = config["model"]["layer"]

    base_vector = torch.load(vectors_dir / "base_vector.pt", weights_only=True)
    base_vector = base_vector.to(device, dtype=param_dtype)

    # Determine which vector to use
    if args.variant == "baseline":
        vector = None
        scale = 0.0
    elif args.variant == "steered":
        vector = base_vector
        scale = 1.0
    elif args.variant == "mlp_mc":
        mlp = SteeringMLP(input_dim=base_vector.shape[0]).to(device, dtype=param_dtype)
        mlp.load_state_dict(torch.load(vectors_dir / "mlp_mc_state_dict.pt", weights_only=True))
        mlp.eval()
        with torch.no_grad():
            vector = mlp(base_vector.unsqueeze(0)).squeeze(0)
        scale = 1.0
    elif args.variant == "mlp_gen":
        mlp = SteeringMLP(input_dim=base_vector.shape[0]).to(device, dtype=param_dtype)
        mlp.load_state_dict(torch.load(vectors_dir / "mlp_gen_state_dict.pt", weights_only=True))
        mlp.eval()
        with torch.no_grad():
            vector = mlp(base_vector.unsqueeze(0)).squeeze(0)
        scale = 1.0

    # Generation config
    steering_cfg = config.get("steering", {})
    eval_cfg = config.get("evaluation", {})
    gen_cfg = {
        "preset": eval_cfg.get("preset", "qa"),
        "temperature": eval_cfg.get("temperature", 0.3),
        "top_p": eval_cfg.get("top_p", 0.9),
        "max_new_tokens": eval_cfg.get("max_new_tokens", 64),
        "max_length": steering_cfg.get("max_length", 512),
        "stop_sequences": eval_cfg.get("stop_sequences", []),
    }

    print(f"Generating for variant: {args.variant} (scale={scale})")

    # Run MC eval
    mc_result = evaluate_multiple_choice(
        model, tokenizer, mc_items,
        layer_index=layer_index, steering_vector=vector, scale=scale,
        max_length=steering_cfg.get("max_length", 512), primary_device=device,
    )

    # Run generation
    gen_result = evaluate_generation(
        model, tokenizer, test_items,
        layer_index=layer_index, steering_vector=vector, scale=scale,
        generation_cfg=gen_cfg, primary_device=device,
        judge=None, semantic_judge=None,
    )

    # Save results
    variant_dir = args.run_dir / args.variant / f"scale_{scale:.2f}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    with open(variant_dir / "mc_details.json", "w") as f:
        json.dump(mc_result["details"], f, indent=2)
    with open(variant_dir / "generation_details.json", "w") as f:
        json.dump(gen_result["details"], f, indent=2)

    print(f"Results saved to: {variant_dir}")
    print(f"MC accuracy: {mc_result['stats'].accuracy:.2%}")
    print(f"Generation samples: {gen_result['stats'].total}")


if __name__ == "__main__":
    main()
