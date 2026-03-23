#!/usr/bin/env python3
"""CAA MLP Steering - Simplified Entry Point

Usage:
    python run.py --model gemma3_12b_it              # Full pipeline
    python run.py --model gemma3_12b_it --stage train   # Extract + train MLPs + generate
    python run.py --model gemma3_12b_it --stage eval    # Judge existing outputs
    python run.py --model gemma3_12b_it --stage extract # Only extract vectors
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.judge import LLMBinaryJudge
from src.evaluation.truthfulqa import evaluate_generation, evaluate_multiple_choice
from src.models.loader import load_causal_model
from src.steering.apply import steering_hook
from src.steering.extract import ActivationExtractor, compute_caa_vector
from src.steering.mlp import SteeringMLP
from src.steering.training import MCTrainingConfig, GenTrainingConfig, train_mc_mlp, train_gen_mlp
from src.steering.vector_bank import VectorBankBuilder
from src.utils.config import load_config, dump_config

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="CAA MLP Steering")
    p.add_argument("--model", required=True, help="Model config name (e.g., gemma3_12b_it)")
    p.add_argument("--stage", choices=["all", "train", "eval", "extract"], default="all",
                   help="Pipeline stage: all (default), train (extract+train+generate), eval (judge only), extract (vectors only)")
    p.add_argument("--run-dir", type=Path, help="Resume from existing run directory (for eval stage)")
    p.add_argument("--splits-file", type=Path, help="Custom splits JSON file (for 2-fold CV)")
    p.add_argument("--output-dir", type=Path, help="Custom output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def setup_environment(config: Dict):
    """Configure cache paths and HF authentication."""
    paths = config.get("paths", {})
    hf_cache = paths.get("hf_cache", "cache/transformers")
    Path(hf_cache).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
    
    # Try to load HF token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        os.environ.setdefault("HF_TOKEN", token_path.read_text().strip())


def create_run_dir(config: Dict, model_name: str) -> Path:
    """Create timestamped run directory."""
    output_root = Path(config.get("paths", {}).get("output_root", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{timestamp}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def stage_extract(config: Dict, run_dir: Path, splits_file: Optional[Path] = None) -> Dict:
    """Stage 1: Load model, extract activations, compute steering vectors."""
    LOG.info("=== STAGE: EXTRACT ===")

    model_cfg = config["model"]
    loaded = load_causal_model(
        model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
    )
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()

    # Load dataset
    tqa_cfg = config.get("truthfulqa", {})
    dataset = TruthfulQADatasetManager(
        dataset_name=tqa_cfg.get("dataset_name", "truthful_qa"),
        dataset_config=tqa_cfg.get("dataset_config", "generation"),
        cache_dir=tqa_cfg.get("cache_dir"),
        seed=config.get("run", {}).get("seed", 42),
    )

    # Load splits from file or create new ones
    if splits_file and splits_file.exists():
        LOG.info("Loading splits from: %s", splits_file)
        with splits_file.open() as f:
            splits_dict = json.load(f)
        from src.data.truthfulqa import TruthfulQAPipelineSplits
        splits = TruthfulQAPipelineSplits(
            steering_pool=splits_dict["steering_pool"],
            train=splits_dict["train"],
            test=splits_dict["test"],
            val=splits_dict.get("val", []),
        )
        LOG.info("Loaded splits: steering_pool=%d, train=%d, test=%d",
                 len(splits.steering_pool), len(splits.train), len(splits.test))
    else:
        split_cfg = tqa_cfg.get("split", {})
        splits = dataset.create_pipeline_splits(
            steering_pool_size=split_cfg.get("steering_pool", 100),
            train_size=split_cfg.get("train", 309),
            val_size=split_cfg.get("val", 0),
            test_size=split_cfg.get("test", 0),
        )
    
    # Save splits
    (run_dir / "metadata").mkdir(exist_ok=True)
    with (run_dir / "metadata" / "splits.json").open("w") as f:
        json.dump({"steering_pool": splits.steering_pool, "train": splits.train, 
                   "val": splits.val, "test": splits.test}, f, indent=2)
    
    # Extract activations
    steering_cfg = config.get("steering", {})
    extractor = ActivationExtractor(
        loaded, model_cfg["layer"],
        max_length=steering_cfg.get("max_length", 512),
        batch_size=steering_cfg.get("batch_size", 8),
    )
    
    pool_pos, pool_neg, _ = dataset.build_caa_prompts(splits.steering_pool)
    LOG.info("Extracting activations from %d prompt pairs", len(pool_pos))
    
    pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
    neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)
    
    # Filter to valid pairs
    valid_pairs = sorted(set(pos_valid) & set(neg_valid))
    pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
    neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
    pos_acts, neg_acts = pos_acts[pos_mask], neg_acts[neg_mask]
    
    LOG.info("Valid activation pairs: %d", len(pos_acts))
    
    # Compute base vector and vector bank
    normalize = steering_cfg.get("normalize_vector", False)
    base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=normalize)
    
    bank_cfg = steering_cfg.get("vector_bank", {})
    bank_mode = bank_cfg.get("mode", "subset")  # "subset" (default), "individual", or "none"

    if bank_mode == "individual":
        from src.steering.vector_bank import create_individual_bank
        vector_bank = create_individual_bank(pos_acts, neg_acts)
    else:
        builder = VectorBankBuilder(pos_acts, neg_acts, normalize=normalize, seed=config.get("run", {}).get("seed", 42))
        vector_bank = builder.build(
            num_vectors=bank_cfg.get("num_vectors", 12),
            sample_size_range=(bank_cfg.get("min_samples", 30), bank_cfg.get("max_samples", 50)),
        )
    
    # Ablation: replace vectors with random noise (same shape/norm)
    ablation = steering_cfg.get("ablation")
    if ablation == "noise":
        from src.steering.vector_bank import create_noise_bank
        LOG.info("ABLATION MODE: replacing steering vectors with random noise")
        base_vector, vector_bank = create_noise_bank(
            base_vector, vector_bank, seed=config.get("run", {}).get("seed", 42)
        )

    # Save vectors
    (run_dir / "vectors").mkdir(exist_ok=True)
    torch.save(base_vector.cpu(), run_dir / "vectors" / "base_vector.pt")
    torch.save({
        "base_vector": vector_bank.base_vector.cpu(),
        "vectors": [v.cpu() for v in vector_bank.vectors],
        "indices": vector_bank.indices,
    }, run_dir / "vectors" / "vector_bank.pt")

    return {
        "model": model, "tokenizer": tokenizer, "device": device,
        "dataset": dataset, "splits": splits, "vector_bank": vector_bank,
    }


def stage_train(config: Dict, run_dir: Path, ctx: Dict) -> Dict:
    """Stage 2: Train MC and Gen MLPs, generate responses."""
    LOG.info("=== STAGE: TRAIN ===")
    
    model, tokenizer, device = ctx["model"], ctx["tokenizer"], ctx["device"]
    dataset, splits, vector_bank = ctx["dataset"], ctx["splits"], ctx["vector_bank"]
    
    hidden_dim = vector_bank.base_vector.shape[0]
    param_dtype = next(model.parameters()).dtype
    
    mlp_cfg = config.get("mlp", {})
    arch_cfg = mlp_cfg.get("architecture", {})
    
    # Create MLPs
    bn_dim = arch_cfg.get("bottleneck_dim")
    mlp_mc = SteeringMLP(
        input_dim=hidden_dim,
        hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
        dropout=arch_cfg.get("dropout", 0.1),
        bottleneck_dim=bn_dim,
    ).to(device, dtype=param_dtype)

    mlp_gen = SteeringMLP(
        input_dim=hidden_dim,
        hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
        dropout=arch_cfg.get("dropout", 0.1),
        bottleneck_dim=bn_dim,
    ).to(device, dtype=param_dtype)
    
    # Train MC MLP
    LOG.info("Training MC MLP")
    mc_cfg = MCTrainingConfig(**mlp_cfg.get("mc_training", {}))
    mc_history = train_mc_mlp(
        mlp_mc, model=model, tokenizer=tokenizer, dataset=dataset,
        train_indices=splits.train, vector_bank=vector_bank,
        layer_index=config["model"]["layer"], primary_device=device,
        max_length=config.get("steering", {}).get("max_length", 512),
        config=mc_cfg, seed=config.get("run", {}).get("seed", 42) + 1,
    )
    
    # Train Gen MLP  
    LOG.info("Training Gen MLP")
    gen_cfg = GenTrainingConfig(**mlp_cfg.get("gen_training", {}))
    gen_history = train_gen_mlp(
        mlp_gen, model=model, tokenizer=tokenizer, dataset=dataset,
        train_indices=splits.train, vector_bank=vector_bank,
        layer_index=config["model"]["layer"], primary_device=device,
        max_length=config.get("steering", {}).get("max_length", 512),
        config=gen_cfg, seed=config.get("run", {}).get("seed", 42) + 2,
    )
    
    # Save training history and MLP weights
    with (run_dir / "training_history.json").open("w") as f:
        json.dump({"mc": mc_history, "gen": gen_history}, f, indent=2)
    
    torch.save(mlp_mc.state_dict(), run_dir / "vectors" / "mlp_mc_state_dict.pt")
    torch.save(mlp_gen.state_dict(), run_dir / "vectors" / "mlp_gen_state_dict.pt")
    
    # Generate responses for all variants
    LOG.info("Generating responses")
    _generate_all_responses(config, run_dir, model, tokenizer, device, dataset, splits, vector_bank, mlp_mc, mlp_gen)
    
    return {**ctx, "mlp_mc": mlp_mc, "mlp_gen": mlp_gen}


def _generate_all_responses(config, run_dir, model, tokenizer, device, dataset, splits, vector_bank, mlp_mc, mlp_gen):
    """Generate responses for baseline and all steering variants."""
    param_dtype = next(model.parameters()).dtype
    layer_index = config["model"]["layer"]
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
    
    test_items = dataset.get_items(splits.test)
    mc_indices = [i for i in splits.test if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)
    
    # Define variants
    base = vector_bank.base_vector.to(device, dtype=param_dtype)
    variants = {
        "baseline": None,
        "steered": base,
        "mlp_mc": mlp_mc(base.unsqueeze(0)).squeeze(0).detach() if mlp_mc else None,
        "mlp_gen": mlp_gen(base.unsqueeze(0)).squeeze(0).detach() if mlp_gen else None,
    }
    variants = {k: v for k, v in variants.items() if v is not None or k == "baseline"}

    # Filter to enabled variants if specified
    enabled = steering_cfg.get("enabled_variants")
    if enabled:
        variants = {k: v for k, v in variants.items() if k in enabled}
        LOG.info("Enabled variants: %s", list(variants.keys()))
    
    results = {}
    for name, vector in variants.items():
        LOG.info("Generating for variant: %s", name)
        scale = 0.0 if vector is None else 1.0
        
        mc_result = evaluate_multiple_choice(
            model, tokenizer, mc_items,
            layer_index=layer_index, steering_vector=vector, scale=scale,
            max_length=steering_cfg.get("max_length", 512), primary_device=device,
        )
        gen_result = evaluate_generation(
            model, tokenizer, test_items,
            layer_index=layer_index, steering_vector=vector, scale=scale,
            generation_cfg=gen_cfg, primary_device=device,
            judge=None, semantic_judge=None,  # Judge later
        )
        
        # Save details
        variant_dir = run_dir / name / f"scale_{scale:.2f}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        with (variant_dir / "mc_details.json").open("w") as f:
            json.dump(mc_result["details"], f, indent=2)
        with (variant_dir / "generation_details.json").open("w") as f:
            json.dump(gen_result["details"], f, indent=2)
        
        results[name] = {f"scale_{scale:.2f}": {"mc": mc_result["stats"], "generation": gen_result["stats"]}}
    
    with (run_dir / "results_raw.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)


def stage_eval(config: Dict, run_dir: Path):
    """Stage 3: Load existing outputs and run judges."""
    LOG.info("=== STAGE: EVAL ===")
    
    judge_cfg = config.get("evaluation", {}).get("judge", {})
    model_name = judge_cfg.get("model")
    
    if not model_name:
        LOG.warning("No judge model configured, skipping evaluation")
        return
    
    LOG.info("Loading judge model: %s", model_name)
    judge = LLMBinaryJudge(
        model_name,
        dtype=judge_cfg.get("dtype", "bfloat16"),
        device_map=judge_cfg.get("device_map", "auto"),
        max_new_tokens=judge_cfg.get("max_new_tokens", 128),
    )
    
    # Find all generation_details.json files and score them
    for gen_file in run_dir.rglob("generation_details.json"):
        LOG.info("Scoring: %s", gen_file)
        with gen_file.open() as f:
            details = json.load(f)
        
        scored = judge.score_responses(details)
        
        with gen_file.open("w") as f:
            json.dump(scored, f, indent=2)
    
    LOG.info("Evaluation complete")


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    base_config = Path("configs/base.yaml")
    model_config = Path(f"configs/models/{args.model}.yaml")
    
    if not model_config.exists():
        LOG.error("Model config not found: %s", model_config)
        return 1
    
    config = load_config(base_config, overrides=[model_config])
    config.setdefault("run", {})["seed"] = args.seed
    
    setup_environment(config)
    
    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
        if not run_dir.exists():
            LOG.error("Run directory not found: %s", run_dir)
            return 1
    elif args.output_dir:
        run_dir = args.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(config, args.model)
    
    LOG.info("Run directory: %s", run_dir)
    dump_config(config, run_dir / "config.yaml")
    
    # Execute stages
    if args.stage in ["all", "train", "extract"]:
        ctx = stage_extract(config, run_dir, splits_file=args.splits_file)
        
        if args.stage in ["all", "train"]:
            ctx = stage_train(config, run_dir, ctx)
            
            # Free model before eval
            del ctx["model"]
            import gc; gc.collect()
            torch.cuda.empty_cache()
    
    if args.stage in ["all", "eval"]:
        stage_eval(config, run_dir)
    
    LOG.info("Done! Results in: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
