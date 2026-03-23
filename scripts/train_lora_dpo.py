#!/usr/bin/env python3
"""Train LoRA with DPO on TruthfulQA, then run our steering pipeline on top.

This demonstrates that activation steering (ours) and parameter-space
fine-tuning (LoRA DPO) are complementary — they operate in different spaces
and their improvements should stack.

Usage:
    python scripts/train_lora_dpo.py --output-dir data/outputs/lora_dpo
    python scripts/train_lora_dpo.py --output-dir data/outputs/lora_dpo --skip-lora  # reuse existing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def prepare_dpo_dataset(seed=42):
    """Convert TruthfulQA into DPO preference pairs using our train split."""
    from src.data.truthfulqa import TruthfulQADatasetManager

    dataset = TruthfulQADatasetManager(seed=seed)
    splits = dataset.create_pipeline_splits(
        steering_pool_size=100, train_size=309, test_size=0
    )

    pairs = []
    for idx in splits.train:
        item = dataset.get_item(int(idx))
        question = item["question"]
        best_answer = item.get("best_answer") or item["correct_answers"][0]
        incorrect = item.get("incorrect_answers") or []
        if not incorrect:
            continue

        # Use first incorrect answer as rejected
        prompt = f"Question: {question}\nAnswer:"
        pairs.append({
            "prompt": prompt,
            "chosen": f" {best_answer}",
            "rejected": f" {incorrect[0]}",
        })

    LOG.info("Prepared %d DPO preference pairs from train split", len(pairs))
    return pairs


def train_lora_dpo(model_name, output_dir, pairs, num_epochs=2, lr=5e-5, lora_r=8):
    """Train LoRA adapters with DPO on TruthfulQA preference pairs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    LOG.info("Loading base model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA config — apply to Q, V projections like standard LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO training config
    lora_output = Path(output_dir) / "lora_adapter"
    training_args = DPOConfig(
        output_dir=str(lora_output),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        max_length=512,
    )

    # Create HF dataset
    hf_dataset = Dataset.from_list(pairs)

    LOG.info("Starting DPO training (LoRA r=%d, lr=%s, epochs=%d)", lora_r, lr, num_epochs)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    # Save adapter
    trainer.save_model(str(lora_output))
    tokenizer.save_pretrained(str(lora_output))
    LOG.info("LoRA adapter saved to %s", lora_output)

    # Free memory
    del trainer, model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return str(lora_output)


def run_steering_pipeline(model_name, lora_path, output_dir, seed=42, bottleneck_dim=8):
    """Run our full steering pipeline on the LoRA-finetuned model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data.truthfulqa import TruthfulQADatasetManager
    from src.steering.extract import ActivationExtractor, compute_caa_vector
    from src.steering.vector_bank import create_individual_bank
    from src.steering.mlp import SteeringMLP
    from src.steering.training import MCTrainingConfig, GenTrainingConfig, train_mc_mlp, train_gen_mlp
    from src.evaluation.truthfulqa import evaluate_multiple_choice, evaluate_generation

    layer_index = 8
    run_dir = Path(output_dir) / "steering_results"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load LoRA model
    LOG.info("Loading LoRA model from %s", lora_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()  # Merge LoRA into base for inference
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    # Dataset + splits (same as always)
    dataset = TruthfulQADatasetManager(seed=seed)
    splits = dataset.create_pipeline_splits(
        steering_pool_size=100, train_size=309, test_size=0
    )

    # Extract CAA vectors from LoRA model
    LOG.info("Extracting CAA vectors from LoRA-finetuned model")
    extractor = ActivationExtractor(
        type('LoadedModel', (), {'model': model, 'tokenizer': tokenizer, 'primary_device': device})(),
        layer_index, max_length=512, batch_size=8,
    )

    pool_pos, pool_neg, _ = dataset.build_caa_prompts(splits.steering_pool)
    pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
    neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)
    valid_pairs = sorted(set(pos_valid) & set(neg_valid))
    pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
    neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
    pos_acts, neg_acts = pos_acts[pos_mask], neg_acts[neg_mask]

    base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=False)
    vector_bank = create_individual_bank(pos_acts, neg_acts)

    # Save vectors
    (run_dir / "vectors").mkdir(exist_ok=True)
    torch.save(base_vector.cpu(), run_dir / "vectors" / "base_vector.pt")
    torch.save(vector_bank.base_vector.cpu(), run_dir / "vectors" / "vector_bank_base.pt")

    # Train MLP
    LOG.info("Training MC MLP on LoRA model")
    hidden_dim = base_vector.shape[0]
    param_dtype = next(model.parameters()).dtype

    mlp_mc = SteeringMLP(input_dim=hidden_dim, bottleneck_dim=bottleneck_dim).to(device, dtype=param_dtype)
    mc_cfg = MCTrainingConfig(lr=5e-4, epochs=2, steps_per_epoch=50, batch_size=4)
    train_mc_mlp(
        mlp_mc, model=model, tokenizer=tokenizer, dataset=dataset,
        train_indices=splits.train, vector_bank=vector_bank,
        layer_index=layer_index, primary_device=device,
        max_length=512, config=mc_cfg, seed=seed + 1,
    )
    torch.save(mlp_mc.state_dict(), run_dir / "vectors" / "mlp_mc_state_dict.pt")

    # Generate
    LOG.info("Generating responses with LoRA + MLP steering")
    test_items = dataset.get_items(splits.test)
    mc_indices = [i for i in splits.test if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)

    transformed = mlp_mc(vector_bank.base_vector.to(device, dtype=param_dtype).unsqueeze(0)).squeeze(0).detach()

    # MC eval
    mc_result = evaluate_multiple_choice(
        model, tokenizer, mc_items,
        layer_index=layer_index, steering_vector=transformed, scale=1.0,
        max_length=512, primary_device=device,
    )
    LOG.info("MC accuracy (LoRA+MLP): %.4f", mc_result["stats"].accuracy)

    # Generation
    gen_cfg = {
        "preset": "qa", "temperature": 0.3, "top_p": 0.9,
        "max_new_tokens": 64, "max_length": 512,
        "stop_sequences": ["\n\n", "\nQuestion:"],
    }
    gen_result = evaluate_generation(
        model, tokenizer, test_items,
        layer_index=layer_index, steering_vector=transformed, scale=1.0,
        generation_cfg=gen_cfg, primary_device=device,
        judge=None, semantic_judge=None,
    )

    # Save in GPT judge-compatible format
    gen_dir = run_dir / "mlp_mc" / "scale_1.00"
    gen_dir.mkdir(parents=True, exist_ok=True)
    with (gen_dir / "generation_details.json").open("w") as f:
        json.dump(gen_result["details"], f, indent=2)

    LOG.info("Done! MC accuracy: %.4f, %d responses saved to %s",
             mc_result["stats"].accuracy, len(gen_result["details"]), gen_dir)


def main():
    parser = argparse.ArgumentParser(description="LoRA DPO + steering pipeline")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/lora_dpo"))
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA training, reuse existing adapter")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lora_path = args.output_dir / "lora_adapter"

    if not args.skip_lora:
        # Step 1: Prepare DPO data
        pairs = prepare_dpo_dataset(seed=args.seed)

        # Step 2: Train LoRA DPO
        lora_path = train_lora_dpo(
            args.model, args.output_dir, pairs,
            num_epochs=args.epochs, lr=args.lr, lora_r=args.lora_r,
        )
    else:
        LOG.info("Skipping LoRA training, using existing adapter at %s", lora_path)

    # Step 3: Run steering pipeline on LoRA model
    run_steering_pipeline(args.model, str(lora_path), str(args.output_dir),
                          seed=args.seed, bottleneck_dim=args.bottleneck_dim)


if __name__ == "__main__":
    main()
