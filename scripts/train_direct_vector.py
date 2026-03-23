#!/usr/bin/env python3
"""Optimize a steering vector directly via GD (no MLP).

Baseline comparison: does the bottleneck MLP add anything over
directly optimizing the raw vector with the same loss function?

Usage:
    python scripts/train_direct_vector.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.data.truthfulqa import TruthfulQADatasetManager
from src.steering.extract import ActivationExtractor, compute_caa_vector
from src.steering.apply import steering_hook
from src.steering.vector_bank import VectorBankBuilder
from src.evaluation.truthfulqa import evaluate_multiple_choice, evaluate_generation
from src.utils.batching import build_prompt_answer_batch
from src.utils.scoring import compute_answer_logprobs

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def _select_mc_answers(dataset, item):
    question = item.get("question", "")
    mc = dataset.get_mc_targets(question)
    if not mc:
        return None
    choices = mc.get("choices") or []
    labels = mc.get("labels") or []
    correct = [c for c, l in zip(choices, labels) if l == 1]
    incorrect = [c for c, l in zip(choices, labels) if l == 0]
    if not correct or not incorrect:
        return None
    return correct[0], incorrect[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/direct_vector"))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--init", choices=["caa", "random"], default="caa",
                        help="Initialize from CAA vector or random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    LOG.info("Loading model: %s", args.model)
    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()
    param_dtype = next(model.parameters()).dtype

    # Dataset + splits
    dataset = TruthfulQADatasetManager(seed=args.seed)
    splits = dataset.create_pipeline_splits(steering_pool_size=100, train_size=309, test_size=0)

    # Extract CAA vector (needed for init and/or comparison)
    LOG.info("Extracting CAA vector")
    extractor = ActivationExtractor(loaded, args.layer, max_length=512, batch_size=8)
    pool_pos, pool_neg, _ = dataset.build_caa_prompts(splits.steering_pool)
    pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
    neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)
    valid_pairs = sorted(set(pos_valid) & set(neg_valid))
    pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
    neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
    pos_acts, neg_acts = pos_acts[pos_mask], neg_acts[neg_mask]
    base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=False)

    # Initialize the optimizable vector
    if args.init == "caa":
        v = base_vector.clone().to(device, dtype=param_dtype).requires_grad_(True)
        LOG.info("Initialized from CAA vector (norm=%.4f)", v.norm().item())
    else:
        v = torch.randn(base_vector.shape[0], device=device, dtype=param_dtype)
        v = v * (base_vector.norm().item() / v.norm().item())
        v = v.requires_grad_(True)
        LOG.info("Initialized from random vector (norm=%.4f)", v.norm().item())

    optimizer = torch.optim.Adam([v], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    valid_indices = [idx for idx in splits.train if dataset.is_valid_mc(idx)]

    # Train: optimize v directly with margin loss
    LOG.info("Optimizing vector directly (%d steps, lr=%s)", args.steps, args.lr)
    for step in range(args.steps):
        optimizer.zero_grad()

        batch_idx = rng.choice(valid_indices, size=min(args.batch_size, len(valid_indices)), replace=False)
        prompts, ans_c, ans_i = [], [], []
        for idx in batch_idx:
            item = dataset.get_item(int(idx))
            pair = _select_mc_answers(dataset, item)
            if pair is None:
                continue
            prompts.append(f"Question: {item['question']}\nAnswer:")
            ans_c.append(pair[0])
            ans_i.append(pair[1])

        if not prompts:
            continue

        with steering_hook(model, args.layer, v, scale=1.0):
            inp_c = build_prompt_answer_batch(tokenizer, prompts, ans_c, max_length=512)
            inp_i = build_prompt_answer_batch(tokenizer, prompts, ans_i, max_length=512)

            lp_c, _ = compute_answer_logprobs(model, input_ids=inp_c[0].to(device),
                                               attention_mask=inp_c[1].to(device), answer_mask=inp_c[2].to(device))
            lp_i, _ = compute_answer_logprobs(model, input_ids=inp_i[0].to(device),
                                               attention_mask=inp_i[1].to(device), answer_mask=inp_i[2].to(device))

        margin_vals = lp_i - lp_c + args.margin
        loss = F.relu(margin_vals).mean()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            with torch.no_grad():
                acc = (lp_c > lp_i).float().mean().item()
            LOG.info("Step %d/%d: loss=%.4f, acc=%.3f, v_norm=%.4f",
                     step + 1, args.steps, loss.item(), acc, v.norm().item())

    LOG.info("Final vector norm: %.4f", v.norm().item())
    LOG.info("Cosine sim (optimized vs CAA): %.6f",
             F.cosine_similarity(v.detach().cpu().unsqueeze(0), base_vector.unsqueeze(0)).item())

    # Save optimized vector
    torch.save(v.detach().cpu(), args.output_dir / "optimized_vector.pt")

    # Evaluate MC
    optimized = v.detach()
    mc_indices = [i for i in splits.test if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)
    mc_result = evaluate_multiple_choice(model, tokenizer, mc_items, layer_index=args.layer,
                                          steering_vector=optimized, scale=1.0, max_length=512, primary_device=device)
    LOG.info("MC accuracy: %.4f", mc_result["stats"].accuracy)

    # Generate
    test_items = dataset.get_items(splits.test)
    gen_cfg = {"preset": "qa", "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 64,
               "max_length": 512, "stop_sequences": ["\n\n", "\nQuestion:"]}
    gen_result = evaluate_generation(model, tokenizer, test_items, layer_index=args.layer,
                                      steering_vector=optimized, scale=1.0, generation_cfg=gen_cfg,
                                      primary_device=device, judge=None, semantic_judge=None)

    gen_dir = args.output_dir / "mlp_mc" / "scale_1.00"
    gen_dir.mkdir(parents=True, exist_ok=True)
    with (gen_dir / "generation_details.json").open("w") as f:
        json.dump(gen_result["details"], f, indent=2)

    LOG.info("Done! %d responses saved", len(gen_result["details"]))


if __name__ == "__main__":
    main()
