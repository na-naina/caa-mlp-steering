#!/usr/bin/env python3
"""Directly optimized steering vector baseline (no MLP) — hyperparameter-matched.

Reviewer-requested baseline (ARR 2026): optimize the 4096-dim steering vector v
directly with EXACTLY the same objective and training protocol as MAST's MC MLP
training (src/steering/training.py:train_mc_mlp):

  - hinge margin loss  relu(lp_incorrect - lp_correct + m).mean(),  m = 1.0
  - anchor             lambda * F.mse_loss(v, v_init)   (per-dim mean, lambda = 0.01)
  - AdamW lr 5e-4, weight_decay 0, grad clip 1.0
  - 2 epochs x 50 steps, batch 8 sampled via np.default_rng(seed).choice(replace=False)
  - prompts "Question: {q}\nAnswer:", primary (a+, a-) pair, teacher forcing
  - identical splits (pool 100 / train 309 / test 408 from --seed)
  - identical generation settings to run.py for the judged eval

The only un-mirrorable component is the MLP's dropout (a property of the
parameterization itself); note this when reporting.

Init modes:
  caa    v0 = v_CAA (the MAST-matched baseline; anchor pulls toward v_CAA)
  zero   v0 = 0     (the "learned bias from scratch" baseline; anchor = ridge)
  random v0 ~ N(0,I) scaled to ||v_CAA||  (noise-control)

Usage:
    python scripts/train_direct_vector.py --init caa --torch-seed 1 \
        --output-dir data/outputs/directv_caa_ts1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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
from src.evaluation.truthfulqa import evaluate_generation
from src.utils.batching import build_prompt_answer_batch
from src.utils.scoring import compute_answer_logprobs

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def _select_mc_answers(dataset, item):
    mc = dataset.get_mc_targets(item.get("question", ""))
    if not mc:
        return None
    choices, labels = mc.get("choices") or [], mc.get("labels") or []
    correct = [c for c, l in zip(choices, labels) if l == 1]
    incorrect = [c for c, l in zip(choices, labels) if l == 0]
    if not correct or not incorrect:
        return None
    return correct[0], incorrect[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init", choices=["caa", "zero", "random"], default="caa")
    # Matched to configs/models/llama2_7b_chat_L8_bn8.yaml + base.yaml mc_training
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--anchor-lambda", type=float, default=0.01,
                   help="Weight on F.mse_loss(v, v_init); mirrors mse_reg=0.01. 0 disables.")
    p.add_argument("--loss", choices=["hinge", "bipo"], default="hinge",
                   help="hinge = paper margin loss; bipo = Cao et al. 2024 bi-directional "
                        "preference loss (sum log-probs vs unsteered reference, d~U{-1,1})")
    p.add_argument("--bipo-beta", type=float, default=0.1,
                   help="BiPO deviation coefficient beta (check BiPO appendix for their value)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42, help="Split + batch-order seed")
    p.add_argument("--torch-seed", type=int, default=None,
                   help="Torch/python RNG seed (init noise, decoding); defaults to --seed")
    p.add_argument("--skip-generation", action="store_true")
    args = p.parse_args()

    torch_seed = args.torch_seed if args.torch_seed is not None else args.seed
    random.seed(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading model: %s", args.model)
    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()
    param_dtype = next(model.parameters()).dtype

    dataset = TruthfulQADatasetManager(seed=args.seed)
    splits = dataset.create_pipeline_splits(steering_pool_size=100, train_size=309, test_size=0)
    (args.output_dir / "metadata").mkdir(exist_ok=True)
    (args.output_dir / "metadata" / "splits.json").write_text(json.dumps({
        "steering_pool": splits.steering_pool, "train": splits.train,
        "test": splits.test, "val": splits.val,
    }))

    LOG.info("Extracting CAA vector (pool=100)")
    extractor = ActivationExtractor(loaded, args.layer, max_length=args.max_length, batch_size=8)
    pool_pos, pool_neg, _ = dataset.build_caa_prompts(splits.steering_pool)
    pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
    neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)
    valid_pairs = sorted(set(pos_valid) & set(neg_valid))
    pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
    neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
    v_caa = compute_caa_vector(pos_acts[pos_mask], neg_acts[neg_mask], normalize=False)

    if args.init == "caa":
        v_init = v_caa.clone()
    elif args.init == "zero":
        v_init = torch.zeros_like(v_caa)
    else:
        r = torch.randn(v_caa.shape[0])
        v_init = r * (v_caa.norm() / r.norm())

    v_init = v_init.to(device, dtype=param_dtype)
    v = v_init.clone().requires_grad_(True)
    LOG.info("Init '%s': ||v0||=%.4f (||v_CAA||=%.4f)", args.init, v_init.norm().item(), v_caa.norm().item())

    optimizer = torch.optim.AdamW([v], lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    valid_indices = [idx for idx in splits.train if dataset.is_valid_mc(idx)]

    history = {"loss": [], "margin": [], "accuracy": []}
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
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

            inp_c = build_prompt_answer_batch(tokenizer, prompts, ans_c, max_length=args.max_length)
            inp_i = build_prompt_answer_batch(tokenizer, prompts, ans_i, max_length=args.max_length)
            ids_c, at_c, m_c = (t.to(device) for t in inp_c)
            ids_i, at_i, m_i = (t.to(device) for t in inp_i)
            ntok_c, ntok_i = m_c.sum(dim=1), m_i.sum(dim=1)

            if args.loss == "bipo":
                # reference (unsteered) sum log-probs, no grad
                with torch.no_grad():
                    ref_c, _ = compute_answer_logprobs(model, input_ids=ids_c,
                                                       attention_mask=at_c, answer_mask=m_c)
                    ref_i, _ = compute_answer_logprobs(model, input_ids=ids_i,
                                                       attention_mask=at_i, answer_mask=m_i)
                d = 1.0 if rng.random() < 0.5 else -1.0
                with steering_hook(model, args.layer, v, scale=d):
                    lp_c, _ = compute_answer_logprobs(model, input_ids=ids_c,
                                                      attention_mask=at_c, answer_mask=m_c)
                    lp_i, _ = compute_answer_logprobs(model, input_ids=ids_i,
                                                      attention_mask=at_i, answer_mask=m_i)
                # per-token means -> sums, as in BiPO Eq. 3
                dlt_c = (lp_c - ref_c) * ntok_c
                dlt_i = (lp_i - ref_i) * ntok_i
                margin_values = -(d * args.bipo_beta * (dlt_c - dlt_i))  # for logging
                loss_main = -F.logsigmoid(d * args.bipo_beta * (dlt_c - dlt_i)).mean()
            else:
                with steering_hook(model, args.layer, v, scale=1.0):
                    lp_c, _ = compute_answer_logprobs(model, input_ids=ids_c,
                                                      attention_mask=at_c, answer_mask=m_c)
                    lp_i, _ = compute_answer_logprobs(model, input_ids=ids_i,
                                                      attention_mask=at_i, answer_mask=m_i)
                margin_values = lp_i - lp_c + args.margin
                loss_main = F.relu(margin_values).mean()

            anchor = F.mse_loss(v, v_init) if args.anchor_lambda > 0 else torch.tensor(0.0, device=device)
            loss = loss_main + args.anchor_lambda * anchor

            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_([v], args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                history["loss"].append(loss.item())
                history["margin"].append(margin_values.mean().item())
                history["accuracy"].append((lp_c > lp_i).float().mean().item())

        LOG.info("epoch %d/%d - loss %.4f, acc %.3f, ||v||=%.4f, ||v-v0||=%.4f",
                 epoch + 1, args.epochs, history["loss"][-1], history["accuracy"][-1],
                 v.norm().item(), (v - v_init).norm().item())

    optimized = v.detach()
    vf = optimized.float().cpu()
    vc = v_caa.float()
    meta = {
        "init": args.init, "lr": args.lr, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
        "margin": args.margin, "anchor_lambda": args.anchor_lambda,
        "grad_clip": args.grad_clip, "seed": args.seed, "torch_seed": torch_seed,
        "layer": args.layer, "model": args.model,
        "v_caa_norm": vc.norm().item(), "v_init_norm": v_init.float().cpu().norm().item(),
        "v_final_norm": vf.norm().item(),
        "cos_final_vs_caa": F.cosine_similarity(vf, vc, dim=0).item(),
        "delta_norm_vs_init": (vf - v_init.float().cpu()).norm().item(),
        "note": "hyperparameter-matched to MAST mc_training; dropout has no vector analogue",
    }
    (args.output_dir / "vectors").mkdir(exist_ok=True)
    torch.save(optimized.cpu(), args.output_dir / "vectors" / "optimized_vector.pt")
    torch.save(vc, args.output_dir / "vectors" / "base_vector.pt")
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (args.output_dir / "training_history.json").write_text(json.dumps({"mc": history}))
    LOG.info("meta: %s", json.dumps(meta, indent=1))

    if args.skip_generation:
        return

    # Generation identical to run.py._generate_all_responses gen_cfg
    gen_cfg = {"preset": "qa", "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 64,
               "max_length": args.max_length, "stop_sequences": ["\n\n", "\nQuestion:"]}
    test_items = dataset.get_items(splits.test)
    LOG.info("Generating %d test responses", len(test_items))
    gen_result = evaluate_generation(model, tokenizer, test_items, layer_index=args.layer,
                                     steering_vector=optimized, scale=1.0,
                                     generation_cfg=gen_cfg, primary_device=device,
                                     judge=None, semantic_judge=None)
    gen_dir = args.output_dir / "mlp_mc" / "scale_1.00"  # judge-script-compatible layout
    gen_dir.mkdir(parents=True, exist_ok=True)
    with (gen_dir / "generation_details.json").open("w") as f:
        json.dump(gen_result["details"], f, indent=2)
    LOG.info("Done! %d responses -> %s", len(gen_result["details"]), gen_dir)


if __name__ == "__main__":
    main()
