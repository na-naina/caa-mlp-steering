#!/usr/bin/env python3
"""Reproduction of RaLFiT (Li, Mao & Wang, Findings ACL 2025) under our 2-fold CV protocol.

RaLFiT = LoRA-DPO on W_O (attention output) and W_down (FFN output) with per-module ranks
allocated by probing (no public code; implemented from the paper, Sec. 3 + 4.1):

  1. Probing. Each training question's truthful and untruthful answers are concatenated with
     the question and run through the frozen model; the output of every MHA and FFN module at
     the LAST token is recorded (2N modules). One sklearn MLPClassifier with default settings
     per module is trained on a random 4:1 split; Corr = 2|Acc - 0.5| on the held-out fifth.
  2. Allocation. rank_i = Corr_i^a / sum_j Corr_j^a * budget, a = 1, budget = 8 * 2N
     (average rank 8, as in their main setting).
  3. Training. DPO with that rank pattern, everything else identical to our LoRA-DPO
     baseline (scripts/train_lora_dpo.py: lr 1e-4, 5 epochs, batch 8, beta 0.1, same pairs),
     so RaLFiT vs LoRA-DPO differs only in the rank allocation, as in RaLFiT's Table 1.

Judgement calls (the paper does not specify) are listed in docs/ralfit_repro_oct9.md:
one (best, first incorrect) pair per question for both probing and DPO; random 4:1 split
over probe samples; ranks rounded to integers, rank 0 => module not adapted; lora_alpha =
2 * rank per module (keeps our baseline's alpha/r = 16/8 scaling).

Output layout matches the other revision runs (data/outputs/rcv_ralfit_s<seed>/fold<k>/),
so evaluate_with_gpt_judge.py and aggregate_revision.py pick it up as method "ralfit".

Usage:
    python scripts/ralfit_repro.py --splits-file data/splits/cv2_s42_fold1.json --seed 42 \
        --output-dir data/outputs/rcv_ralfit_s42/fold1 [--train-only | --generate-only]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from src.prompts.chat import qa_prompt  # noqa: E402
from train_lora_dpo import load_splits, prepare_dpo_dataset, run_lora_only_generation  # noqa: E402

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger("ralfit")
MODULES = ("self_attn.o_proj", "mlp.down_proj")


def probe_texts(seed, splits_file, include_pool):
    """(text, label) per answer: question + truthful (1) / untruthful (0) answer."""
    pairs = prepare_dpo_dataset(seed=seed, splits_file=splits_file, include_pool=include_pool)
    texts, labels = [], []
    for p in pairs:
        texts += [p["prompt"] + p["chosen"], p["prompt"] + p["rejected"]]
        labels += [1, 0]
    return texts, np.array(labels)


@torch.no_grad()
def collect_module_outputs(model, tokenizer, texts, batch_size=8):
    """Last-token output of every W_O and W_down module -> {module_name: [n, d] float32}."""
    names = [n for n, _ in model.named_modules() if n.endswith(MODULES)]
    feats = {n: [] for n in names}
    cur = {}
    hooks = [m.register_forward_hook(lambda _m, _i, out, n=n: cur.__setitem__(n, out))
             for n, m in model.named_modules() if n in feats]
    tokenizer.padding_side = "right"
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(last.numel(), device=device)
        for n in names:
            feats[n].append(cur[n][rows, last].float().cpu())
    for h in hooks:
        h.remove()
    return {n: torch.cat(v).numpy() for n, v in feats.items()}


def probe_correlations(feats, labels, seed):
    """Corr = 2|Acc - 0.5| of a default sklearn MLPClassifier per module (4:1 split)."""
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    idx_tr, idx_va = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=seed)
    out = {}
    for n, X in feats.items():
        clf = MLPClassifier(random_state=seed).fit(X[idx_tr], labels[idx_tr])
        acc = float(clf.score(X[idx_va], labels[idx_va]))
        out[n] = {"acc": acc, "corr": 2 * abs(acc - 0.5)}
        LOG.info("probe %-40s acc %.3f", n, acc)
    return out


def allocate_ranks(probe, avg_rank=8, sharpness=1.0):
    corr = {n: v["corr"] ** sharpness for n, v in probe.items()}
    budget, total = avg_rank * len(corr), sum(corr.values())
    return {n: int(round(c / total * budget)) for n, c in corr.items()}


def train_ralfit(model_name, output_dir, pairs, ranks, lr, epochs, batch_size, beta, seed):
    """DPO with per-module LoRA ranks (copy of train_lora_dpo.train_lora_dpo + rank_pattern)."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    active = {n: r for n, r in ranks.items() if r > 0}
    pat = {re.escape(n): r for n, r in active.items()}
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=sorted(active),
        rank_pattern=pat, alpha_pattern={k: 2 * r for k, r in pat.items()},
    )
    lora_output = Path(output_dir) / "lora_adapter"
    args = DPOConfig(
        output_dir=str(lora_output), num_train_epochs=epochs,
        per_device_train_batch_size=min(batch_size, 2),
        gradient_accumulation_steps=max(1, batch_size // 2),
        learning_rate=lr, beta=beta, seed=seed, bf16=True, logging_steps=10,
        save_strategy="no", remove_unused_columns=False, max_length=512,
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(pairs),
                         processing_class=tokenizer, peft_config=lora_config)
    # fidelity check: the realised per-module ranks and trainable-parameter count
    realised = {n: m.lora_A["default"].weight.shape[0] for n, m in trainer.model.named_modules()
                if hasattr(m, "lora_A") and "default" in m.lora_A}
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    mismatch = [n for n, r in active.items()
                if not any(k.endswith(n) and v == r for k, v in realised.items())]
    if mismatch or len(realised) != len(active):
        raise RuntimeError(f"rank_pattern mismatch: {len(realised)} adapted vs {len(active)} "
                           f"allocated; wrong rank on e.g. {mismatch[:3]}")
    LOG.info("RaLFiT: %d adapted modules, %.2fM trainable params", len(active), n_train / 1e6)
    trainer.train()
    trainer.save_model(str(lora_output))
    tokenizer.save_pretrained(str(lora_output))
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return str(lora_output), n_train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--splits-file", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--avg-rank", type=int, default=8)
    ap.add_argument("--sharpness", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--no-pool", action="store_true",
                    help="exclude the 100-question steering pool (default: use the whole training fold, like LoRA-DPO)")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: limit training pairs")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    include_pool = not args.no_pool

    if not args.generate_only:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        texts, labels = probe_texts(args.seed, args.splits_file, include_pool)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     device_map="auto").eval()
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        feats = collect_module_outputs(model, tok, texts)
        del model
        gc.collect(); torch.cuda.empty_cache()
        probe = probe_correlations(feats, labels, args.seed)
        ranks = allocate_ranks(probe, args.avg_rank, args.sharpness)

        pairs = prepare_dpo_dataset(seed=args.seed, splits_file=args.splits_file,
                                    include_pool=include_pool)
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        _, n_train = train_ralfit(args.model, args.output_dir, pairs, ranks, args.lr,
                                  args.epochs, args.batch_size, args.beta, args.seed)
        (args.output_dir / "ralfit_allocation.json").write_text(json.dumps(
            {"probe": probe, "ranks": ranks, "rank_sum": sum(ranks.values()),
             "trainable_params": n_train, "n_probe_samples": int(len(labels))}, indent=2))
        (args.output_dir / "meta.json").write_text(json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    if not args.train_only:
        run_lora_only_generation(args.model, str(args.output_dir / "lora_adapter"),
                                 str(args.output_dir), seed=args.seed, splits_file=args.splits_file)


if __name__ == "__main__":
    main()
