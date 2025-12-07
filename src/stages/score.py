#!/usr/bin/env python3
"""Stage 4: Score generated responses with judges.

Resource requirements: Inference only (judge model)
- Can run on smaller GPU than main model
- ~24GB for 12B judge

Inputs:
- responses/{variant}/generation.json

Outputs:
- scores/{variant}/scored.json
- scores/summary.json
- metadata/score.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.evaluation.judge import LLMBinaryJudge
from src.evaluation.informativeness import LLMInformativenessJudge
from src.evaluation.semantic import SemanticJudge, SemanticJudgeConfig
from src.evaluation.truthfulqa import _summarize_generation
from src.models.loader import load_causal_model
from src.stages.common import (
    CheckpointManager,
    RunContext,
    check_stage_complete,
    get_or_create_run,
    load_config,
    save_stage_metadata,
    setup_environment,
    setup_logging,
)

LOG = logging.getLogger(__name__)


def run_scoring(ctx: RunContext, force: bool = False) -> dict:
    """Score all generated responses with configured judges.

    Each variant is checkpointed after scoring, so if interrupted,
    the job can resume from the last completed variant.
    """
    config = ctx.config
    eval_cfg = config.get("evaluation", {})

    # Initialize checkpoint manager
    ckpt = CheckpointManager(ctx, "score")
    if force:
        ckpt.clear()

    # Find all variants with generation outputs
    variants = []
    for variant_dir in ctx.responses_dir.iterdir():
        if variant_dir.is_dir() and (variant_dir / "generation.json").exists():
            variants.append(variant_dir.name)

    if not variants:
        LOG.warning("No generated responses found to score")
        return {"variants": [], "results": {}}

    # Determine pending variants
    pending_variants = ckpt.get_pending(variants)

    # Load cached results for completed variants
    results = {}
    for variant in variants:
        if ckpt.is_complete(variant):
            cached = ckpt.get_result(variant)
            if cached:
                results[variant] = cached

    if not pending_variants:
        LOG.info("All variants already scored (resuming from checkpoint)")
        # Save summary with cached results
        with open(ctx.scores_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        return {"variants": variants, "judges": [], "results": results}

    LOG.info("Variants to score: %s (skipping completed: %s)",
             pending_variants, [v for v in variants if v not in pending_variants])

    # Build judges (only if we have work to do)
    judges = {}

    # Truthfulness judge
    judge_cfg = eval_cfg.get("judge", {})
    if judge_cfg.get("model"):
        LOG.info("Loading truthfulness judge: %s", judge_cfg["model"])

        # Check if we can share model with informativeness
        info_cfg = eval_cfg.get("informativeness", {})
        share_model = (
            info_cfg.get("enabled", False) and
            info_cfg.get("model") == judge_cfg.get("model")
        )

        if share_model:
            LOG.info("Sharing model between truth and info judges")
            shared = load_causal_model(
                judge_cfg["model"],
                dtype=judge_cfg.get("dtype", "bfloat16"),
                device_map=judge_cfg.get("device_map", "auto"),
            )
            judges["truth"] = LLMBinaryJudge(
                judge_cfg["model"],
                dtype=judge_cfg.get("dtype", "bfloat16"),
                device_map=judge_cfg.get("device_map", "auto"),
                max_new_tokens=judge_cfg.get("max_new_tokens", 128),
                shared_model=shared,
            )
            judges["info"] = LLMInformativenessJudge(
                info_cfg["model"],
                dtype=info_cfg.get("dtype", "bfloat16"),
                device_map=info_cfg.get("device_map", "auto"),
                max_new_tokens=info_cfg.get("max_new_tokens", 128),
                shared_model=shared,
            )
        else:
            judges["truth"] = LLMBinaryJudge(
                judge_cfg["model"],
                dtype=judge_cfg.get("dtype", "bfloat16"),
                device_map=judge_cfg.get("device_map", "auto"),
                max_new_tokens=judge_cfg.get("max_new_tokens", 128),
            )

    # Informativeness judge (if not shared)
    info_cfg = eval_cfg.get("informativeness", {})
    if info_cfg.get("enabled", False) and "info" not in judges and info_cfg.get("model"):
        LOG.info("Loading informativeness judge: %s", info_cfg["model"])
        judges["info"] = LLMInformativenessJudge(
            info_cfg["model"],
            dtype=info_cfg.get("dtype", "bfloat16"),
            device_map=info_cfg.get("device_map", "auto"),
            max_new_tokens=info_cfg.get("max_new_tokens", 128),
        )

    # Semantic judge (lightweight, always load)
    semantic_cfg = eval_cfg.get("semantic", {})
    if semantic_cfg.get("enabled", True):
        LOG.info("Loading semantic judge")
        judges["semantic"] = SemanticJudge(SemanticJudgeConfig(
            model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            similarity_threshold=semantic_cfg.get("similarity_threshold", 0.6),
        ))

    if not judges:
        LOG.warning("No judges configured")
        return {"variants": variants, "results": results}

    # Score each pending variant
    for variant in pending_variants:
        LOG.info("Scoring variant: %s", variant)

        gen_path = ctx.responses_dir / variant / "generation.json"
        with open(gen_path) as f:
            gen_data = json.load(f)

        details = gen_data["details"]

        # Apply each judge
        if "semantic" in judges:
            details = judges["semantic"].score_responses(details)
        if "truth" in judges:
            details = judges["truth"].score_responses(details)
        if "info" in judges:
            details = judges["info"].score_responses(details)

        # Compute summary stats
        stats = _summarize_generation(
            details,
            judged="truth" in judges,
            informativeness_used="info" in judges,
            semantic_used="semantic" in judges,
            bleurt_used=False,
        )

        # Save scored results
        score_dir = ctx.scores_dir / variant
        score_dir.mkdir(exist_ok=True)

        with open(score_dir / "scored.json", "w") as f:
            json.dump({
                "stats": asdict(stats),
                "details": details,
            }, f, indent=2)

        variant_stats = asdict(stats)
        results[variant] = variant_stats

        # Checkpoint this variant's completion
        ckpt.mark_complete(variant, variant_stats)

        LOG.info("  Accuracy: %.3f, Info mean: %.3f (checkpointed)",
                 stats.accuracy or 0, stats.informativeness_mean or 0)

    # Save summary
    with open(ctx.scores_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return {
        "variants": variants,
        "judges": list(judges.keys()),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Score generated responses")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")
    args = parser.parse_args()

    config = load_config(args.model, args.config_dir)
    setup_environment(config)

    ctx = get_or_create_run(args.model, config, args.run_id)
    setup_logging(args.verbose, ctx.run_dir / "logs" / "score.log")

    LOG.info("Run directory: %s", ctx.run_dir)

    # Check prerequisites
    if not check_stage_complete(ctx, "generate"):
        LOG.error("Generation stage not complete. Run generate stage first.")
        return 1

    if check_stage_complete(ctx, "score") and not args.force:
        LOG.info("Scoring already complete, skipping (use --force to re-run)")
        return 0

    try:
        metadata = run_scoring(ctx, force=args.force)
        save_stage_metadata(ctx, "score", metadata)
        LOG.info("Scoring stage complete")
        return 0
    except Exception as e:
        LOG.exception("Scoring failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
