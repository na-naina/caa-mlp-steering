#!/usr/bin/env python3
"""
Score pre-generated responses with judge models.
Use this after generation completes but judge loading fails due to OOM.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.judge import LLMBinaryJudge
from src.evaluation.informativeness import LLMInformativenessJudge
from src.evaluation.semantic import SemanticJudge, SemanticJudgeConfig
from src.evaluation.truthfulqa import _summarize_generation
from src.models.loader import load_causal_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Score pre-generated responses")
    parser.add_argument("run_dir", type=Path, help="Run output directory")
    parser.add_argument(
        "--judge-model",
        default="google/gemma-3-12b-it",
        help="Judge model name (used for both truth and informativeness if same)",
    )
    parser.add_argument(
        "--judge-device",
        default="auto",
        help="Judge device map",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip truthfulness judge scoring",
    )
    parser.add_argument(
        "--no-informativeness",
        action="store_true",
        help="Skip informativeness judge scoring",
    )
    parser.add_argument(
        "--informativeness-model",
        default=None,
        help="Informativeness judge model (defaults to same as --judge-model)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic judge scoring",
    )
    parser.add_argument(
        "--no-bleurt",
        action="store_true",
        help="Skip BLEURT judge scoring",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    logger.info(f"Scoring responses in {run_dir}")

    # Determine if we can share the judge model
    informativeness_model = args.informativeness_model or args.judge_model
    shared_judge_model = None

    if not args.no_judge and not args.no_informativeness and args.judge_model == informativeness_model:
        logger.info(f"Both judges use '{args.judge_model}' - loading once and sharing")
        shared_judge_model = load_causal_model(
            args.judge_model,
            dtype="bfloat16",
            device_map=args.judge_device,
        )

    # Load truthfulness judge
    judge = None
    if not args.no_judge:
        if shared_judge_model:
            logger.info(f"Using shared model for truthfulness judge")
            judge = LLMBinaryJudge(
                args.judge_model,
                dtype="bfloat16",
                device_map=args.judge_device,
                max_new_tokens=128,
                shared_model=shared_judge_model,
            )
        else:
            logger.info(f"Loading truthfulness judge model: {args.judge_model}")
            judge = LLMBinaryJudge(
                args.judge_model,
                dtype="bfloat16",
                device_map=args.judge_device,
                max_new_tokens=128,
            )

    # Load informativeness judge
    informativeness_judge = None
    if not args.no_informativeness:
        if shared_judge_model:
            logger.info(f"Using shared model for informativeness judge")
            informativeness_judge = LLMInformativenessJudge(
                informativeness_model,
                dtype="bfloat16",
                device_map=args.judge_device,
                max_new_tokens=128,
                shared_model=shared_judge_model,
            )
        else:
            logger.info(f"Loading informativeness judge model: {informativeness_model}")
            informativeness_judge = LLMInformativenessJudge(
                informativeness_model,
                dtype="bfloat16",
                device_map=args.judge_device,
                max_new_tokens=128,
            )

    # Load semantic judge
    semantic_judge = None
    if not args.no_semantic:
        logger.info("Loading semantic judge")
        config = SemanticJudgeConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.6,
        )
        semantic_judge = SemanticJudge(config)

    # Load BLEURT judge
    bleurt_judge = None
    if not args.no_bleurt:
        try:
            from src.evaluation.bleurt_judge import BLEURTJudge, BLEURTJudgeConfig
            logger.info("Loading BLEURT judge")
            config = BLEURTJudgeConfig(checkpoint="bleurt-base-128")
            bleurt_judge = BLEURTJudge(config)
        except Exception as exc:
            logger.warning(f"Failed to load BLEURT judge: {exc}")

    # Find all generation_details.json files
    evaluation = {}

    for variant_dir in run_dir.iterdir():
        if not variant_dir.is_dir() or variant_dir.name in {"metadata", "vectors"}:
            continue

        variant_name = variant_dir.name
        logger.info(f"Processing variant: {variant_name}")
        variant_results = {}

        for scale_dir in variant_dir.iterdir():
            if not scale_dir.is_dir():
                continue

            scale_name = scale_dir.name
            gen_file = scale_dir / "generation_details.json"
            mc_file = scale_dir / "mc_details.json"

            if not gen_file.exists():
                logger.warning(f"  Skipping {scale_name}: no generation_details.json")
                continue

            logger.info(f"  Scoring {scale_name}")

            # Load generated responses
            with gen_file.open() as f:
                details = json.load(f)

            # Score with judges (in order: semantic, truth, informativeness, bleurt)
            annotated = details
            if semantic_judge:
                logger.info(f"    Applying semantic judge")
                annotated = semantic_judge.score_responses(annotated)
            if judge:
                logger.info(f"    Applying truthfulness judge")
                annotated = judge.score_responses(annotated)
            if informativeness_judge:
                logger.info(f"    Applying informativeness judge")
                annotated = informativeness_judge.score_responses(annotated)
            if bleurt_judge:
                logger.info(f"    Applying BLEURT judge")
                annotated = bleurt_judge.score_responses(annotated)

            # Recompute stats
            stats = _summarize_generation(
                annotated,
                judge is not None,
                informativeness_judge is not None,
                semantic_judge is not None,
                bleurt_judge is not None,
            )

            # Save updated results
            with gen_file.open("w") as f:
                json.dump(annotated, f, indent=2)

            # Load MC results if they exist
            mc_stats = None
            if mc_file.exists():
                with mc_file.open() as f:
                    mc_data = json.load(f)
                    if mc_data:  # Not empty
                        mc_stats = mc_data

            # Convert stats to dict for JSON serialization
            from dataclasses import asdict
            stats_dict = asdict(stats) if hasattr(stats, '__dataclass_fields__') else stats

            # Flatten stats into generation dict (matching main pipeline structure)
            gen_data = stats_dict.copy()
            gen_data["details"] = annotated

            variant_results[scale_name] = {
                "generation": gen_data,
            }
            if mc_stats:
                variant_results[scale_name]["mc"] = mc_stats

            # Log key metrics
            metrics_log = f"    Truth: {stats_dict.get('accuracy', 'N/A')}"
            if 'informativeness_rate' in stats_dict:
                metrics_log += f", Info: {stats_dict['informativeness_rate']}"
            if 'semantic_accuracy' in stats_dict:
                metrics_log += f", Sem: {stats_dict['semantic_accuracy']}"
            if 'semantic_diff_mean' in stats_dict:
                metrics_log += f", SemDiff: {stats_dict['semantic_diff_mean']:.3f}"
            logger.info(metrics_log)

        evaluation[variant_name] = variant_results

    # Save results.json
    results_file = run_dir / "results.json"
    logger.info(f"Saving results to {results_file}")
    with results_file.open("w") as f:
        json.dump(evaluation, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
