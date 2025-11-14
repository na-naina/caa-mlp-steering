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
from src.evaluation.semantic import SemanticJudge, SemanticJudgeConfig
from src.evaluation.truthfulqa import _summarize_generation

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
        help="Judge model name",
    )
    parser.add_argument(
        "--judge-device",
        default="auto",
        help="Judge device map",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge scoring",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic judge scoring",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    logger.info(f"Scoring responses in {run_dir}")

    # Load judge
    judge = None
    if not args.no_judge:
        logger.info(f"Loading judge model: {args.judge_model}")
        judge = LLMBinaryJudge(
            args.judge_model,
            dtype="bfloat16",
            device_map=args.judge_device,
            max_new_tokens=32,
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

            # Score with judges
            annotated = details
            if semantic_judge:
                annotated = semantic_judge.score_responses(annotated)
            if judge:
                annotated = judge.score_responses(annotated)

            # Recompute stats
            stats = _summarize_generation(
                annotated,
                judge is not None,
                semantic_judge is not None,
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

            variant_results[scale_name] = {
                "generation": {"stats": stats_dict, "details": annotated},
            }
            if mc_stats:
                variant_results[scale_name]["mc"] = mc_stats

            logger.info(f"    Accuracy: {stats_dict.get('accuracy', 'N/A')}")

        evaluation[variant_name] = variant_results

    # Save results.json
    results_file = run_dir / "results.json"
    logger.info(f"Saving results to {results_file}")
    with results_file.open("w") as f:
        json.dump(evaluation, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
