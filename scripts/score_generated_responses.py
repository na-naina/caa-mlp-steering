#!/usr/bin/env python3
"""
Score pre-generated responses with judge models.
Use this after generation completes but judge loading fails due to OOM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Disable TensorFlow imports; we only use PyTorch backends here.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("SENTENCE_TRANSFORMERS_NO_TF_IMPORT", "1")

import yaml

# Ensure repository root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config to pull judge/semantic settings and cache paths",
    )
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
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Semantic judge model (only used if semantic judge is enabled)",
    )
    parser.add_argument(
        "--no-bleurt",
        action="store_true",
        help="Skip BLEURT judge scoring",
    )
    args = parser.parse_args()

    # Load optional config to set cache paths and judge defaults
    cfg = {}
    if args.config:
        if not args.config.exists():
            raise ValueError(f"Config file does not exist: {args.config}")
        with args.config.open() as f:
            cfg = yaml.safe_load(f) or {}
        paths_cfg = cfg.get("paths", {})
        hf_cache = paths_cfg.get("hf_cache")
        cache_root = paths_cfg.get("cache_root")
        if hf_cache:
            Path(hf_cache).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(hf_cache))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_cache) / "transformers"))
            os.environ.setdefault("HF_DATASETS_CACHE", str(Path(hf_cache) / "datasets"))
        if cache_root:
            Path(cache_root).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache_root) / "datasets"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_root) / "transformers"))

        eval_cfg = cfg.get("evaluation", {})
        judge_cfg = eval_cfg.get("judge", {})
        info_cfg = eval_cfg.get("informativeness", {})
        semantic_cfg = eval_cfg.get("semantic", {})

        # Override defaults only if user didn't set different values explicitly
        if args.judge_model == parser.get_default("judge_model") and judge_cfg.get("model"):
            args.judge_model = judge_cfg["model"]
        if args.judge_device == parser.get_default("judge_device") and judge_cfg.get("device_map"):
            args.judge_device = judge_cfg["device_map"]
        if args.informativeness_model is None and info_cfg.get("model"):
            args.informativeness_model = info_cfg["model"]
        if args.semantic_model == parser.get_default("semantic_model") and semantic_cfg.get("model"):
            args.semantic_model = semantic_cfg["model"]

        # Respect semantic enabled flag from config if user didn't override with --no-semantic
        if not args.no_semantic:
            if semantic_cfg.get("enabled") is False:
                args.no_semantic = True

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
            model_name=args.semantic_model,
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
