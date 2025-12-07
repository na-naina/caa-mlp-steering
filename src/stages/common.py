"""Common utilities for pipeline stages."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import yaml


@dataclass
class RunContext:
    """Shared context for a pipeline run."""
    model_name: str
    run_id: str
    run_dir: Path
    config: Dict[str, Any]

    @property
    def vectors_dir(self) -> Path:
        return self.run_dir / "vectors"

    @property
    def responses_dir(self) -> Path:
        return self.run_dir / "responses"

    @property
    def scores_dir(self) -> Path:
        return self.run_dir / "scores"

    @property
    def metadata_dir(self) -> Path:
        return self.run_dir / "metadata"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"


@dataclass
class CheckpointManager:
    """Track partial progress within a stage for resumable runs.

    Usage:
        ckpt = CheckpointManager(ctx, "generate")

        for variant in variants:
            if ckpt.is_complete(variant):
                LOG.info("Skipping %s (already complete)", variant)
                continue

            # Do work...
            ckpt.mark_complete(variant, {"accuracy": 0.85})

        ckpt.finalize()  # Mark stage complete
    """
    ctx: RunContext
    stage: str
    completed: Set[str] = field(default_factory=set)
    results: Dict[str, Any] = field(default_factory=dict)
    _loaded: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._load()

    @property
    def checkpoint_file(self) -> Path:
        return self.ctx.checkpoints_dir / f"{self.stage}_progress.json"

    def _load(self) -> None:
        """Load existing checkpoint if available."""
        if self._loaded:
            return

        self.ctx.checkpoints_dir.mkdir(exist_ok=True)

        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                self.completed = set(data.get("completed", []))
                self.results = data.get("results", {})
            except (json.JSONDecodeError, IOError):
                pass  # Start fresh if corrupt

        self._loaded = True

    def _save(self) -> None:
        """Persist checkpoint to disk."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "stage": self.stage,
                "completed": list(self.completed),
                "results": self.results,
                "updated": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    def is_complete(self, item: str) -> bool:
        """Check if an item has been completed."""
        return item in self.completed

    def mark_complete(self, item: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark an item as complete and save immediately."""
        self.completed.add(item)
        if result:
            self.results[item] = result
        self._save()

    def get_result(self, item: str) -> Optional[Dict[str, Any]]:
        """Get cached result for a completed item."""
        return self.results.get(item)

    def get_pending(self, items: List[str]) -> List[str]:
        """Get items that haven't been completed yet."""
        return [item for item in items if item not in self.completed]

    def finalize(self) -> Dict[str, Any]:
        """Return aggregated results (call after all items complete)."""
        return {
            "completed": list(self.completed),
            "results": self.results,
        }

    def clear(self) -> None:
        """Clear checkpoint to force full re-run."""
        self.completed.clear()
        self.results.clear()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure logging for a stage."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
        handlers=handlers,
        force=True,
    )


def setup_environment(config: Dict) -> None:
    """Configure HF cache paths and authentication."""
    paths = config.get("paths", {})

    hf_cache = paths.get("hf_cache")
    if hf_cache:
        Path(hf_cache).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)

    # Load HF token if available
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)


def load_config(model_name: str, base_path: Path = Path("configs")) -> Dict[str, Any]:
    """Load merged config from base + model-specific files."""
    base_file = base_path / "base.yaml"
    model_file = base_path / "models" / f"{model_name}.yaml"

    if not base_file.exists():
        raise FileNotFoundError(f"Base config not found: {base_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model config not found: {model_file}")

    with open(base_file) as f:
        config = yaml.safe_load(f)

    with open(model_file) as f:
        model_config = yaml.safe_load(f)

    # Deep merge model config into base
    _deep_merge(config, model_config)
    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_or_create_run(
    model_name: str,
    config: Dict,
    run_id: Optional[str] = None,
) -> RunContext:
    """Get existing run or create new one.

    Output structure:
        outputs/{family}/{model_name}/{timestamp}/
        e.g., outputs/gemma3/gemma3_12b_it/20251201_120000/

    For resuming, supports both new and legacy formats:
        - New: outputs/{family}/{model_name}/{run_id}
        - Legacy: outputs/{family}/{run_id}
    """
    output_root = Path(config.get("paths", {}).get("output_root", "outputs"))

    # Organize by model family and model name
    model_family = config.get("model", {}).get("family", "unknown")
    model_dir = output_root / model_family / model_name

    if run_id:
        # Resume existing run - check both new and legacy locations
        run_dir = model_dir / run_id
        if not run_dir.exists():
            # Try legacy location: outputs/{family}/{run_id}
            legacy_dir = output_root / model_family / run_id
            if legacy_dir.exists():
                run_dir = legacy_dir
            else:
                # Also try if run_id contains model_name prefix (legacy format)
                legacy_dir2 = output_root / model_family / f"{model_name}_{run_id}"
                if legacy_dir2.exists():
                    run_dir = legacy_dir2
                else:
                    raise FileNotFoundError(
                        f"Run directory not found. Tried:\n"
                        f"  - {model_dir / run_id}\n"
                        f"  - {output_root / model_family / run_id}"
                    )
    else:
        # Create new run with timestamp-only ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = timestamp
        run_dir = model_dir / run_id

    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "vectors").mkdir(exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    (run_dir / "scores").mkdir(exist_ok=True)
    (run_dir / "metadata").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    return RunContext(
        model_name=model_name,
        run_id=run_id,
        run_dir=run_dir,
        config=config,
    )


def find_latest_run(model_name: str, config: Dict) -> Optional[str]:
    """Find the most recent run for a model.

    Returns the run_id of the latest run, or None if no runs exist.
    """
    output_root = Path(config.get("paths", {}).get("output_root", "outputs"))
    model_family = config.get("model", {}).get("family", "unknown")
    model_dir = output_root / model_family / model_name

    if not model_dir.exists():
        # Check legacy location
        legacy_dir = output_root / model_family
        if legacy_dir.exists():
            # Find runs matching model_name_* pattern
            runs = [d for d in legacy_dir.iterdir()
                    if d.is_dir() and d.name.startswith(f"{model_name}_")]
            if runs:
                latest = max(runs, key=lambda d: d.stat().st_mtime)
                return latest.name
        return None

    # Find latest run in new structure
    runs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not runs:
        return None

    latest = max(runs, key=lambda d: d.stat().st_mtime)
    return latest.name


def save_stage_metadata(ctx: RunContext, stage: str, metadata: Dict) -> None:
    """Save metadata for a completed stage."""
    metadata["stage"] = stage
    metadata["timestamp"] = datetime.now().isoformat()

    meta_file = ctx.metadata_dir / f"{stage}.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_stage_metadata(ctx: RunContext, stage: str) -> Optional[Dict]:
    """Load metadata from a previous stage."""
    meta_file = ctx.metadata_dir / f"{stage}.json"
    if not meta_file.exists():
        return None
    with open(meta_file) as f:
        return json.load(f)


def check_stage_complete(ctx: RunContext, stage: str) -> bool:
    """Check if a stage has been completed."""
    return load_stage_metadata(ctx, stage) is not None


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
