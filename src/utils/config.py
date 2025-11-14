import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update mapping ``base`` with values from ``overlay``."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file into a dictionary."""
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    base_path: Path,
    overrides: Optional[Iterable[Path]] = None,
    cli_overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Load configuration from YAML files and simple CLI overrides.

    ``cli_overrides`` accepts items in the form ``section.key=value`` and
    updates the configuration at the dotted path. Nested dictionaries can be
    specified with dot notation (e.g. ``model.layer=12``).
    """
    config = load_yaml(base_path)
    if overrides:
        for override in overrides:
            config = _deep_update(config, load_yaml(override))

    if cli_overrides:
        for entry in cli_overrides:
            if "=" not in entry:
                raise ValueError(f"Invalid override '{entry}' (expected key=value)")
            key_path, value = entry.split("=", 1)
            _apply_cli_override(config, key_path.strip(), value.strip())

    if not config.get("run"):
        config["run"] = {}
    if not config["run"].get("id"):
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        config["run"]["id"] = f"run_{timestamp}"
    return config


def _apply_cli_override(config: Dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply dotted-path override to configuration dictionary."""
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]

    # Try to parse JSON for richer types; fallback to string
    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed_value = raw_value
    target[keys[-1]] = parsed_value


def dump_config(config: Dict[str, Any], path: Path) -> None:
    """Persist configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

