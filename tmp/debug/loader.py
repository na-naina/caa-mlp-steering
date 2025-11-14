from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    primary_device: torch.device


_DTYPE_ALIAS = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _parse_dtype(dtype: Optional[str]) -> torch.dtype:
    if dtype is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_ALIAS:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return _DTYPE_ALIAS[key]


def _resolve_primary_device(model: PreTrainedModel) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)

    if hasattr(model, "hf_device_map"):
        device_strings = set(model.hf_device_map.values())
        # Filter out pseudo entries like 'disk' or 'cpu'
        devices = [
            torch.device(dev) for dev in device_strings if dev not in {"disk"}
        ]
        if devices:
            return devices[0]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_causal_model(
    model_name: str,
    *,
    dtype: Optional[str] = None,
    device_map: Optional[str] = "auto",
    max_memory: Optional[dict] = None,
    revision: Optional[str] = None,
) -> LoadedModel:
    """Load a causal language model and tokenizer with sensible defaults."""
    torch_dtype = _parse_dtype(dtype)
    logger.info(
        "Loading model '%s' (dtype=%s, device_map=%s, max_memory=%s)",
        model_name,
        torch_dtype,
        device_map,
        max_memory,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        revision=revision,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    except Exception:
        from transformers import AutoProcessor

        tokenizer = AutoProcessor.from_pretrained(model_name, revision=revision)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    primary_device = _resolve_primary_device(model)
    logger.info(
        "Model loaded. Primary device: %s, dtype: %s",
        primary_device,
        torch_dtype,
    )
    return LoadedModel(model=model, tokenizer=tokenizer, primary_device=primary_device)

