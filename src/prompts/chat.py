"""Optional chat-template wrapping of every TruthfulQA prompt (training, CAA extraction, generation).

Enabled only when the environment variable TQA_CHAT_TEMPLATE is set to a HF model id whose
tokenizer provides the chat template (e.g. google/gemma-4-E4B-it). Unset (the default, used for
LLaMA-2 / Gemma-3 / Qwen3 runs) every function returns the legacy raw-text prompt unchanged.

Why: instruction-tuned 2026 models (Gemma-4-E4B) degenerate under the raw six-shot QA prompt
("I have no comment." on ~44% of questions unsteered). In chat mode the protocol keeps the
LLaMA-2 structure: training / CAA use the zero-shot "Question: ..." prompt and generation the
six-shot QA primer, each placed in the user turn; answers are the assistant turn. The template's
textual BOS is stripped because callers add BOS themselves.
"""
from __future__ import annotations

import os
from functools import lru_cache


def chat_model() -> str | None:
    return os.environ.get("TQA_CHAT_TEMPLATE") or None


@lru_cache(maxsize=2)
def _tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _render(messages, add_generation_prompt: bool) -> str:
    tok = _tokenizer(chat_model())
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)
    bos = tok.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def qa_prompt(question: str) -> str:
    """Training / LoRA-DPO prompt that the answer is appended to."""
    if not chat_model():
        return f"Question: {question}\nAnswer:"
    return _render([{"role": "user", "content": f"Question: {question}"}], add_generation_prompt=True)


def qa_text(question: str, answer: str) -> str:
    """Full question+answer text for CAA extraction."""
    if not chat_model():
        return f"Question: {question}\nAnswer: {answer}"
    return qa_prompt(question) + answer


def generation_prompt(user_text: str) -> str:
    """Wrap an already formatted (e.g. six-shot QA) generation prompt as the user turn."""
    if not chat_model():
        return user_text
    if user_text.rstrip().endswith("\nA:"):  # the QA preset ends with "Q: ...\nA:" -> question last
        user_text = user_text.rstrip()[:-len("\nA:")]
    return _render([{"role": "user", "content": user_text}], add_generation_prompt=True)
