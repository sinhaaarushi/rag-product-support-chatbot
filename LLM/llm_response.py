"""
Generate answers from retrieved context using local Hugging Face models only.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

import config

Role = Literal["customer", "sales"]

# Module-level cache so the ~1 GB flan-t5 weights are loaded once per process
# rather than on every query. Thread-safe because the Streamlit dashboard and
# any future server will serve requests concurrently.
_llm_tokenizer: Any | None = None
_llm_model: Any | None = None
_llm_lock = threading.Lock()


def _role_instructions(role: Role) -> str:
    if role == "customer":
        return (
            "Audience: end customer. Use clear, concise language. "
            "Focus on practical steps and safety. Avoid internal sales jargon."
        )
    if role == "sales":
        return (
            "Audience: sales team. You may highlight positioning, differentiation, "
            "and customer-facing value, but still ground every claim in the context below."
        )
    return "Audience: general."


def _build_prompt(query: str, context_blocks: list[str], role: Role) -> str:
    """Compose the final prompt sent to the seq2seq model.

    Ordering matters: we put the QUESTION first and the CONTEXT last. Two reasons:
      1. Flan-T5's encoder window is 512 tokens. If we overflow, tokenizer
         truncation chops the *end* of the prompt. Putting context last means
         we lose context chunks on overflow -- not the actual question, not the
         "Answer:" trigger. That single change fixes the failure mode where the
         model echoes a random heading from the PDFs instead of answering.
      2. "Lost in the middle" -- transformer attention concentrates on the
         start and end of the input. Question at the top + answer cue at the
         bottom keeps the model anchored on the right task.
    """
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    role_line = _role_instructions(role)
    return f"""You are a product documentation assistant.

{role_line}

Rules (must follow):
- Answer ONLY based on the provided context below. Do not invent facts.
- If the context does not contain enough information, say so plainly and
  suggest what part of the documentation is missing.
- Write a direct, well-formed answer; do not echo headings from the context.

Question:
{query}

Context (use this to answer the question above):
{context}

Answer:"""


def _resolve_chat_model_ref() -> tuple[str, bool]:
    """Resolve (model_ref, local_files_only) honoring OFFLINE_ONLY."""
    if config.OFFLINE_ONLY:
        if not config.HF_CHAT_MODEL_LOCAL_PATH:
            raise RuntimeError("OFFLINE_ONLY is enabled but HF_CHAT_MODEL_LOCAL_PATH is not set.")
        model_path = Path(config.HF_CHAT_MODEL_LOCAL_PATH).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"HF chat model path not found: {model_path}")
        return str(model_path), True
    model_ref = config.HF_CHAT_MODEL_LOCAL_PATH or config.HF_CHAT_MODEL
    return model_ref, bool(config.HF_CHAT_MODEL_LOCAL_PATH)


def _load_chat_model() -> tuple[Any, Any]:
    """Load (and cache) the tokenizer and seq2seq model.

    Uses ``AutoModelForSeq2SeqLM`` + ``.generate()`` directly rather than
    ``transformers.pipeline("text2text-generation", ...)`` because that pipeline
    task was removed in transformers 5.x. The explicit API is also stable across
    4.x and 5.x and lets us cache weights between queries instead of reloading
    the ~1 GB checkpoint on every call.
    """
    global _llm_tokenizer, _llm_model
    with _llm_lock:
        if _llm_tokenizer is not None and _llm_model is not None:
            return _llm_tokenizer, _llm_model

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_ref, local_files_only = _resolve_chat_model_ref()
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_files_only)
        _llm_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_ref, local_files_only=local_files_only
        )
        _llm_model.eval()
        return _llm_tokenizer, _llm_model


def _answer_hf_local(prompt: str) -> str:
    """Local seq2seq model for security-focused, no-external-API inference."""
    tokenizer, model = _load_chat_model()

    # Truncate at the tokenizer boundary (not character count) so the model
    # sees a valid window; character-based clipping can land mid-token and
    # confuse the encoder. Budgets are centralised in config.py so swapping
    # in a different model doesn't require touching this file.
    encoded_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.LLM_MAX_INPUT_TOKENS,
    )
    output_ids = model.generate(
        **encoded_prompt,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        do_sample=False,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate_answer(
    query: str,
    retrieved_texts: list[str],
    role: Role = "customer",
) -> tuple[str, str]:
    """
    Returns (answer, backend) where backend is always 'huggingface_local'.
    """
    prompt = _build_prompt(query, retrieved_texts, role)
    return _answer_hf_local(prompt), "huggingface_local"
