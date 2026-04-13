"""
Generate answers from retrieved context using local Hugging Face models only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import config

Role = Literal["customer", "sales"]


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
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    role_line = _role_instructions(role)
    return f"""You are a product documentation assistant.

{role_line}

Rules (must follow):
- Answer ONLY based on the provided context. Do not hallucinate.
- If the context does not contain enough information, say you do not have enough information in the documents and suggest what is missing.
- Cite ideas generally by referring to the document content (no fake page numbers).

Context:
{context}

Question:
{query}

Answer:"""


def _answer_hf_local(prompt: str) -> str:
    """Local seq2seq model for security-focused, no-external-API inference."""
    from transformers import pipeline

    # Flan-T5 and similar models have a small encoder window; keep inputs bounded.
    max_chars = 3500
    clipped = prompt if len(prompt) <= max_chars else prompt[:max_chars]

    if config.OFFLINE_ONLY:
        if not config.HF_CHAT_MODEL_LOCAL_PATH:
            raise RuntimeError(
                "OFFLINE_ONLY is enabled but HF_CHAT_MODEL_LOCAL_PATH is not set."
            )
        model_path = Path(config.HF_CHAT_MODEL_LOCAL_PATH).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"HF chat model path not found: {model_path}")
        model_ref = str(model_path)
        local_files_only = True
    else:
        model_ref = (
            config.HF_CHAT_MODEL_LOCAL_PATH
            if config.HF_CHAT_MODEL_LOCAL_PATH
            else config.HF_CHAT_MODEL
        )
        local_files_only = bool(config.HF_CHAT_MODEL_LOCAL_PATH)

    gen = pipeline(
        "text2text-generation",
        model=model_ref,
        max_new_tokens=256,
        local_files_only=local_files_only,
    )
    out = gen(clipped, do_sample=False)[0]["generated_text"]
    return str(out).strip()


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
