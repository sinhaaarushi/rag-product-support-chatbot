"""Generate answers from retrieved context using a local GGUF-quantized LLM.

Why llama-cpp and not transformers: this project runs on a laptop-class CPU
with limited RAM. Full-precision weights from ``transformers`` are 3-4x
slower and 3-4x bigger than a Q4_K_M GGUF file with essentially the same
answer quality. llama-cpp-python also gives us a real chat-completion API
(role-tagged messages) instead of us hand-rolling a seq2seq prompt template.

Public API: ``generate_answer(query, retrieved_texts, role)`` returns
``(answer, backend)``. Nothing upstream of this module had to change --
``App/app.py`` and the dashboard still work unmodified.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

import config

Role = Literal["customer", "sales"]

# Module-level cache: loading a 1 GB GGUF and spinning up the ggml context
# takes ~5-10 s on CPU. Doing that once per process (not once per query) is
# what makes the dashboard feel interactive after the first question.
_llm: Any | None = None
_llm_lock = threading.Lock()

# System prompt -- the rules the model follows on every turn. Kept separate
# from the per-query context so it lives in one place and the chat template
# can apply it as a proper ``system`` role message (modern instruct models
# attend to system prompts differently from user turns).
_SYSTEM_PROMPT = """You are a product documentation assistant.

Rules (must follow):
- Use ONLY the facts contained in the context the user provides. Never invent
  information that is not in the context.
- Rephrase the answer in your own words. Do NOT copy sentences, bullet lists,
  or headings verbatim from the context -- synthesise a direct, natural reply.
- Interpret the user's intent charitably: answer the question they clearly
  meant, even if the wording is informal or incomplete.
- If the context does not contain enough information to answer the question,
  reply with one short sentence saying so. Do not pad with unrelated facts.
- Keep answers concise: 2-5 sentences unless the user asks for detail."""


def _role_instructions(role: Role) -> str:
    """Return the audience-specific guidance appended to the system prompt.

    Kept as a function (not a dict) so the signatures stay stable even if we
    later want to branch on tone, reading level, etc.
    """
    if role == "customer":
        return (
            "Audience: end customer. Use clear, concise language. "
            "Focus on practical information and avoid internal sales jargon."
        )
    if role == "sales":
        return (
            "Audience: internal sales team. You may highlight positioning, "
            "differentiation, and customer-facing value, but still ground "
            "every claim in the provided context."
        )
    return "Audience: general."


def _resolve_chat_model_path() -> Path:
    """Locate the GGUF file on disk, honouring OFFLINE_ONLY.

    Accepts either a folder (we'll pick the first .gguf inside) or a direct
    .gguf path in ``HF_CHAT_MODEL_LOCAL_PATH``. Folder mode is convenient
    when ``scripts/download_models.py`` placed the file into a model-named
    directory -- the caller doesn't have to know the exact filename.
    """
    raw_path = config.HF_CHAT_MODEL_LOCAL_PATH
    if not raw_path:
        raise RuntimeError(
            "HF_CHAT_MODEL_LOCAL_PATH is not set. Point it at a .gguf file or a "
            "folder containing one (see scripts/download_models.py)."
        )
    path = Path(raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"HF chat model path not found: {path}")

    if path.is_dir():
        # Pick the first GGUF we find. If there are multiple quantization levels
        # in the same folder, callers should point at a specific file instead.
        gguf_files = sorted(path.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf file under: {path}")
        return gguf_files[0]
    return path


def _load_chat_model() -> Any:
    """Load and cache the llama.cpp ``Llama`` instance.

    Lazy-imported so ``pytest`` and other entry points that never call into
    the LLM don't pay the C-extension import cost, and so a broken install
    fails loudly *at query time* with an actionable error rather than on
    module import where the traceback is harder to place.
    """
    global _llm
    with _llm_lock:
        if _llm is not None:
            return _llm

        from llama_cpp import Llama

        model_path = _resolve_chat_model_path()
        # n_ctx: max tokens in the (prompt + generation) window. 4096 is plenty
        # for RAG with top-3 chunks and leaves room for future tweaks without
        # overflowing. Qwen2.5 natively supports 32K, but a smaller n_ctx keeps
        # the KV cache (and therefore RAM) small on this machine.
        # n_threads: match physical core count. Hyperthreads (logical cores)
        # usually *hurt* llama.cpp throughput because they contend for the
        # same AVX execution units.
        _llm = Llama(
            model_path=str(model_path),
            n_ctx=config.LLM_CONTEXT_TOKENS,
            n_threads=config.LLM_CPU_THREADS,
            n_gpu_layers=0,
            verbose=False,
        )
        return _llm


def _compose_user_message(query: str, context_blocks: list[str], role: Role) -> str:
    """Build the user-role message that combines question + retrieved context.

    We do NOT paste the context into the system prompt -- instruct models are
    trained to treat system prompts as global rules and user prompts as the
    actual task. Putting the context in the user turn keeps that contract
    intact and makes the model more likely to actually use the context.
    """
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context provided)"
    role_line = _role_instructions(role)
    return f"""{role_line}

Question:
{query}

Context (use this to answer the question above):
{context}"""


# Kept under their old names so existing tests keep passing. These are pure
# string-assembly helpers -- no model required.
def _build_prompt(query: str, context_blocks: list[str], role: Role) -> str:
    """Legacy prompt assembler kept for tests and for API parity.

    The GGUF chat path uses ``_compose_user_message`` + the model's chat
    template, but pure string tests still assert on the old structure
    (question before context, grounding rules present, etc.). Returning a
    flattened version here means those assertions stay meaningful without
    duplicating the rule copy.
    """
    user_message = _compose_user_message(query, context_blocks, role)
    return f"{_SYSTEM_PROMPT}\n\n{user_message}\n\nAnswer:"


def generate_answer(
    query: str,
    retrieved_texts: list[str],
    role: Role = "customer",
) -> tuple[str, str]:
    """Run a single chat completion and return ``(answer, backend_label)``.

    ``backend_label`` stays ``"huggingface_local"`` for API compatibility with
    callers that already log / branch on it. The actual backend is now
    llama.cpp -- but that's an implementation detail, not a contract.
    """
    llm = _load_chat_model()
    user_message = _compose_user_message(query, retrieved_texts, role)

    # ``create_chat_completion`` applies the model's chat template (special
    # tokens for system / user / assistant turns) internally. Qwen2.5 uses
    # ChatML, and getting those tokens wrong by hand would silently degrade
    # quality -- letting llama.cpp do it is both safer and less code.
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=0.2,
        top_p=0.9,
        # Repetition penalty keeps the model from looping on long contexts;
        # 1.1 is the llama.cpp convention for "gentle nudge, no damage".
        repeat_penalty=1.1,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer, "huggingface_local"
