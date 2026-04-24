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

import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import config

Role = Literal["customer", "internal", "sales"]

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
- NEVER invent numbers, prices, dates, percentages, SLA figures or capacities
  that are not present in the context. If a specific figure is requested but
  the context does not contain it, say so plainly.
- NEVER substitute placeholder tokens for missing figures. Do NOT write
  "$X", "$Y", "Rs. X", "INR Y", "[TBD]", "XX.XX", "<price>", "(price)",
  "Contact sales for pricing", or similar stand-ins. Either give the real
  figure from the context or state that the documentation does not list it.
- Prices in this documentation are in Indian Rupees (INR, symbol ₹), NOT
  US dollars. Do not introduce a "$" symbol unless it literally appears in
  the context.
- When the context names specific plan/SKU identifiers (e.g. VDaaSGP.1X-Large),
  preserve those exact names -- do not rename them to friendlier labels.
- Keep answers concise: 2-5 sentences unless the user asks for detail."""


# ---------------------------------------------------------------------------
# Fabrication guard
# ---------------------------------------------------------------------------

# Placeholder patterns to detect when the model couldn't find a real figure
# in the context and fell back to inventing a template-looking answer.
#
# Why regex and not an LLM judge: this is a safety net, not a reviewer. Speed
# and determinism matter more than nuance; a fast cheap check that catches
# the 90% obvious cases is worth more than a slow accurate one.
#
# Each pattern is carefully narrow so it does NOT match legitimate output:
#   - r"\$\s*[A-Z]\b"      catches "$X", "$ Y" -- but not "$5" or "$12.50"
#   - r"₹\s*[A-Z]\b"       catches "₹X"        -- but not "₹5,000"
#   - r"(?i)Rs\.?\s*[A-Z]\b"          "Rs. X"  -- but not "Rs. 500"
#   - r"(?i)INR\s+[A-Z]\b"            "INR Y"  -- but not "INR 1,200"
#   - r"\bXX+\.\d+"        catches "XX.XX", "XXX.99"
#   - r"\[(TBD|XX+|price|amount|figure|value)\]"  "[TBD]", "[price]" etc.
#   - r"<(price|amount|figure|value)>"            "<price>", "<amount>"
_FABRICATION_PATTERNS = [
    re.compile(r"\$\s*[A-Z]\b"),
    re.compile(r"₹\s*[A-Z]\b"),
    re.compile(r"\bRs\.?\s*[A-Z]\b", re.IGNORECASE),
    re.compile(r"\bINR\s+[A-Z]\b", re.IGNORECASE),
    re.compile(r"\bXX+(?:\.\d+)?\b"),
    re.compile(r"\[(?:TBD|XX+|price|amount|figure|value)\]", re.IGNORECASE),
    re.compile(r"<(?:price|amount|figure|value)>", re.IGNORECASE),
]


def _looks_like_fabricated_answer(answer: str) -> bool:
    """Return True if the answer contains obvious placeholder hallucinations.

    Separated from ``generate_answer`` so tests can exercise the detector
    directly and so we can reuse it from the streaming path without
    duplicating the regex list.
    """
    if not answer:
        return False
    for pattern in _FABRICATION_PATTERNS:
        if pattern.search(answer):
            return True
    return False


def _fabrication_fallback() -> str:
    """Message returned when the detector trips.

    Deliberately tells the user the figure isn't in the documentation and
    points them at the support email rather than silently returning a blank
    bubble -- blank bubbles make users suspect the app is broken.
    """
    return (
        "I don't have that specific figure in the documents I can see. "
        f"For current numbers, please reach out to {config.SUPPORT_EMAIL} "
        "and our team will share the latest schedule."
    )


def _role_instructions(role: Role) -> str:
    """Return the audience-specific guidance appended to the system prompt.

    Kept as a function (not a dict) so the signatures stay stable even if we
    later want to branch on tone, reading level, etc.
    """
    if role == "customer":
        return (
            "Audience: end customer. Use clear, concise language. "
            "Focus on practical information and avoid internal jargon, "
            "pricing positioning, or competitive talking points."
        )
    if role == "internal":
        return (
            "Audience: internal employee (non-sales -- e.g. support, "
            "operations, engineering). You may include technical detail, "
            "implementation notes, and architectural context where the "
            "source material supports it. Prefer precision over marketing tone."
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


def _compose_user_message(
    query: str,
    context_blocks: list[str],
    role: Role,
) -> str:
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
def _build_prompt(
    query: str,
    context_blocks: list[str],
    role: Role,
) -> str:
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

    # Safety net: even with strong prompt rules, the model will occasionally
    # fabricate placeholder-looking figures when the context doesn't contain
    # a requested number. Replacing such answers with an honest fallback is
    # worth more for user trust than the rare false positive.
    if _looks_like_fabricated_answer(answer):
        return _fabrication_fallback(), "fabrication_guard"

    return answer, "huggingface_local"


def generate_answer_streaming(
    query: str,
    retrieved_texts: list[str],
    role: Role = "customer",
) -> Iterator[str]:
    """Yield answer tokens as they are generated.

    Functionally identical to ``generate_answer`` -- same prompt, same
    inference parameters, same model -- but flips ``stream=True`` so the
    UI can render words the moment they arrive. Total wall-clock time to
    a finished answer is unchanged; what changes is *perceived* latency:
    the user sees something happen at t=0 instead of staring at a spinner
    for 30+ seconds.

    Yields raw string deltas. Callers are expected to concatenate them
    (or pass this generator directly to ``st.write_stream`` which does
    the concatenation and returns the final string).
    """
    llm = _load_chat_model()
    user_message = _compose_user_message(query, retrieved_texts, role)

    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        stream=True,
    )
    # llama-cpp-python streams chunks with the same shape OpenAI uses:
    # {"choices": [{"delta": {"content": "..."}}]}. We only care about the
    # text delta; role/finish_reason/etc. don't affect the rendered answer.
    for chunk in stream:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {}).get("content", "")
        if delta:
            yield delta
