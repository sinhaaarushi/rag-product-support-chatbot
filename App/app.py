"""Local service entrypoint (no API server required)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import config
from Embeddings.embedding_generator import embed_text
from LLM.llm_response import generate_answer, generate_answer_streaming
from pipeline import (
    list_pdf_files_under,
    run_batch_indexing_pipeline,
    run_indexing_pipeline,
)
from Retrieval.retriever import chunks_to_context_dicts, retrieve_for_query
from Utils.backup_utils import create_vector_store_backup, restore_vector_store_backup
from Utils.logging_utils import get_logger
from VectorStore.faiss_store import (
    clear_store,
    get_store_stats,
    unique_indexed_document_names,
)

logger = get_logger("app")


def _record_asked_question(query: str, role: str) -> None:
    """Best-effort append to the local question log; never raises."""
    q = (query or "").strip()
    if not q or len(q) > config.MAX_QUERY_CHARS:
        return
    try:
        from Utils.question_history import append_user_question

        append_user_question(q, role)
    except Exception as exc:
        logger.warning("user question log failed: %s", exc)


def _resolve_path(file_path: str | Path) -> Path:
    p = Path(file_path)
    if not p.is_absolute():
        p = (config.PROJECT_ROOT / p).resolve()
    docs_dir = config.DOCUMENTS_DIR.resolve()
    if not config.ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR and docs_dir not in p.parents:
        raise ValueError(f"File must be inside documents directory: {docs_dir}")
    return p


def _resolve_folder_path(folder_path: str | Path) -> Path:
    p = Path(folder_path)
    if not p.is_absolute():
        p = (config.PROJECT_ROOT / p).resolve()
    docs_dir = config.DOCUMENTS_DIR.resolve()
    if not config.ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR and p != docs_dir:
        raise ValueError(f"Folder must be documents directory: {docs_dir}")
    return p


def documents_sync_report() -> dict:
    """
    Compare PDFs on disk (recursive under documents dir) to document_name keys in the index.
    Use after add/remove/rename of files; run rebuild_index_from_documents() if drift appears.
    """
    docs_dir = config.DOCUMENTS_DIR.resolve()
    on_disk = {config.document_storage_key(p) for p in list_pdf_files_under(docs_dir)}
    in_index = set(unique_indexed_document_names())
    return {
        "documents_dir": str(docs_dir),
        "pdf_count_on_disk": len(on_disk),
        "indexed_document_count": len(in_index),
        "only_on_disk": sorted(on_disk - in_index),
        "only_in_index": sorted(in_index - on_disk),
        "in_sync": on_disk == in_index,
    }


def diagnostics() -> dict:
    """Local diagnostics before document onboarding."""
    docs_dir = config.DOCUMENTS_DIR.resolve()
    pdfs = list_pdf_files_under(docs_dir)
    return {
        "documents_dir": str(docs_dir),
        "pdf_files_present": len(pdfs),
        "documents_sync": documents_sync_report(),
        "faiss_store": get_store_stats(),
        "doc_name_pattern": config.DOC_NAME_PATTERN,
        "enforce_doc_name_pattern": config.ENFORCE_DOC_NAME_PATTERN,
    }


def preflight_check() -> dict:
    """
    Validate local models and local vector store prerequisites.
    """
    checks: dict = {
        "offline_only": config.OFFLINE_ONLY,
        "embedding_model_path_set": bool(config.EMBEDDING_MODEL_LOCAL_PATH),
        "chat_model_path_set": bool(config.HF_CHAT_MODEL_LOCAL_PATH),
        "embedding_model_path_exists": False,
        "chat_model_path_exists": False,
        "embedding_runtime_ok": False,
    }
    if config.EMBEDDING_MODEL_LOCAL_PATH:
        checks["embedding_model_path_exists"] = Path(config.EMBEDDING_MODEL_LOCAL_PATH).exists()
    if config.HF_CHAT_MODEL_LOCAL_PATH:
        checks["chat_model_path_exists"] = Path(config.HF_CHAT_MODEL_LOCAL_PATH).exists()
    try:
        _ = embed_text("preflight")
        checks["embedding_runtime_ok"] = True
    except Exception as exc:
        checks["embedding_error"] = str(exc)
    checks["ready"] = all(
        [
            checks["embedding_model_path_set"],
            checks["chat_model_path_set"],
            checks["embedding_model_path_exists"],
            checks["chat_model_path_exists"],
            checks["embedding_runtime_ok"],
        ]
    )
    return checks


def startup_validate_or_raise() -> None:
    checks = preflight_check()
    if not checks.get("ready", False):
        raise RuntimeError(f"Startup validation failed: {checks}")


def index_document(file_path: str | Path, document_type: str | None = None) -> dict:
    path = _resolve_path(file_path)
    result = run_indexing_pipeline(path, document_type=document_type)
    logger.info("index_document file=%s result=%s", path.name, result.get("chunks_indexed"))
    return result


def index_batch(folder_path: str | Path = config.DOCUMENTS_DIR) -> dict:
    folder = _resolve_folder_path(folder_path)
    result = run_batch_indexing_pipeline(folder)
    logger.info(
        "index_batch folder=%s docs=%s chunks=%s",
        folder,
        result.get("documents_indexed"),
        result.get("total_chunks_indexed"),
    )
    return result


def rebuild_index_from_documents() -> dict:
    """
    Clear the vector store and index every PDF under DOCUMENTS_DIR (recursive).

    Use this after adding, removing, or renaming PDFs so the index matches the folder.
    Appends-only batch indexing does not remove stale chunks for deleted files.
    """
    _resolve_folder_path(config.DOCUMENTS_DIR)
    clear_store()
    folder = config.DOCUMENTS_DIR.resolve()
    result = run_batch_indexing_pipeline(folder)
    logger.info(
        "rebuild_index_from_documents docs=%s chunks=%s",
        result.get("documents_indexed"),
        result.get("total_chunks_indexed"),
    )
    return result


def _out_of_scope_message() -> str:
    """Polite fallback shown when retrieval can't find anything relevant.

    Centralised here (not in the UI) so the server API and the dashboard
    return the same wording, and so swapping the copy or the support email
    is a one-line change.
    """
    return (
        "I'm sorry, I can't answer that from the documentation I have access to. "
        f"Please reach out to us at {config.SUPPORT_EMAIL} and our team will help."
    )


def _source_match_label(top_raw: float) -> str:
    """Defensible high/medium/low from retrieval score (not a made-up percentage)."""
    if top_raw >= 0.55:
        return "High"
    if top_raw >= 0.45:
        return "Medium"
    return "Low"


def _build_reasoning_steps_in_scope(n_chunks: int, top_raw: float) -> list[str]:
    """Honest, pipeline-accurate steps (no model chain-of-thought)."""
    steps: list[str] = [
        f"Retrieved {n_chunks} text chunk(s) from the local FAISS index to use as context.",
    ]
    if config.USE_RERANKER:
        steps.append(
            "Re-ranked the candidate set with a cross-encoder and combined that with the "
            "vector score (document-type weighting can still act as a tie-breaker)."
        )
    else:
        steps.append(
            "Ranked chunks by vector similarity to the question embedding, with retriever "
            "document-type weighting as configured."
        )
    steps.append(
        f"Top retrieved chunk had raw cosine similarity {top_raw:.2f} "
        f"(this product answers only when the best match is ≥ {config.MIN_RETRIEVAL_SCORE:.2f})."
    )
    steps.append(
        "Generated a reply with the local model under a strict 'use only this context' rule, "
        "plus the audience (role) you selected."
    )
    return steps


def _build_reasoning_steps_out_of_scope(n_candidates: int, top_raw: float) -> list[str]:
    return [
        f"Embedded the question and considered up to {n_candidates} candidate chunk(s) from the index.",
        f"The strongest match had raw similarity {top_raw:.2f}, below the minimum "
        f"{config.MIN_RETRIEVAL_SCORE:.2f} used to trust the documentation for an answer.",
        "Skipped the language model for this turn to avoid a confident-sounding but poorly grounded reply.",
    ]


def _heuristic_follow_ups(sources: list[dict]) -> list[str]:
    """Cheap follow-ups from other cited document names; no second LLM call."""
    if not sources:
        return []
    order = list(
        dict.fromkeys(
            str(Path(s.get("document_name", "")).name) for s in sources if s.get("document_name")
        )
    )
    if not order:
        return []
    if len(order) == 1:
        return [f"Summarize other details from {order[0]} that relate to this question."]
    return [
        f"How do {order[0]} and {order[1]} differ on this topic?",
        f"What else in {order[0]} should I know after this answer?",
    ][:2]


def query_documents(
    query: str,
    role: Literal["customer", "internal", "sales", "partners"] = "customer",
    include_sources: bool = False,
) -> dict:
    """Run retrieval + LLM answer for one query and return the response payload.

    Flow:
      1. Embed + kNN retrieve top chunks.
      2. If the best chunk's raw similarity score is below
         ``config.MIN_RETRIEVAL_SCORE``, short-circuit with the support-contact
         fallback -- we don't send the LLM off to hallucinate an answer from
         unrelated context.
      3. Otherwise build the prompt from non-empty chunks and ask the local
         seq2seq model to synthesise an answer.

    ``include_sources`` controls whether the retrieved chunks come back in the
    payload. Sources are useful for debugging and for UIs that want to show
    citations, but not every caller needs them -- keeping them opt-in means
    the hot path stays lean.

    The ``out_of_scope`` flag lets the UI suppress source chips and any
    "grounded in docs" affordances when the bot is punting.
    """
    _record_asked_question(query, role)
    retrieved_chunks = retrieve_for_query(query)

    # The raw (unboosted) score is the truer signal of semantic relevance --
    # the boosted score mixes in document-type weight, which can push an
    # unrelated PSS chunk above the threshold just because PSS has a 1.5x
    # weight. We want to decide "is this question in scope?" before weights.
    top_raw_score = max((c.score for c in retrieved_chunks), default=0.0)

    if not retrieved_chunks or top_raw_score < config.MIN_RETRIEVAL_SCORE:
        logger.info(
            "query role=%s out_of_scope=true top_score=%.3f threshold=%.3f",
            role,
            top_raw_score,
            config.MIN_RETRIEVAL_SCORE,
        )
        n_c = len(retrieved_chunks)
        response: dict = {
            "answer": _out_of_scope_message(),
            "role": role,
            "llm_backend": "fallback",
            "out_of_scope": True,
            "top_retrieval_score": top_raw_score,
            "source_match": "No match",
            "reasoning_steps": _build_reasoning_steps_out_of_scope(n_c, top_raw_score),
            "follow_ups": [],
        }
        if include_sources:
            response["sources"] = []
        return response

    llm_context_chunks = retrieved_chunks[: config.LLM_CONTEXT_TOP_K]
    context_texts = [c.text for c in llm_context_chunks if c.text.strip()]
    answer, backend = generate_answer(query, context_texts, role=role)

    src_dicts = chunks_to_context_dicts(retrieved_chunks)
    response = {
        "answer": answer,
        "role": role,
        "llm_backend": backend,
        "out_of_scope": False,
        "top_retrieval_score": top_raw_score,
        "source_match": _source_match_label(top_raw_score),
        "reasoning_steps": _build_reasoning_steps_in_scope(len(context_texts), top_raw_score),
        "follow_ups": _heuristic_follow_ups(src_dicts),
    }
    if include_sources:
        response["sources"] = src_dicts
    logger.info(
        "query role=%s sources=%s top_score=%.3f",
        role,
        len(response.get("sources", [])),
        top_raw_score,
    )
    return response


def query_documents_streaming(
    query: str,
    role: Literal["customer", "internal", "sales", "partners"] = "customer",
) -> tuple[dict, Iterator[str]]:
    """Streaming variant of ``query_documents``.

    Returns a ``(metadata, token_iterator)`` tuple. Why split the return
    value:

    - Retrieval outcomes (sources, out-of-scope decision, top score) are
      known *before* the LLM runs. Putting them in ``metadata`` lets the
      UI render source chips as soon as the stream ends, without a second
      call back through the pipeline.
    - ``token_iterator`` yields string deltas from the LLM. For the
      out-of-scope case we still return an iterator (yielding the single
      fallback message) so the caller has a uniform contract: "always
      iterate, concatenate the result, render the sources". That keeps
      the dashboard code free of special-casing for the fast path.
    """
    _record_asked_question(query, role)
    retrieved_chunks = retrieve_for_query(query)
    top_raw_score = max((c.score for c in retrieved_chunks), default=0.0)

    if not retrieved_chunks or top_raw_score < config.MIN_RETRIEVAL_SCORE:
        logger.info(
            "query role=%s out_of_scope=true top_score=%.3f threshold=%.3f streaming=true",
            role,
            top_raw_score,
            config.MIN_RETRIEVAL_SCORE,
        )
        fallback = _out_of_scope_message()
        n_c = len(retrieved_chunks)
        metadata: dict = {
            "role": role,
            "out_of_scope": True,
            "top_retrieval_score": top_raw_score,
            "sources": [],
            "llm_backend": "fallback",
            "source_match": "No match",
            "reasoning_steps": _build_reasoning_steps_out_of_scope(n_c, top_raw_score),
            "follow_ups": [],
        }

        def _fallback_stream() -> Iterator[str]:
            yield fallback

        return metadata, _fallback_stream()

    llm_context_chunks = retrieved_chunks[: config.LLM_CONTEXT_TOP_K]
    context_texts = [c.text for c in llm_context_chunks if c.text.strip()]
    sources = chunks_to_context_dicts(retrieved_chunks)
    metadata = {
        "role": role,
        "out_of_scope": False,
        "top_retrieval_score": top_raw_score,
        "sources": sources,
        "llm_backend": "huggingface_local",
        "source_match": _source_match_label(top_raw_score),
        "reasoning_steps": _build_reasoning_steps_in_scope(len(context_texts), top_raw_score),
        "follow_ups": _heuristic_follow_ups(sources),
    }

    logger.info(
        "query role=%s sources=%s top_score=%.3f streaming=true",
        role,
        len(sources),
        top_raw_score,
    )
    return metadata, generate_answer_streaming(query, context_texts, role=role)


def backup_vector_store() -> str:
    archive = create_vector_store_backup()
    logger.info("backup created path=%s", archive)
    return archive


def restore_vector_store(zip_path: str | Path) -> None:
    restore_vector_store_backup(zip_path)
    logger.info("backup restored from=%s", zip_path) # here i included zip path to see the path of the file and see if it is working or not
