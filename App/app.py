"""Local service entrypoint (no API server required)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import config
from Embeddings.embedding_generator import embed_text
from LLM.llm_response import generate_answer
from Retrieval.retriever import chunks_to_context_dicts, retrieve_for_query
from Utils.backup_utils import create_vector_store_backup, restore_vector_store_backup
from Utils.logging_utils import get_logger
from VectorStore.faiss_store import get_store_stats
from pipeline import run_batch_indexing_pipeline, run_indexing_pipeline

logger = get_logger("app")


def _resolve_path(file_path: str | Path) -> Path:
    p = Path(file_path)
    if not p.is_absolute():
        p = (config.PROJECT_ROOT / p).resolve()
    docs_dir = config.DOCUMENTS_DIR.resolve()
    if not config.ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR and docs_dir not in p.parents:
        raise ValueError(
            f"File must be inside documents directory: {docs_dir}"
        )
    return p


def _resolve_folder_path(folder_path: str | Path) -> Path:
    p = Path(folder_path)
    if not p.is_absolute():
        p = (config.PROJECT_ROOT / p).resolve()
    docs_dir = config.DOCUMENTS_DIR.resolve()
    if not config.ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR and p != docs_dir:
        raise ValueError(f"Folder must be documents directory: {docs_dir}")
    return p


def diagnostics() -> dict:
    """Local diagnostics before document onboarding."""
    docs_dir = config.DOCUMENTS_DIR.resolve()
    pdfs = sorted(docs_dir.glob("*.pdf"))
    return {
        "documents_dir": str(docs_dir),
        "pdf_files_present": len(pdfs),
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
        checks["embedding_model_path_exists"] = Path(
            config.EMBEDDING_MODEL_LOCAL_PATH
        ).exists()
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


def query_documents(
    query: str,
    role: Literal["customer", "sales"] = "customer",
    include_sources: bool = False,
) -> dict:
    chunks = retrieve_for_query(query)
    texts = [c.text for c in chunks if c.text.strip()]
    answer, backend = generate_answer(query, texts, role=role)
    out: dict = {"answer": answer, "role": role, "llm_backend": backend}
    if include_sources:
        out["sources"] = chunks_to_context_dicts(chunks)
    logger.info("query role=%s sources=%s", role, len(out.get("sources", [])))
    return out


def backup_vector_store() -> str:
    archive = create_vector_store_backup()
    logger.info("backup created path=%s", archive)
    return archive


def restore_vector_store(zip_path: str | Path) -> None:
    restore_vector_store_backup(zip_path)
    logger.info("backup restored from=%s", zip_path)
