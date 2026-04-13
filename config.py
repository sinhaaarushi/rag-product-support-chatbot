"""
Central configuration for the RAG pipeline (FAISS, embeddings, chunking, retrieval).
Override values via environment variables where noted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# --- FAISS vector store ---
VECTOR_STORE_DIR: Path = PROJECT_ROOT / "Data" / "vector_store"
FAISS_INDEX_FILE: Path = VECTOR_STORE_DIR / "index.faiss"
FAISS_METADATA_FILE: Path = VECTOR_STORE_DIR / "metadata.json"
BACKUP_DIR: Path = PROJECT_ROOT / "Data" / "backups"
LOGS_DIR: Path = PROJECT_ROOT / "Data" / "logs"
APP_LOG_FILE: Path = LOGS_DIR / "rag_app.log"

# --- Embeddings (sentence-transformers) ---
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# Optional local filesystem path for embedding model.
EMBEDDING_MODEL_LOCAL_PATH: str = os.getenv("EMBEDDING_MODEL_LOCAL_PATH", "")
# Dimension for all-MiniLM-L6-v2
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# --- Chunking (reused by Processing/chunking.py) ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", "5"))
# Retrieve extra candidates so weighted re-ranking can surface PSS-heavy hits
RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))

# --- Document weighting (applied after kNN scores; higher = more priority) ---
DOCUMENT_WEIGHTS: dict[str, float] = {
    "PSS": float(os.getenv("WEIGHT_PSS", "1.5")),
    "FAQ": float(os.getenv("WEIGHT_FAQ", "1.2")),
    "manual": float(os.getenv("WEIGHT_MANUAL", "1.0")),
    "guide": float(os.getenv("WEIGHT_GUIDE", "1.0")),
    "default": float(os.getenv("WEIGHT_DEFAULT", "1.0")),
}

# --- LLM (local-only for security) ---
HF_CHAT_MODEL: str = os.getenv("HF_CHAT_MODEL", "google/flan-t5-base")
HF_CHAT_MODEL_LOCAL_PATH: str = os.getenv("HF_CHAT_MODEL_LOCAL_PATH", "")
# Enforce no external LLM APIs in this project.
LOCAL_LLM_ONLY: bool = os.getenv("LOCAL_LLM_ONLY", "true").lower() == "true"
# Strictly enforce offline local model loading from disk.
OFFLINE_ONLY: bool = os.getenv("OFFLINE_ONLY", "true").lower() == "true"

# Default folder for uploaded PDFs (relative to project root)
DOCUMENTS_DIR: Path = PROJECT_ROOT / "Data" / "documents"

# --- API safety controls ---
# Allow indexing only from DOCUMENTS_DIR by default.
ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR: bool = (
    os.getenv("ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR", "false").lower() == "true"
)
# Cap user query size to reduce abuse and runaway prompt payloads.
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))

# Optional API key for server endpoints (except /health).
# If empty, auth is disabled for local development.
API_KEY: str = os.getenv("RAG_API_KEY", "")

# --- Document naming rules ---
# Example accepted names: PSS_motor_v1.pdf, FAQ_returns.pdf, manual_installation.pdf
DOC_NAME_PATTERN: str = os.getenv(
    "DOC_NAME_PATTERN",
    r"^(PSS|FAQ|manual|guide)_[A-Za-z0-9._-]+\.pdf$",
)
ENFORCE_DOC_NAME_PATTERN: bool = (
    os.getenv("ENFORCE_DOC_NAME_PATTERN", "false").lower() == "true"
)


def is_valid_doc_name(file_name: str) -> bool:
    """Validate the PDF filename (basename), including under nested folders."""
    basename = Path(file_name).name
    return bool(re.match(DOC_NAME_PATTERN, basename))


def document_storage_key(path: Path) -> str:
    """
    Stable document id for the index: posix path relative to DOCUMENTS_DIR when
    possible, else the file basename. Keeps nested layouts unambiguous.
    """
    resolved = path.resolve()
    root = DOCUMENTS_DIR.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name
