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

# Optional: load project .env so Streamlit/IDE runs see model paths
# without using run_local.bat. Keep .env out of version control.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# --- Assistant branding ---
# Name the UI uses when introducing itself ("Hello, my name's <X>..."). Change
# the default here or set ASSISTANT_NAME in the environment to rebrand without
# touching dashboard code. Keep it short -- it sits inside a chat bubble.
ASSISTANT_NAME: str = os.getenv("ASSISTANT_NAME", "Yntraa")

# Support contact shown in the out-of-scope fallback. Override via env var or
# edit the default here -- it's intentionally kept in one place so the real
# address isn't duplicated across the codebase.
# TODO: replace "support@example.com" with the real support inbox.
SUPPORT_EMAIL: str = os.getenv("SUPPORT_EMAIL", "support@example.com")

# Minimum cosine similarity a retrieved chunk must reach for us to trust the
# index contains something relevant to the user's question. Below this we
# skip the LLM entirely and return the support-contact fallback -- that's
# how we avoid confidently wrong answers to off-topic questions.
# MiniLM-L6 scores typically land 0.45-0.75 for on-topic, 0.15-0.30 for
# unrelated. 0.35 is a safe midpoint; raise if the bot answers too liberally,
# lower if it punts on legitimate questions.
MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))

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
# 1200 chars (~250-300 tokens) is the sweet spot for RAG with a 4K-token
# context window: chunks are big enough to contain a coherent paragraph or
# a full FAQ Q&A pair, but small enough that a top-5 retrieval still fits
# comfortably in the LLM prompt. The previous 500-char setting fragmented
# most answers across 3-4 chunks and hurt synthesis quality.
# 150-char overlap (~30 tokens) gives us a safety margin so sentences that
# straddle chunk boundaries are captured on both sides.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
# We now run a Qwen2.5 model with a 4K-token context window, so pulling
# 5 chunks still leaves plenty of room for the system prompt, question, and
# generation. Raise to 7-8 if the corpus grows and answers start missing
# cross-document detail; drop to 3 if you swap back to a tiny-context model.
TOP_K: int = int(os.getenv("TOP_K", "5"))
# How many candidates FAISS pulls before re-ranking. More = better recall (we're
# less likely to miss a relevant chunk) at the cost of a slightly slower
# re-ranker step. 25 gives the cross-encoder enough signal without being slow.
RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "25"))

# --- Cross-encoder re-ranker ---
# After FAISS fetches candidates, a small cross-encoder re-scores them by
# reading (question, chunk) pairs jointly -- something pure kNN can't do
# because it only compares two embeddings. This step is the single largest
# quality lever in most RAG stacks. Disable by setting to false in env.
USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_MODEL_LOCAL_PATH: str = os.getenv("RERANKER_MODEL_LOCAL_PATH", "")
# Blend the re-ranker score with the FAISS + document-weight score so PSS/FAQ
# priority still has a voice. 0.85 means 85% re-ranker, 15% weighted kNN --
# works well because cross-encoder scores are far more trustworthy than kNN.
RERANKER_BLEND: float = float(os.getenv("RERANKER_BLEND", "0.85"))

# --- Document weighting (applied after kNN scores; higher = more priority) ---
DOCUMENT_WEIGHTS: dict[str, float] = {
    "PSS": float(os.getenv("WEIGHT_PSS", "1.5")),
    "FAQ": float(os.getenv("WEIGHT_FAQ", "1.2")),
    "manual": float(os.getenv("WEIGHT_MANUAL", "1.0")),
    "guide": float(os.getenv("WEIGHT_GUIDE", "1.0")),
    "default": float(os.getenv("WEIGHT_DEFAULT", "1.0")),
}

# --- LLM (local-only for security) ---
# ``HF_CHAT_MODEL`` is kept for documentation and for tooling that resolves a
# download source; the runtime loader only ever reads from a local path. The
# default is the GGUF repo we fetch in scripts/download_models.py.
HF_CHAT_MODEL: str = os.getenv("HF_CHAT_MODEL", "bartowski/Qwen2.5-1.5B-Instruct-GGUF")
# Point at either a .gguf file directly or a folder containing one.
HF_CHAT_MODEL_LOCAL_PATH: str = os.getenv("HF_CHAT_MODEL_LOCAL_PATH", "")
# Enforce no external LLM APIs in this project.
LOCAL_LLM_ONLY: bool = os.getenv("LOCAL_LLM_ONLY", "true").lower() == "true"
# Strictly enforce offline local model loading from disk.
OFFLINE_ONLY: bool = os.getenv("OFFLINE_ONLY", "true").lower() == "true"

# Max tokens of (prompt + completion) the model can hold in its KV cache.
# Qwen2.5 natively supports 32K; we cap at 4K here to keep the KV-cache RAM
# footprint small on 8 GB machines. Raise to 8192 / 16384 if you have RAM to
# spare and want longer contexts (more retrieved chunks, longer answers).
LLM_CONTEXT_TOKENS: int = int(os.getenv("LLM_CONTEXT_TOKENS", "4096"))
# CPU threads for llama.cpp. Default of 0 tells llama.cpp to pick for itself;
# in practice matching *physical* core count (not logical) gives the best
# throughput because hyperthreads fight for the same AVX units.
LLM_CPU_THREADS: int = int(os.getenv("LLM_CPU_THREADS", "0"))
# Output budget. Larger values = more verbose answers + slower inference.
# 256 is a good default for grounded Q&A; raise only if answers look truncated.
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
# Legacy alias retained for any callers still importing this name.
LLM_MAX_INPUT_TOKENS: int = LLM_CONTEXT_TOKENS
# Cap total characters for retrieved excerpts in the LLM user message. The chat
# template, system prompt, and reply budget also consume n_ctx; overflowing
# prompts cause llama_decode -1 and empty failures. ~8.5k chars is a safe fit
# with LLM_CONTEXT_TOKENS=4096 and typical English tokenization.
LLM_MAX_RETRIEVED_CHARS: int = int(os.getenv("LLM_MAX_RETRIEVED_CHARS", "8500"))

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
# Default is permissive: any *.pdf is accepted. Document type is inferred from
# the folder layout under DOCUMENTS_DIR (see Retrieval.retriever.infer_document_type_from_name),
# not from a strict filename prefix.
#
# Teams that want a strict naming convention can override via env var, e.g.:
#   DOC_NAME_PATTERN='^(PSS|FAQ|manual|guide)_[A-Za-z0-9._-]+\\.pdf$'
#   ENFORCE_DOC_NAME_PATTERN=true
DOC_NAME_PATTERN: str = os.getenv("DOC_NAME_PATTERN", r"^.+\.pdf$")
ENFORCE_DOC_NAME_PATTERN: bool = os.getenv("ENFORCE_DOC_NAME_PATTERN", "false").lower() == "true"


def is_valid_doc_name(file_name: str) -> bool:
    """Validate the PDF filename (basename), including under nested folders.

    Match is case-insensitive on the file extension so ``Report.PDF`` is accepted
    alongside ``Report.pdf``.
    """
    basename = Path(file_name).name
    return bool(re.match(DOC_NAME_PATTERN, basename, flags=re.IGNORECASE))


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
