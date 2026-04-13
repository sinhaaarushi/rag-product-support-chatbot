# RAG System Overview

## Goal

Build an enterprise-style Retrieval-Augmented Generation system for product PDFs (PSS/manual/guide) that gives grounded answers.

## Current implemented modules

- `Ingestion/document_loader.py`
  - Uses `pdfplumber` to read PDF pages.
  - Cleans whitespace and returns one normalized text block.

- `Processing/chunking.py`
  - Splits text into overlapping chunks.
  - Validates chunk configuration.

- `Embeddings/embedding_generator.py`
  - Uses `sentence-transformers/all-MiniLM-L6-v2`.
  - Lazy-loads model once (thread-safe).
  - Supports single and batch embedding.

- `VectorStore/faiss_store.py`
  - Uses local FAISS index (`IndexFlatIP`) for similarity search.
  - Persists vectors in `Data/vector_store/index.faiss`.
  - Persists chunk metadata in `Data/vector_store/metadata.json`.

- `Retrieval/retriever.py`
  - Converts query to embedding.
  - Retrieves candidate chunks from FAISS.
  - Applies score re-ranking using `score * weight`.
  - Gives priority to PSS docs via config weight.

- `LLM/llm_response.py`
  - Prompt enforces grounded answers:
    - "Answer ONLY based on provided context. Do not hallucinate."
  - Supports role-aware style (`customer`, `sales`).
  - Uses local Hugging Face generation only (no external LLM API calls).

- `pipeline.py`
  - End-to-end indexing:
    - load PDF -> chunk -> embed -> index
  - Supports progress callback for status tracking.

- `App/app.py`
  - Local orchestration functions (no API server):
    - `diagnostics()`
    - `preflight_check()`
    - `index_document(...)`
    - `index_batch(...)`
    - `query_documents(...)`
    - `backup_vector_store()` / `restore_vector_store(...)`

- `App/dashboard.py`
  - Streamlit internal dashboard:
    - preflight checks
    - diagnostics
    - backup/restore vector store
    - single/batch indexing
    - query interface with role toggle

## Core configuration

All major controls are in `config.py`:

- FAISS storage: local index/metadata file paths
- Embeddings: model and vector dimension
- Chunking: chunk size and overlap
- Retrieval: top-k and candidate pool
- Document weights: PSS/manual/guide/default
- Security toggles: query length cap, document path restrictions, local/offline enforcement
- Naming rules: regex pattern and strict-enforcement option for incoming PDF names
