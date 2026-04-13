# Change Log (Personal Tracker)

## 2026-04-08 13:58:10

### Major build-out completed

- Created complete modular RAG structure and package layout:
  - `App/`, `Ingestion/`, `Processing/`, `Embeddings/`, `VectorStore/`, `Retrieval/`, `LLM/`
  - Added missing `__init__.py` files
- Added central config in `config.py`
- Added dependencies in `requirements.txt`

### Core RAG features added

- PDF loader switched to `pdfplumber` with text cleanup:
  - `Ingestion/document_loader.py`
- Chunking finalized with validation:
  - `Processing/chunking.py`
- Embedding generation with `all-MiniLM-L6-v2`:
  - `Embeddings/embedding_generator.py`
- OpenSearch index + kNN + metadata storage:
  - `VectorStore/opensearch_client.py`
- Retrieval + PSS document weightage boost:
  - `Retrieval/retriever.py`
- LLM response generation with strict grounded prompt:
  - `LLM/llm_response.py`
- End-to-end indexing pipeline:
  - `pipeline.py`

### API and progress additions

- FastAPI endpoints created:
  - `POST /index`
  - `POST /query`
- Added async indexing and progress polling:
  - `POST /index/async`
  - `GET /index/status/{job_id}`
- Added `/health` endpoint

### Security improvements

- Restricted indexing path to `Data/documents` by default
- Added optional API key auth (`RAG_API_KEY` via `x-api-key`)
- Capped query length with `MAX_QUERY_CHARS`
- Set `/query` to not return source chunks by default
- Replaced internal exception leakage with safer generic 500 messages

### Visual progress UI

- Added internal Streamlit dashboard:
  - `App/dashboard.py`
- Dashboard supports:
  - async index start
  - live status poll
  - chunk progress bar
  - query UI with role selection

### Documentation updates

- Refreshed root `README.md` with secure run instructions and examples
- Added personal notes folder:
  - `My_Project_Notes/`

## 2026-04-08 14:10:00

### Pre-document onboarding readiness added

- Added batch indexing support before real document onboarding:
  - `pipeline.py`: `run_batch_indexing_pipeline(folder_path, progress_callback=None)`
  - `App/app.py`: `POST /index/batch`
- Added diagnostics endpoint for operational checks:
  - `App/app.py`: `GET /diagnostics`
  - Verifies OpenSearch reachability, index stats, and PDF count in `Data/documents`
- Added OpenSearch stats utility:
  - `VectorStore/opensearch_client.py`: `get_index_stats()`

### Dashboard enhancements

- `App/dashboard.py` now includes:
  - Diagnostics tab (`/diagnostics`)
  - Batch indexing controls (`/index/batch`)
  - Existing async single-document indexing and query tab retained

### Documentation and tracker updates

- Updated root `README.md` with new endpoints and API examples
- Updated personal tracker:
  - `My_Project_Notes/PROJECT_OVERVIEW.md`
  - `My_Project_Notes/updates/CHANGELOG.md`

## 2026-04-08 14:20:00

### Security policy update: no external LLM APIs

- Enforced local-only LLM response flow:
  - `LLM/llm_response.py` now uses Hugging Face local generation only
  - Removed OpenAI execution path
- Config updated for explicit local policy:
  - `config.py` adds `LOCAL_LLM_ONLY` flag
- Dependency hardening:
  - `requirements.txt` removed `openai`
- Documentation updated:
  - `README.md` environment section now reflects local-only setup
  - `My_Project_Notes/PROJECT_OVERVIEW.md` updated
  - `My_Project_Notes/SECURITY_NOTES.md` updated

## 2026-04-08 14:27:00

### Offline-only enforcement added (air-gap friendly)

- Added strict offline config controls in `config.py`:
  - `OFFLINE_ONLY`
  - `EMBEDDING_MODEL_LOCAL_PATH`
  - `HF_CHAT_MODEL_LOCAL_PATH`
- Enforced local model path checks:
  - `Embeddings/embedding_generator.py` fails fast if offline mode is enabled and embedding path is missing/invalid
  - `LLM/llm_response.py` fails fast if offline mode is enabled and chat model path is missing/invalid
- Added `local_files_only=True` model loading behavior in offline mode
- Updated `README.md` with exact offline environment variables
- Updated security notes with the new offline control policy

## 2026-04-08 14:45:00

### Major architecture migration: OpenSearch/API -> FAISS/local

- Removed API-driven runtime behavior and switched to full local orchestration:
  - `App/app.py` now exposes local Python functions (no FastAPI endpoints)
  - `App/dashboard.py` now calls local modules directly (no HTTP requests)
- Added FAISS vector store implementation:
  - `VectorStore/faiss_store.py`
  - Persistent storage:
    - `Data/vector_store/index.faiss`
    - `Data/vector_store/metadata.json`
- Switched indexing and retrieval to FAISS:
  - `pipeline.py` now writes chunk embeddings to FAISS store
  - `Retrieval/retriever.py` now searches FAISS and applies PSS-weighted re-ranking
- Updated dependencies:
  - removed API/OpenSearch dependencies
  - added `faiss-cpu`
- Updated docs and tracker:
  - `README.md` rewritten for fully local, no-API operation
  - `My_Project_Notes/PROJECT_OVERVIEW.md` and `SECURITY_NOTES.md` updated

## 2026-04-08 15:05:00

### Pre-document hardening implemented

- Added preflight validation:
  - `App/app.py`: `preflight_check()` and `startup_validate_or_raise()`
  - Checks local model path presence/existence and embedding runtime readiness
- Added local logging:
  - `Utils/logging_utils.py`
  - Runtime logs stored in `Data/logs/rag_app.log`
- Added backup/restore for FAISS store:
  - `Utils/backup_utils.py`
  - Dashboard buttons for backup/restore
- Added document naming convention controls:
  - `config.py`: `DOC_NAME_PATTERN`, `ENFORCE_DOC_NAME_PATTERN`
  - `pipeline.py`: warning/strict rejection for non-matching names
- Added lightweight evaluation tooling:
  - `Eval/sample_queries.json`
  - `Eval/evaluate_queries.py`
- Updated docs and personal notes accordingly

## 2026-04-13

### Localhost one-click launcher

- Added `run_local.bat` to start Streamlit on `127.0.0.1:8501` with offline env vars
- Updated `README.md` with double-click instructions

### Industry-grade hardening

- Added `pytest` suite under `tests/` (chunking, config, FAISS store, retriever weighting)
- Added `requirements-dev.txt`, `pyproject.toml` (pytest + ruff config)
- Added GitHub Actions CI `.github/workflows/ci.yml`
- Added `docs/ARCHITECTURE.md` and `docs/PRODUCTION_CHECKLIST.md`
- Added `Dockerfile` for reproducible server deployment
- Fixed `config.py` module docstring (FAISS, not OpenSearch)

### GitHub / sensitive data policy

- Expanded `.gitignore` for `Data/documents`, `Data/vector_store`, `Data/logs`, `Data/backups`, `.env`, local `models/`
- Added `Data/documents/.gitkeep` and `Data/vector_store/.gitkeep` so empty folders exist in git without real data
- Added `.env.example` (no secrets) and `docs/GITHUB_AND_SENSITIVE_DATA.md` (what to commit vs keep local)
- Updated `README.md` with explicit “code-only on GitHub” note
