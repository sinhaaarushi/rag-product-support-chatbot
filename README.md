# Enterprise RAG Chatbot (Fully Local, No API)

This project is a fully local Retrieval-Augmented Generation (RAG) system for product PDFs (PSS/manual/guide/FAQ).  
It uses HuggingFace models for both embeddings and answer generation, and FAISS as the local vector database.

**GitHub / sensitive data:** this repository is intended to hold **code and documentation only**. PDFs, FAISS indexes, logs, and backups stay **local** (see `.gitignore` and [docs/GITHUB_AND_SENSITIVE_DATA.md](docs/GITHUB_AND_SENSITIVE_DATA.md)). Use `.env.example` as a template; never commit `.env` or `Data/**` content beyond empty folder placeholders.

## Architecture

Pipeline:
1. Load PDF text (`pdfplumber`)
2. Chunk text (`Processing/chunking.py`)
3. Generate embeddings (`sentence-transformers`)
4. Store vectors + metadata in FAISS + local metadata JSON
5. Embed query and perform similarity search
6. Re-rank with document weightage (PSS priority)
7. Build grounded prompt + role adaptation (`customer` / `sales`)
8. Return final answer

## Project Structure

`App/app.py` - local orchestration functions (`diagnostics`, `index_document`, `index_batch`, `query_documents`)  
`App/dashboard.py` - Streamlit local dashboard (no HTTP/API calls)  
`Ingestion/document_loader.py` - PDF loading and text extraction  
`Processing/chunking.py` - chunk generation  
`Embeddings/embedding_generator.py` - embedding model wrapper  
`VectorStore/faiss_store.py` - FAISS index persistence and vector search  
`Retrieval/retriever.py` - retrieval + weighted ranking  
`LLM/llm_response.py` - grounded answer generation (local HF only)  
`pipeline.py` - indexing orchestration  
`config.py` - configuration and offline security controls  

## Security and Storage

- No external LLM API usage.
- Offline-only model loading supported and enforced by config.
- FAISS index and metadata are stored locally under `Data/vector_store`:
  - `index.faiss`: binary vector index
  - `metadata.json`: chunk metadata (`text`, `document_name`, `document_type`, `weight`, `chunk_index`)
- Safety notes:
  - Data stays on local machine unless you move/share files.
  - `metadata.json` contains chunk text; protect this folder with OS access controls.
  - Consider disk encryption for sensitive projects.

## Recommended Before Documents

1. Run **Preflight** in dashboard (validates local model paths and embedding runtime).
2. Confirm naming convention:
   - default pattern: `^(PSS|FAQ|manual|guide)_[A-Za-z0-9._-]+\.pdf$`
   - examples:
     - `PSS_motor_v1.pdf`
     - `FAQ_returns.pdf`
     - `manual_installation.pdf`
     - `guide_quickstart.pdf`
   - PDFs may live in subfolders under `Data/documents/`; the index stores a stable relative path key.
3. After **adding, removing, or renaming** PDFs, use **Rebuild index (clear + full re-index)** in the dashboard (or call `rebuild_index_from_documents()`). Batch indexing alone only appends and does not delete stale chunks.
4. Create a vector-store backup baseline from dashboard.
5. Keep `ENFORCE_DOC_NAME_PATTERN=true` if you want strict naming validation.

## Setup

1) Create and activate venv

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Configure strict local/offline model paths

```powershell
$env:LOCAL_LLM_ONLY = "true"
$env:OFFLINE_ONLY = "true"
$env:EMBEDDING_MODEL_LOCAL_PATH = "C:\models\all-MiniLM-L6-v2"
$env:HF_CHAT_MODEL_LOCAL_PATH = "C:\models\flan-t5-base"
```

## Run Local Dashboard

**Quick start (Windows):** double-click `run_local.bat` in the project folder (edit the model paths inside the file if needed). It binds to **http://127.0.0.1:8501** only.

Or from PowerShell:

```powershell
.\.venv\Scripts\streamlit run App/dashboard.py
```

Dashboard URL: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Logging and Evaluation

- Logs are written to: `Data/logs/rag_app.log`
- Sample evaluation script:

```powershell
.\.venv\Scripts\python Eval/evaluate_queries.py
```

## Industry-grade tooling

- **Tests:** `pytest` — run `pip install -r requirements-dev.txt` then `pytest`.
- **CI:** GitHub Actions workflow `.github/workflows/ci.yml` runs tests on push/PR.
- **Docs:** `docs/ARCHITECTURE.md`, `docs/PRODUCTION_CHECKLIST.md`.
- **Container:** `Dockerfile` for server deployment (mount models + `Data/` at runtime).

## How FAISS is used

- Embedding vectors are normalized float32 arrays.
- FAISS `IndexFlatIP` is used (inner product on normalized vectors = cosine similarity behavior).
- New document chunks are appended to FAISS and synced with metadata JSON.
- After changes to files on disk, use a **full rebuild** so the index matches the folder (see Recommended Before Documents).
- Retrieval returns top-k vector matches, then applies weight boost:
  - `boosted_score = similarity_score * weight`
- PSS chunks get higher default weight via `config.DOCUMENT_WEIGHTS` (FAQ and other types have their own weights).
