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

## Loading your documents

Drop PDFs anywhere under `Data/documents/`. The folder layout is what drives
**document type** (PSS, FAQ, manual, guide), and document type drives the
retrieval weighting in `Retrieval/retriever.py`.

### Recommended layout

```
Data/documents/
├── PSS/
│   ├── ProductA_PSS.pdf
│   ├── ProductB_ProductSpecSheet.pdf
│   └── FAQs/
│       ├── ProductA_FAQ.pdf
│       └── ProductB_FrequentlyAskedQuestions.pdf
├── manuals/
│   └── installation_manual.pdf
└── guides/
    └── quickstart_guide.pdf
```

### How a file gets classified

`infer_document_type_from_name()` resolves type using the path, most-specific
folder first, then falling back to filename tokens:

| Path under `Data/documents/`            | Inferred type |
|------------------------------------------|---------------|
| `PSS/anything.pdf`                       | `PSS`         |
| `PSS/FAQs/anything.pdf`                  | `FAQ` *(subfolder wins over PSS parent)* |
| `manuals/setup.pdf`                      | `manual`      |
| `guides/quickstart.pdf`                  | `guide`       |
| `MotorXYZ_ProductSpecSheet.pdf`          | `PSS` *(filename token)* |
| `BoilerABC_FrequentlyAskedQuestions.pdf` | `FAQ` *(filename token)* |
| `random_report.pdf`                      | `default`     |

Matching is case-insensitive and ignores `_`, `-`, and spaces.

### Filename validation (optional)

By default any `*.pdf` is accepted. If you want a strict naming convention
(e.g. for governance), set:

```powershell
$env:DOC_NAME_PATTERN = '^(PSS|FAQ|manual|guide)_[A-Za-z0-9._-]+\.pdf$'
$env:ENFORCE_DOC_NAME_PATTERN = "true"
```

### Indexing workflow

1. Run **Preflight** in the dashboard to confirm local model paths resolve and
   the embedding runtime works.
2. Drop PDFs under `Data/documents/` following the layout above.
3. Open the dashboard, go to the **Index Document** tab, and click
   **Rebuild index (clear + full re-index)**. This is the safe default after
   any add/remove/rename — batch indexing only appends and won't delete stale
   chunks for files you removed.
4. (Optional but recommended) Create a baseline backup from the **Diagnostics**
   tab so you can restore the index without re-embedding everything.

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
