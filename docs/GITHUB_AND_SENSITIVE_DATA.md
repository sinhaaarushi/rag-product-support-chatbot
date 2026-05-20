# GitHub + sensitive data (how this repo is meant to be used)

## What belongs on GitHub

- **Source code** (`App/`, `Embeddings/`, `LLM/`, `Retrieval/`, `VectorStore/`, `Processing/`, `Ingestion/`, `Utils/`, `Eval/`, `tests/`, etc.)
- **Configuration templates** (e.g. `.env.example`, `config.py` defaults)
- **Documentation** (`README.md`, `docs/`)
- **CI** (`.github/workflows/`)

This shows you are **actively building** a real system without exposing customer or internal documents.

## What must NOT go to GitHub

- **PDFs / source documents** → stay in `Data/documents/` locally (ignored).
- **Vector index + metadata** → `Data/vector_store/` (ignored; metadata can contain chunk text).
- **Logs** → `Data/logs/` (ignored).
- **Backups** → `Data/backups/` (ignored).
- **Secrets** → `.env` (ignored); use `.env.example` only as a template.


Avoid committing large binaries; if you need assets, use **Git LFS** only for non-sensitive files or host artifacts internally.

## If something sensitive was committed by mistake

1. Remove it from the **latest** commit before pushing: `git reset` / amend.
2. If it was **already pushed**, rotate any exposed credentials and ask your admin about **history rewrite** (e.g. `git filter-repo`) — do not rely on a normal delete commit alone for secrets.

## What reviewers see

They see **architecture, tests, CI, and runbooks** — standard for enterprise RAG work — while **data remains explicitly out of scope** of the repository, which is the correct security posture.
