# Production / Handover Checklist

Use this before calling the system “production ready” outside a personal laptop.

## Models and runtime

- [ ] Embedding and chat models are on **approved internal paths** (or mounted volumes).
- [ ] `OFFLINE_ONLY=true` and preflight passes (`ready: true`) on the target host.
- [ ] Disk space verified for models + FAISS growth + logs.

## Data governance

- [ ] PDFs follow naming convention if `ENFORCE_DOC_NAME_PATTERN=true`.
- [ ] `Data/vector_store/` and `Data/documents/` have **restricted OS permissions** (and backups).
- [ ] Decide whether `metadata.json` may contain **PII**; if yes, encrypt at rest or redact.

## Operations

- [ ] Log rotation or size cap for `Data/logs/rag_app.log`.
- [ ] Backup/restore procedure tested (`backup` from dashboard or scripted).
- [ ] Runbook for re-indexing after model or chunk config change.

## Quality

- [ ] `pytest` passes in CI.
- [ ] Sample queries evaluated (`Eval/evaluate_queries.py`) on real docs.

## Deployment (server)

- [ ] Run Streamlit behind **HTTPS** reverse proxy; do not expose raw port to public internet without auth.
- [ ] Add **authentication** (SSO/VPN or app-level) for internal tools.
- [ ] Use **Docker** or systemd unit for reproducible deploy (see `Dockerfile`).
