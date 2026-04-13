# Security Notes

## Security controls currently implemented

1. Document path restriction
   - Default behavior allows indexing only from `Data/documents`.
   - Controlled by `ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR` in `config.py`.

2. Optional API authentication
   - `RAG_API_KEY` can protect API endpoints.
   - Header used: `x-api-key`.
   - If `RAG_API_KEY` is empty, auth is disabled (local development mode).

3. Query size guardrail
   - `MAX_QUERY_CHARS` limits query payload to avoid abuse and oversized prompts.

4. Reduced data leakage by default
   - `/query` returns answer only by default.
   - Source chunk details are optional (`include_sources=true`).

5. Safer error handling
   - Generic 500 error messages for indexing/query flows avoid exposing internals.

6. No external LLM API usage
   - Answer generation is local Hugging Face only.
   - `openai` dependency removed.
   - `LOCAL_LLM_ONLY=true` is used as a policy toggle.

7. Offline-only model loading enforcement
   - `OFFLINE_ONLY=true` requires local filesystem model paths.
   - Required env vars in offline mode:
     - `EMBEDDING_MODEL_LOCAL_PATH`
     - `HF_CHAT_MODEL_LOCAL_PATH`
   - System fails fast if local paths are missing/invalid.

8. Local vector store (FAISS) only
   - No remote vector DB dependency.
   - Vector index is persisted locally (`Data/vector_store/index.faiss`).
   - Metadata (including chunk text) is persisted locally (`Data/vector_store/metadata.json`).
   - Protect the local vector store folder with filesystem permissions/encryption.

9. Document naming governance
   - Default naming regex is enforced optionally:
     - `^(PSS|manual|guide)_[A-Za-z0-9._-]+\.pdf$`
   - This keeps document type inference and PSS prioritization consistent.
   - Enable strict mode with `ENFORCE_DOC_NAME_PATTERN=true`.

## Safe usage guidance

- Keep API and dashboard bound to localhost for development.
- Do not expose Streamlit publicly without auth and network controls.
- Keep API keys in environment variables, never hardcode.
- Avoid logging raw customer-sensitive document text.

## Suggested next hardening steps

- Add rate limiting for `/query`.
- Add structured auth (JWT/SSO) if exposed beyond localhost.
- Add audit logging (who queried, when, which docs were used).
- Encrypt data at rest if persistent storage moves to shared infrastructure.
