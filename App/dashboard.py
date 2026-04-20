"""Internal Streamlit dashboard for local-only RAG runtime."""

from __future__ import annotations

import streamlit as st

from App.app import (
    backup_vector_store,
    diagnostics,
    documents_sync_report,
    index_batch,
    index_document,
    preflight_check,
    query_documents,
    rebuild_index_from_documents,
    restore_vector_store,
)

st.set_page_config(page_title="RAG Control Panel", page_icon=":mag:", layout="wide")
st.title("RAG Control Panel (Internal)")
st.caption("Fully local mode: no API calls, no external service endpoint required.")


tab_health, tab_index, tab_query = st.tabs(["Diagnostics", "Index Document", "Query Documents"])

with tab_health:
    st.subheader("System Diagnostics")
    if st.button("Run Preflight", use_container_width=True):
        try:
            st.json(preflight_check())
        except Exception as exc:
            st.error(f"Preflight failed: {exc}")
    if st.button("Run Diagnostics", use_container_width=True):
        try:
            st.json(diagnostics())
        except Exception as exc:
            st.error(f"Diagnostics failed: {exc}")
    if st.button("Document / index sync (no indexing)", use_container_width=True):
        try:
            st.json(documents_sync_report())
        except Exception as exc:
            st.error(f"Sync report failed: {exc}")
    st.markdown("---")
    st.subheader("Backup / Restore Vector Store")
    if st.button("Create Backup ZIP", use_container_width=True):
        try:
            archive = backup_vector_store()
            st.success(f"Backup created: {archive}")
        except Exception as exc:
            st.error(f"Backup failed: {exc}")
    backup_path = st.text_input("Restore from backup zip path", value="")
    if st.button("Restore Backup ZIP", use_container_width=True):
        try:
            restore_vector_store(backup_path)
            st.success("Backup restored")
        except Exception as exc:
            st.error(f"Restore failed: {exc}")

with tab_index:
    st.subheader("Index Single PDF")
    file_path = st.text_input("PDF path", value="Data/documents/sample.pdf")
    document_type = st.selectbox(
        "Document type",
        ["auto", "PSS", "FAQ", "manual", "guide", "default"],
    )

    if st.button("Index PDF", use_container_width=True):
        try:
            dtype = None if document_type == "auto" else document_type
            result = index_document(file_path=file_path, document_type=dtype)
            st.success("Document indexed")
            st.json(result)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    st.markdown("---")
    st.subheader("Batch Indexing (All PDFs in Folder)")
    batch_folder = st.text_input("Batch folder path", value="Data/documents")
    if st.button("Run Batch Indexing", use_container_width=True):
        try:
            result = index_batch(batch_folder)
            st.success("Batch indexing completed")
            st.json(result)
        except Exception as exc:
            st.error(f"Batch indexing failed: {exc}")

    st.markdown("---")
    st.subheader("Rebuild index (folder = source of truth)")
    st.caption(
        "Clears FAISS + metadata, then re-indexes every PDF under the folder recursively. "
        "Use after you add, remove, or rename files so the index matches disk."
    )
    if st.button("Rebuild index (clear + full re-index)", type="primary", use_container_width=True):
        try:
            result = rebuild_index_from_documents()
            st.success("Rebuild completed")
            st.json(result)
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")

with tab_query:
    st.subheader("Ask Question")
    query = st.text_area("Query", height=120, placeholder="Ask based on indexed documents...")
    role = st.selectbox("Role", ["customer", "sales"])
    include_sources = st.checkbox("Include retrieved sources", value=False)

    if st.button("Run Query", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                data = query_documents(
                    query=query.strip(),
                    role=role,
                    include_sources=include_sources,
                )
                st.success("Response received")
                st.write(data.get("answer", ""))
                if include_sources and data.get("sources"):
                    st.subheader("Sources")
                    st.json(data["sources"])
            except Exception as exc:
                st.error(f"Query failed: {exc}")
