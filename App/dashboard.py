"""Internal Streamlit dashboard for local-only RAG runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


def _run_action(
    button_label: str,
    action: Callable[[], Any],
    *,
    failure_prefix: str,
    success_message: str | None = None,
    button_type: str | None = None,
) -> None:
    """Render a button that runs an action and renders its result or error.

    Every tab in this dashboard has the same shape: "click button, call a
    backend function, show JSON on success, show an st.error on failure."
    Pulling it into one helper keeps each tab readable and makes error
    handling impossible to forget on a new button.
    """
    kwargs: dict[str, Any] = {"use_container_width": True}
    if button_type is not None:
        kwargs["type"] = button_type
    if not st.button(button_label, **kwargs):
        return

    try:
        result = action()
    except Exception as exc:
        # Intentionally broad: the dashboard is the single UI boundary where we
        # turn anything unexpected into a user-visible error instead of letting
        # Streamlit's traceback take over the page.
        st.error(f"{failure_prefix}: {exc}")
        return

    if success_message:
        st.success(success_message)
    # Most backend functions return dicts; fall back to st.write for strings.
    if isinstance(result, dict) or isinstance(result, list):
        st.json(result)
    elif result is not None:
        st.write(result)


tab_health, tab_index, tab_query = st.tabs(["Diagnostics", "Index Document", "Query Documents"])

with tab_health:
    st.subheader("System Diagnostics")

    _run_action("Run Preflight", preflight_check, failure_prefix="Preflight failed")
    _run_action("Run Diagnostics", diagnostics, failure_prefix="Diagnostics failed")
    _run_action(
        "Document / index sync (no indexing)",
        documents_sync_report,
        failure_prefix="Sync report failed",
    )

    st.markdown("---")
    st.subheader("Backup / Restore Vector Store")

    _run_action(
        "Create Backup ZIP",
        backup_vector_store,
        failure_prefix="Backup failed",
        success_message="Backup created",
    )
    backup_path = st.text_input("Restore from backup zip path", value="")
    _run_action(
        "Restore Backup ZIP",
        lambda: restore_vector_store(backup_path),
        failure_prefix="Restore failed",
        success_message="Backup restored",
    )

with tab_index:
    st.subheader("Index Single PDF")
    single_pdf_path = st.text_input("PDF path", value="Data/documents/sample.pdf")
    # "auto" means let the pipeline infer from the folder; any explicit choice
    # overrides the folder-based classifier for this one file.
    doc_type_choice = st.selectbox(
        "Document type",
        ["auto", "PSS", "FAQ", "manual", "guide", "default"],
    )

    _run_action(
        "Index PDF",
        lambda: index_document(
            file_path=single_pdf_path,
            document_type=None if doc_type_choice == "auto" else doc_type_choice,
        ),
        failure_prefix="Indexing failed",
        success_message="Document indexed",
    )

    st.markdown("---")
    st.subheader("Batch Indexing (All PDFs in Folder)")
    batch_folder = st.text_input("Batch folder path", value="Data/documents")
    _run_action(
        "Run Batch Indexing",
        lambda: index_batch(batch_folder),
        failure_prefix="Batch indexing failed",
        success_message="Batch indexing completed",
    )

    st.markdown("---")
    st.subheader("Rebuild index (folder = source of truth)")
    st.caption(
        "Clears FAISS + metadata, then re-indexes every PDF under the folder recursively. "
        "Use after you add, remove, or rename files so the index matches disk."
    )
    _run_action(
        "Rebuild index (clear + full re-index)",
        rebuild_index_from_documents,
        failure_prefix="Rebuild failed",
        success_message="Rebuild completed",
        button_type="primary",
    )

with tab_query:
    st.subheader("Ask Question")
    user_query = st.text_area("Query", height=120, placeholder="Ask based on indexed documents...")
    query_role = st.selectbox("Role", ["customer", "sales"])
    show_sources = st.checkbox("Include retrieved sources", value=False)

    if st.button("Run Query", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                response = query_documents(
                    query=user_query.strip(),
                    role=query_role,
                    include_sources=show_sources,
                )
            except Exception as exc:
                st.error(f"Query failed: {exc}")
            else:
                st.success("Response received")
                st.write(response.get("answer", ""))
                if show_sources and response.get("sources"):
                    st.subheader("Sources")
                    st.json(response["sources"])
