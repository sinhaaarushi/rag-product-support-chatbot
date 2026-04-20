"""Internal Streamlit dashboard for local-only RAG runtime.

The dashboard has three surfaces:
  * Chat       — the demo-ready conversation UI with grounded answers + sources
  * Documents  — index / batch-index / rebuild tools (admin surface)
  * System     — preflight, sync report, backup / restore (admin surface)

The chat surface carries the bulk of the UX work. The two admin tabs are
plain on purpose: they're internal tooling, not a customer-facing demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit invokes `streamlit run App/dashboard.py`, which places only this
# file's directory on sys.path -- not the project root. Without this bootstrap,
# `from App.app import ...` fails with ModuleNotFoundError even though the
# same import works from any other entry point (pytest, scripts/, REPL).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collections.abc import Callable  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from App.app import (  # noqa: E402
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

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PSS Knowledge Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A small amount of CSS to pull the UI away from the default Streamlit look.
# Everything here is cosmetic: colored badges for document types, tighter
# spacing on chat bubbles, rounder corners on source cards. If we ever add
# a real design system we'd replace this with proper component tokens.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Document-type badge pills */
    .doc-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 6px;
    }
    .doc-badge-PSS     { background: #4FD1C51A; color: #4FD1C5; border: 1px solid #4FD1C555; }
    .doc-badge-FAQ     { background: #B794F41A; color: #B794F4; border: 1px solid #B794F455; }
    .doc-badge-manual  { background: #F6AD551A; color: #F6AD55; border: 1px solid #F6AD5555; }
    .doc-badge-guide   { background: #63B3ED1A; color: #63B3ED; border: 1px solid #63B3ED55; }
    .doc-badge-default { background: #A0AEC01A; color: #A0AEC0; border: 1px solid #A0AEC055; }

    /* Subtle score pill beside the badge */
    .score-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        color: #A0AEC0;
        background: #1A1F2B;
        border: 1px solid #2D3748;
    }

    /* Tighten chat message padding */
    [data-testid="stChatMessage"] { padding: 0.75rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PSS Knowledge Assistant")
st.caption(
    "Grounded answers from your internal product documentation — fully local, no external APIs."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_action(
    button_label: str,
    action: Callable[[], Any],
    *,
    failure_prefix: str,
    success_message: str | None = None,
    button_type: str | None = None,
) -> None:
    """Render a button that runs an action and renders its result or error.

    Every admin tab has the same shape: "click button, call a backend function,
    show result on success, show error on failure." Pulling it into one helper
    keeps each tab readable and makes error handling impossible to forget on
    a new button.
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
    if isinstance(result, (dict, list)):
        st.json(result)
    elif result is not None:
        st.write(result)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved chunks as expandable cards with doc-type badges.

    We avoid ``st.json`` for sources because it escapes newlines (the raw
    ``\\n`` is meaningless to a reader) and dumps every field even when most
    are irrelevant at a glance. The card layout shows the three things a
    reviewer actually cares about per source: which doc it came from, how
    strong the match was, and a readable preview of the text.
    """
    if not sources:
        st.caption("No sources retrieved for this question.")
        return

    st.markdown("##### Sources")
    for rank, source in enumerate(sources, start=1):
        doc_type = source.get("document_type", "default")
        doc_name = source.get("document_name", "<unknown>")
        raw_score = float(source.get("score", 0.0))
        boosted_score = float(source.get("boosted_score", 0.0))
        weight = float(source.get("weight", 1.0))
        chunk_text = str(source.get("text", "")).strip()

        badge_class = (
            f"doc-badge-{doc_type}"
            if doc_type
            in {
                "PSS",
                "FAQ",
                "manual",
                "guide",
            }
            else "doc-badge-default"
        )

        header_html = (
            f'<span class="doc-badge {badge_class}">{doc_type}</span>'
            f'<span class="score-pill">match {raw_score:.2f} · weight {weight:.1f} · final {boosted_score:.2f}</span>'
        )

        with st.expander(f"{rank}. {Path(doc_name).name}", expanded=False):
            st.markdown(header_html, unsafe_allow_html=True)
            st.caption(doc_name)
            # st.text preserves original newlines and monospaces the body,
            # which reads much better than st.markdown for raw PDF extractions
            # that contain hyphens, bullets, and incidental formatting.
            st.text(chunk_text if chunk_text else "(empty chunk)")


# ---------------------------------------------------------------------------
# Sidebar — settings and session controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    query_role = st.radio(
        "Answer style",
        options=["customer", "sales"],
        index=0,
        help=(
            "Customer answers use plain language and safety framing. "
            "Sales answers emphasise positioning and differentiation — "
            "still grounded in the same retrieved context."
        ),
    )
    show_sources = st.toggle(
        "Show retrieved sources",
        value=True,
        help="Display the chunks the answer is grounded in, with scores.",
    )

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Fully local: embeddings and LLM run from `C:\\models\\`. No external "
        "API calls. Dashboard binds to 127.0.0.1 only."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_docs, tab_sys = st.tabs(["Chat", "Documents", "System"])


# ---- Chat --------------------------------------------------------------------

with tab_chat:
    # Conversation history lives in session state so messages persist across
    # reruns. Each entry is a dict so we can render sources alongside the
    # matching assistant turn without hunting through two parallel lists.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.chat_history:
        st.info(
            "Ask anything about the indexed product documentation. "
            "Try: *What is a virtual desktop?* or *How do I reset my password?*"
        )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            # Only assistant turns carry sources; the check is defensive so
            # a future refactor can't accidentally attach sources to user turns.
            if turn["role"] == "assistant" and turn.get("sources"):
                _render_sources(turn["sources"])

    user_prompt = st.chat_input("Ask a question about your documents…")
    if user_prompt:
        user_prompt = user_prompt.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            # Spinner is meaningful here: the first query does a cold model
            # load that can take 1-2 minutes. Without feedback users assume
            # the app is broken and refresh the tab.
            with st.spinner("Retrieving context and generating a grounded answer…"):
                try:
                    response = query_documents(
                        query=user_prompt,
                        role=query_role,
                        include_sources=True,
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"*(error: {exc})*",
                            "sources": [],
                        }
                    )
                else:
                    answer_text = response.get("answer", "").strip() or "*(no answer produced)*"
                    st.markdown(answer_text)
                    sources = response.get("sources", []) if show_sources else []
                    if sources:
                        _render_sources(sources)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": response.get("sources", []),
                        }
                    )


# ---- Documents ---------------------------------------------------------------

with tab_docs:
    st.subheader("Index a single PDF")
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
    st.subheader("Batch index a folder")
    batch_folder = st.text_input("Folder path", value="Data/documents")
    _run_action(
        "Run batch indexing",
        lambda: index_batch(batch_folder),
        failure_prefix="Batch indexing failed",
        success_message="Batch indexing completed",
    )

    st.markdown("---")
    st.subheader("Rebuild index from scratch")
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


# ---- System ------------------------------------------------------------------

with tab_sys:
    st.subheader("Health checks")

    _run_action("Run preflight", preflight_check, failure_prefix="Preflight failed")
    _run_action("Run diagnostics", diagnostics, failure_prefix="Diagnostics failed")
    _run_action(
        "Document / index sync (no indexing)",
        documents_sync_report,
        failure_prefix="Sync report failed",
    )

    st.markdown("---")
    st.subheader("Backup / restore")

    _run_action(
        "Create backup ZIP",
        backup_vector_store,
        failure_prefix="Backup failed",
        success_message="Backup created",
    )
    backup_path = st.text_input("Restore from backup zip path", value="")
    _run_action(
        "Restore backup ZIP",
        lambda: restore_vector_store(backup_path),
        failure_prefix="Restore failed",
        success_message="Backup restored",
    )
