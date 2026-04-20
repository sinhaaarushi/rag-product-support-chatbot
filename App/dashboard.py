"""Single-surface chatbot dashboard for the PSS knowledge assistant.

One page, one job: greet the user, let them pick a role (Customer or
Internal Sales Team), then answer questions grounded in the indexed PDFs
with source citations that expand to a page number.

Admin tooling (rebuild index, backup, etc.) intentionally lives in the
`scripts/` CLI, not here — this surface is for the end user.
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

import base64  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from App.app import query_documents  # noqa: E402

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PSS Knowledge Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Logo is optional. Drop a PNG at Data/assets/logo.png and it appears as a
# faded watermark behind the chat. Missing file = no watermark, no error.
_LOGO_PATH = _PROJECT_ROOT / "Data" / "assets" / "logo.png"


def _encoded_logo() -> str | None:
    """Read the brand logo once and return a base64 data-URI, or None.

    We inline the logo as a CSS background rather than using ``st.image`` so
    it can sit behind the chat content as a subtle watermark without stealing
    vertical space from the conversation.
    """
    if not _LOGO_PATH.is_file():
        return None
    try:
        raw_bytes = _LOGO_PATH.read_bytes()
    except OSError:
        return None
    return base64.b64encode(raw_bytes).decode("ascii")


_LOGO_B64 = _encoded_logo()

# Theme + layout CSS. Keeping this inline (instead of a separate stylesheet)
# is deliberate: Streamlit does not serve arbitrary static files by default,
# and the amount of CSS here is small enough that a file split would add
# more indirection than it removes.
_watermark_css = (
    f"""
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/png;base64,{_LOGO_B64}");
        background-repeat: no-repeat;
        background-position: center 35%;
        background-size: min(520px, 60vw);
        opacity: 0.05;
        pointer-events: none;
        z-index: 0;
    }}
    """
    if _LOGO_B64
    else ""
)

st.markdown(
    f"""
    <style>
    /* Tighter top padding so the greeting sits comfortably at the top */
    .block-container {{ padding-top: 2.5rem; padding-bottom: 6rem; max-width: 820px; }}

    /* Brand header */
    .brand-title {{
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }}
    .brand-tagline {{
        color: #A0AEC0;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }}

    /* Document-type badge pills inside source cards */
    .doc-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 6px;
    }}
    .doc-badge-PSS     {{ background: #4FD1C51A; color: #4FD1C5; border: 1px solid #4FD1C555; }}
    .doc-badge-FAQ     {{ background: #B794F41A; color: #B794F4; border: 1px solid #B794F455; }}
    .doc-badge-manual  {{ background: #F6AD551A; color: #F6AD55; border: 1px solid #F6AD5555; }}
    .doc-badge-guide   {{ background: #63B3ED1A; color: #63B3ED; border: 1px solid #63B3ED55; }}
    .doc-badge-default {{ background: #A0AEC01A; color: #A0AEC0; border: 1px solid #A0AEC055; }}

    .page-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #E6EDF3;
        background: #1A1F2B;
        border: 1px solid #2D3748;
    }}

    /* Role pick buttons grow to fill the row evenly */
    div[data-testid="column"] .stButton > button {{
        width: 100%;
        padding: 0.9rem 1rem;
        font-weight: 600;
    }}

    /* Make sure real content sits above the logo watermark */
    .block-container > * {{ position: relative; z-index: 1; }}

    {_watermark_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Brand header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="brand-title">PSS Knowledge Assistant</div>'
    '<div class="brand-tagline">Grounded answers from your product documentation — fully local, no external APIs.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# ``role`` is None until the user picks one on the first turn. Once set, the
# role is locked for the session and drives the LLM prompt framing.
if "role" not in st.session_state:
    st.session_state.role = None
# ``messages`` holds the rolling conversation. Each entry is a dict with a
# role ("user" | "assistant") and content; assistant entries may also carry
# a ``sources`` list for citation rendering.
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sources renderer
# ---------------------------------------------------------------------------

_ROLE_LABELS = {
    "customer": "Customer",
    "sales": "Internal Sales Team",
}


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render one collapsed ``Sources (N)`` block with filename + page only.

    We deliberately do not show the chunk text here. Reviewers asked for a
    clean citation -- document name with an expandable page reference -- not
    a copy of the paragraph the answer was written from. Keeping citations
    thin also avoids the ugly ``\\n`` escaping that ``st.json`` produces.
    """
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for rank, source in enumerate(sources, start=1):
            doc_type = source.get("document_type", "default")
            doc_name = source.get("document_name", "<unknown>")
            page_number = int(source.get("page_number", 0))
            badge_class = (
                f"doc-badge-{doc_type}"
                if doc_type in {"PSS", "FAQ", "manual", "guide"}
                else "doc-badge-default"
            )

            page_label = f"Page {page_number}" if page_number > 0 else "Page unknown"
            # One row per source: badge + filename up top, then a nested
            # expander that reveals only the page number when clicked.
            st.markdown(
                f'<div style="margin: 0.35rem 0 0.1rem 0;">'
                f'<span class="doc-badge {badge_class}">{doc_type}</span>'
                f"<strong>{rank}. {Path(doc_name).name}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Location", expanded=False):
                st.markdown(
                    f'<span class="page-pill">{page_label}</span>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Conversation replay
# ---------------------------------------------------------------------------

# Every rerun replays the full history so the chat feels persistent. We do
# this before rendering any new widgets so the newest message is always at
# the bottom of the scroll position.
for turn in st.session_state.messages:
    with st.chat_message(turn["role"], avatar="🧑" if turn["role"] == "user" else "💬"):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            _render_sources(turn["sources"])


# ---------------------------------------------------------------------------
# First-turn greeting + role pick
# ---------------------------------------------------------------------------


def _greet_and_collect_role() -> None:
    """Render the bot's opening greeting and the Customer / Sales role pick.

    The two buttons are columns so they sit side-by-side on any screen.
    Clicking one seeds the conversation with the user's choice and the
    assistant's acknowledgement, then triggers a rerun so the chat input
    shows up for the next turn.
    """
    with st.chat_message("assistant", avatar="💬"):
        st.markdown(
            "**Hello, how may I help you today?**  \n"
            "Before we start, please choose the option that best describes you:"
        )
        left, right = st.columns(2)
        with left:
            if st.button("I'm a customer", key="pick_customer"):
                _seed_role("customer")
                st.rerun()
        with right:
            if st.button("Internal sales team", key="pick_sales"):
                _seed_role("sales")
                st.rerun()


def _seed_role(picked_role: str) -> None:
    """Lock the role for the session and add a short ack to the transcript."""
    st.session_state.role = picked_role
    label = _ROLE_LABELS.get(picked_role, picked_role)
    st.session_state.messages.append({"role": "user", "content": f"*Selected role: {label}*"})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"Great — I'll answer as if you're from **{label}**. "
                "Ask me anything about the product documentation and I'll pull the "
                "relevant facts with source citations."
            ),
        }
    )


if st.session_state.role is None:
    _greet_and_collect_role()


# ---------------------------------------------------------------------------
# Chat input (only enabled after a role is picked)
# ---------------------------------------------------------------------------

chat_placeholder = (
    "Ask me anything about the product documentation…"
    if st.session_state.role
    else "Pick a role above to start chatting"
)

user_prompt = st.chat_input(chat_placeholder, disabled=st.session_state.role is None)

if user_prompt:
    user_prompt = user_prompt.strip()
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar="💬"):
        # Spinner is meaningful here: the first question triggers a cold model
        # load that can take 1-2 minutes on CPU. Without feedback, users assume
        # the UI is broken and refresh.
        with st.spinner("Searching the documentation and drafting a grounded answer…"):
            try:
                response = query_documents(
                    query=user_prompt,
                    role=st.session_state.role,
                    include_sources=True,
                )
            except Exception as exc:
                # Surface the failure in-line and record it in history so
                # the user can scroll back and see what went wrong.
                error_message = f"Sorry — I hit an error while answering: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
            else:
                answer_text = (
                    response.get("answer", "").strip()
                    or "I don't have enough information in the documents to answer that confidently."
                )
                st.markdown(answer_text)
                retrieved_sources = response.get("sources", []) or []
                if retrieved_sources:
                    _render_sources(retrieved_sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": retrieved_sources,
                    }
                )
