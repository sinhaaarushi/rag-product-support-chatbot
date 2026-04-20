"""Single-surface chatbot dashboard for the PSS knowledge assistant.

One page, one job: the assistant introduces itself by name, asks the user
which role best describes them (Customer or Internal Sales Team), then
answers questions grounded in the indexed PDFs. Each answer is followed by
one small source chip per unique document citation -- filename + page only,
no paragraph re-cite.

Admin tooling (rebuild index, backup, etc.) intentionally lives in the
``scripts/`` CLI, not here.
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

import config  # noqa: E402
from App.app import query_documents  # noqa: E402

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{config.ASSISTANT_NAME} — Knowledge Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Logo is optional. Drop a PNG at Data/assets/logo.png and it appears as a
# faded watermark behind the chat. Missing file = no watermark, no error.
_LOGO_PATH = _PROJECT_ROOT / "Data" / "assets" / "logo.png"


def _encoded_logo() -> str | None:
    """Read the brand logo once and return a base64 data-URI, or None.

    We inline the logo as a CSS background so it can sit behind the chat
    content as a subtle watermark without stealing vertical space. Reading
    it once at module load keeps the file IO out of every rerun.
    """
    if not _LOGO_PATH.is_file():
        return None
    try:
        raw_bytes = _LOGO_PATH.read_bytes()
    except OSError:
        return None
    return base64.b64encode(raw_bytes).decode("ascii")


_LOGO_B64 = _encoded_logo()

_WATERMARK_CSS = (
    f"""
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/png;base64,{_LOGO_B64}");
        background-repeat: no-repeat;
        background-position: center 40%;
        background-size: min(560px, 60vw);
        opacity: 0.04;
        pointer-events: none;
        z-index: 0;
    }}
    """
    if _LOGO_B64
    else ""
)

# Layout and theming CSS. We keep this inline (rather than in a separate
# stylesheet) because Streamlit doesn't serve static CSS by default and the
# amount here is small enough that another file would hurt more than help.
# The goal: match the reference mock -- rounded chat frame, user bubbles on
# the right, assistant on the left, icon-pill role buttons, PDF source chips.
st.markdown(
    f"""
    <style>
    /* Constrain the main canvas so the chat feels like a product window,
       not a full-width webpage. */
    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 7rem;
        max-width: 780px;
    }}

    /* Brand header */
    .brand-title {{
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.15rem;
    }}
    .brand-tagline {{
        color: #A0AEC0;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }}

    /* Chat container -- the rounded bordered frame around the whole dialog.
       Adding the border only when there's real content (handled in Python
       below by wrapping messages in a dedicated div) keeps the empty-state
       greeting from floating inside an awkward empty box. */
    .chat-frame {{
        border: 1px solid #2D3748;
        border-radius: 18px;
        padding: 1.25rem 1rem 0.75rem 1rem;
        background: rgba(26, 31, 43, 0.55);
        backdrop-filter: blur(2px);
    }}

    /* Default chat-message width and rounding. User messages float right,
       assistant messages float left -- mimicking the reference mock. */
    [data-testid="stChatMessage"] {{
        max-width: 85%;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
    }}
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {{
        margin-left: auto;
        background: #2A3B5B;
    }}
    .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background: #1A1F2B;
        border: 1px solid #2D3748;
    }}

    /* Role-pick pill buttons: wide, rounded, emoji-led. Matches the three
       chips in the reference mock but we only need two. */
    .role-row .stButton > button {{
        width: 100%;
        padding: 0.9rem 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid #2D3748;
        background: #1A1F2B;
    }}
    .role-row .stButton > button:hover {{
        border-color: #4FD1C5;
        background: #1F2A3C;
    }}

    /* Source chips sit as a horizontal row below the answer. Each chip
       carries a PDF icon, the document filename, and the page number. */
    .source-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }}
    .source-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #2A1F2E;
        border: 1px solid #B794F455;
        color: #E6EDF3;
        font-size: 0.78rem;
        white-space: nowrap;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .source-chip .pdf-glyph {{
        color: #F56565;
        font-weight: 700;
        font-size: 0.7rem;
        background: #F5656522;
        padding: 1px 5px;
        border-radius: 4px;
    }}
    .source-chip .page-tag {{
        color: #A0AEC0;
        font-size: 0.72rem;
    }}

    /* Keep rendered content above the logo watermark. */
    .block-container > * {{ position: relative; z-index: 1; }}

    {_WATERMARK_CSS}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "role" not in st.session_state:
    st.session_state.role = None
if "messages" not in st.session_state:
    # Each message is a dict: {"role": "user"|"assistant", "content": str,
    # "sources": optional list of source dicts}. Keeping sources inline with
    # the matching assistant turn means replay is a single loop, not two.
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Brand header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="brand-title">{config.ASSISTANT_NAME} · Knowledge Assistant</div>'
    '<div class="brand-tagline">Grounded answers from your product documentation — fully local, no external APIs.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Source chip renderer
# ---------------------------------------------------------------------------

_ROLE_LABELS = {
    "customer": "Customer",
    "sales": "Internal Sales Team",
}


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse sources that point to the same document + page.

    Retrieval often returns two chunks from the same page because a
    paragraph straddled a chunk boundary. The user doesn't care about
    that detail -- they want one chip per (document, page) citation.
    """
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = (str(source.get("document_name", "")), int(source.get("page_number", 0)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _render_source_chips(sources: list[dict[str, Any]]) -> None:
    """Render one PDF chip per unique (document, page) citation.

    The chip is a thin, non-interactive visual pill for now -- it tells the
    reader which document and page the answer came from, matching the
    reference mock. Wiring up an actual "open the PDF" action is a separate
    concern and intentionally deferred (see project notes).
    """
    unique_sources = _dedupe_sources(sources)
    if not unique_sources:
        return

    chips_html: list[str] = ['<div class="source-row">']
    for source in unique_sources:
        doc_name = Path(str(source.get("document_name", ""))).name or "unknown"
        page_number = int(source.get("page_number", 0))
        page_tag = f"Page {page_number}" if page_number > 0 else "Page unknown"
        chips_html.append(
            '<span class="source-chip">'
            '<span class="pdf-glyph">PDF</span>'
            f"<span>{doc_name}</span>"
            f'<span class="page-tag">· {page_tag}</span>'
            "</span>"
        )
    chips_html.append("</div>")
    st.markdown("".join(chips_html), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Role picker (only shown on first turn)
# ---------------------------------------------------------------------------


def _seed_role(picked_role: str) -> None:
    """Lock the role for the session and log the pick + assistant ack."""
    st.session_state.role = picked_role
    label = _ROLE_LABELS.get(picked_role, picked_role)
    st.session_state.messages.append({"role": "user", "content": f"*I'm from the {label} team.*"})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"Great — I'll answer with the **{label}** audience in mind. "
                "Ask me anything about the product documentation and I'll pull "
                "the relevant facts with source citations."
            ),
        }
    )


def _render_role_picker() -> None:
    """First-turn greeting + role pick row.

    We render two ``st.chat_message`` blocks (greeting, pick prompt) and then
    a two-column button row. The buttons sit inside a div with class
    ``role-row`` so the CSS above can style them as wide pill buttons.
    """
    with st.chat_message("assistant", avatar="💬"):
        st.markdown(f"**Hello! My name's {config.ASSISTANT_NAME}. How may I help you today?**")
    with st.chat_message("assistant", avatar="💬"):
        st.markdown("Please pick what identifies you the best:")

    st.markdown('<div class="role-row">', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        if st.button("👤  Customer", key="pick_customer", use_container_width=True):
            _seed_role("customer")
            st.rerun()
    with right:
        if st.button(
            "💼  Internal Sales Team",
            key="pick_sales",
            use_container_width=True,
        ):
            _seed_role("sales")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Conversation replay
# ---------------------------------------------------------------------------


def _replay_history() -> None:
    """Re-render every past turn so the chat feels persistent across reruns.

    Streamlit reruns the whole script on every interaction, so we replay
    history from session_state rather than trying to keep widgets mounted.
    This also means the role picker is conditional: it's only re-rendered
    when ``st.session_state.role`` is still None.
    """
    for turn in st.session_state.messages:
        avatar = "🧑" if turn["role"] == "user" else "💬"
        with st.chat_message(turn["role"], avatar=avatar):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("sources"):
                _render_source_chips(turn["sources"])


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

_replay_history()

if st.session_state.role is None:
    _render_role_picker()


# ---------------------------------------------------------------------------
# Chat input (disabled until a role is picked)
# ---------------------------------------------------------------------------

chat_placeholder = (
    f"Ask {config.ASSISTANT_NAME} anything about the product documentation…"
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
        # Spinner is meaningful here: the first question triggers a cold
        # model load that can take 1-2 minutes on CPU. Without feedback,
        # users assume the UI is broken and refresh.
        with st.spinner("Searching the documentation and drafting an answer…"):
            try:
                response = query_documents(
                    query=user_prompt,
                    role=st.session_state.role,
                    include_sources=True,
                )
            except Exception as exc:
                # Surface failures in-line and record them in history so
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
                # Don't show source chips for an out-of-scope fallback. The
                # fallback text already tells the user to contact support --
                # showing random "relevant" sources here would be misleading.
                out_of_scope = bool(response.get("out_of_scope", False))
                retrieved_sources = [] if out_of_scope else (response.get("sources", []) or [])
                if retrieved_sources:
                    _render_source_chips(retrieved_sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": retrieved_sources,
                    }
                )
