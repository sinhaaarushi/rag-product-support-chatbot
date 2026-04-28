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
import time
from pathlib import Path

# Streamlit invokes `streamlit run App/dashboard.py`, which places only this
# file's directory on sys.path -- not the project root. Without this bootstrap,
# `from App.app import ...` fails with ModuleNotFoundError even though the
# same import works from any other entry point (pytest, scripts/, REPL).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import base64  # noqa: E402
from html import escape  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

import config  # noqa: E402
from App.app import query_documents  # noqa: E402

# Cap so a burst of submits cannot enqueue unbounded work in one session.
_QUERY_QUEUE_MAX = 25


def _enqueue_user_queries(*parts: str | None) -> None:
    """Append non-empty stripped strings to the FIFO; oldest is answered first."""
    q: list[str] = st.session_state._query_queue
    for p in parts:
        if not p:
            continue
        s = str(p).strip()
        if not s:
            continue
        if len(q) >= _QUERY_QUEUE_MAX:
            break
        q.append(s)


def _schedule_queue_ack_or_rerun() -> None:
    """After an answer, pause before the next queued question so the user can cancel."""
    if st.session_state._query_queue:
        st.session_state._queue_prompt_ack_needed = True
    st.rerun()


def _truncate_queue_text(text: str, max_len: int = 160) -> str:
    collapsed = " ".join(str(text).strip().split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1] + "…"


def _render_pending_queue_strip(queued_snap: list[str]) -> None:
    """Show FIFO questions waiting above the chat input (after enqueue; before dequeue)."""
    if not queued_snap or st.session_state.role is None:
        return
    n = len(queued_snap)
    head_label = "In queue" if n > 1 else "Next question"
    parts: list[str] = [
        f'<div class="pending-queue-strip" role="status" aria-label="Queued questions">'
        f'<div class="pq-head">{escape(head_label)} · {n} total</div>'
    ]
    limit = min(n, 8)
    for i in range(limit):
        line = escape(_truncate_queue_text(queued_snap[i]))
        parts.append(f'<div class="pq-row"><span class="pq-idx">{i + 1}.</span>{line}</div>')
    if n > 8:
        parts.append(
            f'<div class="pq-row"><span class="pq-idx"></span>+ {n - 8} more in line</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_queue_continuation_gate() -> None:
    """Between turns: let the user continue to the next queued question or ✕ cancel all.

    Streamlit cannot interrupt a running ``query_documents`` call; cancelling only
    applies to questions not yet sent.
    """
    if not st.session_state.get("_queue_prompt_ack_needed"):
        return
    q: list[str] = st.session_state._query_queue
    if not q:
        st.session_state._queue_prompt_ack_needed = False
        return
    st.info(
        f"**{len(q)} question(s)** in queue. Continue to answer the next, or cancel everything "
        "you have not sent yet."
    )
    for i, text in enumerate(q[:5]):
        preview = text if len(text) <= 120 else text[:117] + "…"
        st.caption(f"{i + 1}. {preview}")
    if len(q) > 5:
        st.caption(f"…and {len(q) - 5} more.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Continue — answer next", type="primary", key="queue_continue_btn"):
            st.session_state._queue_prompt_ack_needed = False
            st.rerun()
    with c2:
        if st.button("✕ Cancel all queued", type="secondary", key="queue_cancel_btn"):
            st.session_state._query_queue.clear()
            st.session_state._queue_prompt_ack_needed = False
            st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{config.ASSISTANT_NAME} — Knowledge Assistant",
    page_icon="💬",
    # "wide" layout removes Streamlit's default narrow column so the chat
    # can breathe on large monitors. Bubble widths are capped in CSS below
    # to keep lines readable (~75-90 chars) even on a 1440p+ display.
    layout="wide",
    # Sidebar hosts the persistent role switcher (Customer / Internal /
    # Sales Team). Expanding by default on wide screens makes the active
    # audience visible at a glance and lets the user switch mid-chat
    # without hunting through a menu.
    initial_sidebar_state="expanded",
)

# Logo is optional. Drop a PNG at Data/assets/logo.png and it appears as a
# faded watermark behind the chat. Missing file = no watermark, no error.
_LOGO_PATH = _PROJECT_ROOT / "Data" / "assets" / "logo.png"
# Optional PNG for the main header chip (mascot / icon). If missing, a default
# SVG is shown. Independent of ``logo.png`` (sidebar wordmark + watermark).
_HEADER_ICON_PATH = _PROJECT_ROOT / "Data" / "assets" / "header_icon.png"


def _read_png_b64(path: Path) -> str | None:
    """Read a PNG from disk and return base64, or None if missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


_LOGO_B64 = _read_png_b64(_LOGO_PATH)
_HEADER_ICON_B64 = _read_png_b64(_HEADER_ICON_PATH)
# Default cute mascot (SVG) when no ``header_icon.png`` is on disk — override by adding the PNG.
_CUTE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48" height="48">\
<defs><radialGradient id="g" cx="28%" cy="28%" r="75%">\
<stop offset="0" stop-color="#5eead4"/><stop offset="0.5" stop-color="#22d3ee"/><stop offset="1" stop-color="#0e7490"/>\
</radialGradient></defs>\
<rect width="48" height="48" rx="12" fill="url(#g)"/>\
<circle cx="18" cy="20" r="3.2" fill="#0c1222"/><circle cx="30" cy="20" r="3.2" fill="#0c1222"/>\
<ellipse cx="24" cy="22" rx="2" ry="1.2" fill="#164e63" opacity="0.4"/>\
<path d="M16 30q8 6 16 0" fill="none" stroke="#0c1222" stroke-width="1.8" stroke-linecap="round"/>\
<path d="M12 16l-3-3M36 16l3-3" stroke="#0c1222" stroke-width="1.5" stroke-linecap="round" opacity="0.35"/>\
</svg>"""
_DEFAULT_HEADER_ICON_B64 = base64.b64encode(_CUTE_SVG.encode("utf-8")).decode("ascii")

# First letter of the assistant name for the header/sidebar tile when no
# custom logo is present. Re-skins when ``ASSISTANT_NAME`` is changed via env.
_logo_initial = (config.ASSISTANT_NAME[:1] or "?").upper()


def _brand_logo_img_src() -> str | None:
    """Data-URL for ``Data/assets/logo.png`` when the file is readable."""
    if not _LOGO_B64:
        return None
    return f"data:image/png;base64,{_LOGO_B64}"


def _header_icon_img_src() -> str:
    """Data-URL: ``header_icon.png`` if present, else the built-in SVG mascot."""
    if _HEADER_ICON_B64:
        return f"data:image/png;base64,{_HEADER_ICON_B64}"
    return f"data:image/svg+xml;base64,{_DEFAULT_HEADER_ICON_B64}"


def _app_header_block_html(assistant: str, subtitle: str) -> str:
    """Main brand header: mascot icon (custom PNG or default SVG) + titles."""
    safe_title = escape(assistant)
    mark = f'<div class="app-logo app-logo--icon" aria-hidden="true"><img src="{_header_icon_img_src()}" alt="" /></div>'
    return f"""<div class="app-header">
    {mark}
    <div>
        <div class="app-title">{safe_title}</div>
        <div class="app-subtitle">{escape(subtitle)}</div>
    </div>
</div>"""


def _sidebar_top_block_html(assistant: str) -> str:
    """Sidebar top: wordmark image above the name when ``logo.png`` is present."""
    src = _brand_logo_img_src()
    safe = escape(assistant)
    if src:
        return f"""
            <div class="sidebar-top">
                <div class="sidebar-brand sidebar-brand--wordmark">
                    <div class="sidebar-wordmark">
                        <img src="{src}" alt="{safe}" />
                    </div>
                    <div>
                        <div class="name">{safe}</div>
                        <div class="tag">Knowledge Assistant</div>
                    </div>
                </div>
                <div class="sidebar-audience-block">
                    <div class="sidebar-section-label">I'm here as</div>
                    <div class="sidebar-section-hint">Tailors tone and detail to your role</div>
                </div>
            </div>
            """
    return f"""
            <div class="sidebar-top">
                <div class="sidebar-brand">
                    <div class="mark">{_logo_initial}</div>
                    <div>
                        <div class="name">{safe}</div>
                        <div class="tag">Knowledge Assistant</div>
                    </div>
                </div>
                <div class="sidebar-audience-block">
                    <div class="sidebar-section-label">I'm here as</div>
                    <div class="sidebar-section-hint">Tailors tone and detail to your role</div>
                </div>
            </div>
            """


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

# ---------------------------------------------------------------------------
# PDF static mirror -- serves source documents over HTTP so citation chips
# can deep-link into them. Populated once at app startup.
# ---------------------------------------------------------------------------

# Browsers block file:// links from http:// origins for security, so we
# can't just point a link at the PDF on disk. Streamlit's static-serving
# feature (enableStaticServing = true in .streamlit/config.toml) exposes
# anything under ./static/ at /app/static/<path>, which the browser will
# happily open in a new tab.
#
# To avoid copying gigabytes of PDFs into the repo tree, we mirror by
# *hardlink* where possible (same bytes on disk, two directory entries)
# and fall back to a real copy only when the filesystem refuses -- e.g.
# if the source and ./static/ live on different drives. This keeps repo
# layout clean: Data/documents/... stays the source of truth, and
# static/pdfs/... is a derived artifact safe to delete and rebuild.
# The actual file-system work lives in ``App.pdf_mirror`` so the exact
# same logic can run as a pre-launch step from ``run_local.bat``.
# Streamlit checks for the static folder at server startup -- running
# the mirror *before* Streamlit boots means the folder is always ready
# by the time the first request arrives.
from App.pdf_mirror import mirror_pdfs as _mirror_pdfs  # noqa: E402


@st.cache_resource(show_spinner=False)
def _mirror_pdfs_into_static() -> dict[str, str]:
    """Streamlit-cached wrapper around ``pdf_mirror.mirror_pdfs``.

    The decorator ensures the filesystem walk happens once per process
    rather than on every rerun. Restart Streamlit (or clear the cache)
    after adding new documents so they become link-able.
    """
    return _mirror_pdfs()


_PDF_URL_MAP = _mirror_pdfs_into_static()

# Layout and theming CSS. Inline because Streamlit doesn't serve static
# files by default, and the scope is still small enough that a file split
# would hurt more than it helps.
#
# Design principles the stylesheet below encodes:
#   - Premium enterprise feel, not a demo. Generous whitespace, restrained
#     accent palette (teal on deep navy), soft shadows rather than hard
#     borders, subtle transitions on anything interactive.
#   - User-on-right / assistant-on-left chat convention, with user bubbles
#     accented so the eye can scan a conversation at a glance.
#   - Role picker is a card grid with icon + title + description -- much
#     more inviting than text pills on first load.
#   - Source chips look like real product citations, not debug output.
st.markdown(
    f"""
    <style>
    /* Font on .stApp only: ``.stApp *`` overrode Streamlit's icon font and
       showed the expander name (e.g. _arrow_right) on top of the label. */
    .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                     "SF Pro Display", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Main canvas. We went from Streamlit's 820 px default to a much
       wider centered column so the UI fills most of the viewport on
       laptop-size and larger monitors without going fully edge-to-edge
       (edge-to-edge on a 1440p+ display looks sparse because chat
       bubbles cap themselves for readability -- see below). */
    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 8rem;
        padding-left: clamp(1.25rem, 4vw, 3rem);
        padding-right: clamp(1.25rem, 4vw, 3rem);
        max-width: min(1280px, 95vw);
        margin-left: auto;
        margin-right: auto;
    }}

    /* Brand header -- initial letter tile + two lines of title/subtitle. */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.35rem;
    }}
    .app-logo {{
        width: 46px;
        height: 46px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #22D3EE 0%, #0891B2 100%);
        color: #0A0F1A;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        box-shadow: 0 8px 24px rgba(34, 211, 238, 0.18),
                    inset 0 1px 0 rgba(255, 255, 255, 0.12);
        flex-shrink: 0;
        animation: logo-pulse 3.2s ease-in-out infinite;
    }}
    .app-logo.app-logo--icon {{
        padding: 4px;
        color: transparent;
        background: linear-gradient(160deg, rgba(18, 32, 52, 0.97), rgba(8, 90, 120, 0.65));
        border: 1px solid rgba(34, 211, 238, 0.32);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(34, 211, 238, 0.12);
        animation: app-icon-bob 3.4s ease-in-out infinite;
    }}
    .app-logo--icon img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        border-radius: 9px;
    }}
    .app-title {{
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #E6EDF3;
        line-height: 1.15;
    }}
    .app-subtitle {{
        font-size: 0.85rem;
        color: #8B95A7;
        margin-top: 0.15rem;
    }}

    /* Status pill row -- live "grounded in N documents" badge. */
    .status-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.85rem 0 1.75rem 0;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.07);
        border: 1px solid rgba(34, 211, 238, 0.22);
        color: #7DD3FC;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }}
    .status-pill.neutral {{
        background: rgba(148, 163, 184, 0.06);
        border-color: rgba(148, 163, 184, 0.18);
        color: #94A3B8;
    }}
    .status-pill--cyan .status-pill__icon,
    .status-pill--purple .status-pill__icon {{
        display: inline-flex;
        flex-shrink: 0;
    }}
    .status-pill--purple {{
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.35);
        color: #C4B5FD;
    }}

    /* Stat strip (dashboard row below header pills) */
    .stat-strip {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.6rem;
        margin: 0.25rem 0 1.5rem 0;
    }}
    @media (max-width: 900px) {{
        .stat-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    .stat-card {{
        border-radius: 12px;
        padding: 0.65rem 0.7rem 0.7rem 0.7rem;
        background: linear-gradient(165deg, rgba(20, 28, 42, 0.95) 0%, rgba(10, 14, 22, 0.98) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.22);
    }}
    .stat-card--cyan {{
        border-color: rgba(34, 211, 238, 0.28);
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.1), 0 4px 20px rgba(34, 211, 238, 0.08);
    }}
    .stat-card--emerald {{
        border-color: rgba(52, 211, 153, 0.22);
        box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.08), 0 4px 18px rgba(16, 185, 129, 0.07);
    }}
    .stat-card--violet {{
        border-color: rgba(167, 139, 250, 0.3);
        box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.1), 0 4px 20px rgba(124, 58, 237, 0.1);
    }}
    .stat-card--amber {{
        border-color: rgba(251, 191, 36, 0.25);
        box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.08), 0 4px 18px rgba(245, 158, 11, 0.08);
    }}
    .stat-card__val {{
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .stat-card__sub {{
        font-size: 0.68rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }}
    .stat-card__ic {{
        font-size: 1.1rem;
        margin-bottom: 0.25rem;
    }}

    /* Onboarding "hero" cards (empty state, no role yet) */
    .hero-wrap {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 0 0 1.25rem 0;
    }}
    .hero-card {{
        flex: 1 1 300px;
        min-width: min(100%, 280px);
        border-radius: 16px;
        padding: 1rem 1rem 1rem 1.1rem;
        display: flex;
        flex-direction: row;
        align-items: stretch;
        gap: 0.9rem;
        background: linear-gradient(165deg, rgba(15, 23, 42, 0.65) 0%, rgba(8, 12, 22, 0.92) 100%);
    }}
    .hero-card--cyan {{
        border: 1px solid rgba(34, 211, 238, 0.4);
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.12), 0 8px 32px rgba(34, 211, 238, 0.12);
    }}
    .hero-card--violet {{
        border: 1px solid rgba(139, 92, 246, 0.45);
        box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.15), 0 8px 32px rgba(88, 28, 135, 0.18);
    }}
    .hero-card__text {{
        flex: 1 1 auto;
        min-width: 0;
        color: #E2E8F0;
        font-size: 0.9rem;
        line-height: 1.5;
    }}
    .hero-card__text strong {{
        color: #F8FAFC;
        display: block;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }}
    .hero-card__art {{
        flex: 0 0 100px;
        width: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .hero-card__art svg {{
        max-width: 100%;
        height: auto;
        filter: drop-shadow(0 0 10px rgba(34, 211, 238, 0.25));
    }}
    .hero-card--violet .hero-card__art svg {{
        filter: drop-shadow(0 0 10px rgba(124, 58, 237, 0.35));
    }}

    /* Chat bubbles. Compact padding, soft radius, subtle shadow for depth.
       The pixel cap + percentage fallback keeps text lines to roughly
       75-90 characters on a wide monitor (optimal readability) while
       still letting the bubble shrink naturally on narrow screens. */
    [data-testid="stChatMessage"] {{
        max-width: min(720px, 72%);
        border-radius: 16px;
        padding: 0.75rem 1.05rem;
        margin-bottom: 0.7rem;
        line-height: 1.55;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
        border: none;
    }}
    /* User bubble: accent gradient, right-aligned. */
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {{
        margin-left: auto;
        background: linear-gradient(135deg, #164e63 0%, #0E3B4E 100%);
        color: #E0F2FE;
        border: 1px solid rgba(34, 211, 238, 0.28);
    }}
    /* Assistant bubble: neutral surface, left-aligned. */
    .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background: #141925;
        border: 1px solid #1F2937;
        color: #E6EDF3;
    }}
    /* Slightly larger, rounder avatars. */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {{
        width: 32px !important;
        height: 32px !important;
        border-radius: 10px !important;
    }}

    /* ---------------------------------------------------------------
       Sidebar: persistent role switcher.
       ---------------------------------------------------------------
       Three stacked cards (Customer / Internal / Sales Team). The
       active role is rendered as Streamlit's "primary" button type so
       we can target it with a single CSS selector -- the alternative,
       nth-child targeting, breaks the moment anything else is added
       to the sidebar. That keeps the "glow on the selected card" logic
       robust even if we later append a "New chat" button, a version
       footer, etc. below the picker. */
    [data-testid="stSidebar"] {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, #0D121C 0%, #0B0F1A 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.08);
        /* Pin the sidebar to a fixed width so its content (brand tile,
           "Yntraa" label, role cards) never gets squished into a narrow
           strip where text wraps one character per line. Streamlit
           otherwise lets the user drag the resize handle down to ~20px,
           which turns the brand into a vertical ladder of letters. */
        min-width: 280px !important;
        max-width: 280px !important;
        width: 280px !important;
    }}
    [data-testid="stSidebar"]::before {{
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse 120% 80% at 0% 0%,
                rgba(34, 211, 238, 0.11) 0%, transparent 55%),
            radial-gradient(ellipse 100% 60% at 100% 85%,
                rgba(8, 145, 178, 0.07) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }}
    [data-testid="stSidebar"]::after {{
        content: "";
        position: absolute;
        inset: 0;
        opacity: 0.22;
        background-image: radial-gradient(
            rgba(148, 163, 184, 0.5) 1px, transparent 1px
        );
        background-size: 18px 18px;
        pointer-events: none;
        z-index: 0;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] .block-container {{
        position: relative;
        z-index: 1;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0.45rem;
        min-width: 280px;
    }}

    /* Kill the collapse button and the resize handle. The role switcher
       is the primary navigation of the app -- the sidebar is not an
       optional panel and shouldn't be collapsible, draggable, or
       resizable to a broken state. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarResizeHandle"],
    button[data-testid="stSidebarNavCollapseButton"],
    button[kind="headerNoPadding"] {{
        display: none !important;
    }}
    /* Neutralise any residual transform Streamlit applies when it thinks
       the sidebar is collapsed -- we never want it off-screen. */
    [data-testid="stSidebar"][aria-expanded="false"] {{
        transform: none !important;
        visibility: visible !important;
    }}

    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.25rem 0.4rem 0.9rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }}
    .sidebar-brand--wordmark {{
        flex-direction: column;
        align-items: flex-start;
        gap: 0.6rem;
        /* Pull the custom logo closer to the top of the sidebar. */
        padding-top: 0;
    }}
    .sidebar-wordmark {{
        width: 100%;
        line-height: 0;
    }}
    .sidebar-wordmark img {{
        width: 100%;
        max-height: 84px;
        object-fit: contain;
        object-position: left center;
        display: block;
        filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.4));
    }}
    .sidebar-brand .mark {{
        width: 34px; height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #22D3EE 0%, #0891B2 100%);
        color: #071018;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800;
        font-size: 1.05rem;
        box-shadow: 0 6px 18px rgba(34, 211, 238, 0.35);
        flex-shrink: 0;
    }}
    .sidebar-brand .mark--image {{
        background: linear-gradient(145deg, rgba(12, 20, 32, 0.96), rgba(7, 24, 38, 0.92));
        border: 1px solid rgba(34, 211, 238, 0.28);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35),
                    0 0 20px rgba(34, 211, 238, 0.15);
        padding: 3px;
    }}
    .sidebar-brand .mark--image img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 7px;
        display: block;
    }}
    .sidebar-brand .name {{
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #F1F5F9;
        font-size: 1rem;
    }}
    .sidebar-brand .tag {{
        color: #94A3B8;
        font-size: 0.75rem;
    }}

    .sidebar-top {{
        margin-bottom: 0.15rem;
    }}
    .sidebar-audience-block {{
        margin: 0.5rem 0.15rem 0.6rem;
        padding-left: 0.55rem;
        border-left: 2px solid rgba(34, 211, 238, 0.35);
    }}
    .sidebar-section-label {{
        color: #CBD5E1;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 0.2rem 0;
        font-weight: 600;
    }}
    .sidebar-section-hint {{
        color: #64748B;
        font-size: 0.68rem;
        line-height: 1.35;
        margin: 0 0 0.25rem 0;
    }}

    /* Default (inactive) role card -- Streamlit "secondary" buttons. */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        min-height: 52px;
        margin-bottom: 0.55rem;
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
        background: linear-gradient(180deg, #141B28 0%, #10151F 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #CBD5E1;
        font-size: 0.92rem;
        font-weight: 600;
        text-align: left;
        letter-spacing: 0.01em;
        transition: all 0.18s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: rgba(34, 211, 238, 0.45);
        background: linear-gradient(180deg, #17202F 0%, #121826 100%);
        color: #E6EDF3;
        transform: translateY(-1px);
    }}
    [data-testid="stSidebar"] .stButton > button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
    }}

    /* Active role card -- rendered as Streamlit "primary" kind. */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[kind="primaryFormSubmit"] {{
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.22), rgba(14, 116, 144, 0.22)) !important;
        border: 1.5px solid rgba(34, 211, 238, 0.75) !important;
        color: #E6FDFF !important;
        /* Double shadow: tight ring + outer bloom for the "glow" effect. */
        box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.30),
                    0 0 28px rgba(34, 211, 238, 0.45),
                    0 2px 6px rgba(0, 0, 0, 0.3) !important;
        animation: sidebar-role-pulse 2.6s ease-in-out infinite;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        /* Don't lift on hover -- the active card is already elevated.
           Just bump the glow slightly to acknowledge the hover. */
        transform: none;
        box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.40),
                    0 0 36px rgba(34, 211, 238, 0.55),
                    0 2px 6px rgba(0, 0, 0, 0.3) !important;
    }}
    @keyframes sidebar-role-pulse {{
        0%, 100% {{ box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.28),
                                0 0 24px rgba(34, 211, 238, 0.40),
                                0 2px 6px rgba(0, 0, 0, 0.3); }}
        50%      {{ box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.36),
                                0 0 34px rgba(34, 211, 238, 0.55),
                                0 2px 6px rgba(0, 0, 0, 0.3); }}
    }}

    /* Inactive role buttons: per-role left accent. Expects: sidebar markdown,
       then three ``stButton`` blocks as children 2--4 of the vertical block. */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:nth-child(2) .stButton > button[kind="secondary"] {{
        border-left: 3px solid #22D3EE !important;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:nth-child(3) .stButton > button[kind="secondary"] {{
        border-left: 3px solid #A78BFA !important;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:nth-child(4) .stButton > button[kind="secondary"] {{
        border-left: 3px solid #FB923C !important;
    }}

    .sidebar-footnote {{
        color: #64748B;
        font-size: 0.72rem;
        line-height: 1.5;
        margin-top: 1rem;
        padding: 0.6rem 0.2rem 0;
        border-top: 1px solid rgba(148, 163, 184, 0.08);
        display: flex;
        align-items: flex-start;
        gap: 0.4rem;
    }}
    .sidebar-footnote__icon {{
        flex-shrink: 0;
        width: 1.1rem;
        height: 1.1rem;
        margin-top: 0.12rem;
        color: #22D3EE;
    }}
    .sidebar-footnote p {{ margin: 0; }}

    /* Suggested-prompt chips -- shown once the role is picked and before
       the first question. Rendered via Streamlit buttons inside a
       .suggest-row wrapper. Look deliberately different from role cards
       (rounded pill, softer contrast) so users don't confuse them. */
    .suggest-label {{
        color: #94A3B8;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.6rem 0.1rem;
    }}
    .suggest-row .stButton > button {{
        width: 100%;
        padding: 0.6rem 0.9rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.04);
        border: 1px solid rgba(34, 211, 238, 0.22);
        color: #CBD5E1;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: center;
        line-height: 1.35;
        transition: all 0.15s ease;
        min-height: 42px;
    }}
    .suggest-row .stButton > button:hover {{
        background: rgba(34, 211, 238, 0.1);
        border-color: rgba(34, 211, 238, 0.5);
        color: #E6EDF3;
        transform: translateY(-1px);
    }}

    /* Typing indicator -- three dots that bounce while the model thinks.
       Shown inside the assistant bubble in place of the default Streamlit
       spinner, which is a generic gray ring that doesn't match the theme.

       IMPORTANT: the dot styles must be scoped to ``.typing-dots .dot``,
       NOT ``.typing-dots span``. The caption is also a <span>, and an
       unscoped selector gave it width:8px + the bounce animation, which
       collapsed the caption text into an 8-pixel column where every
       letter wrapped onto its own line. This was the "vertical ladder
       of letters next to the chat bubble" bug. */
    .typing-dots {{
        display: flex;
        align-items: center;
        flex-wrap: nowrap;
        gap: 10px;
        padding: 0.3rem 0;
        /* Flex item inside the chat bubble should take whatever width
           the bubble offers so the caption can flow normally to the
           right of the dots. */
        width: 100%;
        min-width: 0;
    }}
    .typing-dots .dot {{
        flex: 0 0 8px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22D3EE;
        opacity: 0.35;
        animation: typing-bounce 1.25s infinite ease-in-out;
        display: inline-block;
    }}
    .typing-dots .dot:nth-child(2) {{ animation-delay: 0.18s; }}
    .typing-dots .dot:nth-child(3) {{ animation-delay: 0.36s; }}
    @keyframes typing-bounce {{
        0%, 80%, 100% {{ transform: translateY(0); opacity: 0.35; }}
        40% {{ transform: translateY(-5px); opacity: 1; }}
    }}
    .typing-caption {{
        color: #94A3B8;
        font-size: 0.82rem;
        font-style: italic;
        /* Keep the caption on a single horizontal line regardless of the
           bubble width. If it ever doesn't fit, ellipsis rather than
           wrap -- wrapping made it interleave with the next message. */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex: 1 1 auto;
        min-width: 0;
        line-height: 1;
    }}

    /* New messages slide in from below rather than popping. Small motion
       but it makes the chat feel responsive instead of abrupt. */
    @keyframes fadeSlideIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    [data-testid="stChatMessage"] {{
        animation: fadeSlideIn 0.28s ease-out both;
    }}

    /* Logo tile breathes -- a subtle pulsing outer glow that signals the
       assistant is live without being distracting. Pure box-shadow, no
       repaint, so it's cheap. */
    @keyframes logo-pulse {{
        0%, 100% {{
            box-shadow: 0 8px 24px rgba(34, 211, 238, 0.18),
                        0 0 0 rgba(34, 211, 238, 0),
                        inset 0 1px 0 rgba(255, 255, 255, 0.12);
        }}
        50% {{
            box-shadow: 0 10px 28px rgba(34, 211, 238, 0.3),
                        0 0 32px rgba(34, 211, 238, 0.22),
                        inset 0 1px 0 rgba(255, 255, 255, 0.14);
        }}
    }}
    @keyframes app-icon-bob {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-2px); }}
    }}

    /* Source chips -- one per unique (document, page) citation. Now
       clickable <a> elements that open the cited PDF at the cited page
       in a new tab. Styling needs to *signal* clickability clearly
       (hover lift, pointer cursor, external-link glyph) so people don't
       need a tooltip to realise it's interactive. */
    .source-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }}
    .source-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.85rem;
        border-radius: 10px;
        background: rgba(34, 211, 238, 0.06);
        border: 1px solid rgba(34, 211, 238, 0.22);
        color: #B8C4D3;
        font-size: 0.8rem;
        text-decoration: none !important;
        transition: transform 0.15s ease, background 0.15s ease,
                    border-color 0.15s ease, box-shadow 0.15s ease;
        cursor: pointer;
        max-width: 100%;
    }}
    a.source-chip:hover {{
        background: rgba(34, 211, 238, 0.12);
        border-color: rgba(34, 211, 238, 0.55);
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(34, 211, 238, 0.1);
        color: #E6EDF3;
    }}
    a.source-chip:active {{
        transform: translateY(0);
        box-shadow: 0 2px 6px rgba(34, 211, 238, 0.12);
    }}
    .source-chip--inactive {{
        cursor: not-allowed;
        opacity: 0.75;
    }}
    .source-chip .pdf-glyph {{
        color: #F87171;
        background: rgba(248, 113, 113, 0.12);
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.66rem;
        letter-spacing: 0.04em;
        flex-shrink: 0;
    }}
    .source-chip .doc-name {{
        font-weight: 500;
        color: #E6EDF3;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .source-chip .page-tag {{
        color: #94A3B8;
        font-size: 0.72rem;
        flex-shrink: 0;
    }}
    /* Small up-right arrow -- the universal "opens in new tab" glyph. Only
       rendered for clickable chips (not the inactive fallback span). */
    .source-chip .chip-arrow {{
        color: #22D3EE;
        font-size: 0.72rem;
        margin-left: 0.1rem;
        opacity: 0.7;
        transition: opacity 0.15s ease, transform 0.15s ease;
        flex-shrink: 0;
    }}
    a.source-chip:hover .chip-arrow {{
        opacity: 1;
        transform: translate(1px, -1px);
    }}
    .source-section {{
        margin-top: 0.6rem;
    }}
    .source-section__title {{
        color: #94A3B8;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }}
    .source-cards {{
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }}
    .source-cards a.source-chip {{
        width: 100%;
        box-sizing: border-box;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.1),
            0 4px 18px rgba(0, 0, 0, 0.22);
    }}
    .conf-badge {{
        display: inline-block;
        font-size: 0.72rem;
        margin: 0.3rem 0 0.15rem;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
    }}
    .conf-badge .sm-text {{ font-weight: 600; }}
    .conf-badge--high {{
        color: #7dd3fc;
        background: rgba(34, 211, 238, 0.1);
        border: 1px solid rgba(34, 211, 238, 0.32);
    }}
    .conf-badge--mid {{
        color: #fde68a;
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.28);
    }}
    .conf-badge--low {{
        color: #fb923c;
        background: rgba(251, 146, 60, 0.08);
        border: 1px solid rgba(251, 146, 60, 0.28);
    }}
    .conf-badge--none {{
        color: #94a3b8;
        background: rgba(100, 116, 139, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.22);
    }}
    .followup-block__t {{
        color: #94A3B8;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.65rem 0 0.4rem 0.05rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Kill every piece of Streamlit chrome that shows in a dev/demo run.
       The "File change · Rerun · Always rerun" banner is a floating widget
       that Streamlit injects when it detects a source file edit; multiple
       test-ids have shipped across versions, so we nuke all the known ones
       *and* the container they sit in. Using display:none (not
       visibility:hidden) so they stop reserving layout space. */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stDeployButton"],
    [data-testid="stDecoration"],
    [data-testid="stAppToolbar"],
    [data-testid="stAppHeader"],
    [data-testid="manage-app-button"],
    [class*="viewerBadge"],
    [class*="_profileContainer"],
    #MainMenu,
    footer {{
        display: none !important;
    }}
    /* Belt-and-suspenders: some Streamlit builds put the rerun prompt in a
       floating notification at top-right that matches none of the test-ids
       above. Catch by aria/role as a fallback. */
    div[role="status"][aria-live="polite"]:has(button) {{
        display: none !important;
    }}

    /* Keep rendered content above the logo watermark. */
    .block-container > * {{ position: relative; z-index: 1; }}

    /* Chat input: width-capped, pill bar, cyan glow */
    [data-testid="stChatInput"] {{
        max-width: min(1280px, 95vw) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%);
        border: 1px solid rgba(34, 211, 238, 0.28) !important;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.1), 0 4px 20px rgba(0, 0, 0, 0.35);
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: rgba(34, 211, 238, 0.55) !important;
        box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.15), 0 0 24px rgba(34, 211, 238, 0.12);
    }}
    [data-testid="stChatInput"] textarea {{
        color: #E2E8F0 !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: #64748B !important;
    }}

    /* Snapshot of queued questions directly above the chat bar */
    .pending-queue-strip {{
        max-width: min(1280px, 95vw);
        margin: 0 auto 10px auto;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(34, 211, 238, 0.22);
        background: rgba(15, 23, 42, 0.82);
        font-size: 0.92rem;
        color: #E2E8F0;
    }}
    .pending-queue-strip .pq-head {{
        color: #67E8F9;
        font-weight: 600;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }}
    .pending-queue-strip .pq-row {{
        margin: 4px 0 0 0;
        padding-left: 0.35rem;
        line-height: 1.35;
    }}
    .pending-queue-strip .pq-idx {{
        color: #94A3B8;
        margin-right: 6px;
        font-variant-numeric: tabular-nums;
    }}

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
if "_response_time_ms" not in st.session_state:
    # Last few successful query wall times for the stat card average.
    st.session_state._response_time_ms = []
if "_query_queue" not in st.session_state:
    # FIFO of user questions when another answer is still being generated;
    # drained one per rerun so nothing is dropped on double-Enter or chip+type.
    st.session_state._query_queue = []
if "_queue_prompt_ack_needed" not in st.session_state:
    # When True, we show Continue / Cancel before dequeueing the next queued question.
    st.session_state._queue_prompt_ack_needed = False


# ---------------------------------------------------------------------------
# Brand header
# ---------------------------------------------------------------------------


def _indexed_document_count() -> int:
    """Report how many unique PDFs are currently in the vector store.

    Used as a trust signal in the header pill -- "Grounded in N documents"
    beats a generic "ready" message. Failures are swallowed: a missing
    index shouldn't stop the UI from rendering, and the status pill just
    falls back to a neutral "local & private" badge.
    """
    try:
        from VectorStore.faiss_store import unique_indexed_document_names

        return len(unique_indexed_document_names())
    except Exception:
        return 0


_SVG_CHECK = (
    '<svg class="status-pill__icon" width="12" height="12" viewBox="0 0 12 12" '
    'fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M10 3L4.5 8.5L2 6" stroke="currentColor" stroke-width="1.4" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_SVG_SHIELD = (
    '<svg class="status-pill__icon" width="12" height="12" viewBox="0 0 24 24" '
    'fill="currentColor" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>'
    "</svg>"
)


def _status_docs_pill(doc_count: int) -> str:
    """Return the HTML for the "Grounded in N documents" header pill."""
    if doc_count <= 0:
        return '<span class="status-pill neutral">Index not loaded</span>'
    label = "document" if doc_count == 1 else "documents"
    return (
        f'<span class="status-pill status-pill--cyan">{_SVG_CHECK}'
        f"<span>Grounded in {doc_count} {label}</span></span>"
    )


def _local_status_pill() -> str:
    """Second header pill: local-only, purple accent to pair with the cyan index pill."""
    return (
        f'<span class="status-pill status-pill--purple">{_SVG_SHIELD}'
        f"<span>100% local · no external APIs</span></span>"
    )


def _format_avg_s(times_ms: list[float]) -> str:
    """Format rolling average of response times, or a dash if none yet."""
    if not times_ms:
        return "—"
    s = (sum(times_ms) / len(times_ms)) / 1000.0
    return f"{s:.1f}s"


def _stats_strip_html(doc_count: int, times_ms: list[float]) -> str:
    """Four metric tiles below the status pills (mock-inspired dashboard row)."""
    dval = str(doc_count) if doc_count > 0 else "0"
    dsub = "Indexed document" if doc_count == 1 else "Indexed documents"
    return f"""
<div class="stat-strip" role="list">
  <div class="stat-card stat-card--cyan" role="listitem">
    <div class="stat-card__ic" aria-hidden="true">📚</div>
    <div class="stat-card__val">{dval}</div>
    <div class="stat-card__sub">{dsub}</div>
  </div>
  <div class="stat-card stat-card--emerald" role="listitem">
    <div class="stat-card__ic" aria-hidden="true">🔒</div>
    <div class="stat-card__val">100%</div>
    <div class="stat-card__sub">Local &amp; private</div>
  </div>
  <div class="stat-card stat-card--violet" role="listitem">
    <div class="stat-card__ic" aria-hidden="true">⚡</div>
    <div class="stat-card__val">{_format_avg_s(times_ms)}</div>
    <div class="stat-card__sub">Avg. response (session)</div>
  </div>
  <div class="stat-card stat-card--amber" role="listitem">
    <div class="stat-card__ic" aria-hidden="true">✓</div>
    <div class="stat-card__val">Cited</div>
    <div class="stat-card__sub">Source-backed answers</div>
  </div>
</div>"""


def _empty_state_hero_html(assistant: str) -> str:
    """Two feature-style cards for the no-role / welcome state (inline SVG art)."""
    a = escape(assistant)
    book_svg = (
        '<svg class="hero-illu" viewBox="0 0 120 100" width="100" height="85" '
        'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="b" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0" stop-color="#22d3ee" stop-opacity="0.5"/>'
        '<stop offset="1" stop-color="#0891b2" stop-opacity="0.15"/>'
        "</linearGradient></defs>"
        '<rect x="20" y="12" width="70" height="70" rx="6" fill="#0f172a" '
        'stroke="url(#b)" stroke-width="1.2"/>'
        '<line x1="32" y1="32" x2="78" y2="32" stroke="#64748b" stroke-width="1.2"/>'
        '<line x1="32" y1="45" x2="72" y2="45" stroke="#64748b" stroke-width="1.2"/>'
        '<line x1="32" y1="58" x2="68" y2="58" stroke="#64748b" stroke-width="1.2"/>'
        '<circle cx="88" cy="30" r="12" fill="none" stroke="#22d3ee" stroke-width="2.5"/>'
        '<line x1="99" y1="41" x2="104" y2="46" stroke="#22d3ee" stroke-width="2"/>'
        "</svg>"
    )
    people_svg = (
        '<svg class="hero-illu" viewBox="0 0 120 100" width="100" height="85" '
        'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="p" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0" stop-color="#a78bfa" stop-opacity="0.5"/>'
        '<stop offset="1" stop-color="#4c1d95" stop-opacity="0.2"/>'
        "</linearGradient></defs>"
        '<circle cx="38" cy="30" r="9" fill="#0f172a" stroke="url(#p)" stroke-width="1.2"/>'
        '<circle cx="60" cy="24" r="9" fill="#0f172a" stroke="url(#p)" stroke-width="1.2"/>'
        '<circle cx="82" cy="32" r="9" fill="#0f172a" stroke="url(#p)" stroke-width="1.2"/>'
        '<rect x="20" y="50" width="80" height="7" rx="2" fill="#1e1b4b" opacity="0.55"/>'
        "</svg>"
    )
    return f"""
<div class="hero-wrap">
  <div class="hero-card hero-card--cyan" role="region" aria-label="About the assistant">
    <div class="hero-card__text">
      <strong>Hello — I&apos;m {a}.</strong>
      I can answer questions grounded in your product documentation. Every answer comes
      with the exact source I used.
    </div>
    <div class="hero-card__art" aria-hidden="true">{book_svg}</div>
  </div>
  <div class="hero-card hero-card--violet" role="region" aria-label="Getting started">
    <div class="hero-card__text">
      <strong>Pick an audience to begin</strong>
      Choose <strong>Customer</strong> or <strong>Internal</strong> or <strong>Sales Team</strong>
      on the left. I&apos;ll match tone and detail to that role — and you can switch any time.
    </div>
    <div class="hero-card__art" aria-hidden="true">{people_svg}</div>
  </div>
</div>"""


_doc_count = _indexed_document_count()

_APP_SUBTITLE = "Knowledge Assistant · Grounded in your product documentation"
st.markdown(
    _app_header_block_html(config.ASSISTANT_NAME, _APP_SUBTITLE)
    + f"""
    <div class="status-row">
        {_status_docs_pill(_doc_count)}
        {_local_status_pill()}
    </div>
    {_stats_strip_html(_doc_count, st.session_state.get("_response_time_ms", []))}
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Source chip renderer
# ---------------------------------------------------------------------------

_ROLE_LABELS = {
    "customer": "Customer",
    "internal": "Internal",
    "sales": "Sales Team",
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


def _conf_badge_class_for_match(label: str) -> str:
    """Map API ``source_match`` to a theme class for the badge row."""
    if label == "No match":
        return "conf-badge--none"
    if label == "High":
        return "conf-badge--high"
    if label == "Low":
        return "conf-badge--low"
    return "conf-badge--mid"


def _render_source_section(sources: list[dict[str, Any]]) -> None:
    """Sources block: section title + one glowing row per (document, page)."""
    unique_sources = _dedupe_sources(sources)
    if not unique_sources:
        return

    parts: list[str] = [
        '<div class="source-section" role="group" aria-label="Document sources">',
        '<div class="source-section__title">Sources</div>',
        '<div class="source-cards">',
    ]
    for source in unique_sources:
        doc_name = Path(str(source.get("document_name", ""))).name or "unknown"
        safe_name = escape(doc_name)
        page_number = int(source.get("page_number", 0))
        page_tag = f"Page {page_number}" if page_number > 0 else "Page unknown"
        url_path = _PDF_URL_MAP.get(doc_name)
        if url_path:
            href = f"{url_path}#page={page_number}" if page_number > 0 else url_path
            parts.append(
                f'<a class="source-chip" href="{href}" target="_blank" '
                f'rel="noopener noreferrer" '
                f'title="Open {safe_name} at page {page_number} in a new tab">'
                '<span class="pdf-glyph">PDF</span>'
                f'<span class="doc-name">{safe_name}</span>'
                f'<span class="page-tag">· {page_tag}</span>'
                '<span class="chip-arrow" aria-hidden="true">↗</span>'
                "</a>"
            )
        else:
            parts.append(
                '<span class="source-chip source-chip--inactive" '
                f'title="PDF file could not be located: {safe_name}">'
                '<span class="pdf-glyph">PDF</span>'
                f'<span class="doc-name">{safe_name}</span>'
                f'<span class="page-tag">· {page_tag}</span>'
                "</span>"
            )
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_assistant_extras(turn: dict[str, Any], turn_index: int) -> None:
    """After the answer body: defensible source-match badge, reasoning expander, sources, follow-ups."""
    sm = turn.get("source_match")
    if sm:
        c = _conf_badge_class_for_match(sm)
        st.markdown(
            f'<div class="conf-badge {c}">'
            f'Source match: <span class="sm-text">{escape(sm)}</span></div>',
            unsafe_allow_html=True,
        )
    steps = turn.get("reasoning_steps")
    if steps:
        with st.expander("How this answer was built", expanded=False):
            for s in steps:
                st.markdown(f"• {s}")
    if turn.get("sources"):
        _render_source_section(turn["sources"])
    fups: list = turn.get("follow_ups") or []
    if fups:
        st.markdown(
            '<div class="followup-block"><p class="followup-block__t">You might also ask</p></div>',
            unsafe_allow_html=True,
        )
        fu_cols = st.columns(len(fups))
        for j, p in enumerate(fups):
            with fu_cols[j]:
                if st.button(
                    p,
                    key=f"fol_{turn_index}_{j}",
                    use_container_width=True,
                ):
                    st.session_state._pending_query = p
                    st.rerun()


# ---------------------------------------------------------------------------
# Role picker (only shown on first turn)
# ---------------------------------------------------------------------------


# Ordered list so the sidebar renders cards in a predictable top-to-bottom
# sequence. Tuple entries: (role_id, emoji, label, short description).
_ROLE_OPTIONS: list[tuple[str, str, str, str]] = [
    ("customer", "👤", "Customer", "External user — plain-language answers, no internal jargon."),
    ("internal", "🛠", "Internal", "Support / engineering / ops — technical detail welcome."),
    ("sales", "💼", "Sales Team", "Positioning, differentiators, and plan fit."),
]


def _seed_initial_role(picked_role: str) -> None:
    """Seed the greeting/ack messages on the *first* role pick of a session.

    Kept separate from ``_switch_role`` because the two have different
    conversational semantics: the first pick deserves a proper greeting
    and a "you can ask me anything" nudge, whereas subsequent switches
    are a quiet audience change -- we don't want to re-greet the user
    halfway through a conversation.
    """
    label = _ROLE_LABELS.get(picked_role, picked_role)
    st.session_state.messages.append({"role": "user", "content": f"*I'm here as **{label}**.*"})
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


def _switch_role(picked_role: str) -> None:
    """Update the active role and (if mid-conversation) log the switch.

    We append a lightweight italic system-note message rather than a
    full greeting so the transcript stays readable: the switch is a
    *meta* event, not a user turn. The note also makes it obvious in
    the transcript exactly where the audience changed, which matters
    when a reviewer later asks "why did the tone change here?".
    """
    if st.session_state.role == picked_role:
        # No-op click on the already-active card. Don't rerun or log.
        return

    is_first_pick = st.session_state.role is None
    st.session_state.role = picked_role

    if is_first_pick:
        _seed_initial_role(picked_role)
    else:
        label = _ROLE_LABELS.get(picked_role, picked_role)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"*Audience switched to **{label}**. "
                    "Future answers will be tailored accordingly.*"
                ),
            }
        )


def _render_sidebar() -> None:
    """Render the persistent role switcher in the left sidebar.

    The currently active role's button is rendered with
    ``type="primary"``, which lets a single CSS rule target
    ``button[kind="primary"]`` to apply the cyan glow. This is more
    robust than nth-child selectors because adding or removing
    sidebar elements doesn't shift the highlight.
    """
    with st.sidebar:
        st.markdown(
            _sidebar_top_block_html(config.ASSISTANT_NAME),
            unsafe_allow_html=True,
        )

        for role_id, emoji, label, _desc in _ROLE_OPTIONS:
            is_active = st.session_state.role == role_id
            if st.button(
                f"{emoji}  {label}  ›",
                key=f"sb_pick_{role_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=_desc,
            ):
                _switch_role(role_id)
                st.rerun()

        st.markdown(
            (
                '<div class="sidebar-footnote">'
                '<span class="sidebar-footnote__icon" aria-hidden="true">'
                '<svg viewBox="0 0 20 20" width="1em" height="1em" fill="currentColor" '
                'xmlns="http://www.w3.org/2000/svg" focusable="false">'
                '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 '
                '9a.75.75 0 000 1.5V13a.75.75 0 001.5 0V9.75A.75.75 0 009 9z"/>'
                "</svg></span>"
                "<p>Switch audience any time — the next answer adapts immediately.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _render_initial_greeting() -> None:
    """Main-area welcome while no role: wide hero cards (mock-style), not chat bubbles."""
    st.markdown(_empty_state_hero_html(config.ASSISTANT_NAME), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Suggested prompts -- shown once the role is picked and before the first
# real user question. Gives demo watchers an obvious starting point and
# doubles as visual proof that the bot is ready.
# ---------------------------------------------------------------------------

# Keep these grounded in what the PSS corpus actually covers. A suggested
# prompt the bot can't answer is worse than none at all -- it makes the
# first demo interaction a fallback message.
_SUGGESTED_PROMPTS = [
    "What is vDaaS and who is it for?",
    "What plans are available?",
    "How is data secured?",
    "What's the SLA commitment?",
]


def _render_suggested_prompts() -> None:
    """Render a row of one-click example questions.

    Clicking a chip stores the prompt in ``st.session_state._pending_query``
    and reruns; the main loop picks it up on the next frame and sends it
    through the normal query path. This routes suggested prompts through
    the exact same code the chat input uses, so there's no risk of the
    two paths drifting apart in how they render retrieval errors, sources,
    fabrication fallbacks, etc.
    """
    st.markdown('<div class="suggest-label">Try asking</div>', unsafe_allow_html=True)
    st.markdown('<div class="suggest-row">', unsafe_allow_html=True)
    columns = st.columns(len(_SUGGESTED_PROMPTS), gap="small")
    for column, prompt in zip(columns, _SUGGESTED_PROMPTS, strict=True):
        with column:
            if st.button(prompt, key=f"suggest_{prompt}", use_container_width=True):
                st.session_state._pending_query = prompt
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
    for idx, turn in enumerate(st.session_state.messages):
        avatar = "🧑" if turn["role"] == "user" else "💬"
        with st.chat_message(turn["role"], avatar=avatar):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                _render_assistant_extras(turn, idx)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

# Sidebar first so its widgets are registered before the main-area
# widgets call st.rerun(). A click on a role card flips session state,
# reruns, and the main body re-renders on the next frame with the new
# audience in effect.
_render_sidebar()

_replay_history()

_render_queue_continuation_gate()

if st.session_state.role is None:
    _render_initial_greeting()
elif not any(
    msg["role"] == "user" and not msg["content"].startswith("*")
    for msg in st.session_state.messages
):
    # Role picked, but the user hasn't typed a real question yet (the
    # seeded "*I'm here as Customer*" messages start with "*" and don't
    # count). Surfacing suggested prompts here gives demo watchers an
    # obvious starting point and makes the bot feel pre-loaded.
    _render_suggested_prompts()


# ---------------------------------------------------------------------------
# Chat input (disabled until a role is picked)
# ---------------------------------------------------------------------------

_pending_queue_banner = st.empty()

chat_placeholder = (
    f"Ask {config.ASSISTANT_NAME} anything about the product documentation…"
    if st.session_state.role
    else "Pick a role in the sidebar to start chatting"
)

typed_prompt = st.chat_input(chat_placeholder, disabled=st.session_state.role is None)

# Suggested chip / follow-up buttons set _pending_query and rerun; pop so it fires once.
pending_prompt = st.session_state.pop("_pending_query", None)
# Typed input and pending chip both enqueue; order is typed first, then pending.
_enqueue_user_queries(typed_prompt, pending_prompt)
queued_snap = list(st.session_state._query_queue)
with _pending_queue_banner.container():
    _render_pending_queue_strip(queued_snap)

user_prompt = st.session_state._query_queue.pop(0) if st.session_state._query_queue else None

if user_prompt:
    still_queued = len(st.session_state._query_queue)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)
        if still_queued:
            st.caption(
                f"{still_queued} more question{'s' if still_queued != 1 else ''} "
                "in line — after this answer you can continue or ✕ cancel before the next runs."
            )

    with st.chat_message("assistant", avatar="💬"):
        # Custom typing indicator in place of Streamlit's generic gray
        # spinner. Using st.empty() so we can overwrite the dots with the
        # real answer in the same bubble -- no flashing "spinner bubble
        # disappears / new bubble appears" transition.
        answer_slot = st.empty()
        answer_slot.markdown(
            '<div class="typing-dots">'
            '<span class="dot"></span>'
            '<span class="dot"></span>'
            '<span class="dot"></span>'
            '<span class="typing-caption">'
            f"{config.ASSISTANT_NAME} is reading the documentation…"
            "</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        _q_t0 = time.perf_counter()
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
            answer_slot.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )
            if st.session_state._query_queue:
                _schedule_queue_ack_or_rerun()
        else:
            _elapsed_s = time.perf_counter() - _q_t0
            st.session_state._response_time_ms = (
                st.session_state.get("_response_time_ms", []) + [_elapsed_s * 1000.0]
            )[-5:]
            answer_text = (
                response.get("answer", "").strip()
                or "I don't have enough information in the documents to answer that confidently."
            )
            answer_slot.markdown(answer_text)
            out_of_scope = bool(response.get("out_of_scope", False))
            retrieved_sources = [] if out_of_scope else (response.get("sources", []) or [])
            turn_idx = len(st.session_state.messages)
            asst_msg: dict[str, Any] = {
                "role": "assistant",
                "content": answer_text,
                "sources": retrieved_sources,
                "source_match": response.get("source_match"),
                "reasoning_steps": response.get("reasoning_steps") or [],
                "follow_ups": response.get("follow_ups") or [],
                "out_of_scope": out_of_scope,
            }
            _render_assistant_extras(asst_msg, turn_idx)
            st.session_state.messages.append(asst_msg)
            if st.session_state._query_queue:
                _schedule_queue_ack_or_rerun()
