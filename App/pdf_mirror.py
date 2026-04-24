"""Mirror the source PDFs into ``App/static/pdfs/`` for Streamlit to serve.

The Streamlit dashboard cites sources as clickable chips that link to the
PDF at a specific page. Browsers block ``file://`` links from an ``http://``
origin, so we can't just point at the file on disk -- the PDFs have to be
reachable over HTTP. Streamlit's built-in static-file serving handles that
as long as the files live under ``<main_script_dir>/static/`` (the folder
name is hardcoded to ``static`` inside Streamlit itself).

Why a separate module instead of doing this inside ``dashboard.py``:

1. Streamlit checks for the existence of the static folder at server
   *startup*, before the app script runs. If the folder isn't there yet
   it logs a warning and (in large-folder cases) disables static serving
   outright. Running the mirror as a *pre-launch* step from
   ``run_local.bat`` means the folder is guaranteed to exist before
   Streamlit boots.
2. It's also nice to be able to rebuild the mirror from a plain Python
   REPL or a CI job without dragging in Streamlit and torch and
   sentence-transformers.

Mirrors are done with hardlinks where the filesystem supports them (same
bytes on disk, two directory entries, zero extra storage). On a
filesystem that refuses hardlinks (e.g. across drives on Windows) we
fall back to a real ``shutil.copy2``. Either way the resulting
``App/static/pdfs/<basename>.pdf`` is a read-only mirror of the source
tree, safe to delete and regenerate.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)

# All paths are resolved relative to this file so the function works the
# same whether it's invoked from the project root, from ``App/``, or from
# a background process started elsewhere.
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent

# Source tree: every PDF under Data/documents/ (PSS, FAQs, etc.)
SOURCE_ROOT = _PROJECT_ROOT / "Data" / "documents"
# Destination: the ``static`` folder that Streamlit looks for at startup.
# Must sit next to the main script (``App/dashboard.py``) -- see
# ``streamlit.file_util.get_app_static_dir`` which hardcodes
# ``Path(main_script_path).parent / "static"``.
STATIC_ROOT = _APP_DIR / "static" / "pdfs"
# URL prefix used by the chip renderer. Absolute path so it resolves
# correctly regardless of the caller's current location in the app.
URL_PREFIX = "/app/static/pdfs"


def mirror_pdfs() -> dict[str, str]:
    """Mirror every PDF under ``Data/documents/`` into ``App/static/pdfs/``.

    Returns a ``{basename: relative_url_path}`` map the UI can use to
    resolve a document name coming back from retrieval into the URL that
    opens the file in a new tab. Idempotent: re-running is a cheap no-op
    because existing links are detected via ``samefile`` and skipped.

    Name collisions (two PDFs with the same basename but in different
    source subfolders) are resolved by prefixing the parent directory --
    e.g. ``PSS__Foo.pdf`` alongside ``FAQs__Foo.pdf`` -- so both remain
    reachable.
    """
    if not SOURCE_ROOT.is_dir():
        logger.warning("PDF source root not found: %s", SOURCE_ROOT)
        return {}

    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    linked = copied = skipped = 0
    for pdf_path in sorted(SOURCE_ROOT.rglob("*.pdf")):
        basename = pdf_path.name
        dest_name = basename
        dest = STATIC_ROOT / dest_name

        # Collision with a different file -- namespace by parent folder.
        if dest.exists() and not dest.samefile(pdf_path):
            dest_name = f"{pdf_path.parent.name}__{basename}"
            dest = STATIC_ROOT / dest_name

        if dest.exists():
            skipped += 1
        else:
            try:
                # Hardlink: no duplicate bytes, no admin rights required,
                # instantly visible to any HTTP server that walks the tree.
                os.link(pdf_path, dest)
                linked += 1
            except OSError:
                # Fallback path for cross-device mirrors or filesystems
                # that don't support hardlinks (rare on NTFS but possible
                # across drive letters).
                shutil.copy2(pdf_path, dest)
                copied += 1

        # URL-encode each segment so filenames with spaces and parens
        # (common in vendor PDFs) survive the round-trip to the browser.
        mapping[basename] = f"{URL_PREFIX}/{quote(dest_name)}"

    logger.info(
        "pdf mirror complete: linked=%d copied=%d skipped=%d total=%d dest=%s",
        linked,
        copied,
        skipped,
        len(mapping),
        STATIC_ROOT,
    )
    return mapping


if __name__ == "__main__":
    # Invoked by ``run_local.bat`` as a pre-launch step so the static
    # folder exists before Streamlit starts its server. Logging output
    # is intentionally terse -- this runs on every boot.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = mirror_pdfs()
    print(f"pdf mirror ready: {len(result)} file(s) under {STATIC_ROOT}")
