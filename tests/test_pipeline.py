"""Pipeline filesystem helpers (no PDF loading)."""

from __future__ import annotations

from pipeline import list_pdf_files_under


def test_list_pdf_files_under_recursive(tmp_path):
    root = tmp_path / "documents"
    (root / "PSS").mkdir(parents=True)
    (root / "PSS" / "PSS_a.pdf").write_bytes(b"%PDF")
    (root / "FAQ_b.PDF").write_bytes(b"%PDF")
    found = list_pdf_files_under(root)
    assert len(found) == 2
