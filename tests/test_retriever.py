"""Retriever ranking: mock embedding + FAISS to avoid loading models."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import config
from Retrieval import retriever as r


@pytest.fixture
def isolated_vector_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    vs = tmp_path / "vs"
    vs.mkdir()
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", vs)
    monkeypatch.setattr(config, "FAISS_INDEX_FILE", vs / "index.faiss")
    monkeypatch.setattr(config, "FAISS_METADATA_FILE", vs / "metadata.json")


def test_infer_document_type_from_prefix():
    assert r.infer_document_type_from_name("PSS_x.pdf") == "PSS"
    assert r.infer_document_type_from_name("FAQ_shipping.pdf") == "FAQ"
    assert r.infer_document_type_from_name("dir/FAQ_shipping.pdf") == "FAQ"


def test_infer_from_folder_path_matches_pss():
    assert r.infer_document_type_from_name("PSS/motor_overview.pdf") == "PSS"
    assert r.infer_document_type_from_name("pss/anything.pdf") == "PSS"


def test_infer_more_specific_subfolder_wins():
    """PSS/FAQs/whatever.pdf must resolve to FAQ, not PSS."""
    assert r.infer_document_type_from_name("PSS/FAQs/returns.pdf") == "FAQ"
    assert r.infer_document_type_from_name("PSS/FAQ/returns.pdf") == "FAQ"


def test_infer_from_full_form_synonyms_in_filename():
    assert r.infer_document_type_from_name("MotorXYZ_ProductSpecSheet.pdf") == "PSS"
    assert r.infer_document_type_from_name("BoilerABC_FrequentlyAskedQuestions.pdf") == "FAQ"


def test_infer_token_anywhere_in_basename():
    """The legacy heuristic only checked the leading token; now any token works."""
    assert r.infer_document_type_from_name("Acme-Pump-PSS-v3.pdf") == "PSS"
    assert r.infer_document_type_from_name("returns_FAQ_2024.pdf") == "FAQ"


def test_infer_falls_back_to_default():
    assert r.infer_document_type_from_name("random_file.pdf") == "default"
    assert r.infer_document_type_from_name("nested/folder/random.pdf") == "default"


def test_infer_handles_pathlib_input():
    from pathlib import Path

    assert r.infer_document_type_from_name(Path("PSS") / "FAQs" / "returns.pdf") == "FAQ"


def test_retrieve_applies_weight_boost(isolated_vector_store, monkeypatch):
    monkeypatch.setattr(config, "TOP_K", 2)
    monkeypatch.setattr(config, "RETRIEVAL_CANDIDATES", 10)

    def fake_search(query_vector, k):
        return [
            {
                "score": 1.0,
                "metadata": {
                    "text": "low weight",
                    "document_name": "manual_x.pdf",
                    "document_type": "manual",
                    "weight": 1.0,
                },
            },
            {
                "score": 0.9,
                "metadata": {
                    "text": "high weight pss",
                    "document_name": "PSS_x.pdf",
                    "document_type": "PSS",
                    "weight": 1.5,
                },
            },
        ]

    with patch.object(r, "embed_text", return_value=[0.0] * config.EMBEDDING_DIMENSION):
        with patch.object(r, "search", side_effect=fake_search):
            chunks = r.retrieve_for_query("test query")
    assert len(chunks) == 2
    # PSS: 0.9 * 1.5 = 1.35 vs manual: 1.0 * 1.0 = 1.0 -> PSS first
    assert chunks[0].document_type == "PSS"
    assert chunks[0].boosted_score >= chunks[1].boosted_score
