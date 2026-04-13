"""Config helpers and document naming rules."""

from __future__ import annotations

import re

import pytest

import config


def test_valid_doc_names_match_pattern():
    assert config.is_valid_doc_name("PSS_motor_v1.pdf") is True
    assert config.is_valid_doc_name("manual_installation.pdf") is True
    assert config.is_valid_doc_name("guide_quickstart.pdf") is True


def test_invalid_doc_names():
    assert config.is_valid_doc_name("random.pdf") is False
    assert config.is_valid_doc_name("PSS.pdf") is False


def test_doc_name_pattern_is_valid_regex():
    re.compile(config.DOC_NAME_PATTERN)
