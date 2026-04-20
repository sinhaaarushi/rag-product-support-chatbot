"""Pure-function tests for the grounded prompt builder.

These do not load any model: they exercise the prompt template only,
which is the part most likely to regress under refactors.
"""

from __future__ import annotations

from LLM.llm_response import _build_prompt, _role_instructions


def test_role_instructions_distinct_for_customer_and_sales():
    customer = _role_instructions("customer")
    sales = _role_instructions("sales")
    assert customer != sales
    assert "customer" in customer.lower()
    assert "sales" in sales.lower()


def test_prompt_includes_query_and_context():
    prompt = _build_prompt(
        query="What is the warranty period?",
        context_blocks=["Warranty: 2 years from purchase."],
        role="customer",
    )
    assert "What is the warranty period?" in prompt
    assert "Warranty: 2 years from purchase." in prompt
    assert "Answer:" in prompt


def test_prompt_with_no_context_uses_placeholder():
    prompt = _build_prompt(query="Anything?", context_blocks=[], role="customer")
    assert "(no context)" in prompt


def test_prompt_grounding_rules_present():
    prompt = _build_prompt(query="x", context_blocks=["y"], role="sales")
    lower = prompt.lower()
    assert "only based on the provided context" in lower
    assert "do not hallucinate" in lower


def test_prompt_separates_multiple_context_blocks():
    prompt = _build_prompt(
        query="q",
        context_blocks=["chunk one", "chunk two", "chunk three"],
        role="customer",
    )
    for chunk in ("chunk one", "chunk two", "chunk three"):
        assert chunk in prompt
    assert prompt.count("---") >= 2
