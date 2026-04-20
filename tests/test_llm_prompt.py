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
    # Accept either the old or new phrasing so tightening the wording later
    # doesn't keep breaking this test. What matters is that the anti-invention
    # rule is present somewhere in the prompt.
    assert "do not invent" in lower or "do not hallucinate" in lower


def test_prompt_puts_question_before_context():
    """Truncation safety: Flan-T5's 512-token encoder window means overflow
    drops the *end* of the prompt. Keeping the question ahead of the context
    guarantees the actual task survives truncation even on long contexts.
    """
    prompt = _build_prompt(
        query="MARKER_QUESTION",
        context_blocks=["MARKER_CONTEXT"],
        role="customer",
    )
    assert prompt.index("MARKER_QUESTION") < prompt.index("MARKER_CONTEXT")
    # And "Answer:" should trail everything else so the model's generation
    # cue isn't buried mid-prompt.
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_separates_multiple_context_blocks():
    prompt = _build_prompt(
        query="q",
        context_blocks=["chunk one", "chunk two", "chunk three"],
        role="customer",
    )
    for chunk in ("chunk one", "chunk two", "chunk three"):
        assert chunk in prompt
    assert prompt.count("---") >= 2
