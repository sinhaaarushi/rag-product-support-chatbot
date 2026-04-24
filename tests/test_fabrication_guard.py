"""Unit tests for the fabrication detector.

The detector is the last line of defence against the model inventing
placeholder figures when the retrieved context doesn't actually contain a
number the user asked for (prices, SLA percentages, capacities, etc.).

These tests split into two halves:

  1. *must trip*  -- sample answers that are clearly hallucinated and should
     be swapped for the support-email fallback at generation time.
  2. *must pass*  -- sample answers containing legitimate, document-grounded
     figures (real INR amounts, percentages, SKU names). The detector must
     NOT flag these, otherwise we replace a good answer with a fallback --
     a worse failure mode than the one we're guarding against.

If either half fails, we'd rather find out here than in a demo.
"""

from __future__ import annotations

import pytest

from LLM.llm_response import _looks_like_fabricated_answer


@pytest.mark.parametrize(
    "answer",
    [
        "Personal Desktop costs $X per month.",
        "The plan is priced at $ Y per user.",
        "Pricing is ₹X per month for the basic tier.",
        "The charge is ₹ Z per user per month.",
        "It costs Rs. X per month.",
        "The fee is Rs X per user.",
        "Monthly cost: INR Y per seat.",
        "Monthly cost: INR  Z per user.",
        "The price is XX.XX per month.",
        "The price is XXX.99 for the Large plan.",
        "Contact sales -- the rate is [TBD] at the moment.",
        "Licensing is [price] per user per month.",
        "The figure is <price> per seat.",
        "Bandwidth quota is <amount> GB per month.",
    ],
)
def test_detector_trips_on_obvious_placeholders(answer: str) -> None:
    assert _looks_like_fabricated_answer(answer) is True, (
        f"expected detector to trip on: {answer!r}"
    )


@pytest.mark.parametrize(
    "answer",
    [
        # Real INR figures with the rupee symbol, different formats.
        "The plan costs ₹5,000 per month for the Personal tier.",
        "Add-on data is ₹50 per GB per month.",
        "Large plan starts at ₹12,500 monthly.",
        # Same in text form -- must remain untouched.
        "The plan costs Rs. 5,000 per month.",
        "Monthly fee is INR 1,200 for the basic tier.",
        # Percentages / SLAs -- nothing currency-like at all.
        "The SLA guarantees 99.9% uptime.",
        "Storage is 100 GB root disk and 100 GB data disk.",
        "The plan provides 16 vCPUs and 32 GB RAM.",
        # SKU names containing single X's -- must not false-positive on the
        # "XX" pattern because they only have one X in a row.
        "The VDaaSGP.1X-Large plan has 4 vCPUs and 16 GB RAM.",
        "Available tiers: VDaaSCI.2X-Large and VDaaSCI.4X-Large.",
        "vHDaaSCI.4X-Large supports 16 vCPUs / 32 GB RAM.",
        # Documentation excerpts that mention the word "price" but
        # contain no placeholder bracket syntax.
        "All prices are in INR and exclusive of taxes and statutory levies.",
        "Pricing is governed by the latest price schedule on the official portal.",
        # Legitimate USD mention (edge case: some docs do have $ in them).
        "Comparable AWS Workspaces plans start at $23 per user per month.",
    ],
)
def test_detector_does_not_trip_on_legitimate_answers(answer: str) -> None:
    assert _looks_like_fabricated_answer(answer) is False, (
        f"expected detector to stay quiet on: {answer!r}"
    )


def test_detector_handles_empty_input_gracefully() -> None:
    """Empty / None-ish answers are a no-op: no fabrication to flag, but we
    also shouldn't crash or raise. The caller will handle "empty answer"
    with its own fallback path."""
    assert _looks_like_fabricated_answer("") is False
    assert _looks_like_fabricated_answer("   ") is False
