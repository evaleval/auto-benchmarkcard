"""Tests for the HF/RAG source-widening of three Stage-A curator fields.

methodology.validation / validity_justification / calculation were paper-only, so they went
Not-specified for no-paper cards (e.g. aa-lcr) even when the HF card carried the content. They
now also draw from html + hf via CURATOR_FIELDS["...']["sources"], picked up by fields_for_source.
The verbatim verifier is unchanged, so widening cannot introduce fabrication.
"""

import pytest

from auto_benchmarkcard.tools.composer import evidence as E

WIDENED = [
    "methodology.validation",
    "methodology.validity_justification",
    "methodology.calculation",
]


@pytest.mark.parametrize("field", WIDENED)
@pytest.mark.parametrize("source", ["hf", "html"])
def test_widened_fields_drawn_from_hf_and_html(field, source):
    assert field in E.fields_for_source(source)


@pytest.mark.parametrize("field", WIDENED)
def test_widened_fields_still_drawn_from_paper(field):
    assert field in E.fields_for_source("paper")          # regression: paper not lost


def test_size_breakdown_already_hf():
    # No-change confirmation: contrary to the audit's "paper-only", base already allows hf.
    assert "data.size_breakdown" in E.fields_for_source("hf")


def test_similar_benchmarks_now_drawn_from_html_and_hf():
    # was paper-only; comparisons named in README/HTML are now reachable (no lineage mining)
    for source in ("paper", "html", "hf"):
        assert "benchmark_details.similar_benchmarks" in E.fields_for_source(source)


def test_similar_benchmarks_gap_query_sharpened():
    q = E.CURATOR_FIELDS["benchmark_details.similar_benchmarks"]["gap_query"]
    for term in ("compared to", "unlike", "builds on"):
        assert term in q
