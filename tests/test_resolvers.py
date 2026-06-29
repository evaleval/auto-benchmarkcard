"""Offline unit tests for the curated override tables and name normalization.

These exercise the deterministic short-circuit paths (KNOWN_PAPERS and
HF_REPO_OVERRIDES) and the name-normalization helpers. They make no network
or LLM calls, so they are fast and safe to run anywhere.
"""

from auto_benchmarkcard.tools.eee.eee_tool import (
    _normalize_benchmark_name,
    resolve_hf_repo,
)
from auto_benchmarkcard.tools.eee.paper_resolver import (
    KNOWN_PAPERS,
    _normalize_key,
    resolve_paper,
)


def test_known_papers_shortcircuit_returns_table_url():
    # A name in the curated table resolves from it with no API/LLM call.
    result = resolve_paper(suite_name="gsm8k")
    assert isinstance(result, dict)
    assert result["url"] == KNOWN_PAPERS["gsm8k"]


def test_normalize_key_folds_case_and_separators():
    assert _normalize_key("MMLU-Pro") == "mmlu_pro"
    assert _normalize_key("MMLU Pro") == "mmlu_pro"
    assert _normalize_key("MMLU-Pro") in KNOWN_PAPERS


def test_hf_repo_override_returns_curated_repo():
    assert resolve_hf_repo("math") == "hendrycks/competition_math"


def test_hf_repo_override_known_unmatchable_returns_none():
    assert resolve_hf_repo("wmt 2014") is None


def test_normalize_benchmark_name_merges_variants():
    assert _normalize_benchmark_name("MMLU-PRO") == _normalize_benchmark_name("MMLU Pro")
