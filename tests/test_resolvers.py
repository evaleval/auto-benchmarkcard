"""Offline unit tests for HF-repo resolution fast-path and name normalization.

These exercise the deterministic paths that make no network or LLM calls: the
existing-repo passthrough in resolve_hf_repo, and the name-normalization helper.
(The curated KNOWN_PAPERS / HF_REPO_OVERRIDES tables were removed, so paper and
HF-repo resolution now always go through live search.)
"""

from auto_benchmarkcard.tools.eee.eee_tool import (
    _normalize_benchmark_name,
    resolve_hf_repo,
)


def test_existing_hf_repo_is_returned_without_search():
    # When EEE already provides a repo, it is returned directly (no API call).
    assert resolve_hf_repo("anything", existing_hf_repo="allenai/quac") == "allenai/quac"


def test_normalize_benchmark_name_merges_variants():
    assert _normalize_benchmark_name("MMLU-PRO") == _normalize_benchmark_name("MMLU Pro")
    assert _normalize_benchmark_name("MMLU_Pro") == _normalize_benchmark_name("mmlu pro")
