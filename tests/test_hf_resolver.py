"""Tests for resolve_hf_repo's name-match gate (composer-redesign v2).

Network-free: HfApi.list_datasets is monkeypatched with a fake returning fixed
candidates, and the manual override table is emptied so the search path runs.
"""

import pytest

from auto_benchmarkcard.tools.eee import eee_tool


class _FakeDataset:
    def __init__(self, id, downloads):
        self.id = id
        self.downloads = downloads


def _fake_hf_api(datasets):
    class _FakeApi:
        def list_datasets(self, search=None, sort=None, limit=None):
            return list(datasets)
    return _FakeApi


def test_exact_name_match_accepts_low_download_dataset(monkeypatch):
    # BioLP-Bench -> baceolus/biolp-bench is an exact normalized-name match: accepted despite
    # only 7 downloads (below MIN_HF_DOWNLOADS).
    monkeypatch.setattr(eee_tool, "HfApi", _fake_hf_api([_FakeDataset("baceolus/biolp-bench", 7)]))
    assert eee_tool.resolve_hf_repo("BioLP-Bench") == "baceolus/biolp-bench"


def test_malformed_eee_repo_falls_through_to_search(monkeypatch):
    # A placeholder pseudo-id from the EEE data (example://...) is not a repo: it must not be
    # returned (the HF API can only raise on it) and must not block the search path.
    monkeypatch.setattr(eee_tool, "HfApi",
                        _fake_hf_api([_FakeDataset("grimulkan/theory-of-mind", 9000)]))
    assert eee_tool.resolve_hf_repo(
        "theory-of-mind", existing_hf_repo="example://theory_of_mind") == "grimulkan/theory-of-mind"


def test_wellformed_eee_repo_still_short_circuits(monkeypatch):
    def _boom():  # pragma: no cover - search must not run
        raise AssertionError("search must not run when EEE provides a valid repo")
    monkeypatch.setattr(eee_tool, "HfApi", _boom)
    assert eee_tool.resolve_hf_repo("x", existing_hf_repo="allenai/quac") == "allenai/quac"


def test_token_low_download_match_stays_gated(monkeypatch):
    # "BioLP" is a whole token of "biolp-bench" but not an exact match -> the download gate applies,
    # and 7 downloads is below MIN_HF_DOWNLOADS, so nothing resolves.
    monkeypatch.setattr(eee_tool, "HfApi", _fake_hf_api([_FakeDataset("baceolus/biolp-bench", 7)]))
    assert eee_tool.resolve_hf_repo("BioLP") is None


def test_token_high_download_match_resolves(monkeypatch):
    # "BioLP" (5 chars, distinctive) is a whole token of "biolp-bench": a token-boundary match,
    # not a mid-word slice, so it resolves once downloads clear the gate.
    monkeypatch.setattr(eee_tool, "HfApi", _fake_hf_api([_FakeDataset("baceolus/biolp-bench", 1000)]))
    assert eee_tool.resolve_hf_repo("BioLP") == "baceolus/biolp-bench"


def test_short_name_substring_never_fabricates(monkeypatch):
    # Regression: "arc" is a mid-word substring of "banned-historical-archives" but not a whole
    # token -> must never resolve to it, even at very high downloads.
    monkeypatch.setattr(
        eee_tool, "HfApi",
        _fake_hf_api([_FakeDataset("someorg/banned-historical-archives", 99999)]),
    )
    assert eee_tool.resolve_hf_repo("arc") is None


def test_short_name_requires_exact_even_as_token(monkeypatch):
    # "arc" IS a whole token of "ai2_arc", but a <=4 char name is too ambiguous for token overlap
    # and requires an exact normalized match, so the high-download token candidate is rejected.
    monkeypatch.setattr(eee_tool, "HfApi", _fake_hf_api([_FakeDataset("allenai/ai2_arc", 99999)]))
    assert eee_tool.resolve_hf_repo("arc") is None


def test_short_name_exact_match_still_resolves(monkeypatch):
    # A short name that exactly matches the dataset name still resolves (exact tier, any downloads).
    monkeypatch.setattr(eee_tool, "HfApi", _fake_hf_api([_FakeDataset("ought/raft", 3)]))
    assert eee_tool.resolve_hf_repo("RAFT") == "ought/raft"


def test_delimiterless_prefix_is_not_a_token_match(monkeypatch):
    # When the dataset name has no delimiter ("biolpbench" is one token), "biolp" is a mid-word
    # prefix, not a whole token, so even at high downloads it does not resolve.
    monkeypatch.setattr(eee_tool, "HfApi", _fake_hf_api([_FakeDataset("baceolus/biolpbench", 9000)]))
    assert eee_tool.resolve_hf_repo("BioLP") is None
