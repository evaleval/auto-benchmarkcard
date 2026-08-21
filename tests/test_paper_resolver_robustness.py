"""Paper-resolver robustness: author-fetch retry/cache (F1-3) and name-token recovery (F1-2).

All network-free: requests.get is monkeypatched and time.sleep is stubbed so retries don't
delay the suite. The F1-2 cases drive resolve_paper with the search, metadata and LLM boundaries
mocked, so only the in-resolver acceptance logic is exercised.
"""

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr


class _Resp:
    def __init__(self, *, text="", json_data=None, status=200):
        self._text, self._json, self.status_code = text, json_data, status

    @property
    def text(self):
        return self._text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>BigCodeBench: Benchmarking Code Generation with Diverse Function Calls</title>
    <published>2024-06-22T17:59:59Z</published>
    <author><name>Terry Yue Zhuo</name></author>
    <author><name>Minh Chien Vu</name></author>
  </entry>
</feed>"""

_ARXIV_ATOM_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


@pytest.fixture(autouse=True)
def _no_sleep_clean_cache(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._author_cache.clear()


# --------------------------------------------------------------------------- F1-3 retry/cache
def test_arxiv_meta_retries_then_succeeds(monkeypatch):
    # Two transient network errors then a good response, all within _http_get's retry budget.
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return _Resp(text=_ARXIV_ATOM)

    monkeypatch.setattr(pr.requests, "get", flaky)
    meta = pr._fetch_arxiv_meta("2406.15877")
    assert meta["authors"] == ["Terry Yue Zhuo", "Minh Chien Vu"]
    assert meta["year"] == 2024
    assert calls["n"] == 3


def test_arxiv_meta_retries_empty_parse(monkeypatch):
    # A 200 with an empty Atom feed (no <entry>) is a transient miss: retry once, then succeed.
    calls = {"n": 0}

    def resp_fn(*a, **k):
        calls["n"] += 1
        return _Resp(text=_ARXIV_ATOM_EMPTY) if calls["n"] == 1 else _Resp(text=_ARXIV_ATOM)

    monkeypatch.setattr(pr.requests, "get", resp_fn)
    meta = pr._fetch_arxiv_meta("2406.15877")
    assert meta["authors"] == ["Terry Yue Zhuo", "Minh Chien Vu"]
    assert calls["n"] == 2


def test_successful_fetch_is_cached(monkeypatch):
    calls = {"n": 0}

    def ok(*a, **k):
        calls["n"] += 1
        return _Resp(text=_ARXIV_ATOM)

    monkeypatch.setattr(pr.requests, "get", ok)
    url = "https://arxiv.org/abs/2406.15877"
    first = pr._fetch_paper_authors(url)
    assert first["authors"] == ["Terry Yue Zhuo", "Minh Chien Vu"]
    assert url in pr._author_cache
    n = calls["n"]
    second = pr._fetch_paper_authors(url)
    assert second == first and calls["n"] == n  # served from cache, no new fetch


# --------------------------------------------------------------------------- F1-2 / F1-3 resolve_paper
def _mock_resolver(monkeypatch, candidates, llm_result, *, full_name="AlpacaEval 2.0"):
    """Wire resolve_paper's boundaries: metadata, search source, prefilter, LLM verdict."""
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": full_name, "domain": "chatbot evaluation"})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: full_name)
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [{"_raw": True}], lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda *a, **k: [dict(c) for c in candidates])
    monkeypatch.setattr(pr, "_batch_verify_with_llm", lambda *a, **k: llm_result)


def _cand(title, arxiv_id, sim, authors):
    return {"title": title, "abstract": "", "arxiv_id": arxiv_id, "doi": "", "url": "",
            "year": 2024, "citationCount": 0, "authors": authors,
            "_title_similarity": sim, "_source": "openalex", "_query_used": "q"}


def test_name_token_fallback_recovers_correct_paper(monkeypatch):
    # LLM inconclusively declines (none, LOW confidence -> falls through, not a confident-none).
    # The wrong paper has HIGHER title similarity, so a pure-similarity floor would mis-pick it;
    # only the name-bearing candidate must be accepted.
    correct = _cand("Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators",
                    "2404.04475", 78.0, ["Yann Dubois"])
    wrong = _cand("Self-Rewarding Language Models", "2401.10020", 81.0, ["Weizhe Yuan"])
    _mock_resolver(monkeypatch, [correct, wrong],
                   {"match_index": "none", "confidence": 0.4, "reasoning": "x", "error": False})
    result = pr.resolve_paper("AlpacaEval 2.0", full_name="AlpacaEval 2.0")
    assert result is not None
    assert "2404.04475" in result["url"]
    assert result["authors"] == ["Yann Dubois"]


def test_name_token_fallback_skips_when_ambiguous(monkeypatch):
    # Two candidates bear the name -> not a unique signal -> no fabrication (husk preserved).
    a = _cand("AlpacaEval: An Automatic Evaluator", "1111.11111", 70.0, ["A"])
    b = _cand("Revisiting AlpacaEval Benchmarks", "2222.22222", 72.0, ["B"])
    _mock_resolver(monkeypatch, [a, b],
                   {"match_index": "none", "confidence": 0.4, "reasoning": "x", "error": False})
    assert pr.resolve_paper("AlpacaEval 2.0", full_name="AlpacaEval 2.0") is None


def test_name_token_fallback_skips_short_name(monkeypatch):
    # A short benchmark name (<5 normalized chars) is too ambiguous: even a sole title that bears
    # it is not recovered.
    bearer = _cand("ZZZ the Reasoning Challenge", "3333.33333", 60.0, ["C"])
    _mock_resolver(monkeypatch, [bearer],
                   {"match_index": "none", "confidence": 0.4, "reasoning": "x", "error": False},
                   full_name="ZZZ")
    assert pr.resolve_paper("ZZZ", full_name="ZZZ") is None


def test_search_path_backfills_missing_authors(monkeypatch):
    # LLM accepts a candidate that has an arXiv id but no search-API authors -> authors are
    # backfilled from the paper's own metadata.
    accepted = _cand("API-Bank: A Comprehensive Benchmark", "2304.08244", 95.0, [])
    _mock_resolver(monkeypatch, [accepted],
                   {"match_index": 0, "confidence": 1.0, "reasoning": "ok", "error": False},
                   full_name="API-Bank")
    monkeypatch.setattr(pr, "_fetch_paper_authors",
                        lambda url: {"authors": ["Minghao Li"], "title": "API-Bank", "year": 2023})
    result = pr.resolve_paper("API-Bank", full_name="API-Bank")
    assert result is not None
    assert result["authors"] == ["Minghao Li"]
