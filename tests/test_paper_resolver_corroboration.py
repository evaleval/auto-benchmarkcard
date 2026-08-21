"""F4 entity-resolution: Tier-1 confident-none, Tier-1b parse repair, Tier-2 repo corroboration,
the Tier-3a worker guard, and the F4-1c search-layer fixes.

All synthetic -- no real benchmark names. Network is mocked; the checks are general mechanisms
that must score an unseen card the same way the regression fixtures here do.
"""

import json

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr
from auto_benchmarkcard import workers


class _Resp:
    def __init__(self, *, json_data=None, status=200):
        self._json, self.status_code = json_data, status

    def json(self):
        return self._json or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _no_sleep_clean_state(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._author_cache.clear()
    pr._metadata_cache.clear()
    pr._SOURCE_STATUS.clear()


def _cand(title, arxiv_id, sim, authors=("A",)):
    return {"title": title, "abstract": "", "arxiv_id": arxiv_id, "doi": "", "url": "",
            "year": 2024, "citationCount": 0, "authors": list(authors),
            "_title_similarity": sim, "_source": "openalex", "_query_used": "q"}


def _mock_resolver(monkeypatch, candidates, llm_result, *, full_name="WidgetBench"):
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": full_name, "domain": "widget evaluation"})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: full_name)
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [{"_raw": True}], lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda *a, **k: [dict(c) for c in candidates])
    monkeypatch.setattr(pr, "_batch_verify_with_llm", lambda *a, **k: llm_result)


# Tier-1: confident none is final --------------------------------------------------------------
def test_confident_none_resolves_nothing(monkeypatch):
    # A name-bearing candidate would normally be recovered by name_token_fallback; a confident,
    # non-errored "none" must short-circuit that and resolve nothing (the aa-index defect).
    bearer = _cand("WidgetBench: A Benchmark for Widgets", "2404.00001", 70.0)
    _mock_resolver(monkeypatch, [bearer],
                   {"match_index": "none", "confidence": 1.0, "reasoning": "x", "error": False})
    assert pr.resolve_paper("widgetbench", full_name="WidgetBench") is None


def test_low_confidence_none_still_recovers(monkeypatch):
    # An inconclusive (low-confidence) none falls through to the recovery fallbacks.
    bearer = _cand("WidgetBench: A Benchmark for Widgets", "2404.00002", 70.0)
    _mock_resolver(monkeypatch, [bearer],
                   {"match_index": "none", "confidence": 0.4, "reasoning": "x", "error": False})
    result = pr.resolve_paper("widgetbench", full_name="WidgetBench")
    assert result is not None and "2404.00002" in result["url"]


def test_errored_none_still_recovers(monkeypatch):
    # An LLM-error verdict (error=True, conf 0.0) is inconclusive, not a confident none.
    bearer = _cand("WidgetBench: A Benchmark for Widgets", "2404.00003", 70.0)
    _mock_resolver(monkeypatch, [bearer],
                   {"match_index": "none", "confidence": 0.0, "reasoning": "err", "error": True})
    result = pr.resolve_paper("widgetbench", full_name="WidgetBench")
    assert result is not None and "2404.00003" in result["url"]


# Tier-1b: bounded JSON-parse repair -----------------------------------------------------------
class _FakeLLM:
    def __init__(self, texts):
        self._texts, self.calls = list(texts), 0

    def generate(self, prompt):
        self.calls += 1
        return self._texts.pop(0)


_GOOD_VERDICT = '{"match_index": 0, "confidence": 0.9, "reasoning": "ok"}'
_BAD_VERDICT = "not json at all"
_CANDS = [{"title": "X", "abstract": "a", "year": 2024, "citationCount": 0}]


def test_parse_failure_retries_once_then_succeeds(monkeypatch):
    fake = _FakeLLM([_BAD_VERDICT, _GOOD_VERDICT])
    monkeypatch.setattr(pr, "get_llm_handler", lambda *a, **k: fake)
    result = pr._batch_verify_with_llm(_CANDS, "widgetbench", [], [], None)
    assert result["match_index"] == 0 and not result.get("error")
    assert fake.calls == 2


def test_parse_failure_twice_is_inconclusive(monkeypatch):
    fake = _FakeLLM([_BAD_VERDICT, _BAD_VERDICT])
    monkeypatch.setattr(pr, "get_llm_handler", lambda *a, **k: fake)
    result = pr._batch_verify_with_llm(_CANDS, "widgetbench", [], [], None)
    assert result.get("error") is True and result["match_index"] == "none"
    assert fake.calls == 2


# Tier-2: repo corroboration -------------------------------------------------------------------
def test_corroboration_clean_repo():
    assert pr._repo_corroboration("widgetorg/widgetbench", "widgetbench") is None


def test_corroboration_derivative_repo():
    # basename bears the slug PLUS extra distinctive tokens -> a training-set/variant derivative.
    assert pr._repo_corroboration("acme/Distill-RL-widgetbench-extra", "widgetbench") == "derivative_repo"


def test_corroboration_aggregate_repo():
    # basename shares no whole-token run with the name -> a multi-benchmark / unrelated repo.
    assert pr._repo_corroboration("someorg/MegaSuite", "widgetbench") == "aggregate_repo"


def test_corroboration_arxiv_mismatch_takes_precedence():
    # Disagreeing arxiv ids reject the binding even when the basename would corroborate.
    assert pr._repo_corroboration("widgetorg/widgetbench", "widgetbench",
                                  repo_arxiv_id="1111.11111", paper_arxiv_id="2222.22222") == "arxiv_mismatch"


def test_corroboration_arxiv_agreement_is_clean():
    assert pr._repo_corroboration("widgetorg/widgetbench", "widgetbench",
                                  repo_arxiv_id="1111.11111v2", paper_arxiv_id="1111.11111") is None


def test_corroboration_short_slug_skipped():
    # A slug shorter than the floor is too ambiguous to judge a repo by name.
    assert pr._repo_corroboration("org/totallydifferent", "abc") is None


# Tier-3a: worker drop guard -------------------------------------------------------------------
def test_drop_uncorroborated_hf_clears_overrides_and_ns_display():
    card = {"benchmark_details": {"logo": "https://x/y.png", "org_url": "https://hf/org"}}
    overrides = {"benchmark_details.languages": ["English"], "data.size": "1K<n<10K"}
    out = workers._drop_uncorroborated_hf(card, overrides, "derivative_repo")
    assert overrides == {}
    assert out["benchmark_details"]["logo"] == "Not specified"
    assert out["benchmark_details"]["org_url"] == "Not specified"


def test_drop_uncorroborated_hf_noop_when_corroborated():
    card = {"benchmark_details": {"logo": "https://x/y.png", "org_url": "https://hf/org"}}
    overrides = {"data.size": "1K<n<10K"}
    workers._drop_uncorroborated_hf(card, overrides, None)
    assert overrides == {"data.size": "1K<n<10K"}
    assert card["benchmark_details"]["logo"] == "https://x/y.png"


# D4: _extract_paper_from_hf must NOT anchor on a top-level tag -------------------------------
def test_extract_paper_prefers_readme_over_toplevel_tag():
    # A wrong top-level arxiv tag plus a correct README Paper: link -> the README link wins
    # (the function never reads top-level tags; agentharm-shaped regression).
    hf = {"tags": ["arxiv:1111.11111"], "card_data": {},
          "readme_markdown": "Intro\n**Paper:** https://arxiv.org/abs/2222.22222\nmore"}
    assert workers._extract_paper_from_hf(hf) == "https://arxiv.org/abs/2222.22222"


# F4-1c: search-layer query + telemetry ------------------------------------------------------
def test_suffix_aware_query_adds_base_entity():
    queries = [q.lower() for q in pr._build_search_queries("widgetbench-d")]
    assert "widgetbench" in queries


def test_hf_repo_id_bridges_into_queries():
    queries = pr._build_search_queries("xyz", hf_repo_id="Org-Team/SuperWidget")
    assert "Super Widget" in queries


def test_resolve_dedups_candidates_by_arxiv_id(monkeypatch):
    captured = {}

    def fake_verify(candidates, *a, **k):
        captured["n"] = len(candidates)
        return {"match_index": 0, "confidence": 1.0, "error": False}

    raw = [{"title": "Widget Bench", "arxiv_id": "9999.00001", "doi": "", "_title_similarity": 70,
            "authors": ["A"]},
           {"title": "Widget Bench rewritten", "arxiv_id": "9999.00001", "doi": "",
            "_title_similarity": 70, "authors": ["A"]}]
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": "WidgetBench", "domain": "widgets"})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: "WidgetBench")
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: raw, lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda cands, *a, **k: [dict(c) for c in cands])
    monkeypatch.setattr(pr, "_batch_verify_with_llm", fake_verify)
    monkeypatch.setattr(pr, "_fetch_paper_authors",
                        lambda url: {"authors": ["A"], "title": "", "year": None})
    pr.resolve_paper("widgetbench", full_name="WidgetBench")
    assert captured["n"] == 1


def test_sources_tried_telemetry_written(monkeypatch, tmp_path):
    cand = {"title": "WidgetBench", "arxiv_id": "9999.00002", "doi": "", "_title_similarity": 99,
            "authors": ["A"]}
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": "WidgetBench", "domain": "widgets"})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: "WidgetBench")
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [dict(cand)], lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda cands, *a, **k: [dict(c) for c in cands])
    monkeypatch.setattr(pr, "_batch_verify_with_llm",
                        lambda *a, **k: {"match_index": 0, "confidence": 1.0, "error": False})
    monkeypatch.setattr(pr, "_fetch_paper_authors",
                        lambda url: {"authors": [], "title": "", "year": None})
    pr.resolve_paper("widgetbench", full_name="WidgetBench", output_dir=tmp_path)
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    by_source = {s["source"]: s for s in log["sources_tried"]}
    assert by_source["openalex"]["status"] == "ok"
    assert by_source["openalex"]["results"] >= 1


def test_source_status_records_rate_limit(monkeypatch):
    monkeypatch.setattr(pr.requests, "get", lambda *a, **k: _Resp(status=429))
    pr._SOURCE_STATUS.clear()
    assert pr._search_openalex("q") == []
    assert pr._SOURCE_STATUS.get("openalex") == "rate_limited"
