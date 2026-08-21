"""FINISH-4 Mode-A residue: bfcl must not re-resolve to the wrong paper.

Network-free. Drives resolve_paper with the search/metadata/LLM boundaries mocked
(the robustness-suite pattern) to prove the wrong paper is not re-resolved.
"""

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._author_cache.clear()


def _cand(title, arxiv_id, sim, authors):
    return {"title": title, "abstract": "", "arxiv_id": arxiv_id, "doi": "", "url": "",
            "year": 2024, "citationCount": 0, "authors": authors,
            "_title_similarity": sim, "_source": "openalex", "_query_used": "q"}


def _mock_resolver(monkeypatch, candidates, llm_result, *, full_name):
    """Wire resolve_paper's boundaries: metadata, search source, prefilter, LLM verdict."""
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": full_name, "domain": "x"})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: full_name)
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [{"_raw": True}], lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda *a, **k: [dict(c) for c in candidates])
    monkeypatch.setattr(pr, "_batch_verify_with_llm", lambda *a, **k: llm_result)


# bfcl: removal must not re-resolve to the wrong (robot-trust) paper
def test_bfcl_does_not_resolve_to_robot_trust_paper(monkeypatch):
    # After removal bfcl flows through search; a confident-none verdict leaves it unresolved.
    # Even with the old wrong paper as the only candidate, it must not be returned.
    wrong = _cand("Towards a Participatory and Social Justice-Oriented Measure of Human-Robot Trust",
                  "2402.15671", 60.0, ["Raj Korpan"])
    _mock_resolver(monkeypatch, [wrong],
                   {"match_index": "none", "confidence": 0.9, "reasoning": "unrelated", "error": False},
                   full_name="Berkeley Function Calling Leaderboard")
    result = pr.resolve_paper("bfcl", full_name="Berkeley Function Calling Leaderboard")
    assert result is None or "2402.15671" not in (result.get("url") or "")
