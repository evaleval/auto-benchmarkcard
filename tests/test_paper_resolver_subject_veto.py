"""Accept-time subject veto: a same-name paper from a plainly different domain must not
survive ANY accept path (batch_verification, title_similarity_fallback, name_token_fallback).

The ten replay fixtures in tests/data/subject_veto_fixtures.json are the real 2026-06-25
broad_run accepts (titles, abstracts, identities from the shipped sidecars and cards), pinned
by ACTION (kept / vetoed / abstained) with floor-inequality asserts, never exact floats:
- echo replays the defining defect (a graph-propagation "ECHO" accepted at conf 0.95 for an
  image-generation benchmark) -> vetoed on the batch path.
- covost2 replays the Zenodo model-record accept on the title-similarity path -> vetoed
  (reclassified 2026-07-04: a model card is not the introducing paper; honest-none is the
  intended outcome, logged as recall cost).
- screenspot / studiogan / vip-bench replay correct-domain fallback accepts -> kept.
- hard-problems / disinformation replay thin-identity fallback accepts -> abstained (kept).
- charxiv / mgsm / gsm8k replay known-correct confident batch accepts -> kept.

Network and LLM are mocked; the veto itself is deterministic.
"""

import json
import os

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr

_FIXTURES = json.load(open(os.path.join(os.path.dirname(__file__), "data",
                                        "subject_veto_fixtures.json")))
FLOOR = _FIXTURES["floor"]
CASES = {c["slug"]: c for c in _FIXTURES["cases"]}


@pytest.fixture(autouse=True)
def _no_sleep_clean_state(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._author_cache.clear()
    pr._metadata_cache.clear()
    pr._SOURCE_STATUS.clear()


def _mock_resolver(monkeypatch, case, llm_result):
    cand = {"title": case["candidate_title"], "abstract": case["abstract"],
            "arxiv_id": case["candidate_id"], "doi": "", "url": "",
            "year": 2024, "citationCount": 0, "authors": ["A"],
            "_title_similarity": case["title_similarity"],
            "_source": "openalex", "_query_used": "q"}
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": case["full_name"],
                                         "domain": case["domain"]})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: case["full_name"])
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [{"_raw": True}],
                                          lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda *a, **k: [dict(cand)])
    monkeypatch.setattr(pr, "_batch_verify_with_llm", lambda *a, **k: llm_result)


def _run(monkeypatch, tmp_path, slug):
    case = CASES[slug]
    if case["path"] == "batch":
        llm = {"match_index": 0, "confidence": 0.95, "reasoning": "match", "error": False}
    else:
        # inconclusive verdict -> deterministic recovery (title-similarity fallback)
        llm = {"match_index": "none", "confidence": 0.4, "reasoning": "unsure", "error": False}
    _mock_resolver(monkeypatch, case, llm)
    result = pr.resolve_paper(slug, full_name=case["full_name"],
                              overview=case["overview"], output_dir=tmp_path)
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    return case, result, log


def _assert_action(case, result, log):
    rec = log["subject_overlap"]
    assert rec["action"] == case["expect"]
    assert rec["floor"] == FLOOR
    expected_via = "batch_verification" if case["path"] == "batch" else "title_similarity_fallback"
    assert rec["resolved_via"] == expected_via
    if case["expect"] == "vetoed":
        assert rec["overlap"] is not None and rec["overlap"] < FLOOR
        assert result is None
        assert log["resolved_url"] is None
    elif case["expect"] == "kept":
        assert rec["overlap"] is not None and rec["overlap"] >= FLOOR
        assert result is not None
    else:  # abstained: thin abstract or thin identity must never veto
        assert rec["overlap"] is None
        assert result is not None


@pytest.mark.parametrize("slug", list(CASES))
def test_replayed_accept_action(monkeypatch, tmp_path, slug):
    case, result, log = _run(monkeypatch, tmp_path, slug)
    _assert_action(case, result, log)


def test_covost2_veto_is_on_title_similarity_path(monkeypatch, tmp_path):
    # The exact path that produced the historical Zenodo accept must now be covered.
    case, result, log = _run(monkeypatch, tmp_path, "covost2")
    assert log["subject_overlap"]["resolved_via"] == "title_similarity_fallback"
    assert log["subject_overlap"]["action"] == "vetoed"
    assert result is None


def test_echo_veto_is_on_batch_path(monkeypatch, tmp_path):
    # The echo defect was a CONFIDENT batch accept; fallback-only coverage would miss it.
    case, result, log = _run(monkeypatch, tmp_path, "echo")
    assert log["subject_overlap"]["resolved_via"] == "batch_verification"
    assert log["subject_overlap"]["action"] == "vetoed"
    assert result is None


def test_name_token_fallback_is_also_vetoed(monkeypatch, tmp_path):
    # Same-name wrong-domain accept via the sole-name-bearer recovery (synthetic, structural).
    case = {"slug": "widgetgan", "full_name": "WidgetGAN",
            "domain": "image generation evaluation",
            "overview": "WidgetGAN is a benchmark comparing generative adversarial networks "
                        "for realistic image synthesis quality across resolution settings "
                        "and training regimes on standard vision datasets.",
            "candidate_title": "WidgetGAN for Parametric Furniture Assembly Planning",
            "candidate_id": "9999.99999",
            "abstract": "Automated furniture assembly requires reasoning about screws, "
                        "panels and joints. We present a planner that sequences assembly "
                        "steps for flat-pack furniture using constraint solving over part "
                        "geometry, tested on wardrobe and shelf products in a factory "
                        "simulation environment with robotic arm executions.",
            "title_similarity": 60.0, "path": "name_token", "expect": "vetoed"}
    llm = {"match_index": "none", "confidence": 0.4, "reasoning": "unsure", "error": False}
    _mock_resolver(monkeypatch, case, llm)
    result = pr.resolve_paper("widgetgan", full_name="WidgetGAN",
                              overview=case["overview"], output_dir=tmp_path)
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    assert log["subject_overlap"]["resolved_via"] == "name_token_fallback"
    assert log["subject_overlap"]["action"] == "vetoed"
    assert result is None


def test_empty_abstract_abstains_never_vetoes(monkeypatch, tmp_path):
    # A missing abstract (sparse search record / outage) must keep the accept.
    case = dict(CASES["echo"], abstract="", expect="abstained")
    _mock_resolver(monkeypatch, case,
                   {"match_index": 0, "confidence": 0.95, "reasoning": "m", "error": False})
    result = pr.resolve_paper("echo", full_name=case["full_name"],
                              overview=case["overview"], output_dir=tmp_path)
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    assert log["subject_overlap"]["action"] == "abstained"
    assert log["subject_overlap"]["overlap"] is None
    assert result is not None


def test_empty_identity_abstains_never_vetoes(monkeypatch, tmp_path):
    # A thin identity (no full_name, no overview, no domain) abstains exactly like a
    # missing abstract.
    case = dict(CASES["echo"], full_name=None, domain=None, overview=None)
    _mock_resolver(monkeypatch, case,
                   {"match_index": 0, "confidence": 0.95, "reasoning": "m", "error": False})
    result = pr.resolve_paper("echo", full_name=None, overview=None, output_dir=tmp_path)
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    assert log["subject_overlap"]["action"] == "abstained"
    assert log["subject_overlap"]["overlap"] is None
    assert result is not None


def test_overlap_recorded_on_kept_accepts(monkeypatch, tmp_path):
    # Calibration requirement: the numeric overlap lands in the sidecar on every accept.
    _, _, log = _run(monkeypatch, tmp_path, "gsm8k")
    rec = log["subject_overlap"]
    assert isinstance(rec["overlap"], float)
    assert rec["action"] == "kept"
