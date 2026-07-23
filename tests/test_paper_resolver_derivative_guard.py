"""Derivative-binds-parent guard: a variant-qualified identity (helm_lite, mm-mind2web) must
not bind the PARENT's paper unless the paper mentions the variant. Replaces the deleted
KNOWN_NO_PAPER sentinel with a general mechanism, wired in both resolve_paper (all accept
paths) and the pre-set binding gate (before the Tier-1 accept).

Real titles/abstracts from tests/data/derivative_guard_fixtures.json; network and LLM mocked.
Sharp regression pins:
- helm_capabilities: the HELM abstract contains the bare word 'capabilities' but never 'HELM
  Capabilities' as a unit; bare word-mention must NOT stand the guard down.
- helm_air_bench-shaped identities: the full_name 'AIR-Bench' is title-borne by the correct
  paper, so the guard stands down (harness prefix, not a variant of HELM).
- charxiv-d keeps the CharXiv paper (single-char qualifier exemption: the parent paper
  documents the descriptive split); mgsm and gsm8k keep their own papers.
"""

import json
import os

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr
from auto_benchmarkcard import workers as W

_DATA = json.load(open(os.path.join(os.path.dirname(__file__), "data",
                                    "derivative_guard_fixtures.json")))


@pytest.fixture(autouse=True)
def _no_sleep_clean_state(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._author_cache.clear()
    pr._metadata_cache.clear()
    pr._SOURCE_STATUS.clear()


# --- helper unit: _parent_only_binding --------------------------------------------------------

def test_fires_for_helm_variants_on_the_real_helm_paper():
    helm = _DATA["helm"]
    for suite, fn in (("helm_lite", "HELM Lite"), ("helm_capabilities", "HELM Capabilities")):
        hit = pr._parent_only_binding(suite, fn, helm["title"], helm["abstract"])
        assert hit is not None, suite
        assert hit["parent"] == "helm"


def test_bare_word_mention_does_not_stand_down():
    # 'capabilities' appears as a plain word in the HELM abstract; only the adjacent unit
    # 'HELM Capabilities' would count as a variant mention.
    helm = _DATA["helm"]
    assert "capabilities" in helm["abstract"].lower()
    assert pr._parent_only_binding("helm_capabilities", "HELM Capabilities",
                                   helm["title"], helm["abstract"]) is not None


def test_adjacent_variant_mention_stands_down():
    helm = _DATA["helm"]
    doctored = helm["abstract"] + " We also release HELM Lite, a lightweight subset."
    assert pr._parent_only_binding("helm_lite", "HELM Lite",
                                   helm["title"], doctored) is None


def test_fires_for_mm_mind2web_on_the_real_mind2web_paper():
    m2w = _DATA["mind2web"]
    hit = pr._parent_only_binding("mm-mind2web", "MM-Mind2Web", m2w["title"], m2w["abstract"])
    assert hit is not None
    assert hit["parent"] == "mind2web"


def test_stand_down_when_title_bears_the_full_identity():
    # The paper IS about the variant: harness-prefix slugs keep their own paper.
    assert pr._parent_only_binding(
        "helm_air_bench", "AIR-Bench",
        "AIR-Bench: Benchmarking Large Language Models via Safety Refusal Categories",
        "We present AIR-Bench, a safety benchmark grounded in regulation frameworks.") is None


def test_never_fires_for_single_token_and_single_char_qualifiers():
    mgsm, gsm8k, cx = _DATA["mgsm"], _DATA["gsm8k"], _DATA["charxiv"]
    assert pr._parent_only_binding("mgsm", "Multilingual Grade School Math",
                                   mgsm["title"], mgsm["abstract"]) is None
    assert pr._parent_only_binding("gsm8k", "GSM8K", gsm8k["title"], gsm8k["abstract"]) is None
    assert pr._parent_only_binding("charxiv-d", "CharXiv-D", cx["title"], cx["abstract"]) is None


def test_never_fires_for_version_metric_and_generic_qualifiers():
    text_t = "WMT: Findings of the Workshop on Machine Translation"
    text_a = "Shared task results for WMT and LBPP systems across AITZ settings."
    for suite, fn in (("wmt 2014", "WMT 2014"), ("lbpp (v2)", "LBPP (v2)"),
                      ("seal-0", "Seal-0"), ("amc_2022_23", "AMC_2022_23"),
                      ("aitz_em", "AITZ_EM"),
                      ("terminal-bench", "Terminal-Bench"), ("vip-bench", "ViP-Bench"),
                      ("hard problems", "Hard Problems"),
                      ("natural questions", "Natural Questions")):
        assert pr._parent_only_binding(suite, fn, text_t, text_a) is None, suite


def test_acronym_of_parent_qualifier_never_fires():
    # 'Expansion (Acronym)' identities are one name written twice: the acronym qualifier
    # must not read as a variant marker (the ifeval false positive, 2026-07-05).
    ife = _DATA["ifeval"]
    assert pr._parent_only_binding(
        "ifeval", "Instruction-Following Evaluation (IFEval)",
        ife["title"], ife["abstract"]) is None
    # unit checks on the segmentation itself
    assert pr._is_acronym_of("ifeval", ["instruction", "following", "evaluation"])
    assert pr._is_acronym_of("nq", ["natural", "questions"])
    assert not pr._is_acronym_of("mm", ["mind2web"])
    assert not pr._is_acronym_of("capabilities", ["helm"])
    assert not pr._is_acronym_of("lite", ["helm"])


def test_resolve_binds_ifeval_after_acronym_fix(monkeypatch, tmp_path):
    _mock_resolver(monkeypatch, _DATA["ifeval"],
                   full_name="Instruction-Following Evaluation (IFEval)",
                   domain="instruction following evaluation")
    result = pr.resolve_paper("ifeval", full_name="Instruction-Following Evaluation (IFEval)",
                              overview="IFEval evaluates the ability of large language models "
                                       "to follow verifiable natural language instructions "
                                       "using prompts with checkable constraints.",
                              output_dir=tmp_path)
    assert result is not None and "2311.07911" in result["url"]
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    assert "derivative_guard" not in log


def test_plain_prose_parent_surface_does_not_arm_the_guard():
    # 'Grade School Math' in Title-case prose is not a name-like surface.
    assert pr._parent_only_binding(
        "mgsm", "Multilingual Grade School Math",
        "Solving Grade School Math with Verifiers",
        "We study grade school math problems in English.") is None


# --- resolve_paper integration ----------------------------------------------------------------

def _mock_resolver(monkeypatch, paper, *, full_name, domain, overview_words=""):
    cand = {"title": paper["title"], "abstract": paper["abstract"],
            "arxiv_id": paper["arxiv"], "doi": "", "url": "",
            "year": 2023, "citationCount": 100, "authors": ["A"],
            "_title_similarity": 90.0, "_source": "openalex", "_query_used": "q"}
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": full_name, "domain": domain})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: full_name)
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [{"_raw": True}],
                                          lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda *a, **k: [dict(cand)])
    monkeypatch.setattr(pr, "_batch_verify_with_llm",
                        lambda *a, **k: {"match_index": 0, "confidence": 0.95,
                                         "reasoning": "match", "error": False})


def test_resolve_abstains_helm_lite_and_capabilities(monkeypatch, tmp_path):
    for suite, fn in (("helm_lite", "HELM Lite"), ("helm_capabilities", "HELM Capabilities")):
        out = tmp_path / suite
        _mock_resolver(monkeypatch, _DATA["helm"], full_name=fn, domain="language model evaluation")
        result = pr.resolve_paper(suite, full_name=fn, output_dir=out)
        assert result is None, suite
        log = json.loads((out / "paper-verification.json").read_text())
        assert log["derivative_guard"]["action"] == "abstained"
        assert log["resolved_url"] is None


def test_resolve_abstains_mm_mind2web(monkeypatch, tmp_path):
    _mock_resolver(monkeypatch, _DATA["mind2web"], full_name="MM-Mind2Web",
                   domain="web agents")
    result = pr.resolve_paper("mm-mind2web", full_name="MM-Mind2Web", output_dir=tmp_path)
    assert result is None
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    assert log["derivative_guard"]["parent"] == "mind2web"


def test_resolve_still_binds_mgsm_gsm8k_charxiv_d(monkeypatch, tmp_path):
    veto_cases = {c["slug"]: c for c in json.load(
        open(os.path.join(os.path.dirname(__file__), "data",
                          "subject_veto_fixtures.json")))["cases"]}
    for suite, fn, key in (("mgsm", "Multilingual Grade School Math", "mgsm"),
                           ("gsm8k", "GSM8K", "gsm8k"),
                           ("charxiv-d", "CharXiv-D", "charxiv")):
        out = tmp_path / suite
        base = veto_cases[key]
        _mock_resolver(monkeypatch, _DATA[key], full_name=fn,
                       domain=base["domain"] or "evaluation")
        result = pr.resolve_paper(suite, full_name=fn, overview=base["overview"],
                                  output_dir=out)
        assert result is not None, suite
        assert _DATA[key]["arxiv"] in result["url"], suite
        log = json.loads((out / "paper-verification.json").read_text())
        assert "derivative_guard" not in log, suite


# --- binding gate integration (pre-set paper, before Tier-1) ----------------------------------

def _ctx(suite, full_name):
    return {"suite_name": suite, "full_name": full_name, "overview": None, "domain": None,
            "sub_benchmarks": [], "metrics": [], "eval_library": None,
            "hf_repo_id": None, "identity_subject": ""}


def test_gate_rejects_helm_variant_preset_before_tier1(monkeypatch):
    helm = _DATA["helm"]
    monkeypatch.setattr(W, "_fetch_paper_meta_for_verify",
                        lambda url: {"title": helm["title"], "abstract": helm["abstract"]})
    called = {"llm": 0}

    def _boom(*a, **k):
        called["llm"] += 1
        return {"is_match": True, "confidence": 1.0, "error": False}
    monkeypatch.setattr(W, "verify_paper_binding", _boom)
    action, record = W._verify_paper_binding_gate(
        {}, "https://arxiv.org/abs/2211.09110", "eee_unitxt", _ctx("helm_lite", "HELM Lite"))
    assert action == "reject"
    assert record["derivative_guard"]["parent"] == "helm"
    assert called["llm"] == 0, "guard must fire deterministically, before any LLM call"


def test_gate_keeps_parent_slug_on_its_own_paper(monkeypatch):
    m2w = _DATA["mind2web"]
    monkeypatch.setattr(W, "_fetch_paper_meta_for_verify",
                        lambda url: {"title": m2w["title"], "abstract": m2w["abstract"]})
    monkeypatch.setattr(W, "verify_paper_binding",
                        lambda *a, **k: {"is_match": True, "confidence": 1.0, "error": False,
                                         "match_index": 0})
    action, record = W._verify_paper_binding_gate(
        {}, "https://arxiv.org/abs/2306.06070", "eee_unitxt", _ctx("mind2web", "Mind2Web"))
    assert action == "keep"
