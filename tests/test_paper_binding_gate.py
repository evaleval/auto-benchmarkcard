"""Tests for the resolve-time paper-binding verifier (the mirror of the HF gate) in
run_paper_resolver, plus the orchestrator routing change that makes the gate actually run.

Network-free: _fetch_paper_meta_for_verify, verify_paper_binding and resolve_paper are
monkeypatched. The LLM is mocked via a call-counter so the deterministic fast paths can be asserted
to make ZERO LLM calls (the cost guard). Checks are structural mechanisms, never keyed on a
benchmark-name list.
"""

import pytest

import auto_benchmarkcard.workers as W
from auto_benchmarkcard import workflow
from auto_benchmarkcard.config import Config


class _FakeOM:
    """Captures the per-tool sidecar writes so tests can assert the binding record."""

    def __init__(self):
        self.writes = {}

    def save_tool_output(self, data, tool, filename):
        self.writes[(tool, filename)] = data
        return f"/tmp/{tool}/{filename}"

    def get_tool_output_path(self, tool, create_if_missing=True):
        return f"/tmp/{tool}"

    def paper_sidecar(self):
        return self.writes.get(("paper_resolver", "paper-verification.json"))


def _state(name, *, extracted=None, eee=None, hf_json=None):
    return {
        "query": name,
        "extracted_ids": extracted if extracted is not None else {},
        "eee_metadata": eee if eee is not None else {"benchmark_name": name, "metrics": {}},
        "hf_json": hf_json,
        "output_manager": _FakeOM(),
    }


class _LLMCounter:
    """Mock for verify_paper_binding: counts calls, returns a fixed verdict."""

    def __init__(self, verdict):
        self.calls = 0
        self._verdict = verdict

    def __call__(self, suite, paper_url, **kwargs):
        self.calls += 1
        return dict(self._verdict)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", True)


def _patch_meta(monkeypatch, title="", abstract="", fetch_error=False):
    monkeypatch.setattr(W, "_fetch_paper_meta_for_verify",
                        lambda url: {"title": title, "abstract": abstract, "fetch_error": fetch_error})


# --- BYPASS CLOSED: a pre-set wrong paper is now verified + rejected + falls through ----------

def test_preset_wrong_paper_rejected_and_falls_through(monkeypatch):
    # mgsm class: a pre-set GSM8K paper on an mgsm identity. LLM confident not-a-match -> reject ->
    # drop paper_url + paper_url_from_hf -> fall through to the verified search path.
    counter = _LLMCounter({"is_match": False, "confidence": 0.95, "reasoning": "GSM8K, not MGSM"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="Training Verifiers to Solve Math Word Problems")
    # search fall-through resolves the real MGSM paper
    monkeypatch.setattr(W, "resolve_paper",
                        lambda *a, **k: {"url": "https://arxiv.org/abs/2210.03057",
                                         "title": "Language Models are Multilingual...", "authors": ["S"]})
    st = _state("mgsm", extracted={"paper_url": "https://arxiv.org/abs/2110.14168",
                                    "paper_url_from_hf": "https://arxiv.org/abs/2110.14168"})
    out = W.run_paper_resolver(st)
    assert counter.calls == 1
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2210.03057"
    assert "2110.14168" not in out["extracted_ids"]["paper_url"]
    assert out["paper_resolver_attempted"] is True


def test_preset_wrong_paper_rejected_no_search_match_honest_thin(monkeypatch):
    # Reject + search also yields nothing -> honest-thin (no paper_url) and the dropped binding is gone.
    counter = _LLMCounter({"is_match": False, "confidence": 0.9})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="An Unrelated Paper")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    st = _state("bbq", extracted={"paper_url": "https://arxiv.org/abs/2110.08514"})
    out = W.run_paper_resolver(st)
    assert (out.get("extracted_ids") or {}).get("paper_url") is None
    assert out["paper_resolver_attempted"] is True


def test_preset_realistic_nomatch_zero_conf_rejects_and_falls_through(monkeypatch):
    # FINISH-8 G1 (the dominant defect): the REALISTIC no-match verdict the live bug produced. A genuine
    # LLM "none" returns is_match=False, confidence=0.0 (confidence is on the MATCH scale), error=False.
    # The pre-fix reject branch required conf>=0.7, so this fell to keep-degraded -> the mgsm/GSM8K wrong
    # paper survived. The fix treats any non-errored not-a-match as a confident reject -> drop + fall
    # through to the verified search, which resolves the real MGSM paper and re-sources its authors.
    # (The earlier reject test used conf=0.95, which DID satisfy the dead branch and so masked the bug.)
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": False,
                           "reasoning": "This paper introduces GSM8K, not the multilingual MGSM."})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="Training Verifiers to Solve Math Word Problems",
                abstract="state-of-the-art language models solve grade school math word problems")
    monkeypatch.setattr(W, "resolve_paper",
                        lambda *a, **k: {"url": "https://arxiv.org/abs/2210.03057",
                                         "title": "Language Models are Multilingual Chain-of-Thought "
                                                  "Reasoners",
                                         "authors": ["Freda Shi", "Mirac Suzgun"]})
    st = _state("mgsm", extracted={"paper_url": "https://arxiv.org/abs/2110.14168",
                                   "paper_url_from_hf": "https://arxiv.org/abs/2110.14168"})
    out = W.run_paper_resolver(st)
    assert counter.calls == 1
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2210.03057"
    assert "2110.14168" not in out["extracted_ids"]["paper_url"]
    # fall-through re-sources the correct paper's authors (no GSM8K leak path here)
    assert out["extracted_ids"].get("paper_authors") == ["Freda Shi", "Mirac Suzgun"]


def test_preset_realistic_nomatch_zero_conf_honest_thin(monkeypatch):
    # Same realistic conf=0.0 no-match verdict, but the search fall-through finds nothing -> honest-thin
    # (no paper_url), and the dropped binding is recorded as rejected via prior_binding.
    seen = {}
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": False,
                           "reasoning": "wrong paper for this benchmark"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Different Paper", abstract="unrelated content here for the check")

    def _capture_resolve(*a, **k):
        seen["prior_binding"] = k.get("prior_binding")
        return None
    monkeypatch.setattr(W, "resolve_paper", _capture_resolve)
    st = _state("mgsm", extracted={"paper_url": "https://arxiv.org/abs/2110.14168"})
    out = W.run_paper_resolver(st)
    assert (out.get("extracted_ids") or {}).get("paper_url") is None
    assert seen["prior_binding"] is not None and seen["prior_binding"]["verdict"] == "rejected"


def test_preset_realistic_nomatch_thin_abstract_rejects_gracefully(monkeypatch):
    # A non-errored is_match=False with a THIN/empty abstract still rejects + falls through without
    # crashing (the search fall-through is the safety net; a successful-but-empty fetch is not an outage).
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": False})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="Some Paper", abstract="")   # fetch_error False, empty abstract
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert (out.get("extracted_ids") or {}).get("paper_url") is None


def test_correct_preset_confident_match_not_over_rejected(monkeypatch):
    # gsm8k control: its REAL paper is a confident match (is_match=True, conf=1.0). Fix 1 must not touch
    # the accept path -- the binding is KEPT/confirmed, never over-rejected.
    counter = _LLMCounter({"is_match": True, "confidence": 1.0, "error": False,
                           "reasoning": "this paper introduces GSM8K"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="Training Verifiers to Solve Math Word Problems",
                abstract="grade school math word problems dataset and verifiers")
    st = _state("gsm8k", extracted={"paper_url": "https://arxiv.org/abs/2110.14168"},
                eee={"benchmark_name": "GSM8K", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2110.14168"
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "confirmed"


def test_realistic_nomatch_with_error_keeps_degraded(monkeypatch):
    # Control for the false-drop risk: conf=0.0 AND error=True (an LLM outage) still KEEPS degraded,
    # never drops -- error=True is the only outage signal that reaches the LLM block.
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": True})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Paper", abstract="some content for the verifier to read")
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/1234.56789"
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "unverified_degraded"


def test_nonerrored_missing_match_index_keeps_degraded(monkeypatch):
    # FINISH-8 fix-round nit: a non-errored verdict that carries a `match_index` key with a
    # missing/null/non-integer value (a degenerate parse result that slipped past error=True) is
    # AMBIGUOUS, not a decision -> keep-degraded, never reject. A correct pre-set must not be dropped
    # on a no-decision verdict; only a REAL not-a-match (or match_index "none") rejects.
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": False, "match_index": None})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Paper", abstract="some content for the verifier to read")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)  # must not be needed (no reject)
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/1234.56789"
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "unverified_degraded"


def test_nonerrored_match_index_none_string_still_rejects(monkeypatch):
    # The flip side: an explicit match_index=="none" IS a real decision -> reject + fall through (the
    # guard must not over-suppress a genuine no-match that carries the string sentinel).
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": False, "match_index": "none"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Wrong Paper", abstract="unrelated content for the verifier")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert (out.get("extracted_ids") or {}).get("paper_url") is None


def test_verdict_has_decision_unit():
    # The decision-validity helper: absent match_index -> is_match is authoritative (decision);
    # present-but-garbage -> no decision; valid int / "none" -> decision.
    assert W._verdict_has_decision({"is_match": False}) is True              # no match_index key
    assert W._verdict_has_decision({"is_match": False, "match_index": None}) is False
    assert W._verdict_has_decision({"is_match": False, "match_index": "garbage"}) is False
    assert W._verdict_has_decision({"is_match": False, "match_index": True}) is False  # bool, not int
    assert W._verdict_has_decision({"is_match": True, "match_index": 0}) is True
    assert W._verdict_has_decision({"is_match": False, "match_index": "none"}) is True


def test_reject_passes_prior_binding_to_search(monkeypatch):
    # On reject->fall-through, the dropped binding is carried into resolve_paper (prior_binding) so the
    # sidecar still records the bypass ("sidecar for every binding").
    seen = {}
    counter = _LLMCounter({"is_match": False, "confidence": 0.9, "reasoning": "wrong"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="Wrong Paper")

    def _capture_resolve(*a, **k):
        seen["prior_binding"] = k.get("prior_binding")
        return None
    monkeypatch.setattr(W, "resolve_paper", _capture_resolve)
    st = _state("mgsm", extracted={"paper_url": "https://arxiv.org/abs/2110.14168",
                                   "paper_url_from_hf": "https://arxiv.org/abs/2110.14168"})
    W.run_paper_resolver(st)
    pb = seen["prior_binding"]
    assert pb is not None
    assert pb["verdict"] == "rejected"
    assert pb["binding_source"] == "hf_extractor"  # paper_url_from_hf == paper_url


# --- COST GUARD: deterministic fast paths make ZERO LLM calls --------------------------------

def test_preset_name_bearing_with_positive_overlap_accepts_zero_llm(monkeypatch):
    # 0-LLM deterministic accept requires name-bearing title AND POSITIVE subject overlap (mirror of
    # HF Tier-1). A self-referential pre-set whose abstract coheres with the identity is kept, no LLM.
    counter = _LLMCounter({"is_match": False, "confidence": 1.0})  # would reject if called
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="CharXiv: Charting Gaps in Realistic Chart Understanding",
                abstract="charts realistic chart understanding reasoning over scientific figures and plots")
    eee = {"benchmark_name": "charxiv",
           "metrics": {"acc": {"evaluation_description":
                               "realistic chart understanding reasoning over scientific figures plots charts"}}}
    st = _state("charxiv", extracted={"paper_url": "https://arxiv.org/abs/2406.18521"}, eee=eee)
    out = W.run_paper_resolver(st)
    assert counter.calls == 0                        # cost guard
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2406.18521"
    assert out["paper_verified"] is True
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "confirmed"
    assert st["output_manager"].paper_sidecar()["binding"]["tier"] == "deterministic"


def test_preset_name_bearing_thin_abstract_wrong_domain_routes_to_llm_and_rejects(monkeypatch):
    # MUST-2 regression: a name-bearing pre-set with a THIN/empty abstract (e.g. an abstract-less
    # OpenAlex DOI record) on a WRONG-domain identity must NOT auto-accept at 0 LLM -- it routes to the
    # LLM verifier, which rejects on domain. (The pre-fix bug auto-accepted this with verdict=confirmed.)
    counter = _LLMCounter({"is_match": False, "confidence": 0.95, "reasoning": "video, not text-to-image"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="ActionBench: A Video Action Benchmark", abstract="")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    eee = {"benchmark_name": "ActionBench",
           "metrics": {"fid": {"evaluation_description": "text-to-image generation measured by FID"}}}
    st = _state("actionbench", extracted={"paper_url": "https://arxiv.org/abs/2305.10683"}, eee=eee)
    out = W.run_paper_resolver(st)
    assert counter.calls == 1                        # routed to the LLM, NOT auto-accepted
    assert (out.get("extracted_ids") or {}).get("paper_url") is None


def test_preset_name_bearing_thin_abstract_correct_samename_routes_to_llm_and_keeps(monkeypatch):
    # The flip side: a CORRECT same-name pre-set with a thin abstract also routes to the LLM (no
    # false-accept loss of cost-guard precision) and the LLM keeps it on a confident match.
    counter = _LLMCounter({"is_match": True, "confidence": 0.9, "reasoning": "the CharXiv paper"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="CharXiv: Charting Gaps in Realistic Chart Understanding", abstract="")
    eee = {"benchmark_name": "charxiv",
           "metrics": {"acc": {"evaluation_description": "chart understanding"}}}
    st = _state("charxiv", extracted={"paper_url": "https://arxiv.org/abs/2406.18521"}, eee=eee)
    out = W.run_paper_resolver(st)
    assert counter.calls == 1                        # thin -> LLM, not deterministic auto-accept
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2406.18521"
    assert out["paper_verified"] is True


def test_preset_fetch_error_keeps_degraded_no_llm(monkeypatch):
    # SHOULD-3: a paper-metadata FETCH failure keeps the binding degraded and does NOT call the LLM on
    # a blank candidate (a transient outage must not drop a correct pre-set via an LLM "none").
    counter = _LLMCounter({"is_match": False, "confidence": 0.95})  # would reject if called
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="", abstract="", fetch_error=True)
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert counter.calls == 0                        # never verify a blank candidate
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/1234.56789"
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "unverified_degraded"


def test_preset_different_declared_name_rejects_zero_llm(monkeypatch):
    # A bound paper that plainly DECLARES a different benchmark name rejects deterministically (no LLM).
    counter = _LLMCounter({"is_match": True, "confidence": 1.0})  # would keep if called
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    st = _state("beyond-aime", extracted={"paper_url": "https://arxiv.org/abs/2510.11653"},
                eee={"benchmark_name": "Beyond AIME", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert counter.calls == 0                        # cost guard: deterministic name guard
    assert (out.get("extracted_ids") or {}).get("paper_url") is None


# --- SAME-NAME, DIFFERENT-DOMAIN: the actionbench class ---------------------------------------

def test_samename_domain_reject_via_llm(monkeypatch):
    # Paxion introduces a video-language ActionBench; identity is text-to-image. LLM confident reject.
    counter = _LLMCounter({"is_match": False, "confidence": 0.9, "reasoning": "video-language domain"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="Paxion: Patching Action Knowledge",
                abstract="video-language foundation models for action recognition in videos")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    eee = {"benchmark_name": "ActionBench",
           "metrics": {"fid": {"evaluation_description": "text-to-image generation measured by FID"}}}
    st = _state("actionbench", extracted={"paper_url": "https://arxiv.org/abs/2305.10683"}, eee=eee)
    out = W.run_paper_resolver(st)
    assert counter.calls == 1
    assert (out.get("extracted_ids") or {}).get("paper_url") is None


def test_samename_domain_reject_via_deterministic_backstop(monkeypatch):
    # Even with a HESITANT LLM (conf < 0.7), a same-name binding whose abstract is near-disjoint from
    # the identity subject is dropped by the deterministic subject-overlap backstop.
    counter = _LLMCounter({"is_match": True, "confidence": 0.4})  # hesitant, would-be keep
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="ActionBench: A Video Benchmark",
                abstract="monocular video reconstruction of 3D human body meshes from frames camera")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    eee = {"benchmark_name": "ActionBench",
           "metrics": {"fid": {"evaluation_description":
                               "text-to-image diffusion generation quality scored by frechet distance"}}}
    st = _state("actionbench", extracted={"paper_url": "https://arxiv.org/abs/2305.10683"}, eee=eee)
    out = W.run_paper_resolver(st)
    assert counter.calls == 1
    assert (out.get("extracted_ids") or {}).get("paper_url") is None
    # The dropped binding is carried to search as prior_binding (rejected by the backstop).


# --- CORRECT PAPER STILL VERIFIES ------------------------------------------------------------

def test_correct_preset_kept_via_llm(monkeypatch):
    # A coherent same-name pre-set whose title does NOT bear the distinctive name (so it routes to the
    # LLM) is kept on a confident match.
    counter = _LLMCounter({"is_match": True, "confidence": 0.9, "reasoning": "introduces it"})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Hand-Built Benchmark for Bias",
                abstract="bias in question answering across social dimensions")
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert counter.calls == 1
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/1234.56789"
    assert out["paper_verified"] is True
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "confirmed"


def test_llm_error_keeps_degraded(monkeypatch):
    # Under an LLM outage (error=True), an unverified pre-set is kept DEGRADED, not dropped (mirror of
    # the HF unverified_degraded keep). Never lose a possibly-correct binding under an outage.
    counter = _LLMCounter({"is_match": False, "confidence": 0.0, "error": True})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Descriptive Paper Title", abstract="some content")
    st = _state("widgetqa", extracted={"paper_url": "https://arxiv.org/abs/1234.56789"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/1234.56789"
    assert st["output_manager"].paper_sidecar()["binding"]["verdict"] == "unverified_degraded"


# --- TIER-1 HF-README binding is verified, not trusted raw ------------------------------------

def test_hf_readme_binding_verified_and_rejected(monkeypatch):
    # No pre-set paper_url; _extract_paper_from_hf yields a README arxiv tag. It is VERIFIED before
    # trusting: confident reject -> fall through to search.
    counter = _LLMCounter({"is_match": False, "confidence": 0.9})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Foreign Paper")
    monkeypatch.setattr(W, "_extract_paper_from_hf", lambda hf: "https://arxiv.org/abs/2110.14168")
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)
    st = _state("mgsm", extracted={}, hf_json={"id": "juletxara/mgsm"})
    out = W.run_paper_resolver(st)
    assert counter.calls == 1
    assert (out.get("extracted_ids") or {}).get("paper_url") is None


def test_hf_readme_binding_kept_when_verified(monkeypatch):
    counter = _LLMCounter({"is_match": True, "confidence": 0.9})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    _patch_meta(monkeypatch, title="A Descriptive Self Paper", abstract="aligns with identity")
    monkeypatch.setattr(W, "_extract_paper_from_hf", lambda hf: "https://arxiv.org/abs/9999.00001")
    st = _state("widgetqa", extracted={}, hf_json={"id": "org/widgetqa"},
                eee={"benchmark_name": "WidgetQA", "metrics": {}})
    out = W.run_paper_resolver(st)
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/9999.00001"
    assert out["paper_verified"] is True


# --- FLAG OFF: byte-identical pre-feature behavior -------------------------------------------

def test_flag_off_preset_skips_no_verification(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", False)
    counter = _LLMCounter({"is_match": False, "confidence": 1.0})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    st = _state("mgsm", extracted={"paper_url": "https://arxiv.org/abs/2110.14168"})
    out = W.run_paper_resolver(st)
    assert counter.calls == 0
    assert out == {"paper_resolver_attempted": True,
                   "completed": ["paper_resolver skipped (paper_url already set)"]}


def test_flag_off_hf_readme_trusted_raw(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", False)
    counter = _LLMCounter({"is_match": False, "confidence": 1.0})
    monkeypatch.setattr(W, "verify_paper_binding", counter)
    monkeypatch.setattr(W, "_extract_paper_from_hf", lambda hf: "https://arxiv.org/abs/2110.14168")
    st = _state("mgsm", extracted={}, hf_json={"id": "juletxara/mgsm"})
    out = W.run_paper_resolver(st)
    assert counter.calls == 0
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2110.14168"
    assert "completed" in out and "hf_readme" in out["completed"][0]


# --- ROUTING: the orchestrator routes a set-but-unverified paper to the resolver, no re-loop ---

def _route(state):
    return workflow.orchestrator(state)["next"]


def test_routing_preset_unverified_goes_to_resolver(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", True)
    st = {"completed": ["hf done"], "unitxt_json": None, "docling_output": None,
          "extracted_ids": {"paper_url": "https://arxiv.org/abs/x"},
          "hf_repo": "org/x", "hf_json": {"id": "org/x"}, "eee_metadata": {"benchmark_name": "x"},
          "hf_extraction_attempted": True, "paper_resolver_attempted": False, "paper_verified": None}
    assert _route(st) == "paper_resolver_worker"


def test_routing_no_reloop_after_verification(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", True)
    base = {"completed": ["hf done"], "unitxt_json": None, "docling_output": None,
            "extracted_ids": {"paper_url": "https://arxiv.org/abs/x"},
            "hf_repo": "org/x", "hf_json": {"id": "org/x"}, "eee_metadata": {"benchmark_name": "x"},
            "hf_extraction_attempted": True}
    # After the worker ran once: paper_resolver_attempted True (the loop guard) -> never re-routes here,
    # regardless of paper_verified, even though paper_url is still set.
    for verified in (True, None, False):
        st = dict(base, paper_resolver_attempted=True, paper_verified=verified)
        assert _route(st) != "paper_resolver_worker", verified


def test_routing_flag_off_preset_skips_resolver(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", False)
    st = {"completed": ["hf done"], "unitxt_json": None, "docling_output": None,
          "extracted_ids": {"paper_url": "https://arxiv.org/abs/x"},
          "hf_repo": "org/x", "hf_json": {"id": "org/x"}, "eee_metadata": {"benchmark_name": "x"},
          "hf_extraction_attempted": True, "paper_resolver_attempted": False, "paper_verified": None}
    # Flag OFF: a pre-set paper_url skips the resolver exactly as before -> docling.
    assert _route(st) == "docling_worker"
