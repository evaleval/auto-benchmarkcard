"""Tests for the resolve-time HF-match 3-tier gate in run_hf.

Network-free: hf_dataset_metadata.func and verify_hf_match are monkeypatched. The LLM is mocked via
a call-counter so Tier-0 / Tier-1 can be asserted to make ZERO LLM calls (the cost guard). Checks
are structural mechanisms, never keyed on a benchmark-name list.
"""

import pytest

import auto_benchmarkcard.workers as W
from auto_benchmarkcard.config import Config


class _FakeOM:
    """Captures the per-tool sidecar writes so tests can assert the verdict."""

    def __init__(self):
        self.writes = {}

    def save_tool_output(self, data, tool, filename):
        self.writes[(tool, filename)] = data
        return f"/tmp/{tool}/{filename}"

    def get_tool_output_path(self, tool, create_if_missing=True):
        return f"/tmp/{tool}"

    def hf_sidecar(self):
        return self.writes.get(("hf_verifier", "hf-verification.json"))


def _state(name, repo_id, *, eee=None, extracted=None):
    return {
        "query": name,
        "hf_repo": repo_id,
        "eee_metadata": eee if eee is not None else {"benchmark_name": name, "metrics": {}},
        "extracted_ids": extracted or {},
        "output_manager": _FakeOM(),
    }


def _hf_data(repo_id, *, readme="", pretty_name=None, tags=None, task_categories=None):
    return {
        "id": repo_id,
        "card_data": {"pretty_name": pretty_name, "task_categories": task_categories or []},
        "readme_markdown": readme,
        "tags": tags or [],
    }


def _patch_hf(monkeypatch, hf_data):
    class _T:
        @staticmethod
        def func(repo_id):
            return hf_data
    monkeypatch.setattr(W, "hf_dataset_metadata", _T)


class _LLMCounter:
    def __init__(self, verdict):
        self.calls = 0
        self._verdict = verdict

    def __call__(self, identity, candidate):
        self.calls += 1
        return dict(self._verdict)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(Config, "HF_VERIFICATION_ENABLED", True)


# --- Tier 0: deterministic reject, ZERO LLM ---------------------------------------

def test_tier0_aggregate_repo_rejects_zero_llm(monkeypatch):
    # repo basename bears no whole-token run of the name -> aggregate_repo.
    counter = _LLMCounter({"is_match": True, "confidence": 1.0})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    _patch_hf(monkeypatch, _hf_data("human-centered-eval/some_aggregate",
                                    readme="An aggregate of many benchmarks."))
    st = _state("WidgetBench", "human-centered-eval/some_aggregate")
    out = W.run_hf(st)
    assert counter.calls == 0                       # cost guard: no LLM at Tier 0
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert "hf_json" not in out
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


# --- Tier 1: deterministic accept, ZERO LLM ---------------------------------------

def test_tier1_corroborated_coined_high_overlap_accepts_zero_llm(monkeypatch):
    counter = _LLMCounter({"is_match": False, "confidence": 1.0})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    readme = ("WidgetBench is a benchmark of factory schematics for widget assembly reasoning "
              "over mechanical parts and gears, measuring assembly accuracy.")
    _patch_hf(monkeypatch, _hf_data("acme/WidgetBench", readme=readme))
    eee = {"benchmark_name": "WidgetBench",
           "metrics": {"acc": {"evaluation_description":
                               "widget assembly reasoning over factory schematics mechanical parts gears"}}}
    st = _state("WidgetBench", "acme/WidgetBench", eee=eee)
    out = W.run_hf(st)
    assert counter.calls == 0                       # cost guard: no LLM at Tier 1
    assert out["hf_json"]["id"] == "acme/WidgetBench"
    assert st["output_manager"].hf_sidecar()["verdict"] == "confirmed"
    assert st["output_manager"].hf_sidecar()["tier"] == "accept"


# --- Tier 2: LLM verify --------------------------------------------------------------

def test_tier2_actionbench_reject_closes_paper_leak(monkeypatch):
    # actionbench: corroborated (None) + coined (True) but near-disjoint overlap -> Tier 2, LLM
    # rejects on domain. After reject, run_paper_resolver must NOT surface the repo's arxiv tag.
    counter = _LLMCounter({"is_match": False, "confidence": 0.95, "reasoning": "3D-mesh domain"})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    readme = "ActionMesh: reconstructing 3D human body meshes from monocular video frames."
    hf_data = _hf_data("facebook/actionbench", readme=readme,
                       tags=["arxiv:2601.16148"])
    hf_data["card_data"]["tags"] = ["arxiv:2601.16148"]
    _patch_hf(monkeypatch, hf_data)
    eee = {"benchmark_name": "ActionBench",
           "metrics": {"fid": {"evaluation_description":
                               "text-to-image generation of human actions measured by FID"}}}
    st = _state("ActionBench", "facebook/actionbench", eee=eee)
    out = W.run_hf(st)
    assert counter.calls == 1                       # routed to Tier 2
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert "hf_json" not in out
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"

    # Hermetic: no live OpenAlex/S2/RITS -- mock resolve_paper so the only way 2601.16148 could
    # surface is the HF-README/tag tier (_extract_paper_from_hf reads state["hf_json"]).
    monkeypatch.setattr(W, "resolve_paper", lambda *a, **k: None)

    # Build the paper-resolver state from run_hf's ACTUAL reject output (not a hand-built None):
    # the reject return omits hf_json, so state.get("hf_json") is falsy and the tag is unreachable.
    leak_state = {"query": "ActionBench", "extracted_ids": {}, "eee_metadata": eee,
                  "output_manager": _FakeOM()}
    leak_state.update(out)              # carries hf_repo=None, hf_rejected=True, NO hf_json
    assert not leak_state.get("hf_json")
    pr_out = W.run_paper_resolver(leak_state)
    leaked = (pr_out.get("extracted_ids") or {}).get("paper_url") or ""
    assert "2601.16148" not in leaked

    # Non-vacuous control: had the binding been KEPT (hf_json present), the SAME README arxiv tag WOULD
    # leak via _extract_paper_from_hf -- proving the closed-leak above is the HF reject, not the test
    # construction. Demonstrated with PAPER_VERIFICATION_ENABLED OFF (the pre-feature raw-trust path);
    # with the FINISH-7 paper-binding gate ON that README tag is now itself verified+rejected, which is
    # a SECOND independent closure (covered by tests/test_paper_binding_gate.py).
    monkeypatch.setattr(Config, "PAPER_VERIFICATION_ENABLED", False)
    kept_state = {"query": "ActionBench", "extracted_ids": {}, "eee_metadata": eee,
                  "hf_json": hf_data, "output_manager": _FakeOM()}
    kept_out = W.run_paper_resolver(kept_state)
    kept_leaked = (kept_out.get("extracted_ids") or {}).get("paper_url") or ""
    assert "2601.16148" in kept_leaked


def test_tier2_generic_name_rejects(monkeypatch):
    # Generic (not name-like) exact match with a thin/unrelated README -> Tier 2 -> LLM rejects.
    counter = _LLMCounter({"is_match": False, "confidence": 0.85, "reasoning": "generic name, no corroboration"})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger survival records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert counter.calls == 1
    assert out["hf_repo"] is None and out["hf_rejected"] is True


def test_tier2_mbpp_accept(monkeypatch):
    # mbpp is NOT name-like (lowercase, no caps/digit/camelCase) -> Tier 2, LLM accepts the
    # self-describing repo. (Documents that mbpp does not land at Tier 1.)
    counter = _LLMCounter({"is_match": True, "confidence": 0.9, "reasoning": "mbpp dataset"})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    _patch_hf(monkeypatch, _hf_data("google-research-datasets/mbpp",
                                    readme="Mostly Basic Python Problems (MBPP) coding benchmark."))
    st = _state("mbpp", "google-research-datasets/mbpp")
    out = W.run_hf(st)
    assert counter.calls == 1
    assert out["hf_json"]["id"] == "google-research-datasets/mbpp"
    assert st["output_manager"].hf_sidecar()["verdict"] == "confirmed"


# --- Tier 2 error / low-confidence policy -------------------------------------------

def test_llm_error_corroborated_coined_keeps_degraded(monkeypatch):
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    readme = "DistinctBench is a benchmark; the README is too thin to clear the Tier-1 overlap floor."
    _patch_hf(monkeypatch, _hf_data("acme/DistinctBench", readme=readme))
    st = _state("DistinctBench", "acme/DistinctBench")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/DistinctBench"   # corroborated+coined -> KEEP
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


def test_llm_error_generic_rejects(monkeypatch):
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger survival records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True   # generic -> REJECT
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


# --- type-coercion of the LLM verdict (a quoted bool / string conf must not silently keep) -------

def test_quoted_false_is_match_does_not_keep_generic(monkeypatch):
    # The model emits "is_match": "false" (a quoted boolean -> truthy string). A generic repo must
    # NOT be kept as 'confirmed'; the coercion reads "false" as not-a-match -> REJECT.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": "false", "confidence": 0.97})
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger survival records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


def test_quoted_true_is_match_still_accepts(monkeypatch):
    # "is_match": "true" must still be read as a match (coercion is not over-aggressive).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": "true", "confidence": 0.9})
    _patch_hf(monkeypatch, _hf_data("acme/widget", readme="Widget dataset."))
    st = _state("widget", "acme/widget")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/widget"
    assert st["output_manager"].hf_sidecar()["verdict"] == "confirmed"


def test_string_confidence_does_not_crash_into_keep(monkeypatch):
    # A non-numeric confidence ("high") must not raise (which run_hf's outer guard would mask into a
    # keep). It coerces to 0.0 -> the low-confidence policy rejects a generic repo.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": "high"})
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger survival records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"
    # The verdict is a real reject decision, not the verifier_error catch-all.
    assert "verifier_error" not in st["output_manager"].hf_sidecar()["verdict"]


# --- FINISH-7 G1-HF-1: empty/contentless repo never authority-kept -------------------

def test_empty_repo_rejected_even_if_corroborated_coined(monkeypatch):
    # acadreason fixture: Ross12/Acadreason is corroborated (basename bears the coined name) but its
    # README is only YAML frontmatter and card_data is empty -> contentless. The LLM (live) returns
    # is_match=false conf=0.0; the authority-keep policy would otherwise KEEP it (unverified_degraded)
    # and leak its MIT license. The empty-repo guard must REJECT instead.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0,
                                      "reasoning": "no README, no content to corroborate"})
    _patch_hf(monkeypatch, _hf_data("Ross12/Acadreason",
                                    readme="---\nlicense: mit\n---\n", tags=["license:mit"]))
    st = _state("ACADREASON", "Ross12/Acadreason")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert "hf_json" not in out                                  # no MIT-tag splice reaches the card
    sc = st["output_manager"].hf_sidecar()
    assert sc["verdict"] == "rejected"
    assert "empty/contentless" in sc["reason"]


def test_content_bearing_corroborated_coined_still_kept_on_weak_verdict(monkeypatch):
    # CONTROL (activitynet/charxiv class): a corroborated+coined repo WITH a real README must still be
    # authority-kept on a weak/errored verdict -- the empty-repo guard keys on empty CONTENT, never on
    # subject-overlap or a name, so a content-bearing repo is never collateral.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    readme = ("ActivityNet is a large-scale video benchmark of untrimmed human activity clips with "
              "train, validation and test splits for temporal activity localization.")
    _patch_hf(monkeypatch, _hf_data("YimuWang/ActivityNet", readme=readme))
    st = _state("ActivityNet", "YimuWang/ActivityNet")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "YimuWang/ActivityNet"        # content-bearing -> KEEP (degraded)
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


def test_empty_repo_rejected_frontmatter_without_trailing_newline(monkeypatch):
    # finding [7]: a frontmatter-only README whose closing fence sits at EOF (NO trailing newline)
    # must still be seen as contentless -- the composer's lead-strip needs a trailing '\n', so the
    # predicate strips the EOF frontmatter itself. Without this, the MIT-only repo would be kept.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    _patch_hf(monkeypatch, _hf_data("Ross12/Acadreason",
                                    readme="---\nlicense: mit\n---", tags=["license:mit"]))
    st = _state("ACADREASON", "Ross12/Acadreason")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


def test_contentless_repo_via_card_data_present_is_kept(monkeypatch):
    # An empty README but a real card_data.description IS content -> not contentless -> authority-kept.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    hf = _hf_data("acme/CoinedBench", readme="", pretty_name="Coined Bench")
    hf["card_data"]["description"] = "A benchmark of widget assembly reasoning tasks over schematics."
    _patch_hf(monkeypatch, hf)
    st = _state("CoinedBench", "acme/CoinedBench")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/CoinedBench"
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


def test_content_bearing_readme_ending_in_hr_dashes_is_kept(monkeypatch):
    # Regression for the SHOULD-5 frontmatter-EOF strip: a content-bearing README that STARTS with
    # frontmatter AND ENDS with a bare '---' horizontal rule must NOT be over-stripped to contentless
    # (the EOF-tolerant strip is bounded to a single frontmatter block). On a weak/errored verdict the
    # corroborated+coined repo is KEPT (degraded), not false-rejected.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    readme = ("---\nlicense: mit\npretty_name: RealBench\n---\n\n# RealBench\n\n"
              "RealBench measures spatial reasoning over 5000 examples with train/test splits.\n\n---")
    _patch_hf(monkeypatch, _hf_data("acme/RealBench", readme=readme))
    st = _state("RealBench", "acme/RealBench")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/RealBench"               # real content -> kept (degraded)
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


# --- FINISH-7 G1-HF-2: HF reject clears extracted_ids["hf_repo"] (org/logo/resources leak) --------

def test_reject_clears_extracted_hf_repo_preserving_other_ids(monkeypatch):
    # actionbench fixture: a confident LLM reject must also drop the rejected repo from extracted_ids,
    # so the composer cannot derive org_url/logo (apply_display_fields) or resources[hf] (hf_url) from
    # it. ONLY hf_repo is cleared; every other extracted id is preserved (paper_url etc.).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 1.0,
                                      "reasoning": "name collision, different domain"})
    _patch_hf(monkeypatch, _hf_data("facebook/actionbench",
                                    readme="A paired video-3D synthetic benchmark for animated 3D mesh "
                                           "generation from video."))
    extracted = {"hf_repo": "facebook/actionbench",
                 "paper_url": "https://arxiv.org/abs/2311.15841",
                 "paper_url_from_hf": None, "risk_tags": ["bias"]}
    st = _state("ActionBench", "facebook/actionbench", extracted=extracted)
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert out["extracted_ids"]["hf_repo"] is None               # rejected repo dropped
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/2311.15841"  # preserved
    assert out["extracted_ids"]["risk_tags"] == ["bias"]         # preserved
    assert "paper_url_from_hf" in out["extracted_ids"]           # every other key preserved


def test_accept_does_not_touch_extracted_ids(monkeypatch):
    # A KEPT binding must not alter extracted_ids (no spurious churn on the accept path).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": True, "confidence": 0.95})
    _patch_hf(monkeypatch, _hf_data("acme/widget", readme="Widget dataset."))
    extracted = {"hf_repo": "acme/widget", "paper_url": "u"}
    st = _state("widget", "acme/widget", extracted=extracted)
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/widget"
    assert "extracted_ids" not in out                            # accept path leaves ids untouched


# --- flag-OFF + verifier-exception safety ---------------------------------------------

def test_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.setattr(Config, "HF_VERIFICATION_ENABLED", False)
    counter = _LLMCounter({"is_match": False, "confidence": 1.0})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    _patch_hf(monkeypatch, _hf_data("facebook/actionbench", readme="ActionMesh 3D meshes."))
    st = _state("ActionBench", "facebook/actionbench")
    out = W.run_hf(st)
    assert counter.calls == 0
    # Byte-identical to the pre-feature return: exactly hf_json + completed, nothing else.
    assert set(out) == {"hf_json", "completed"}
    assert out["completed"] == ["hf done"]
    assert out["hf_json"]["id"] == "facebook/actionbench"   # kept, no verification
    assert st["output_manager"].hf_sidecar() is None        # no sidecar when disabled


def test_verifier_exception_keeps_binding(monkeypatch):
    def _boom(identity, candidate):
        raise RuntimeError("verifier bug")
    monkeypatch.setattr(W, "verify_hf_match", _boom)
    # Force Tier 2 (generic name) so the verifier is actually invoked and raises.
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "kaggle/titanic"          # exception -> keep today's behavior
    assert st["output_manager"].hf_sidecar()["verdict"] == "verifier_error"


# --- BLOCKER 1: a PROSE not-a-match must not keep a generic repo ----------------------

@pytest.mark.parametrize("verdict_str", ["no match", "unsure", "n/a", "maybe", "false", "no", "NO MATCH"])
def test_prose_is_match_rejects_generic(monkeypatch, verdict_str):
    # is_match as prose / any non-truthy token must read as NOT a match (positive whitelist).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": verdict_str, "confidence": 0.92})
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger survival records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


@pytest.mark.parametrize("verdict_str", ["yes", "y", "match", "TRUE", "1"])
def test_truthy_tokens_still_accept(monkeypatch, verdict_str):
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": verdict_str, "confidence": 0.9})
    _patch_hf(monkeypatch, _hf_data("acme/widget", readme="Widget dataset."))
    st = _state("widget", "acme/widget")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/widget"
    assert st["output_manager"].hf_sidecar()["verdict"] == "confirmed"


# --- SHOULD-FIX 1: unified Tier-2 policy (authority raises the LLM's bar to override) -

def test_corroborated_coined_hesitant_reject_keeps_degraded(monkeypatch):
    # not-a-match at conf 0.6 (< 0.7 floor) on a corroborated+coined repo -> KEEP (unverified_degraded),
    # so a hesitant LLM does not false-reject a correct repo (activitynet/charxiv class).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.6})
    readme = "DistinctBench is a coined benchmark; README too thin to clear Tier-1 overlap."
    _patch_hf(monkeypatch, _hf_data("acme/DistinctBench", readme=readme))
    st = _state("DistinctBench", "acme/DistinctBench")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "acme/DistinctBench"
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


def test_corroborated_coined_confident_reject_still_rejects(monkeypatch):
    # not-a-match at conf 0.8 (>= 0.7) still rejects even a corroborated+coined repo (actionbench class:
    # a confident domain rejection overrides authority).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.8})
    _patch_hf(monkeypatch, _hf_data("acme/DistinctBench", readme="Thin coined readme."))
    st = _state("DistinctBench", "acme/DistinctBench")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


def test_generic_hesitant_reject_rejects(monkeypatch):
    # not-a-match at conf 0.6 on a GENERIC repo -> REJECT (no authority to keep it).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.6})
    _patch_hf(monkeypatch, _hf_data("kaggle/titanic", readme="Passenger survival records."))
    st = _state("titanic", "kaggle/titanic")
    out = W.run_hf(st)
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


# --- BLOCKER 2: EEE-provided abbreviation repos bypass Tier-0 and survive -------------

def _eee_state(name, repo_id, *, metrics=None):
    """An EEE-provided binding: eee_metadata carries existing_hf_repo == repo_id."""
    return _state(name, repo_id, eee={
        "benchmark_name": name, "existing_hf_repo": repo_id, "metrics": metrics or {}})


def test_eee_abbreviation_bypasses_tier0_and_llm_accepts(monkeypatch):
    # BIG-Bench Hard -> lukaemon/bbh: corrob == aggregate_repo (abbreviation), but repo_source==eee
    # bypasses Tier-0 and routes to Tier-2 where the LLM accepts the README.
    counter = _LLMCounter({"is_match": True, "confidence": 0.9})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    _patch_hf(monkeypatch, _hf_data("lukaemon/bbh", readme="BIG-Bench Hard: 23 challenging tasks."))
    st = _eee_state("BIG-Bench Hard", "lukaemon/bbh")
    out = W.run_hf(st)
    assert counter.calls == 1                                # routed to Tier-2, not Tier-0
    assert out["hf_json"]["id"] == "lukaemon/bbh"
    assert st["output_manager"].hf_sidecar()["repo_source"] == "eee"
    assert st["output_manager"].hf_sidecar()["verdict"] == "confirmed"


def test_eee_abbreviation_survives_llm_down(monkeypatch):
    # Same eee abbreviation under LLM outage: authority (repo_source==eee) keeps it degraded, NOT reject.
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    _patch_hf(monkeypatch, _hf_data("openai/grade-school-math", readme="GSM8K grade school math."))
    st = _eee_state("GSM8K", "openai/grade-school-math")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "openai/grade-school-math"
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


def test_eee_abbreviation_non_namelike_survives_llm_down(monkeypatch):
    # MMLU full name 'Massive Multitask Language Understanding' -> cais/mmlu: aggregate_repo AND
    # _token_is_namelike(full name)==False, so it survives ONLY by the eee authority (not corroborated-coined).
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": False, "confidence": 0.0, "error": True})
    _patch_hf(monkeypatch, _hf_data("cais/mmlu", readme="Massive Multitask Language Understanding."))
    st = _eee_state("Massive Multitask Language Understanding", "cais/mmlu")
    out = W.run_hf(st)
    assert out["hf_json"]["id"] == "cais/mmlu"
    assert st["output_manager"].hf_sidecar()["verdict"] == "unverified_degraded"


def test_search_aggregate_still_tier0_rejects(monkeypatch):
    # A SEARCH-path aggregate (no existing_hf_repo) still hits the Tier-0 reject (the exemption is
    # eee-only). Guards against the exemption leaking to the search path.
    counter = _LLMCounter({"is_match": True, "confidence": 1.0})
    monkeypatch.setattr(W, "verify_hf_match", counter)
    _patch_hf(monkeypatch, _hf_data("someorg/multi_benchmark_suite", readme="Many benchmarks."))
    st = _state("WidgetBench", "someorg/multi_benchmark_suite")   # default eee: no existing_hf_repo
    out = W.run_hf(st)
    assert counter.calls == 0                                # Tier-0, zero LLM
    assert out["hf_repo"] is None and out["hf_rejected"] is True
    assert st["output_manager"].hf_sidecar()["verdict"] == "rejected"


# --- MINOR: repo_source is decision-neutral on the NON-aggregate band -----------------

def test_repo_source_decision_neutral_non_aggregate(monkeypatch):
    # For a non-aggregate repo, eee vs search must yield the SAME keep/reject (only telemetry differs).
    # (For aggregate_repo they intentionally differ at Tier-0 by design -- BLOCKER 2.)
    monkeypatch.setattr(W, "verify_hf_match",
                        lambda i, c: {"is_match": True, "confidence": 0.9})
    readme = "WidgetBench dataset of factory schematics."
    _patch_hf(monkeypatch, _hf_data("acme/WidgetBench", readme=readme))

    st_search = _state("WidgetBench", "acme/WidgetBench")
    out_search = W.run_hf(st_search)
    st_eee = _eee_state("WidgetBench", "acme/WidgetBench")
    out_eee = W.run_hf(st_eee)

    # Same decision (both keep), different repo_source telemetry.
    assert ("hf_json" in out_search) == ("hf_json" in out_eee) is True
    assert st_search["output_manager"].hf_sidecar()["repo_source"] == "search"
    assert st_eee["output_manager"].hf_sidecar()["repo_source"] == "eee"
    assert (st_search["output_manager"].hf_sidecar()["verdict"]
            == st_eee["output_manager"].hf_sidecar()["verdict"])
