"""Wiring tests for HF-match verification: orchestrator routing (no hf_worker loop on reject),
the descriptive gate flag (does not move GO/NO-GO), the _subject_overlap refactor (byte-behavior
identical _hf_subject_conflicts), and the eee_metadata existing_hf_repo passthrough.

Network-free.
"""

import importlib.util
import json
from pathlib import Path

from auto_benchmarkcard.workflow import orchestrator
from auto_benchmarkcard.tools.composer.composer_tool import _subject_overlap, _hf_subject_conflicts


def _load_cgm():
    p = Path(__file__).resolve().parents[1] / "scripts" / "compute_gate_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_gate_metrics", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_state(**over):
    """A complete orchestrator state with everything resolved up to the HF step."""
    s = {
        "query": "demo", "catalog_path": None,
        "unitxt_json": {}, "extracted_ids": {"paper_url": "https://arxiv.org/abs/2401.0"},
        "hf_repo": None, "hf_json": None, "hf_rejected": None,
        "docling_output": {}, "html_content": {}, "github_readme": {},
        "eee_metadata": {"benchmark_name": "demo"}, "hf_extraction_attempted": True,
        "paper_resolver_attempted": True, "composed_card": {}, "risk_enhanced_card": {},
        "rag_results": {}, "factuality_results": {}, "completed": [],
    }
    s.update(over)
    return s


# --- orchestrator routing: reject must not loop into hf_worker -----------------------

def test_reject_state_does_not_route_to_hf_worker():
    # hf_repo cleared to None + hf_rejected True (the run_hf reject return).
    st = _base_state(hf_repo=None, hf_rejected=True, hf_json=None)
    assert orchestrator(st)["next"] != "hf_worker"


def test_hf_rejected_guard_blocks_even_if_repo_set():
    # Belt-and-suspenders: even if hf_repo were still set, hf_rejected blocks the hf_worker clause.
    st = _base_state(hf_repo="acme/x", hf_rejected=True, hf_json=None)
    assert orchestrator(st)["next"] != "hf_worker"


def test_unrejected_unfetched_repo_still_routes_to_hf_worker():
    # The normal pre-fetch case still routes in (no regression to the guard).
    st = _base_state(hf_repo="acme/x", hf_rejected=None, hf_json=None)
    assert orchestrator(st)["next"] == "hf_worker"


# --- gate flag: descriptive, does not change GO/NO-GO --------------------------------

def _write_hf_sidecar(run_dir, card, verdict, repo_id="acme/x", reason="r"):
    d = run_dir / card / "tool_output" / "hf_verifier"
    d.mkdir(parents=True)
    (d / "hf-verification.json").write_text(json.dumps(
        {"benchmark": card, "repo_id": repo_id, "verdict": verdict, "reason": reason}))


def test_gate_hf_match_flags_count_reject_unverified_and_error(tmp_path):
    cgm = _load_cgm()
    _write_hf_sidecar(tmp_path, "cardA", "rejected")
    _write_hf_sidecar(tmp_path, "cardB", "unverified_degraded")
    _write_hf_sidecar(tmp_path, "cardC", "confirmed")        # not flagged
    _write_hf_sidecar(tmp_path, "cardD", "verifier_error")   # LOUD: counted
    flags = cgm._hf_match_flags(str(tmp_path))
    kinds = sorted(f["kind"] for f in flags)
    assert kinds == ["hf_match_error", "hf_match_rejected", "hf_match_unverified"]


def test_gate_hf_verification_summary_exposes_denominator(tmp_path):
    # The per-verdict summary is the denominator: a mass verifier crash (many verifier_error,
    # zero rejected/unverified flags) is visible because total > 0 with the crash count broken out.
    cgm = _load_cgm()
    _write_hf_sidecar(tmp_path, "c1", "confirmed")
    _write_hf_sidecar(tmp_path, "c2", "confirmed")
    _write_hf_sidecar(tmp_path, "c3", "verifier_error")
    summ = cgm._hf_verification_summary(str(tmp_path))
    assert summ["total"] == 3
    assert summ["by_verdict"] == {"confirmed": 2, "verifier_error": 1}


def _write_min_card(run_dir, name):
    """A minimal per-card benchmarkcard so compute()/gather_cards sees one card."""
    d = run_dir / f"{name}_run" / "benchmarkcard"
    d.mkdir(parents=True)
    (d / f"benchmark_card_{name}.json").write_text(json.dumps(
        {"benchmark_details": {"name": name, "overview": "x"}}))
    return run_dir / f"{name}_run"


def test_gate_verdict_independent_of_hf_flag(tmp_path):
    """Differential: two run dirs identical except one carries a rejected HF sidecar. The GO/NO-GO
    verdict block must be byte-identical -- the descriptive HF flag does not feed _verdict."""
    cgm = _load_cgm()
    import inspect
    # Structural guarantee: _verdict cannot even see the HF flags (not in its signature).
    assert "hf" not in str(inspect.signature(cgm._verdict)).lower()

    # Behavioural guarantee: adding the HF sidecar changes the flag count but NOT the verdict.
    dir_a = tmp_path / "a"
    card_a = _write_min_card(dir_a, "demo")
    metrics_a = cgm.compute(str(dir_a))

    dir_b = tmp_path / "b"
    card_b = _write_min_card(dir_b, "demo")
    hf_dir = card_b / "tool_output" / "hf_verifier"
    hf_dir.mkdir(parents=True)
    (hf_dir / "hf-verification.json").write_text(json.dumps(
        {"benchmark": "demo", "repo_id": "acme/x", "verdict": "rejected", "reason": "r"}))
    metrics_b = cgm.compute(str(dir_b))

    assert metrics_b["hf_match_verification"]["count"] == 1   # flag IS counted
    assert metrics_a["hf_match_verification"]["count"] == 0
    assert metrics_a["verdict"] == metrics_b["verdict"]        # but verdict is unchanged


# --- paper-binding gate flag: descriptive, does not change GO/NO-GO (FINISH-7) -------------------

def _binding(verdict, source="eee_unitxt", url="https://arxiv.org/abs/x", reason="r"):
    return {"binding_source": source, "paper_url": url, "verdict": verdict, "reason": reason}


def test_gate_paper_binding_flags_count_reject_and_unverified():
    cgm = _load_cgm()
    # rejected and unverified_degraded are flagged; confirmed / skipped_curated are not.
    assert [f["kind"] for f in cgm._paper_binding_flags("c", {"binding": _binding("rejected")})] \
        == ["paper_binding_rejected"]
    assert [f["kind"] for f in cgm._paper_binding_flags("c", {"binding": _binding("unverified_degraded")})] \
        == ["paper_binding_unverified"]
    assert cgm._paper_binding_flags("c", {"binding": _binding("confirmed")}) == []
    assert cgm._paper_binding_flags("c", {"binding": _binding("skipped_curated")}) == []
    assert cgm._paper_binding_flags("c", {"sources_tried": []}) == []


def test_gate_paper_binding_flags_surface_prior_binding_on_fallthrough():
    # A pre-set rejected then fell through to a curated/search KEEP: the final binding is kept but the
    # dropped pre-set is still surfaced from the top-level prior_binding ("never silent").
    cgm = _load_cgm()
    pv = {"binding": _binding("skipped_curated", source="curated"),
          "prior_binding": _binding("rejected", source="eee_unitxt")}
    kinds = [f["kind"] for f in cgm._paper_binding_flags("c", pv)]
    assert kinds == ["paper_binding_rejected"]


def test_gate_verdict_independent_of_paper_binding_flag(tmp_path):
    """Differential: the descriptive paper-binding flag must not feed GO/NO-GO. _verdict cannot even
    see it (not in its signature), and adding a rejected binding sidecar changes the flag count but
    not the verdict."""
    cgm = _load_cgm()
    import inspect
    sig = str(inspect.signature(cgm._verdict)).lower()
    assert "binding" not in sig and "paper" not in sig

    dir_a = tmp_path / "a"
    _write_min_card(dir_a, "demo")
    metrics_a = cgm.compute(str(dir_a))

    dir_b = tmp_path / "b"
    card_b = _write_min_card(dir_b, "demo")
    pr_dir = card_b / "tool_output" / "paper_resolver"
    pr_dir.mkdir(parents=True)
    (pr_dir / "paper-verification.json").write_text(json.dumps(
        {"suite": "demo", "binding": _binding("rejected")}))
    metrics_b = cgm.compute(str(dir_b))

    assert metrics_b["paper_binding_verification"]["count"] == 1
    assert metrics_a["paper_binding_verification"]["count"] == 0
    assert metrics_a["verdict"] == metrics_b["verdict"]


# --- _subject_overlap refactor: _hf_subject_conflicts unchanged ----------------------

_DISJOINT_README = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
_DISJOINT_IDENTITY = "one two three four five six seven eight nine ten eleven twelve"
_ALIGNED_IDENTITY = "alpha beta gamma delta epsilon zeta eta theta iota kappa"


def test_subject_overlap_none_on_thin_input():
    assert _subject_overlap("alpha beta", "one two") is None


def test_subject_overlap_float_on_full_input():
    ov = _subject_overlap(_DISJOINT_README, _ALIGNED_IDENTITY)
    assert ov is not None and ov > 0.5


def test_hf_subject_conflicts_fires_on_disjoint():
    assert _hf_subject_conflicts(_DISJOINT_README, _DISJOINT_IDENTITY) is True


def test_hf_subject_conflicts_false_on_aligned():
    assert _hf_subject_conflicts(_DISJOINT_README, _ALIGNED_IDENTITY) is False


def test_hf_subject_conflicts_false_on_thin():
    assert _hf_subject_conflicts("alpha beta", "one two") is False


# --- eee_metadata carries the raw existing_hf_repo for telemetry ---------------------

def test_eee_metadata_carries_existing_hf_repo():
    from auto_benchmarkcard.tools.eee.eee_tool import EEEBenchmarkInfo, eee_to_pipeline_inputs
    bench = EEEBenchmarkInfo(name="demo", source_type="hf_dataset", hf_repo="acme/demo")
    out = eee_to_pipeline_inputs(bench)
    assert out["eee_metadata"]["existing_hf_repo"] == "acme/demo"
