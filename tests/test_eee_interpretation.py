"""Tests for the post-composition methodology.interpretation injection (contract S4.6).

Network-free: exercises the deterministic EEE -> interpretation mapping, the
Not-specified overwrite guard, and the provenance-sidecar emission directly.
"""

import json

from auto_benchmarkcard.eee_workflow import (
    _derive_interpretation,
    _enrich_baseline_results,
    _inject_eee_interpretation,
    _record_derived_provenance,
    _record_identity_name_provenance,
)
from auto_benchmarkcard.output import OutputManager

# The EEE-derived provenance entry recorded for benchmark_type / appears_in
# (built inline at the injection site in process_single_benchmark).
_EEE_DERIVED = {
    "source": "eee",
    "evidence": "",
    "status": "derived",
    "verified": True,
    "evidence_ids": [],
}


def _eee(lower_is_better, *, use_metric_config=True, primary="accuracy"):
    """Build minimal EEE metadata carrying a lower_is_better signal."""
    es = {"primary_metric": primary}
    if use_metric_config:
        es["metric_config"] = {"lower_is_better": lower_is_better}
        metrics = {}
    else:
        metrics = {primary: {"lower_is_better": lower_is_better}}
    return {"evaluation_summary": es, "metrics": metrics}


def _ns_card():
    return {"methodology": {"interpretation": "Not specified", "metrics": ["Accuracy"]}}


def test_lower_is_better_true_maps_to_lower():
    assert _derive_interpretation(_eee(True)) == "lower_is_better"
    card = _ns_card()
    entry = _inject_eee_interpretation(card, _eee(True))
    assert card["methodology"]["interpretation"] == "lower_is_better"
    assert entry is not None


def test_lower_is_better_false_maps_to_higher():
    assert _derive_interpretation(_eee(False)) == "higher_is_better"
    card = _ns_card()
    entry = _inject_eee_interpretation(card, _eee(False))
    assert card["methodology"]["interpretation"] == "higher_is_better"
    assert entry is not None


def test_per_metric_fallback_when_no_metric_config():
    eee = _eee(True, use_metric_config=False)
    assert _derive_interpretation(eee) == "lower_is_better"
    card = _ns_card()
    assert _inject_eee_interpretation(card, eee) is not None
    assert card["methodology"]["interpretation"] == "lower_is_better"


def test_no_signal_leaves_interpretation_unchanged():
    for eee in ({}, {"evaluation_summary": {}}, {"metrics": {}, "evaluation_summary": {"primary_metric": "x"}}):
        assert _derive_interpretation(eee) is None
        card = _ns_card()
        entry = _inject_eee_interpretation(card, eee)
        assert entry is None
        assert card["methodology"]["interpretation"] == "Not specified"


def test_does_not_overwrite_existing_value():
    card = {"methodology": {"interpretation": "higher_is_better"}}
    entry = _inject_eee_interpretation(card, _eee(True))
    assert entry is None
    assert card["methodology"]["interpretation"] == "higher_is_better"


def test_injection_clears_stale_missing_fields_entry():
    card = _ns_card()
    card["missing_fields"] = ["methodology.interpretation", "data.source"]
    entry = _inject_eee_interpretation(card, _eee(True))
    assert entry is not None
    assert card["missing_fields"] == ["data.source"]


def test_provenance_entry_shape():
    entry = _inject_eee_interpretation(_ns_card(), _eee(True))
    assert entry == {
        "source": "eee",
        "evidence": "",
        "status": "derived",
        "verified": True,
        "evidence_ids": [],
    }


def test_sidecar_merge_preserves_existing_entries(tmp_path):
    om = OutputManager("demo-bench", base_path=str(tmp_path))
    safe = om.benchmark_name
    filename = f"provenance_{safe}.json"
    base = {"data": {"size": {"source": "deterministic", "evidence": "1319"}}}
    om.save_tool_output(base, "composer", filename)

    entry = _inject_eee_interpretation(_ns_card(), _eee(True))
    _record_derived_provenance(om, safe, "methodology", "interpretation", entry)

    import os
    path = os.path.join(om.get_tool_output_path("composer"), filename)
    with open(path) as f:
        prov = json.load(f)
    assert prov["data"]["size"]["source"] == "deterministic"
    assert prov["methodology"]["interpretation"]["source"] == "eee"
    assert prov["methodology"]["interpretation"]["status"] == "derived"


def test_sidecar_created_when_absent(tmp_path):
    om = OutputManager("demo-bench2", base_path=str(tmp_path))
    safe = om.benchmark_name
    entry = _inject_eee_interpretation(_ns_card(), _eee(False))
    _record_derived_provenance(om, safe, "methodology", "interpretation", entry)

    import os
    path = os.path.join(om.get_tool_output_path("composer"), f"provenance_{safe}.json")
    with open(path) as f:
        prov = json.load(f)
    assert prov["methodology"]["interpretation"]["source"] == "eee"


def _read_prov(om, safe):
    import os
    path = os.path.join(om.get_tool_output_path("composer"), f"provenance_{safe}.json")
    with open(path) as f:
        return json.load(f)


def test_records_benchmark_type_derived_provenance(tmp_path):
    om = OutputManager("demo-btype", base_path=str(tmp_path))
    safe = om.benchmark_name
    _record_derived_provenance(om, safe, "benchmark_details", "benchmark_type", _EEE_DERIVED)

    prov = _read_prov(om, safe)
    assert prov["benchmark_details"]["benchmark_type"]["source"] == "eee"
    assert prov["benchmark_details"]["benchmark_type"]["status"] == "derived"


def test_records_appears_in_derived_provenance(tmp_path):
    om = OutputManager("demo-appears", base_path=str(tmp_path))
    safe = om.benchmark_name
    _record_derived_provenance(om, safe, "benchmark_details", "appears_in", _EEE_DERIVED)

    prov = _read_prov(om, safe)
    assert prov["benchmark_details"]["appears_in"]["source"] == "eee"
    assert prov["benchmark_details"]["appears_in"]["status"] == "derived"


# ----------------------------------------- baseline_results EEE provenance ----

def _eee_eval(n=13):
    """Minimal EEE metadata carrying an evaluation summary baseline_results can use."""
    return {
        "evaluation_summary": {
            "primary_metric": "score",
            "total_models_evaluated": n,
            "top_performers": [
                {"model": "m1", "score": 0.7120},
                {"model": "m2", "score": 0.7000},
            ],
            "score_statistics": {"mean": 0.5695, "std_dev": 0.1958, "min": 0.1, "max": 0.8},
        }
    }


def test_baseline_results_fill_returns_eee_entry():
    card = {"methodology": {"baseline_results": "Not specified"}}
    entry = _enrich_baseline_results(card, _eee_eval())
    assert entry == _EEE_DERIVED
    assert card["methodology"]["baseline_results"].startswith(
        "Based on 13 model evaluations from Every Eval Ever"
    )


def test_baseline_results_wrapped_card_filled_in_place():
    card = {"benchmark_card": {"methodology": {"baseline_results": ""}}}
    entry = _enrich_baseline_results(card, _eee_eval())
    assert entry is not None
    assert "mean score = 0.5695" in card["benchmark_card"]["methodology"]["baseline_results"]


def test_baseline_results_preserves_genuine_stage_b_value():
    card = {"methodology": {"baseline_results": "GPT-4 scores 0.81 (exact match)."}}
    entry = _enrich_baseline_results(card, _eee_eval())
    assert entry is None
    assert card["methodology"]["baseline_results"] == "GPT-4 scores 0.81 (exact match)."


def test_baseline_results_resolves_raw_metric_id_to_human_name():
    # F4: a raw llm_stats.<slug>.score primary_metric is displayed as its human metric_name
    card = {"methodology": {"baseline_results": "Not specified"}}
    eee = _eee_eval()
    eee["evaluation_summary"]["primary_metric"] = "llm_stats.x.score"
    eee["metrics"] = {"llm_stats.x.score": {"metric_name": "X score", "metric_kind": "benchmark_score"}}
    _enrich_baseline_results(card, eee)
    br = card["methodology"]["baseline_results"]
    assert "llm_stats." not in br
    assert "mean X score =" in br


def test_baseline_results_no_eval_summary_returns_none():
    card = {"methodology": {"baseline_results": "Not specified"}}
    assert _enrich_baseline_results(card, {}) is None
    assert card["methodology"]["baseline_results"] == "Not specified"


def test_baseline_results_no_top_or_stats_returns_none():
    card = {"methodology": {"baseline_results": "Not specified"}}
    eee = {"evaluation_summary": {"primary_metric": "score", "total_models_evaluated": 3}}
    assert _enrich_baseline_results(card, eee) is None
    assert card["methodology"]["baseline_results"] == "Not specified"


def test_records_baseline_results_derived_provenance(tmp_path):
    om = OutputManager("demo-baseline", base_path=str(tmp_path))
    safe = om.benchmark_name
    card = {"methodology": {"baseline_results": "Not specified"}}
    entry = _enrich_baseline_results(card, _eee_eval())
    assert entry is not None
    _record_derived_provenance(om, safe, "methodology", "baseline_results", entry)

    prov = _read_prov(om, safe)
    assert prov["methodology"]["baseline_results"]["source"] == "eee"
    assert prov["methodology"]["baseline_results"]["evidence_ids"] == []


# ------------------------------------------ benchmark_details.name identity ----

def _name_card(name="AA-LCR"):
    return {"benchmark_details": {"name": name}}


def test_identity_name_records_deterministic_provenance(tmp_path):
    om = OutputManager("demo-name", base_path=str(tmp_path))
    safe = om.benchmark_name
    entry = _record_identity_name_provenance(
        om, safe, _name_card("AA-LCR"), "AA-LCR", {"benchmark_name": "AA-LCR"}
    )
    assert entry is not None
    prov = _read_prov(om, safe)
    assert prov["benchmark_details"]["name"]["source"] == "deterministic"
    assert prov["benchmark_details"]["name"]["verified"] is True
    assert prov["benchmark_details"]["name"]["evidence_ids"] == []


def test_identity_name_matches_via_eee_benchmark_name(tmp_path):
    # benchmark_name arg differs from the card value; the EEE benchmark_name still matches.
    om = OutputManager("demo-name2", base_path=str(tmp_path))
    safe = om.benchmark_name
    entry = _record_identity_name_provenance(
        om, safe, _name_card("AA-LCR"), "aa-lcr-scan-key", {"benchmark_name": "AA-LCR"}
    )
    assert entry is not None
    assert _read_prov(om, safe)["benchmark_details"]["name"]["source"] == "deterministic"


def test_identity_name_skips_when_stage_b_already_cited(tmp_path):
    om = OutputManager("demo-name3", base_path=str(tmp_path))
    safe = om.benchmark_name
    base = {"benchmark_details": {"name": {"source": "docling/stated", "evidence_ids": ["E1"]}}}
    om.save_tool_output(base, "composer", f"provenance_{safe}.json")

    entry = _record_identity_name_provenance(
        om, safe, _name_card("AA-LCR"), "AA-LCR", {"benchmark_name": "AA-LCR"}
    )
    assert entry is None
    # the genuine Stage-B docling provenance is preserved, not overwritten
    assert _read_prov(om, safe)["benchmark_details"]["name"]["source"] == "docling/stated"


def test_identity_name_skips_when_value_differs_from_identity(tmp_path):
    import os
    om = OutputManager("demo-name4", base_path=str(tmp_path))
    safe = om.benchmark_name
    entry = _record_identity_name_provenance(
        om, safe, _name_card("Some Other Benchmark"), "AA-LCR", {"benchmark_name": "AA-LCR"}
    )
    assert entry is None
    # nothing recorded -> no sidecar written
    path = os.path.join(om.get_tool_output_path("composer"), f"provenance_{safe}.json")
    assert not os.path.exists(path)


def test_identity_name_skips_when_not_specified(tmp_path):
    om = OutputManager("demo-name5", base_path=str(tmp_path))
    safe = om.benchmark_name
    entry = _record_identity_name_provenance(
        om, safe, _name_card("Not specified"), "AA-LCR", {"benchmark_name": "AA-LCR"}
    )
    assert entry is None


def test_derived_provenance_merges_across_sections(tmp_path):
    """The benchmark_details and methodology recordings must coexist in one sidecar:
    each call merges into the existing file rather than clobbering other sections."""
    om = OutputManager("demo-merge", base_path=str(tmp_path))
    safe = om.benchmark_name
    base = {"data": {"size": {"source": "deterministic", "evidence": "1319"}}}
    om.save_tool_output(base, "composer", f"provenance_{safe}.json")

    _record_derived_provenance(om, safe, "benchmark_details", "benchmark_type", _EEE_DERIVED)
    _record_derived_provenance(om, safe, "benchmark_details", "appears_in", _EEE_DERIVED)
    interp = _inject_eee_interpretation(_ns_card(), _eee(True))
    _record_derived_provenance(om, safe, "methodology", "interpretation", interp)

    prov = _read_prov(om, safe)
    assert prov["data"]["size"]["source"] == "deterministic"
    assert prov["benchmark_details"]["benchmark_type"]["source"] == "eee"
    assert prov["benchmark_details"]["appears_in"]["source"] == "eee"
    assert prov["methodology"]["interpretation"]["source"] == "eee"
