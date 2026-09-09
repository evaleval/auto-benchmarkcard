"""Offline tests for the composer gate v2 metrics. Runs the classifier over the
captured D6 pilot cards and asserts the documented buckets plus that the headline
hallucination-rate computes to a number. No model, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import compute_gate_metrics as gm  # noqa: E402

# Captured runs are optional local fixtures. Point this at the output directory
# containing d6_pilot, d6_repilot, etc. to include those regression checks.
CAPTURED_RUNS = os.path.expanduser(os.environ.get(
    "BENCHMARKCARD_CAPTURED_RUNS", os.path.join(os.path.dirname(__file__), "..", "output")
))
D6_DIR = os.path.join(CAPTURED_RUNS, "d6_pilot", "output")
D6_REPILOT_DIR = os.path.join(CAPTURED_RUNS, "d6_repilot", "output")
WAVE2_DIR = os.path.join(CAPTURED_RUNS, "d6_repilot_wave2", "output")

# Applied per-test (not module-wide) so the synthetic schema_invalid / verdict unit
# tests below run even when the captured pilot fixtures are absent.
requires_d6 = pytest.mark.skipif(
    not os.path.isdir(D6_DIR), reason=f"D6 pilot fixture not present at {D6_DIR}")
requires_repilot = pytest.mark.skipif(
    not os.path.isdir(D6_REPILOT_DIR), reason=f"D6 repilot fixture not present at {D6_REPILOT_DIR}")
requires_wave2 = pytest.mark.skipif(
    not os.path.isdir(WAVE2_DIR), reason=f"D6 repilot wave2 fixture not present at {WAVE2_DIR}")

D8_FINISH3_DIR = os.path.join(CAPTURED_RUNS, "d8_finish3", "output")
requires_d8 = pytest.mark.skipif(
    not os.path.isdir(D8_FINISH3_DIR), reason=f"d8_finish3 fixture not present at {D8_FINISH3_DIR}")

D8_FINISH4_DIR = os.path.join(CAPTURED_RUNS, "d8_finish4", "output")
requires_d8_finish4 = pytest.mark.skipif(
    not os.path.isdir(D8_FINISH4_DIR), reason=f"d8_finish4 fixture not present at {D8_FINISH4_DIR}")


@pytest.fixture(scope="module")
def cards():
    return {name: (card, prov) for name, card, prov, hf, eee in gm.gather_cards(D6_DIR)}


def _label(card, prov, path):
    section, _, field = path.partition(".")
    value = card.get(section, {}).get(field)
    return gm.classify(value, gm._prov_entry(prov, section, field))


@requires_d6
def test_three_cards_loaded(cards):
    assert set(cards) == {"aa-lcr", "bigcodebench", "biolp-bench"}


@requires_d6
@pytest.mark.parametrize("path", ["methodology.interpretation", "methodology.calculation"])
def test_interpretation_and_calculation_are_ns(cards, path):
    for name, (card, prov) in cards.items():
        assert _label(card, prov, path) == gm.NS, f"{name}.{path}"


@requires_d6
def test_data_size_grounded_derived(cards):
    # Present (non-NS) on two cards -> DERIVED via source=deterministic; absent on
    # biolp-bench -> NS. Asserted on classifier output, not the rate population.
    for name in ("aa-lcr", "bigcodebench"):
        card, prov = cards[name]
        assert _label(card, prov, "data.size") == gm.DERIVED, name
    card, prov = cards["biolp-bench"]
    assert _label(card, prov, "data.size") == gm.NS


@requires_d6
def test_bigcodebench_methods_extracted_e27(cards):
    card, prov = cards["bigcodebench"]
    assert _label(card, prov, "methodology.methods") == gm.EXTRACTED
    entry = gm._prov_entry(prov, "methodology", "methods")
    assert "E27" in (entry.get("evidence_ids") or [])


@requires_d6
def test_hallucination_rate_is_a_number():
    rate = gm.compute(D6_DIR)["aggregate"]["hallucination_rate"]
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


@requires_d6
def test_no_ns_or_derived_field_is_ungrounded(cards):
    # Core safety property: a field that is genuinely NS or structured-derived must
    # never land in the UNGROUNDED (hallucination) bucket.
    for name, (card, prov) in cards.items():
        for row in gm.classify_card(card, prov):
            if row["label"] != gm.UNGROUNDED:
                continue
            section, _, field = row["path"].partition(".")
            value = card.get(section, {}).get(field)
            assert not gm.is_ns_value(value), f"{name}.{row['path']}: NS read as UNGROUNDED"
            assert not gm._is_derived_source(row["source"]), \
                f"{name}.{row['path']}: derived read as UNGROUNDED"


# Parse failures must not pass as honest abstention -----------------------------

def _card(flagged=None, **sections):
    card = dict(sections)
    if flagged is not None:
        card["flagged_fields"] = flagged
    return card


def test_schema_invalid_fields_picks_only_schema_invalid():
    card = _card(flagged={
        "benchmark_details.name": "[schema_invalid] unparseable group output",
        "data.size": "[schema_invalid] unparseable group output",
        "methodology.methods": "[low_confidence] weak evidence",
    })
    # only the schema_invalid markers, returned sorted; the low_confidence flag is left out
    assert gm.schema_invalid_fields(card) == ["benchmark_details.name", "data.size"]


def test_schema_invalid_fields_empty_when_clean():
    assert gm.schema_invalid_fields(_card()) == []
    assert gm.schema_invalid_fields(_card(flagged={})) == []
    assert gm.schema_invalid_fields(_card(flagged={"x.y": "[low_confidence] note"})) == []
    assert gm.schema_invalid_fields({"flagged_fields": None}) == []


def test_verdict_nogo_on_any_schema_invalid():
    # The bigcodebench failure mode: 13 fields lost to a parse failure, 0% hallucination.
    by_card = [{"card": "bigcodebench", "count": 13, "fields": ["benchmark_details.name"]}]
    v = gm._verdict(13, by_card, 0.0)
    assert v["result"] == "NO-GO"
    assert v["checks"]["schema_invalid"]["pass"] is False
    assert v["checks"]["schema_invalid"]["count"] == 13
    assert v["checks"]["hallucination_rate"]["pass"] is True
    assert any("schema_invalid" in r for r in v["reasons"])


def test_verdict_go_when_clean_and_within_bar():
    for rate in (None, 0.0, 0.01, 0.02):  # 0.02 is the bar itself -> still GO
        v = gm._verdict(0, [], rate)
        assert v["result"] == "GO", rate
        assert v["checks"]["schema_invalid"]["pass"] is True
        assert v["checks"]["hallucination_rate"]["pass"] is True
        assert v["reasons"] == []


def test_verdict_nogo_above_hallucination_bar():
    v = gm._verdict(0, [], 0.05)
    assert v["result"] == "NO-GO"
    assert v["checks"]["hallucination_rate"]["pass"] is False
    assert v["checks"]["schema_invalid"]["pass"] is True


def test_verdict_boundary_just_over_bar():
    assert gm._verdict(0, [], 0.0201)["result"] == "NO-GO"


@requires_repilot
def test_repilot_is_nogo_from_parse_failures():
    m = gm.compute(D6_REPILOT_DIR)
    assert m["verdict"]["result"] == "NO-GO"
    si = m["verdict"]["checks"]["schema_invalid"]
    assert si["pass"] is False
    assert si["count"] == 13
    by_card = {e["card"]: e for e in si["by_card"]}
    assert set(by_card) == {"bigcodebench"}
    assert by_card["bigcodebench"]["count"] == 13
    assert "benchmark_details.name" in by_card["bigcodebench"]["fields"]
    assert m["aggregate"]["schema_invalid"]["count"] == 13


@requires_repilot
def test_repilot_clean_cards_not_blocked_on_parse_axis():
    m = gm.compute(D6_REPILOT_DIR)
    for name in ("aa-lcr", "biolp-bench"):
        assert m["per_card"][name]["schema_invalid"]["count"] == 0, name
    assert m["per_card"]["bigcodebench"]["schema_invalid"]["count"] == 13


@requires_repilot
def test_print_summary_emits_gate_block(capsys):
    gm.print_summary(gm.compute(D6_REPILOT_DIR))
    out = capsys.readouterr().out
    assert "GATE: NO-GO" in out
    assert "parse" in out  # per-card table column header
    assert "benchmark_details.name" in out  # offending field listed


# Gate v3 -- descriptive content checks -----------------------------------------
# All structure-keyed: no test (and no check) depends on a benchmark name or an
# expected value, so the predicates score an unseen card the same way.

def _v3_card(bd=None, meth=None, pu=None):
    return {"benchmark_details": bd or {}, "methodology": meth or {},
            "purpose_and_intended_users": pu or {}}


def _meth_prov(**fields):
    return {"methodology": {f: {"evidence": q} for f, q in fields.items()}}


def test_metric_canonical_predicate():
    # genuine other:<token> (and a bare other:score with no dot) stay canonical
    for good in ("pass@1", "pass@10", "other:elo", "other:score", "accuracy", "F1", "BLEU"):
        assert gm._is_canonical_metric(good), good
    # the degenerate llm-stats proxy other:<slug>.score is no longer whitelisted
    for bad in ("the model is scored on accuracy", "score", "custom thing", "other:",
                "other:bigcodebench.score"):
        assert not gm._is_canonical_metric(bad), bad


def test_bare_url_predicate_keeps_aggregator_flags_garbled():
    for good in ("https://llm-stats.com/benchmarks/x", "https://arxiv.org/abs/2406.15877",
                 "https://doi.org/10.1101/2024.08.21.608694"):
        assert gm._is_bare_url(good), good
    for bad in ("https://a.orghttps://b.org", "see https://x.org for details",
                "ftp://x.org/file", "just some prose", ""):
        assert not gm._is_bare_url(bad), bad


def test_clean_data_type_predicate():
    for good in ("code", "text", "tabular, text", "other:graph"):
        assert gm._is_clean_data_type(good), good
    for bad in ("other:multiple-choice and error-detection", "accuracy", "other:"):
        assert not gm._is_clean_data_type(bad), bad


def test_title_like_name_predicate():
    bad = "BIGCODEBENCH : BENCHMARKING CODE GENERATION WITH DIVERSE FUNCTION CALLS AND COMPLEX INSTRUCTIONS"
    assert gm._is_title_like_name(bad)
    for ok in ("AA-LCR", "BigCodeBench", "Biological Lab Protocol benchmark (BioLP-bench)"):
        assert not gm._is_title_like_name(ok), ok


def test_atomic_element_predicate():
    for ok in ("biology", "natural language processing", "Company Reports", "English"):
        assert gm._is_atomic_element(ok), ok
    for bad in ("7 domains covering 139 libraries",
                "BigCodeBench challenges LLMs to invoke many calls. It serves as a benchmark.",
                "a single element that is far too long to be one atomic enumerated list item here"):
        assert not gm._is_atomic_element(bad), bad


def test_format_lint_names_residual_defects():
    card = _v3_card(
        bd={"name": "AN ALL CAPS PAPER TITLE THAT IS CLEARLY FAR TOO LONG TO BE A NAME",
            "data_type": "other:multiple-choice and error-detection",
            "domains": ["7 categories covering 100 things"],
            "resources": ["https://llm-stats.com/x", "https://a.orghttps://b.org"]},
        meth={"metrics": ["pass@1", "scored on vibes"]})
    kinds = {f["kind"] for f in gm._format_lint("x", card)}
    assert {"name_title_like", "data_type_not_atomic", "list_not_atomic",
            "resource_not_url", "metric_noncanonical"} <= kinds
    # the well-formed aggregator URL is not flagged
    paths = [f["detail"] for f in gm._format_lint("x", card) if f["kind"] == "resource_not_url"]
    assert all("llm-stats" not in d for d in paths)


def test_aboutness_judge_flag_fires_on_exec_only_evidence():
    card = _v3_card(meth={"judge_uses_llm": True, "judge_num": 1, "judge_models": ["GPT-4"]})
    prov = _meth_prov(judge_uses_llm="We use the unbiased version of Pass@K to score correctness.",
                      calculation="Functional correctness via unit tests in a sandbox.")
    flags = gm._aboutness_flags("x", card, prov)
    assert any(f["kind"] == "judge_mis_aboutness" for f in flags)


def test_aboutness_judge_quiet_on_output_scoring_evidence():
    card = _v3_card(meth={"judge_uses_llm": True, "judge_num": 1, "judge_models": ["Qwen"]})
    prov = _meth_prov(judge_uses_llm="Qwen3 is used as the equality checker model.")
    assert not [f for f in gm._aboutness_flags("x", card, prov)
                if f["kind"] == "judge_mis_aboutness"]


def test_aboutness_judge_quiet_when_not_asserted():
    card = _v3_card(meth={"judge_uses_llm": False, "judge_num": "Not specified",
                          "judge_models": ["Not specified"]})
    assert not gm._aboutness_flags("x", card, _meth_prov())


# J5 -- judge-DENIAL mis-aboutness (read-only mirror of the EEE injection detector) ----

def test_judge_denial_fires_on_named_judge_in_hf_readme():
    card = _v3_card(meth={"judge_uses_llm": False, "judge_num": "Not specified",
                          "judge_models": ["Not specified"]})
    hf = {"readme_markdown": "## Scoring Approach\nQwen3 235B A22B 2507 Non-reasoning is used as "
                             "the equality checker model."}
    den = [f for f in gm._aboutness_flags("x", card, {}, hf=hf)
           if f["kind"] == "judge_denial_mis_aboutness"]
    assert den and "Qwen3 235B A22B 2507 Non-reasoning" in den[0]["detail"]


def test_judge_denial_fires_from_eee_description():
    card = _v3_card(meth={"judge_uses_llm": False})
    eee = {"metrics": {"m": {"evaluation_description": "GPT-4 is used as the judge model."}}}
    assert any(f["kind"] == "judge_denial_mis_aboutness"
               for f in gm._aboutness_flags("x", card, {}, eee=eee))


def test_judge_denial_quiet_when_ns_unnamed_or_exec():
    hf_named = {"readme_markdown": "Qwen3 is used as the equality checker model."}
    # NS is honest abstention, not a denial
    ns = _v3_card(meth={"judge_uses_llm": "Not specified"})
    assert not [f for f in gm._aboutness_flags("x", ns, {}, hf=hf_named)
                if f["kind"] == "judge_denial_mis_aboutness"]
    # False but the source names no judge (exact-match scoring) -> quiet
    denial = _v3_card(meth={"judge_uses_llm": False})
    assert not gm._aboutness_flags("x", denial, {},
                                   hf={"readme_markdown": "Scored by exact match accuracy."})
    # False with execution-based scoring (Pass@1) -> exec veto keeps it quiet
    assert not gm._aboutness_flags("x", denial, {},
                                   hf={"readme_markdown": "Each solution is checked with Pass@1 unit tests."})


def test_judge_denial_quiet_on_human_eval_and_construction():
    denial = _v3_card(meth={"judge_uses_llm": False})
    # human evaluation in the source is not an LLM judge -> no denial flag
    assert not gm._aboutness_flags("x", denial, {},
        hf={"readme_markdown": "We use three human evaluators to judge the responses."})
    # a model used for dataset construction is not output scoring -> no denial flag
    assert not gm._aboutness_flags("x", denial, {},
        hf={"readme_markdown": "We use GPT-4 to evaluate the difficulty of each task during dataset construction."})


def test_judge_denial_is_descriptive_not_in_verdict():
    # verdict is a pure function of schema_invalid + hallucination_rate; the denial flag (a v3
    # descriptive item) cannot move it. Guards against a future wiring slip into _verdict.
    import inspect
    src = inspect.getsource(gm._verdict)
    assert "aboutness" not in src and "judge_denial" not in src


def test_aboutness_flags_back_compat_three_args():
    # the hf/eee params default to None so the legacy 3-arg call site still works
    card = _v3_card(meth={"judge_uses_llm": True, "judge_num": 1, "judge_models": ["GPT-4"]})
    prov = _meth_prov(judge_uses_llm="We use the unbiased version of Pass@K to score correctness.")
    assert any(f["kind"] == "judge_mis_aboutness" for f in gm._aboutness_flags("x", card, prov))


def test_aboutness_data_type_metric_only():
    assert any(f["kind"] == "data_type_not_modality"
               for f in gm._aboutness_flags("x", _v3_card(bd={"data_type": "accuracy score"}), {}))
    assert not gm._aboutness_flags("x", _v3_card(bd={"data_type": "code"}), {})


def test_aboutness_related_benchmark_leakage():
    leaking = _v3_card(bd={"name": "SuperBench", "similar_benchmarks": ["GLUE"],
                           "overview": "GLUE evaluates language understanding across tasks."})
    assert any(f["kind"] == "related_benchmark_leakage" for f in gm._aboutness_flags("x", leaking, {}))
    # own name present -> not leakage
    clean = _v3_card(bd={"name": "SuperBench", "similar_benchmarks": ["GLUE"],
                         "overview": "SuperBench extends GLUE with harder tasks."})
    assert not [f for f in gm._aboutness_flags("x", clean, {}) if f["kind"] == "related_benchmark_leakage"]


def test_source_coverage_maps_three_telemetry_shapes(tmp_path):
    shapes = {
        "full": {"full_text": True, "paper_url_resolved": True, "reason": "ok"},
        "abs": {"full_text": False, "degraded_to_abstract_only": True,
                "paper_url_resolved": True, "reason": "fetch_blocked", "http_status": 403},
        "none": {"full_text": False, "degraded_to_abstract_only": False,
                 "paper_url_resolved": False, "reason": "no_paper_url"},
    }
    for name, tel in shapes.items():
        d = tmp_path / f"{name}_x" / "tool_output" / "composer"
        d.mkdir(parents=True)
        (d / f"docling_telemetry_{name}.json").write_text(__import__("json").dumps(tel))
    sc = gm._source_coverage(str(tmp_path))
    assert sc["by_card"]["full"]["label"] == "full_text"
    assert sc["by_card"]["abs"]["label"] == "abstract_only"
    assert sc["by_card"]["none"]["label"] == "no_paper"
    assert sc["counts"] == {"full_text": 1, "abstract_only": 1, "no_paper": 1}


@requires_wave2
def test_v3_additive_verdict_and_grounding_unchanged():
    m = gm.compute(WAVE2_DIR)
    # the two existing bars are untouched by the v3 layer
    assert m["verdict"]["result"] == "GO"
    assert m["aggregate"]["hallucination_rate"] == 0.0
    assert m["aggregate"]["schema_invalid"]["count"] == 0
    # the descriptive blocks are present, including the new dataset-identity + F4 blocks
    assert {"aboutness", "format_lint", "dataset_identity", "source_coverage",
            "none_override", "paper_search_degraded", "shell_detector",
            "repo_paper_mismatch"} <= set(m)
    assert m["dataset_identity"]["count"] == 0  # wave2 repo ids all share a token with their card


@requires_wave2
def test_v3_source_coverage_marks_biolp_abstract_only():
    sc = gm.compute(WAVE2_DIR)["source_coverage"]
    assert sc["by_card"]["biolp-bench"]["label"] == "abstract_only"
    assert sc["by_card"]["bigcodebench"]["label"] == "full_text"
    assert sc["by_card"]["aa-lcr"]["label"] == "no_paper"


@requires_wave2
def test_v3_aboutness_clean_and_format_lint_names_biolp_data_type():
    m = gm.compute(WAVE2_DIR)
    assert m["aboutness"]["count"] == 0  # post Brief-A/B/C the content defects are fixed
    dt_flags = [f for f in m["format_lint"]["flags"]
                if f["card"] == "biolp-bench" and f["path"] == "benchmark_details.data_type"]
    assert dt_flags and dt_flags[0]["kind"] == "data_type_not_atomic"


# F3-2 dataset-identity mismatch ------------------------------------------------

def test_dataset_identity_mismatch_fires_on_disjoint_repo_id():
    # arc shape: card 'arc' resolved to an unrelated repo sharing no name token.
    card = _v3_card(bd={"name": "AI2 Reasoning Challenge (ARC)"})
    hf = {"id": "banned-historical-archives/banned-historical-archives"}
    flags = gm._dataset_identity_flags("arc", card, hf)
    assert any(f["kind"] == "dataset_identity_mismatch" for f in flags)


def test_dataset_identity_quiet_when_repo_id_shares_token():
    # a token shared between the resolved repo id and the card name -> not a mismatch.
    # superglue<->super_glue is the concatenated-vs-separated case: same identity, must stay quiet.
    for name, repo in (("bigcodebench", "bigcode/bigcodebench"),
                       ("api-bank", "liminghao1630/API-Bank"),
                       ("blink", "BLINK-Benchmark/BLINK"),
                       ("aa-lcr", "ArtificialAnalysis/AA-LCR"),
                       ("agentharm", "ai-safety-institute/AgentHarm"),
                       ("superglue", "aps/super_glue")):
        assert gm._dataset_identity_flags(name, _v3_card(), {"id": repo}) == [], name


def test_dataset_identity_quiet_when_hf_missing_or_no_id():
    card = _v3_card(bd={"name": "Whatever Benchmark"})
    assert gm._dataset_identity_flags("foo", card, {}) == []            # no hf file resolved
    assert gm._dataset_identity_flags("foo", card, {"id": None}) == []  # alpacaeval: id is None
    assert gm._dataset_identity_flags("foo", card, {"id": ""}) == []    # empty id


@requires_d6
def test_gather_cards_returns_five_tuples():
    rows = gm.gather_cards(D6_DIR)
    assert rows and all(len(r) == 5 for r in rows)
    assert all(isinstance(r[3], dict) for r in rows)  # 4th element is the resolved-HF dict
    assert all(isinstance(r[4], dict) for r in rows)  # 5th element is the EEE-metadata dict


# F4 entity-resolution gate checks ----------------------------------------------

def test_none_override_fires_on_confident_none_with_resolved_url():
    pv = {"llm_verification": {"match_index": "none", "error": False, "confidence": 0.9},
          "resolved_url": "https://arxiv.org/abs/1234.5678"}
    assert any(f["kind"] == "none_override" for f in gm._none_override_flags("foo", pv))


def test_none_override_quiet_on_error_lowconf_or_no_url():
    url = "https://arxiv.org/abs/1234.5678"
    err = {"llm_verification": {"match_index": "none", "error": True, "confidence": 0.9}, "resolved_url": url}
    low = {"llm_verification": {"match_index": "none", "error": False, "confidence": 0.3}, "resolved_url": url}
    nourl = {"llm_verification": {"match_index": "none", "error": False, "confidence": 0.9}, "resolved_url": None}
    assert gm._none_override_flags("foo", err) == []
    assert gm._none_override_flags("foo", low) == []
    assert gm._none_override_flags("foo", nourl) == []
    assert gm._none_override_flags("foo", None) == []


def test_paper_search_degraded_fires_on_each_degraded_status():
    pv = {"sources_tried": [
        {"source": "openalex", "status": "auth_error", "results": 0},
        {"source": "semantic_scholar", "status": "rate_limited", "results": 0},
        {"source": "crossref", "status": "errored", "results": 0},
    ]}
    flags = gm._paper_search_degraded_flags("foo", pv)
    assert len(flags) == 3
    assert all(f["kind"] == "paper_search_degraded" for f in flags)
    assert {f["path"] for f in flags} == {
        "sources_tried.openalex", "sources_tried.semantic_scholar", "sources_tried.crossref"}


def test_paper_search_degraded_quiet_on_ok_or_zero_results():
    pv = {"sources_tried": [
        {"source": "openalex", "status": "ok", "results": 5},
        {"source": "semantic_scholar", "status": "0-results", "results": 0},
    ]}
    assert gm._paper_search_degraded_flags("foo", pv) == []


def test_paper_search_degraded_quiet_on_missing_or_empty_pv():
    assert gm._paper_search_degraded_flags("foo", None) == []
    assert gm._paper_search_degraded_flags("foo", {}) == []
    assert gm._paper_search_degraded_flags("foo", {"sources_tried": []}) == []


def test_paper_search_degraded_is_descriptive_not_in_verdict():
    # verdict is a pure function of schema_invalid + hallucination_rate; the degraded flag
    # (a descriptive item over sources_tried) cannot move it.
    import inspect
    src = inspect.getsource(gm._verdict)
    assert "paper_search_degraded" not in src and "sources_tried" not in src


def test_shell_detector_fires_on_empty_husk():
    card = _v3_card(bd={"name": "Not specified", "overview": "Not specified"})
    flags = gm._shell_detector_flags("foo", card, {"resolved_url": None})
    assert any(f["kind"] == "shell_detector" for f in flags)


def test_shell_detector_quiet_when_resolved_or_has_content():
    husk = _v3_card(bd={"name": "Not specified", "overview": "Not specified"})
    assert gm._shell_detector_flags("foo", husk, {"resolved_url": "https://arxiv.org/abs/1234.5678"}) == []
    has_content = _v3_card(bd={"name": "WidgetBench", "overview": "Not specified"})
    assert gm._shell_detector_flags("foo", has_content, {"resolved_url": None}) == []


def test_repo_paper_mismatch_derivative_locks_dataset_identity_gap():
    # Derivative shape: the repo shares the slug token (so _dataset_identity_flags stays quiet)
    # but carries extra distinctive tokens -- only the new check catches it.
    card = _v3_card(bd={"name": "WidgetBench"})
    hf = {"id": "acme/Distill-RL-widgetbench-extra"}
    assert gm._dataset_identity_flags("widgetbench", card, hf) == []
    flags = gm._repo_paper_mismatch_flags("widgetbench", card, hf)
    assert any(f["kind"] == "repo_paper_mismatch_derivative_repo" for f in flags)


def test_repo_paper_mismatch_aggregate_repo():
    card = _v3_card(bd={"name": "WidgetBench"})
    flags = gm._repo_paper_mismatch_flags("widgetbench", card, {"id": "someorg/MegaSuite"})
    assert any(f["kind"] == "repo_paper_mismatch_aggregate_repo" for f in flags)


def test_repo_paper_mismatch_quiet_on_clean_repo_or_no_hf():
    card = _v3_card(bd={"name": "WidgetBench"})
    assert gm._repo_paper_mismatch_flags("widgetbench", card, {"id": "widgetorg/widgetbench"}) == []
    assert gm._repo_paper_mismatch_flags("widgetbench", card, {}) == []


# F4 subject-vs-slug + metric_noncanonical (both descriptive) -------------------

def test_subject_vs_slug_fires_on_titlelike_disjoint_name():
    # bfcl shape: a wrong paper's title became the card identity (title-like, shares no slug token).
    card = _v3_card(bd={"name": "Towards a Participatory and Social Justice-Oriented "
                                "Measure of Human-Robot Trust"})
    assert any(f["kind"] == "subject_vs_slug" for f in gm._subject_vs_slug_flags("bfcl", card))


def test_subject_vs_slug_quiet_on_short_disjoint_name():
    # short correct identities that merely use different tokens than the slug must stay quiet:
    # mmlu (acronym vs spelled-out expansion) and tau^2-Bench (unicode strips away) are not title-like.
    assert gm._subject_vs_slug_flags(
        "mmlu", _v3_card(bd={"name": "Measuring Massive Multitask Language Understanding"})) == []
    assert gm._subject_vs_slug_flags(
        "tau-bench-2-telecom", _v3_card(bd={"name": "τ²-Bench"})) == []


def test_subject_vs_slug_quiet_when_titlelike_but_token_overlaps():
    # a long, title-like name that still carries the slug identity is the right subject -> quiet.
    card = _v3_card(bd={"name": "Biological Lab Protocol benchmark for evaluating large "
                                "language models on wet-lab procedures (BioLP-bench)"})
    assert gm._subject_vs_slug_flags("biolp-bench", card) == []


def test_subject_vs_slug_quiet_on_ns_or_husk():
    assert gm._subject_vs_slug_flags("foo", _v3_card(bd={"name": "Not specified"})) == []
    assert gm._subject_vs_slug_flags("foo", _v3_card()) == []


def test_metric_noncanonical_fires_on_degenerate_proxy_only():
    card = _v3_card(meth={"metrics": ["other:aa-index.score", "other:apex_v1.score"]})
    fired = [f for f in gm._format_lint("x", card) if f["kind"] == "metric_noncanonical"]
    assert len(fired) == 2
    # genuine other:<token>, a bare other:score (no dot), and canonical metrics do not fire
    clean = _v3_card(meth={"metrics": ["other:elo", "other:GPQA", "pass@1", "accuracy", "other:score"]})
    assert not [f for f in gm._format_lint("x", clean) if f["kind"] == "metric_noncanonical"]


@requires_d8
def test_d8_subject_vs_slug_fires_exactly_on_bfcl():
    m = gm.compute(D8_FINISH3_DIR)
    assert set(m["subject_vs_slug"]["by_card"]) == {"bfcl"}


@requires_d8
def test_d8_metric_noncanonical_fires_on_other_score_proxy_set():
    m = gm.compute(D8_FINISH3_DIR)
    mn = [f for f in m["format_lint"]["flags"] if f["kind"] == "metric_noncanonical"]
    # every flagged metric is the degenerate other:<slug>.score proxy; 27 such cards in this run
    assert mn and all(".score" in f["detail"] for f in mn)
    assert len({f["card"] for f in mn}) == 27


@requires_d8
def test_d8_verdict_invariant_under_descriptive_changes():
    # the two descriptive flags must not move the GO/NO-GO math (schema_invalid + hallucination_rate).
    m = gm.compute(D8_FINISH3_DIR)
    assert m["verdict"]["result"] == "GO"
    assert m["verdict"]["checks"]["schema_invalid"]["count"] == 0
    assert round(m["aggregate"]["hallucination_rate"], 4) == 0.0127


# FINISH-5 numeric_internal_contradiction + available_but_dropped (both descriptive) ----
# All structure/arithmetic-keyed: no test (and no check) depends on a benchmark name.

def _num_card(size=None, breakdown=None, overview=""):
    return {"data": {"size": size, "size_breakdown": breakdown},
            "benchmark_details": {"overview": overview}}


def test_parse_size_bucket():
    assert gm._parse_size_bucket("Less than 1K examples") == (0.0, 1000)
    assert gm._parse_size_bucket("1K to 10K examples") == (1000, 10000)
    assert gm._parse_size_bucket("100K to 1M examples") == (100000, 1000000)
    assert gm._parse_size_bucket("23,679 English text generation prompts") == (23679, 23679)
    for bad in (None, "Not specified", "a few", ["1K to 10K examples"]):
        assert gm._parse_size_bucket(bad) is None, bad


def test_breakdown_leaf_sum_excludes_aggregate_rows():
    # the arc shape: leaf splits + subtotals + grand total. Only the leaves count.
    bd = {"Challenge_Train": 1119, "Challenge_Test": 1172, "Easy_Train": 2251, "Easy_Test": 2376,
          "Challenge_Total": 2291, "Easy_Total": 4627, "Total": 6918}
    s, n = gm._breakdown_leaf_sum(bd)
    assert s == 1119 + 1172 + 2251 + 2376 and n == 4
    # booleans and non-numbers are ignored; no aggregate keys -> all rows are leaves
    s2, n2 = gm._breakdown_leaf_sum({"a": 10, "b": 60, "c": 30, "note": "x", "flag": True})
    assert s2 == 100 and n2 == 3


def test_prose_magnitude_product_only():
    assert gm._prose_magnitude("203 activity classes with an average of 137 videos per class") == 27811
    # a lone large integer (no 'per <unit>' rate) never triggers a magnitude
    assert gm._prose_magnitude("a corpus of 849 video hours across many clips") is None


def test_numeric_contradiction_fires_on_each_signal():
    # N1 -- breakdown is percentages while the prose states a larger total
    n1 = gm._numeric_contradiction_flags("x", _num_card(
        size="Less than 1K examples", breakdown={"val": 10, "pub": 60, "priv": 30},
        overview="110 tasks (440 with augmentations)"))
    assert n1 and n1[0]["path"] == "data.size_breakdown"
    # N2 -- breakdown sums above the stated size upper bound
    n2 = gm._numeric_contradiction_flags("x", _num_card(
        size="1K to 10K examples", breakdown={"a": 9000, "b": 9000}))
    assert n2 and "above the stated size upper bound" in n2[0]["detail"]
    # N3 -- prose product exceeds the bucket (largest literal alone is within it)
    n3 = gm._numeric_contradiction_flags("x", _num_card(
        size="Less than 1K examples",
        overview="203 activity classes with an average of 137 videos per class, 849 video hours"))
    assert n3 and n3[0]["path"] == "data.size"
    assert all(f["kind"] == "numeric_internal_contradiction" for f in n1 + n2 + n3)


def test_numeric_contradiction_quiet_on_consistent_and_aggregate():
    # a breakdown that is a faithful subset (sums below the bucket) and a consistent literal
    assert gm._numeric_contradiction_flags("x", _num_card(
        size="100K to 1M examples", breakdown={"axb": 1104, "boolq": 9427},
        overview="8 tasks for language understanding")) == []
    # arc trap: splits + Total rows; leaf sum == stated size, so no contradiction
    assert gm._numeric_contradiction_flags("x", _num_card(
        size="7787 questions",
        breakdown={"Train": 3000, "Test": 4787, "Total": 7787})) == []
    # no size / no breakdown -> nothing to compare
    assert gm._numeric_contradiction_flags("x", _num_card(size="Not specified")) == []


def test_available_but_dropped_fires_on_license_and_language():
    # licence dropped: card NS but the same-field evidence names a licence
    lic = gm._available_but_dropped_flags(
        "x", {"ethical_and_legal_considerations": {"data_licensing": "Not specified"}},
        {"ethical_and_legal_considerations": {"data_licensing":
            {"evidence": "All QAs are licensed under CC BY-SA 4.0.", "evidence_ids": ["E29"]}}})
    assert lic and lic[0]["kind"] == "available_but_dropped" and "CC BY-SA 4.0" in lic[0]["detail"]
    # language dropped, same-field evidence
    same = gm._available_but_dropped_flags(
        "x", {"benchmark_details": {"languages": ["Not specified"]}},
        {"benchmark_details": {"languages":
            {"evidence": "collected from 12 popular Python repositories", "evidence_ids": ["E45"]}}})
    assert same and "languages evidence" in same[0]["detail"] and "Python" in same[0]["detail"]
    # language dropped, cross-field fallback to data_type evidence (no languages prov entry)
    xfield = gm._available_but_dropped_flags(
        "x", {"benchmark_details": {"languages": ["Not specified"]}},
        {"benchmark_details": {"data_type":
            {"evidence": "task automation via Python code", "evidence_ids": ["E03"]}}})
    assert xfield and "data_type evidence" in xfield[0]["detail"]


def test_available_but_dropped_quiet_on_abstention_or_present():
    # honest abstention in the evidence -> not a dropped value
    for ev in ("Not specified", "No evidence provided", ""):
        assert gm._available_but_dropped_flags(
            "x", {"ethical_and_legal_considerations": {"data_licensing": "Not specified"}},
            {"ethical_and_legal_considerations": {"data_licensing":
                {"evidence": ev, "evidence_ids": []}}}) == [], ev
    # evidence exists but names no value of the field's type -> quiet (kills tangential quotes)
    assert gm._available_but_dropped_flags(
        "x", {"benchmark_details": {"languages": ["Not specified"]}},
        {"benchmark_details": {"languages":
            {"evidence": "the dataset collects 500 Issue-Pull Request pairs", "evidence_ids": ["E1"]}}}) == []
    # field already filled -> never a dropped value
    assert gm._available_but_dropped_flags(
        "x", {"ethical_and_legal_considerations": {"data_licensing": "MIT License"}},
        {"ethical_and_legal_considerations": {"data_licensing":
            {"evidence": "MIT License", "evidence_ids": ["E1"]}}}) == []


def test_available_but_dropped_seam_not_applicable_clears():
    # PC-Composer seam: code benchmarks get languages='not-applicable' (a FILLED value, not NS).
    # The flag must NOT fire even when data_type evidence names a language -- else the post-fix
    # re-pilot would read a persisting flag as 'the composer fix did not land'.
    card = {"benchmark_details": {"languages": ["not-applicable"]}}
    prov = {"benchmark_details": {"data_type":
            {"evidence": "task automation via Python code", "evidence_ids": ["E03"]}}}
    assert gm._available_but_dropped_flags("x", card, prov) == []
    assert gm.is_not_specified(["not-applicable"]) is False


def test_finish5_checks_are_descriptive_not_in_verdict():
    # verdict is a pure function of schema_invalid + hallucination_rate; neither new check leaks in.
    import inspect
    src = inspect.getsource(gm._verdict)
    assert "numeric_contradiction" not in src and "available_but_dropped" not in src


@requires_d8_finish4
def test_d8f4_numeric_contradiction_fires_exactly():
    m = gm.compute(D8_FINISH4_DIR)
    assert set(m["numeric_contradiction"]["by_card"]) == {"activitynet", "agentharm", "rewardbench"}
    # the clean cards (incl. the arc aggregate-row trap) stay quiet
    for clean in ("bigcodebench", "superglue", "bold", "arc"):
        assert clean not in m["numeric_contradiction"]["by_card"], clean


@requires_d8_finish4
def test_d8f4_available_but_dropped_fire_set():
    m = gm.compute(D8_FINISH4_DIR)
    flags = m["available_but_dropped"]["flags"]
    by = {(f["card"], f["path"]) for f in flags}
    assert ("charxiv-d", "ethical_and_legal_considerations.data_licensing") in by
    assert ("bigcodebench-full", "benchmark_details.languages") in by
    assert set(m["available_but_dropped"]["by_card"]) == {
        "charxiv-d", "bigcodebench-full", "agentharm", "swe-bench"}


@requires_d8_finish4
def test_d8f4_verdict_and_existing_blocks_unchanged_by_new_checks():
    m = gm.compute(D8_FINISH4_DIR)
    assert m["verdict"]["result"] == "GO"
    assert m["verdict"]["checks"]["schema_invalid"]["count"] == 0
    assert round(m["aggregate"]["hallucination_rate"], 4) == 0.0056
    assert {"numeric_contradiction", "available_but_dropped"} <= set(m)
