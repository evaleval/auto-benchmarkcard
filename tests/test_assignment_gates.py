"""Tests for the deterministic assignment gates (aboutness.apply_assignment_gates).

The gates are structural and model-independent, so they are exercised directly with
synthetic v2 evidence records (the 11-key shape verify_items emits) and a mini referent
frame. Cases cover the sibling gate (headline drops, the comparison-sentence keep, the
similar_benchmarks exemption, the subset data.size refile), the region gate, the
claim-role gate, the table-anchor measurement pass, the no-op paths (no frame / v1
records), and the finalize_evidence wiring.
"""

import pytest

from auto_benchmarkcard.tools.composer import aboutness as A
from auto_benchmarkcard.tools.composer import evidence as E


def _frame():
    return {
        "target_id": "target",
        "nodes": [
            {"id": "target", "type": "target", "name": "OurBench", "aliases": []},
            {"id": "sibling:mmlu", "type": "sibling", "name": "MMLU", "aliases": []},
            {"id": "subset:hard", "type": "subset", "name": "Hard", "aliases": []},
            {"id": "split:test", "type": "split", "name": "test", "aliases": []},
        ],
    }


def _rec(field, quote, referent="target", claim_role="primary_result", region="",
         section_path="", table_id="", doc=E.DOC_PAPER, cs=0, note=""):
    return {"field": field, "quote": quote, "char_start": cs, "doc": doc,
            "kind": "stated", "note": note, "referent": referent,
            "claim_role": claim_role, "section_path": section_path,
            "region": region, "table_id": table_id}


def _v1_rec(field, quote, cs=0, doc=E.DOC_PAPER):
    return {"field": field, "quote": quote, "char_start": cs, "doc": doc,
            "kind": "stated", "note": ""}


# ------------------------------------------------------------------ no-op paths ----

def test_no_frame_is_noop():
    recs = [_rec("data.size", "MMLU contains 15,908 questions", referent="sibling:mmlu")]
    telem = {}
    out = A.apply_assignment_gates(recs, None, telem)
    assert out == recs
    assert telem == {}


def test_v1_records_pass_through_untouched():
    recs = [_v1_rec("data.size", "MMLU contains 15,908 questions"),
            _v1_rec("methodology.metrics", "accuracy")]
    telem = {}
    out = A.apply_assignment_gates(recs, _frame(), telem)
    assert out == recs
    assert telem == {}


def test_clean_records_untouched_order_preserved():
    recs = [
        _rec("purpose_and_intended_users.tasks", "answer physics questions"),
        _rec("benchmark_details.overview", "OurBench measures physics reasoning"),
        _rec("methodology.metrics", "We report accuracy"),
    ]
    telem = {}
    out = A.apply_assignment_gates(list(recs), _frame(), telem)
    assert out == recs
    assert telem["actions"] == [] and telem["summary"] == {}


# ---------------------------------------------------------------- sibling gate ----

@pytest.mark.parametrize("field", sorted(A._GATE_HEADLINE_FIELDS))
def test_sibling_referent_dropped_for_headline_fields(field):
    telem = {}
    out = A.apply_assignment_gates(
        [_rec(field, "MMLU contains 15,908 questions", referent="sibling:mmlu")],
        _frame(), telem)
    assert out == []
    assert telem["summary"]["sibling"]["drop"] == 1
    assert telem["actions"][0] == {"gate": "sibling", "field": field,
                                   "referent": "sibling:mmlu", "region": "",
                                   "action": "drop"}


def test_sibling_referent_kept_for_similar_benchmarks_unchanged():
    rec = _rec("benchmark_details.similar_benchmarks", "MMLU covers 57 subjects",
               referent="sibling:mmlu")
    telem = {}
    out = A.apply_assignment_gates([rec], _frame(), telem)
    assert out == [rec]
    assert telem["actions"] == []


def test_sibling_headline_kept_when_quote_names_target():
    # a comparison sentence naming both is kept, annotated, not dropped
    rec = _rec("benchmark_details.overview",
               "Unlike MMLU, OurBench covers open-ended physics tasks",
               referent="sibling:mmlu")
    telem = {}
    out = A.apply_assignment_gates([rec], _frame(), telem)
    assert len(out) == 1
    assert "about sibling: MMLU" in out[0]["note"]
    assert telem["summary"]["sibling"] == {"annotate": 1}


def test_sibling_nonheadline_field_kept_with_annotation():
    rec = _rec("data.annotation", "MMLU was annotated by crowdworkers",
               referent="sibling:mmlu")
    telem = {}
    out = A.apply_assignment_gates([rec], _frame(), telem)
    assert len(out) == 1
    assert out[0]["field"] == "data.annotation"
    assert "about sibling: MMLU" in out[0]["note"]
    assert telem["summary"]["sibling"] == {"annotate": 1}
    assert rec["note"] == ""                      # input record not mutated


def test_subset_size_refiled_to_size_breakdown():
    telem = {}
    out = A.apply_assignment_gates(
        [_rec("data.size", "the Hard subset contains 1,500 questions",
              referent="subset:hard", cs=30)],
        _frame(), telem)
    assert len(out) == 1
    assert out[0]["field"] == "data.size_breakdown"
    assert telem["summary"]["sibling"]["refile"] == 1
    refile = [a for a in telem["actions"] if a["action"] == "refile"][0]
    assert refile["field"] == "data.size"         # logs the ORIGINAL tag


def test_subset_size_refile_dedups_against_existing_breakdown_span():
    existing = _rec("data.size_breakdown", "the Hard subset contains 1,500 questions",
                    referent="subset:hard", cs=30)
    telem = {}
    out = A.apply_assignment_gates(
        [existing,
         _rec("data.size", "the Hard subset contains 1,500 questions",
              referent="subset:hard", cs=30)],
        _frame(), telem)
    breakdown = [r for r in out if r["field"] == "data.size_breakdown"]
    assert len(breakdown) == 1                    # not duplicated
    assert telem["summary"]["sibling"]["drop_duplicate"] == 1


def test_subset_size_dropped_when_breakdown_path_missing(monkeypatch):
    monkeypatch.setattr(E, "CURATOR_FIELDS",
                        {k: v for k, v in E.CURATOR_FIELDS.items()
                         if k != "data.size_breakdown"})
    telem = {}
    out = A.apply_assignment_gates(
        [_rec("data.size", "the Hard subset contains 1,500 questions",
              referent="subset:hard")],
        _frame(), telem)
    assert out == []
    assert telem["summary"]["sibling"]["drop"] == 1


def test_subset_and_split_referents_keep_flowing_elsewhere():
    recs = [
        _rec("methodology.metrics", "on Hard we report exact match", referent="subset:hard"),
        _rec("data.size", "the test set has 1,000 examples", referent="split:test"),
    ]
    telem = {}
    out = A.apply_assignment_gates(list(recs), _frame(), telem)
    assert [r["field"] for r in out] == ["methodology.metrics", "data.size"]
    assert [r["referent"] for r in out] == ["subset:hard", "split:test"]
    # only the table-anchor measurement fires (digit-bearing data.size, no table)
    assert set(telem["summary"]) <= {"table_anchor"}


# ----------------------------------------------------------------- region gate ----

def test_related_work_dropped_for_all_but_similar_benchmarks():
    telem = {}
    out = A.apply_assignment_gates(
        [_rec("purpose_and_intended_users.limitations",
              "prior benchmarks are saturated", region="related_work"),
         _rec("benchmark_details.similar_benchmarks", "MMLU covers 57 subjects",
              referent="sibling:mmlu", region="related_work")],
        _frame(), telem)
    assert [r["field"] for r in out] == ["benchmark_details.similar_benchmarks"]
    assert telem["summary"]["region"]["drop"] == 1


def test_region_gate_is_paper_only():
    rec = _rec("data.source", "collected from prior work", region="related_work",
               doc=E.DOC_HF)
    out = A.apply_assignment_gates([rec], _frame(), {})
    assert out == [rec]


@pytest.mark.parametrize("field,dropped", [
    ("methodology.baseline_results", True),
    ("methodology.metrics", True),
    ("data.size", False),
])
def test_ablation_dropped_for_baselines_and_metrics(field, dropped):
    telem = {}
    out = A.apply_assignment_gates(
        [_rec(field, "removing the retriever lowers accuracy by 12 points",
              region="ablation")],
        _frame(), telem)
    assert (out == []) is dropped


@pytest.mark.parametrize("field,dropped", [
    ("benchmark_details.overview", True),
    ("data.size_breakdown", False),
])
def test_appendix_dropped_for_overview_only(field, dropped):
    out = A.apply_assignment_gates(
        [_rec(field, "full per-category breakdown with 12 categories",
              region="appendix")],
        _frame(), {})
    assert (out == []) is dropped


# ------------------------------------------------------------- claim-role gate ----

@pytest.mark.parametrize("field", sorted(A._GATE_NUMERIC_FIELDS))
def test_illustration_dropped_for_numeric_fields(field):
    telem = {}
    out = A.apply_assignment_gates(
        [_rec(field, "for example, a model scoring 90 would rank first",
              claim_role="illustration")],
        _frame(), telem)
    assert out == []
    assert telem["summary"]["claim_role"]["drop"] == 1


def test_illustration_kept_for_non_numeric_field():
    rec = _rec("purpose_and_intended_users.tasks",
               "for example, identify the mistaken step", claim_role="illustration")
    out = A.apply_assignment_gates([rec], _frame(), {})
    assert out == [rec]


def test_secondary_dropped_for_baseline_results_only():
    telem = {}
    out = A.apply_assignment_gates(
        [_rec("methodology.baseline_results", "GPT-4 scored 86.4 in their study",
              claim_role="secondary"),
         _rec("methodology.metrics", "accuracy as defined by prior work",
              claim_role="secondary")],
        _frame(), telem)
    assert [r["field"] for r in out] == ["methodology.metrics"]
    assert telem["summary"]["claim_role"]["drop"] == 1


# ------------------------------------------------------------ table anchor pass ----

def test_numeric_quote_with_table_id_annotated():
    telem = {}
    out = A.apply_assignment_gates(
        [_rec("methodology.baseline_results", "GPT-4 achieves 86.4",
              table_id="table:3", section_path="4 Results", region="results")],
        _frame(), telem)
    assert len(out) == 1
    assert "table: table:3" in out[0]["note"]
    assert telem["summary"]["table_anchor"]["annotate"] == 1


def test_unanchored_numeric_counted_not_dropped():
    rec = _rec("data.size", "OurBench contains 21,000 questions",
               section_path="1 Introduction", region="intro")
    telem = {}
    out = A.apply_assignment_gates([rec], _frame(), telem)
    assert len(out) == 1                          # measurement only, no drop
    assert out[0]["note"] == ""
    assert telem["summary"]["table_anchor"]["unanchored_numeric"] == 1


def test_unanchored_numeric_not_counted_for_non_paper_docs():
    telem = {}
    A.apply_assignment_gates(
        [_rec("data.size", "21,000 questions", doc=E.DOC_HF)], _frame(), telem)
    assert "table_anchor" not in telem["summary"]


def test_digitless_quote_not_counted():
    telem = {}
    A.apply_assignment_gates(
        [_rec("data.size", "a large collection of questions", region="data")],
        _frame(), telem)
    assert "table_anchor" not in telem["summary"]


# ---------------------------------------------------------- telemetry contract ----

def test_action_entries_carry_the_five_keys_and_match_summary():
    telem = {}
    A.apply_assignment_gates(
        [_rec("data.size", "MMLU contains 15,908 questions", referent="sibling:mmlu"),
         _rec("methodology.baseline_results", "GPT-4 scored 86.4",
              claim_role="secondary"),
         _rec("data.size", "the Hard subset contains 1,500 questions",
              referent="subset:hard", cs=30)],
        _frame(), telem)
    for entry in telem["actions"]:
        assert set(entry) == {"gate", "field", "referent", "region", "action"}
    total_logged = sum(n for gate in telem["summary"].values() for n in gate.values())
    assert total_logged == len(telem["actions"])


def test_telemetry_dict_none_is_tolerated():
    out = A.apply_assignment_gates(
        [_rec("data.size", "MMLU contains 15,908 questions", referent="sibling:mmlu")],
        _frame(), None)
    assert out == []


# ------------------------------------------------- finalize_evidence wiring ----

def test_finalize_evidence_applies_gates_with_frame():
    paper = [
        _rec("benchmark_details.overview", "OurBench measures physics reasoning"),
        _rec("data.size", "MMLU contains 15,908 questions", referent="sibling:mmlu", cs=10),
        _rec("data.size", "OurBench contains 21,000 questions", cs=50),
    ]
    telem = {}
    items, digest = E.finalize_evidence(paper, [], [], [], frame=_frame(),
                                        gate_telemetry=telem)
    quotes = [it["quote"] for it in items]
    assert "MMLU contains 15,908 questions" not in quotes      # sibling size dropped
    assert "OurBench contains 21,000 questions" in quotes
    ids = [it["evidence_id"] for it in items]
    assert ids == [f"E{i:02d}" for i in range(1, len(items) + 1)]   # ids stay dense
    digest_ids = {it["evidence_id"] for v in digest.values() for it in v}
    assert digest_ids == set(ids)
    assert telem["summary"]["sibling"]["drop"] == 1


def test_finalize_evidence_without_frame_keeps_old_behavior():
    paper = [_rec("data.size", "MMLU contains 15,908 questions",
                  referent="sibling:mmlu")]
    items, _ = E.finalize_evidence(paper, [], [], [])
    assert [it["quote"] for it in items] == ["MMLU contains 15,908 questions"]
