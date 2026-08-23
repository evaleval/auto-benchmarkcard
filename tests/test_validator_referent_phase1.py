"""Phase-1 wrong-referent fixes in the Stage-B validator.

Pins the three mechanical guards:
  1. evidence-tag/field agreement: a cited id must be tagged for the citing field
     (exact for non-prose classes, section-mates allowed for prose), enforced only
     when the caller passes the digest as an id->field mapping;
  2. list-cap truncations are recorded in telemetry instead of vanishing;
  3. the enum repair instruction prefers the other:<verbatim> escape over coercing
     to the closest-looking allowed value, and FIELD_CAPS carries negative
     referent clauses for the top-defect fields.
"""

import json

from auto_benchmarkcard.tools.composer import composer_tool as C
from auto_benchmarkcard.tools.composer import field_spec as fs
from auto_benchmarkcard.tools.composer import validator as V
from tests.test_composer_stage_b import SECTION_MODELS, _gen

PURPOSE_DIGEST = {
    "E1": "purpose_and_intended_users.goal",
    "E2": "purpose_and_intended_users.audience",
}


def _purpose_citing(prov_ids):
    prov = {k: {"evidence_ids": v} for k, v in prov_ids.items()}
    return {"purpose_and_intended_users": {
        "goal": "g", "audience": ["safety evaluators"],
        "tasks": ["t"], "limitations": "l", "out_of_scope_uses": ["x"],
        "provenance": prov}}


def _all_e1():
    return {k: ["E1"] for k in ["goal", "audience", "tasks", "limitations", "out_of_scope_uses"]}


def _gen_rec(seq):
    box = list(seq)
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return box.pop(0)

    fn.calls = calls
    return fn


# ------------------------------------------- 1. evidence-tag/field agreement ----

def test_enum_needs_exact_tag_prose_accepts_section_mates():
    # every field cites E1 (tagged ...goal): prose section-mates pass, but the enum
    # field audience needs an exact tag match -> exactly one mismatch error
    _, errors, _ = V.validate_sections(
        _purpose_citing(_all_e1()), ["purpose_and_intended_users"], SECTION_MODELS,
        set(PURPOSE_DIGEST), set(), digest_fields=PURPOSE_DIGEST)
    mism = [e for e in errors if e.type == "evidence_field_mismatch"]
    assert [e.path for e in mism] == ["purpose_and_intended_users.audience"]
    assert not [e for e in errors if e.type == "evidence_missing"]


def _data_citing(source_ids, size_ids=None):
    prov = {"source": {"evidence_ids": source_ids}}
    if size_ids:
        prov["size"] = {"evidence_ids": size_ids}
    return {"data": {"source": "collected from exams", "size": "817", "format": "parquet",
                     "annotation": "Not specified", "size_breakdown": "Not specified",
                     "collection_date": "Not specified",
                     "contamination_controls": "Not specified", "provenance": prov}}


DATA_DIGEST = {"E8": "data.source", "E9": "methodology.methods"}


def test_prose_cross_section_citation_is_mismatch():
    _, errors, _ = V.validate_sections(
        _data_citing(["E9"]), ["data"], SECTION_MODELS,
        set(DATA_DIGEST), set(), digest_fields=DATA_DIGEST)
    assert [e.type for e in errors] == ["evidence_field_mismatch"]
    assert errors[0].path == "data.source"
    assert "methodology.methods" in errors[0].detail


def test_exempt_fields_keep_exemption():
    # data.size is established/evidence-exempt: a mis-tagged citation on it raises nothing
    _, errors, _ = V.validate_sections(
        _data_citing(["E8"], size_ids=["E9"]), ["data"], SECTION_MODELS,
        set(DATA_DIGEST), set(), digest_fields=DATA_DIGEST)
    assert not errors


def test_unknown_id_stays_evidence_missing_not_mismatch():
    _, errors, _ = V.validate_sections(
        _data_citing(["E99"]), ["data"], SECTION_MODELS,
        set(DATA_DIGEST), set(), digest_fields=DATA_DIGEST)
    assert [e.type for e in errors] == ["evidence_missing"]


def test_plain_id_set_skips_agreement_check():
    # legacy call form (iterable of ids): mis-tagged citations cannot be detected
    out = V.run_group_with_repair("data_only", ["data"], SECTION_MODELS, "BASE",
                                  _gen([json.dumps(_data_citing(["E9"]))]), set(DATA_DIGEST))
    assert out.telemetry["converged"] and not out.flagged
    assert out.telemetry["evidence_mismatch_found"] == 0


def test_mismatch_repairs_through_loop():
    wrong = json.dumps(_data_citing(["E9"]))
    right = json.dumps(_data_citing(["E8"]))
    gen = _gen_rec([wrong, right])
    out = V.run_group_with_repair("data_only", ["data"], SECTION_MODELS, "BASE",
                                  gen, DATA_DIGEST)
    assert out.telemetry["converged"] and not out.flagged
    assert out.telemetry["evidence_mismatch_found"] == 1
    assert out.telemetry["evidence_mismatch_repaired"] == 1
    assert out.sections["data"]["source"] == "collected from exams"
    assert "tagged for a different field" in gen.calls[1]  # repair prompt used the new entry


def test_mismatch_unrepaired_demotes_to_ns():
    wrong = json.dumps(_data_citing(["E9"]))
    out = V.run_group_with_repair("data_only", ["data"], SECTION_MODELS, "BASE",
                                  _gen([wrong] * 3), DATA_DIGEST)
    assert not out.telemetry["converged"]
    assert out.sections["data"]["source"] == "Not specified"
    assert "evidence_mismatch" in out.flagged["data.source"]
    assert out.telemetry["evidence_mismatch_found"] == 1
    assert out.telemetry["evidence_mismatch_repaired"] == 0
    assert out.telemetry["evidence_demoted"] >= 1


# --------------------------------------------------- 2. truncation telemetry ----

def test_list_cap_truncation_recorded_in_telemetry():
    over = _purpose_citing(_all_e1())
    over["purpose_and_intended_users"]["tasks"] = [f"task {i}" for i in range(14)]
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(over)]), {"E1"})
    assert len(out.sections["purpose_and_intended_users"]["tasks"]) == 12  # cap still applies
    assert out.telemetry["list_truncations"] == [
        {"field": "purpose_and_intended_users.tasks", "original": 14, "kept": 12}]


def test_no_truncation_means_empty_telemetry():
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(_purpose_citing(_all_e1()))]), {"E1"})
    assert out.telemetry["list_truncations"] == []


def test_list_hygiene_without_collector_still_truncates():
    val = [f"t{i}" for i in range(14)]
    assert len(V.list_hygiene("purpose_and_intended_users.tasks", val)) == 12


# ------------------------------------------- 3. enum escape + negative caps ----

def test_enum_repair_instruction_prefers_escape_over_coercion():
    err = V.VErr("purpose_and_intended_users.audience", "enum_not_member",
                 allowed=["safety evaluators"], observed="ML folks")
    prompt = V.build_repair_prompt("BASE", [err])
    assert "closest" not in prompt
    assert "genuinely the same thing" in prompt
    assert "other:<verbatim term from the evidence>" in prompt


def test_other_verbatim_escape_accepted_by_enum_check():
    payload = _purpose_citing(_all_e1())
    payload["purpose_and_intended_users"]["audience"] = ["other:medical exam designers"]
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(payload)]), {"E1"})
    assert out.telemetry["converged"] and not out.flagged
    assert out.sections["purpose_and_intended_users"]["audience"] == ["other:medical exam designers"]


def test_negative_caps_on_top_defect_fields():
    for path in ("methodology.metrics", "data.size", "data.size_breakdown",
                 "methodology.baseline_results", "benchmark_details.authors",
                 "benchmark_details.languages", "benchmark_details.domains", "data.source"):
        assert "NOT" in fs.FIELD_CAPS[path], path
    assert "NOT a subset, split, or a prior version's count" in fs.FIELD_CAPS["data.size"]


def test_negative_caps_surface_in_prompt_spec():
    spec = C._field_spec_block(SECTION_MODELS, ["data"], {})
    assert "NOT a subset" in spec           # established field still carries its cap
    assert "NOT the hosting platform" in spec
