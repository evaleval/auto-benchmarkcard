"""Tests for the Stage-B formatter + validator-loop (composer-redesign v2, phase [B+V]).

Network-free: the validator loop is driven by a fake generate_fn and the full
compose tool is exercised with a monkeypatched LLM handler.
"""

import json

import pytest

from auto_benchmarkcard.tools.composer import composer_tool as C
from auto_benchmarkcard.tools.composer import field_spec as fs
from auto_benchmarkcard.tools.composer import validator as V
from auto_benchmarkcard.tools.composer.composer_tool import (
    BenchmarkCard, BenchmarkDetails, DataInfo, EthicalAndLegalConsiderations,
    Methodology, PurposeAndIntendedUsers)

SECTION_MODELS = {
    "benchmark_details": BenchmarkDetails,
    "purpose_and_intended_users": PurposeAndIntendedUsers,
    "data": DataInfo,
    "methodology": Methodology,
    "ethical_and_legal_considerations": EthicalAndLegalConsiderations,
}


def _gen(seq):
    box = list(seq)
    return lambda prompt: box.pop(0)


def _gen_meta(responses):
    """Fake returning (text, stop_reason) pairs; records each call's (prompt, max_tokens)."""
    box = list(responses)
    calls = []

    def fn(prompt, max_tokens=None):
        calls.append((prompt, max_tokens))
        return box.pop(0)

    fn.calls = calls
    return fn


def _purpose(aud="safety evaluators"):
    prov = {k: {"evidence_ids": ["E1"]} for k in
            ["goal", "audience", "tasks", "limitations", "out_of_scope_uses"]}
    return {"purpose_and_intended_users": {
        "goal": "g", "audience": [aud],
        "tasks": ["t"], "limitations": "l", "out_of_scope_uses": ["x"], "provenance": prov}}


# ---------------------------------------------------------------- schema ----

def test_models_have_new_fields_and_single_audience():
    assert "audience" in PurposeAndIntendedUsers.model_fields
    assert {"audience_evaluators", "audience_consumers"}.isdisjoint(PurposeAndIntendedUsers.model_fields)
    assert {"size_breakdown", "collection_date", "contamination_controls"} <= set(DataInfo.model_fields)
    assert {"human_baseline", "judge_uses_llm", "judge_num", "judge_models",
            "judge_score_consolidation", "validity_justification"} <= set(Methodology.model_fields)
    assert {"authors", "logo", "org_url"} <= set(BenchmarkDetails.model_fields)


def test_display_fields_optional_excluded_when_unset():
    bd = BenchmarkDetails(name="X", overview="o", data_type="text", domains=["d"],
                          languages=["English"], similar_benchmarks=[], resources=[])
    assert bd.authors is None and bd.logo is None and bd.org_url is None
    assert "authors" not in bd.model_dump(exclude_none=True)


def test_provenance_widening_accepts_dict_and_list():
    me = Methodology(methods=["m"], metrics=["Accuracy"], calculation="c", interpretation="x",
                     baseline_results="b", validation="v",
                     provenance={"metrics": {"source": "eee", "evidence": "q",
                                 "quote_loc": {"doc": "eee", "char_start": 5}, "evidence_ids": ["E07"]}})
    assert me.provenance["metrics"]["quote_loc"] == {"doc": "eee", "char_start": 5}
    assert me.provenance["metrics"]["evidence_ids"] == ["E07"]


def test_new_content_fields_default_to_ns():
    di = DataInfo(source="s", size="1", format="parquet", annotation="a")
    assert di.size_breakdown == "Not specified"
    assert DataInfo(source="s", size="1", format="parquet", annotation="a",
                    size_breakdown={"train": 5}).size_breakdown == {"train": 5}


# ------------------------------------------------------------- field_spec ----

def test_enum_scope_is_exactly_six_paths_trap5():
    assert set(fs.ENUM_REGISTRY) == fs.EXPECTED_ENUM_PATHS
    for free in ("methodology.metrics", "methodology.interpretation", "data.format"):
        assert free not in fs.ENUM_REGISTRY
    for prose in ("benchmark_details.overview", "purpose_and_intended_users.goal",
                  "methodology.calculation"):
        assert prose not in fs.ENUM_REGISTRY


def test_enum_vocab_from_live_maps():
    langs = fs.enum_vocab("benchmark_details.languages")
    assert "English" in langs and "Multilingual" in langs
    assert "MIT License" in fs.enum_vocab("ethical_and_legal_considerations.data_licensing")
    assert fs.enum_vocab("benchmark_details.overview") is None


def test_groups_cover_all_sections():
    fs.assert_groups_cover()
    flat = [s for _, secs in fs.ACTIVE_GROUPS for s in secs]
    assert sorted(flat) == sorted(fs.ALL_SECTIONS)


# ----------------------------------------------------------- json parsing ----

@pytest.mark.parametrize("raw,expected", [
    ('{"a":1}', {"a": 1}),
    ('```json\n{"a":1}\n```', {"a": 1}),
    ('sure:\n{"a": {"b": 2}} trailing', {"a": {"b": 2}}),
    ('{"s":"a } b","n":{"x":1}}', {"s": "a } b", "n": {"x": 1}}),
])
def test_robust_json_loads_ok(raw, expected):
    assert V.robust_json_loads(raw)[0] == expected


def test_robust_json_loads_unparseable():
    obj, err = V.robust_json_loads("not json at all")
    assert obj is None and err


# ------------------------------------------------------------- validator ----

def test_clean_first_pass_ethics():
    ethics = {"ethical_and_legal_considerations": {
        "privacy_and_anonymity": "p", "data_licensing": "MIT License", "consent_procedures": "c",
        "compliance_with_regulations": "r",
        "provenance": {"privacy_and_anonymity": {"evidence_ids": ["E1"]},
                       "consent_procedures": {"evidence_ids": ["E2"]},
                       "compliance_with_regulations": {"evidence_ids": ["E3"]}}}}
    out = V.run_group_with_repair("ethics", ["ethical_and_legal_considerations"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(ethics)]), {"E1", "E2", "E3"})
    assert out.telemetry["first_pass_valid"] and out.telemetry["converged"]
    assert not out.flagged


def test_enum_repair_converges_round_two():
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(_purpose("ML folks")), json.dumps(_purpose())]),
                                  {"E1"})
    assert out.telemetry["converged"] and out.telemetry["iterations_to_valid"] == 2
    assert not out.flagged


def test_enum_miss_kept_best_effort_and_flagged():
    bad = json.dumps(_purpose("ML folks"))
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([bad, bad, bad]), {"E1"})
    assert not out.telemetry["converged"]
    assert out.sections["purpose_and_intended_users"]["audience"] == ["ML folks"]
    assert "purpose_and_intended_users.audience" in out.flagged


def test_other_escape_passes_and_counts():
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(_purpose("other:grant reviewers"))]), {"E1"})
    assert out.telemetry["converged"] and out.telemetry["n_other"] >= 1
    assert out.telemetry["other_rate"] > 0


def test_evidence_forgery_demoted_to_ns():
    forge = _purpose()
    forge["purpose_and_intended_users"]["provenance"]["goal"] = {"evidence_ids": ["E99"]}
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(forge)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["goal"] == "Not specified"
    assert "purpose_and_intended_users.goal" in out.flagged
    assert out.telemetry["evidence_demoted"] >= 1


def test_empty_dict_collapse_and_ns_leak():
    data = {"data": {"source": "no facts provided", "size": "1", "format": "parquet", "annotation": "a",
                     "size_breakdown": {}, "collection_date": "Not specified",
                     "contamination_controls": "Not specified", "provenance": {}}}
    out = V.run_group_with_repair("data_only", ["data"], SECTION_MODELS, "BASE",
                                  _gen([json.dumps(data)]), set())
    sec = out.sections["data"]
    assert sec["size_breakdown"] == "Not specified"   # empty dict collapsed
    assert sec["source"] == "Not specified"           # leak-phrasing normalized
    assert out.telemetry["ns_normalized"] >= 1


def test_keep_best_prefers_parsed_over_unparseable():
    bad = json.dumps(_purpose("ML folks"))
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([bad, "GARBAGE", "GARBAGE"]), {"E1"})
    assert out.sections["purpose_and_intended_users"]["audience"] == ["ML folks"]


def test_established_enum_mismatch_is_soft_not_flagged():
    # data_licensing is established (_ALWAYS_OVERRIDE); an out-of-vocab value is a soft mismatch
    # (no hard flag, still converges) AND is coerced to NS so the raw value never ships.
    ethics = {"ethical_and_legal_considerations": {
        "privacy_and_anonymity": "Not specified", "data_licensing": "Some Weird License",
        "consent_procedures": "Not specified", "compliance_with_regulations": "Not specified",
        "provenance": {}}}
    out = V.run_group_with_repair("ethics", ["ethical_and_legal_considerations"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(ethics)]), set())
    assert out.telemetry["converged"]                 # soft mismatch does not block
    assert not out.flagged
    assert out.telemetry["established_mismatch"] >= 1
    assert out.sections["ethical_and_legal_considerations"]["data_licensing"] == "Not specified"


def _ethics(data_licensing):
    return {"ethical_and_legal_considerations": {
        "privacy_and_anonymity": "Not specified", "data_licensing": data_licensing,
        "consent_procedures": "Not specified", "compliance_with_regulations": "Not specified",
        "provenance": {}}}


def test_out_of_vocab_data_licensing_coerced_to_ns_valid_survives():
    # data_licensing is a scalar established enum: an out-of-vocab license must never ship raw
    # -> coerced to NS (never guess). A vocab license passes through untouched.
    bad, _, stats = V.validate_sections(
        _ethics("Some Weird License"), ["ethical_and_legal_considerations"], SECTION_MODELS, set(), set())
    assert bad["ethical_and_legal_considerations"]["data_licensing"] == "Not specified"
    assert stats["established_mismatch"] >= 1
    good, _, _ = V.validate_sections(
        _ethics("MIT License"), ["ethical_and_legal_considerations"], SECTION_MODELS, set(), set())
    assert good["ethical_and_legal_considerations"]["data_licensing"] == "MIT License"


def _bd(data_type):
    return {"benchmark_details": {
        "name": "X", "overview": "Not specified", "data_type": data_type,
        "domains": ["Not specified"], "languages": ["English"],
        "similar_benchmarks": ["Not specified"], "resources": [], "provenance": {}}}


def test_out_of_vocab_data_type_coerced_to_ns_valid_survives():
    # an out-of-vocab modality must never ship -> coerced to NS (never guess); a valid one stays.
    bad, _, stats = V.validate_sections(
        _bd("biology"), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert bad["benchmark_details"]["data_type"] == "Not specified"
    assert stats["established_mismatch"] >= 1
    good, _, _ = V.validate_sections(
        _bd("code"), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert good["benchmark_details"]["data_type"] == "code"
    # mixed valid+invalid: keep the valid modality, drop only the out-of-vocab token
    mixed, _, _ = V.validate_sections(
        _bd("text, bogus"), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert mixed["benchmark_details"]["data_type"] == "text"


def test_data_type_junk_compound_other_cleaned():
    # A compound / task-format other: in the modality field passes enum_check unchecked
    # (misses=[], used_other=True), so the misses-gated backstop never runs. The other:-
    # content backstop cleans it to a real modality or a single clean other:<modality>.
    out, _, _ = V.validate_sections(
        _bd("other:multiple-choice and error-detection"),
        ["benchmark_details"], SECTION_MODELS, set(), set())
    assert out["benchmark_details"]["data_type"] == "other:error-detection"
    # a clean single other:<modality> is a no-op
    keep, _, _ = V.validate_sections(
        _bd("other:simulation"), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert keep["benchmark_details"]["data_type"] == "other:simulation"
    # a clean in-vocab multi-modality is untouched (used_other is False -> backstop skipped)
    multi, _, _ = V.validate_sections(
        _bd("image, text"), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert multi["benchmark_details"]["data_type"] == "image, text"
    # a compound other: that names a real modality coerces to that modality
    coerce, _, _ = V.validate_sections(
        _bd("other:image and classification"),
        ["benchmark_details"], SECTION_MODELS, set(), set())
    assert coerce["benchmark_details"]["data_type"] == "image"


def _bd_lang(languages):
    return {"benchmark_details": {
        "name": "X", "overview": "Not specified", "data_type": "text",
        "domains": ["Not specified"], "languages": languages,
        "similar_benchmarks": ["Not specified"], "resources": [], "provenance": {}}}


def test_out_of_vocab_languages_dropped_valid_survives():
    # languages is a list-valued established enum: an ungrounded Stage-B non-answer like
    # "already established" must never ship -> dropped element-wise; the field coerces to NS
    # only when nothing valid remains (never guess a language). Valid languages survive.
    junk, _, stats = V.validate_sections(
        _bd_lang(["already established"]), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert junk["benchmark_details"]["languages"] == ["Not specified"]
    assert stats["established_mismatch"] >= 1
    # mixed valid+invalid: keep the valid language, drop only the out-of-vocab token
    mixed, _, _ = V.validate_sections(
        _bd_lang(["English", "already established"]), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert mixed["benchmark_details"]["languages"] == ["English"]
    # all-valid inputs pass through untouched
    good, _, _ = V.validate_sections(
        _bd_lang(["English"]), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert good["benchmark_details"]["languages"] == ["English"]
    na, _, _ = V.validate_sections(
        _bd_lang(["not-applicable"]), ["benchmark_details"], SECTION_MODELS, set(), set())
    assert na["benchmark_details"]["languages"] == ["not-applicable"]


def test_keep_best_prefers_cleaner_earlier_over_dirtier_later():
    # enum_check collapses every out-of-vocab token of a field into ONE VErr, so a
    # 1-bad-token round and a 2-bad-token round tie on len(errors). keep-best must still
    # prefer the cleaner earlier attempt (kept a valid token) over the dirtier later one.
    def _p(aud):
        prov = {k: {"evidence_ids": ["E1"]} for k in
                ["goal", "audience", "tasks", "limitations", "out_of_scope_uses"]}
        return json.dumps({"purpose_and_intended_users": {
            "goal": "g", "audience": aud,
            "tasks": ["t"], "limitations": "l", "out_of_scope_uses": ["x"], "provenance": prov}})
    cleaner = _p(["safety evaluators", "x"])   # 1 valid token kept, 1 miss
    dirtier = _p(["bad1", "bad2"])             # 0 valid tokens, 2 misses
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([cleaner, dirtier, dirtier]), {"E1"})
    assert not out.telemetry["converged"]
    assert "safety evaluators" in out.sections["purpose_and_intended_users"]["audience"]


def test_provenance_nonstring_source_evidence_safe_for_backfill():
    from auto_benchmarkcard.card_utils import backfill_from_provenance
    # Contract §2: source/evidence must stay strings. The LLM here confuses them with the
    # adjacent evidence_ids -> dict source + list evidence. A dict/list survivor crashes
    # _is_structured_source / backfill_from_provenance on .lower(); the validator must drop
    # the non-string keys while preserving the structured quote_loc/evidence_ids.
    prov = {f: {"evidence_ids": ["E1"]} for f in
            ["methods", "calculation", "interpretation", "baseline_results", "validation"]}
    prov["metrics"] = {"source": {"doc": "eee"}, "evidence": ["E07"],
                       "quote_loc": {"doc": "eee", "char_start": 5}, "evidence_ids": ["E1"]}
    methodology = {"methodology": {
        "methods": ["m"], "metrics": ["Accuracy"], "calculation": "c", "interpretation": "i",
        "baseline_results": "b", "validation": "v", "provenance": prov}}
    out = V.run_group_with_repair("data_method", ["methodology"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(methodology)]), {"E1"})
    pm = out.sections["methodology"]["provenance"]["metrics"]
    assert not isinstance(pm.get("source"), (dict, list))    # coerced away / absent
    assert not isinstance(pm.get("evidence"), (dict, list))
    assert pm["quote_loc"] == {"doc": "eee", "char_start": 5}  # structured keys survive
    assert pm["evidence_ids"] == ["E1"]
    section = out.sections["methodology"]
    backfill_from_provenance({"methodology": section}, {"methodology": section["provenance"]}, ["ctx"])


def test_collapse_all_zero_dict_is_placeholder():
    # COLLAPSE_FIELDS promise empty/placeholder -> NS; an all-zero size_breakdown is a
    # placeholder and must not leak as a filled value.
    data = {"data": {"source": "s", "size": "1", "format": "parquet", "annotation": "a",
                     "size_breakdown": {"all": 0}, "collection_date": "Not specified",
                     "contamination_controls": "Not specified",
                     "provenance": {"source": {"evidence_ids": ["E1"]},
                                    "annotation": {"evidence_ids": ["E1"]}}}}
    out = V.run_group_with_repair("data_only", ["data"], SECTION_MODELS, "BASE",
                                  _gen([json.dumps(data)]), {"E1"})
    assert out.sections["data"]["size_breakdown"] == "Not specified"


# ------------------------------------------------ digest -> prompt seams ----

def test_judge_setup_bucket_surfaced_and_four_flat_fields_requested():
    digest = {"methodology.judge_setup": [{"evidence_id": "E1", "field": "methodology.judge_setup",
              "quote": "scored by 3 LLM judges, majority vote", "doc": "paper", "kind": "stated"}]}
    block = C._digest_block(digest, ["methodology"])
    assert "methodology.judge_setup" in block and "E1" in block
    spec = C._field_spec_block(SECTION_MODELS, ["methodology"], {})
    for f in fs.JUDGE_FLAT_FIELDS:
        assert f in spec


def test_field_spec_block_excludes_display_and_shows_audience_vocab():
    spec = C._field_spec_block(SECTION_MODELS, ["benchmark_details", "purpose_and_intended_users"], {})
    assert "authors" not in spec and "logo" not in spec   # display fields not authored
    assert "safety evaluators" in spec                     # audience vocab surfaced


def test_field_spec_block_label_conditional_on_det_value():
    # the "already established" label appears ONLY when det_facts carries the value; an established
    # field with no det value gets the plain spec (so B never echoes the label as a value).
    with_val = C._field_spec_block(SECTION_MODELS, ["benchmark_details"],
                                   {"benchmark_details.data_type": "code"})
    assert "already established" in with_val               # data_type is backed by a det value
    assert with_val.count("already established") == 1      # languages (no det value) stays plain
    without = C._field_spec_block(SECTION_MODELS, ["benchmark_details"], {})
    assert "already established" not in without


# ----------------------------------------- list / gloss / discipline caps ----

_LIST_CLAUSE = "one discrete item per array element"


def _spec_lines(*sections):
    spec = C._field_spec_block(SECTION_MODELS, list(sections), {})
    return {ln.split()[1]: ln for ln in spec.splitlines() if ln.startswith("- ")}


def test_list_fields_get_atomicity_clause_prose_does_not():
    lines = _spec_lines("benchmark_details", "purpose_and_intended_users")
    for f in ("domains", "tasks", "out_of_scope_uses", "similar_benchmarks", "audience"):
        assert _LIST_CLAUSE in lines[f], f
    assert _LIST_CLAUSE not in lines["goal"]          # a non-list prose field stays prose


def test_list_clause_is_general_outside_pilot_fields():
    # generality witness: methods/resources are list fields unrelated to the 3 pilot cards
    lines = _spec_lines("methodology", "benchmark_details")
    for f in ("methods", "resources"):
        assert _LIST_CLAUSE in lines[f], f


def test_stage_b_rules_carve_prose_and_add_list_subrule():
    rules = C._STAGE_B_RULES
    assert "4b." in rules
    assert "Prose (non-list) fields" in rules
    assert _LIST_CLAUSE in rules


def test_rule5_requires_bare_judge_model_identifiers():
    rules = C._STAGE_B_RULES
    assert "BARE model identifiers" in rules
    assert "never in judge_models" in rules


def test_audience_line_present_with_vocab_and_noise_stripped():
    lines = _spec_lines("purpose_and_intended_users")
    aud_line = lines["audience"]
    assert "safety evaluators" in aud_line              # vocab surfaced on the single field
    assert "runs" in aud_line and "uses" in aud_line    # fused gloss covers both roles
    spec = C._field_spec_block(SECTION_MODELS, ["purpose_and_intended_users"], {})
    assert "controlled vocabulary" not in spec and "see field spec" not in spec


def test_field_gloss_keeps_useful_parenthetical_strips_machinery():
    class _F:
        description = "Who runs it (controlled vocabulary; see field spec)"
    assert C._field_gloss(_F()) == "Who runs it"

    class _G:
        description = "Subject areas (e.g., medical, legal)"
    assert "e.g., medical, legal" in C._field_gloss(_G())


def test_uncapped_prose_field_surfaces_description_gloss():
    # data.source has no FIELD_CAPS entry -> its noise-stripped description gloss appears
    assert "data.source" not in fs.FIELD_CAPS
    desc_gloss = C._field_gloss(DataInfo.model_fields["source"])
    assert desc_gloss                                  # the model field carries a description
    assert desc_gloss in _spec_lines("data")["source"]


def test_capped_field_shows_cap_not_raw_description():
    lines = _spec_lines("purpose_and_intended_users", "methodology")
    assert "no scope-framing" in lines["tasks"]                  # B3 cap
    assert "bare model identifier" in lines["judge_models"]      # B4 cap
    assert "not the run/invocation command" in lines["methods"]  # B5 cap


def test_new_field_caps_present_with_expected_substrings():
    caps = fs.FIELD_CAPS
    assert "no scope-framing" in caps["purpose_and_intended_users.tasks"]
    assert "misuse case" in caps["purpose_and_intended_users.out_of_scope_uses"]
    assert "bare model identifier" in caps["methodology.judge_models"]
    comp = caps["ethical_and_legal_considerations.compliance_with_regulations"]
    assert "GDPR" in comp and "do NOT restate the license" in comp
    assert "distinct from methods" in caps["methodology.calculation"]
    assert "do NOT restate" in caps["purpose_and_intended_users.limitations"]
    # metrics is an established list field; a content cap would fight "copy verbatim"
    assert "methodology.metrics" not in caps


# -------------------------------------------------------- full compose ----

def _all_ns_response():
    out = {}
    for sec, cls in SECTION_MODELS.items():
        out[sec] = {f: "Not specified" for f in cls.model_fields if f != "provenance"}
        out[sec]["provenance"] = {}
    return json.dumps(out)


class _FakeHandler:
    model_name = "fake-model"

    def __init__(self, resp):
        self.resp = resp

    def generate(self, prompt, response_format=None):
        return self.resp

    def generate_with_meta(self, prompt, response_format=None, max_completion_tokens=None):
        return self.resp, None


def _compose(query="TestBench"):
    fn = getattr(C.compose_benchmark_card, "func", C.compose_benchmark_card)
    return fn(query=query)


def test_full_compose_clean(monkeypatch):
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler(_all_ns_response()))
    res = _compose()
    card = res["benchmark_card"]
    assert "audience" in card["purpose_and_intended_users"]
    assert {"audience_evaluators", "audience_consumers"}.isdisjoint(card["purpose_and_intended_users"])
    assert {"size_breakdown", "judge_uses_llm", "validity_justification"} <= (
        set(card["data"]) | set(card["methodology"]))
    val = res["composition_metadata"]["validation"]
    assert set(val["per_group"]) == {"identity_purpose", "data_method", "ethics"}
    assert val["card"]["first_pass_validity"] == 1.0
    assert res["composition_metadata"]["generation_method"] == "stage_b_grouped_validated"
    # aboutness guard ran and surfaced telemetry; an all-NS card has nothing to demote
    assert "aboutness" in val
    assert val["aboutness"]["judge_demoted"] is False
    assert val["aboutness"]["data_source_demoted"] is False


def test_full_compose_demotes_ungrounded_value(monkeypatch):
    resp = json.loads(_all_ns_response())
    resp["benchmark_details"]["overview"] = "A real overview claim."
    resp["benchmark_details"]["provenance"] = {"overview": {"source": "paper", "evidence": "q"}}
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler(json.dumps(resp)))
    res = _compose()
    card = res["benchmark_card"]
    assert card["benchmark_details"]["overview"] == "Not specified"
    assert any("overview" in k for k in card.get("flagged_fields", {}))


# ----------------------------------------------------- truncation (BUG 1) ----

# a group output cut off mid-string: unparseable, the way the token cap truncates it
_TRUNCATED = '{"purpose_and_intended_users": {"goal": "g'


def test_truncation_raises_budget_and_keeps_base_prompt():
    gen = _gen_meta([(_TRUNCATED, "length"), (json.dumps(_purpose()), "stop")])
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", gen, {"E1"})
    assert out.telemetry["converged"]
    assert out.telemetry["truncation_retries"] == 1
    assert gen.calls[0] == ("BASE", None)      # first call uses the config-default budget
    assert gen.calls[1] == ("BASE", 32768)     # truncation -> base prompt + bigger budget, not a repair prompt


def test_truncation_exhausted_stops_without_repair_round():
    # still truncating after the single escalation: stop, do NOT run a (re-truncating) repair round
    gen = _gen_meta([(_TRUNCATED, "length"), (_TRUNCATED, "length")])
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", gen, {"E1"})
    assert not out.telemetry["converged"]
    assert out.telemetry["truncation_retries"] == 1
    assert len(gen.calls) == 2                  # short-circuited: no 3rd repair round
    assert gen.calls[1] == ("BASE", 32768)


def test_truncated_then_invalid_takes_normal_repair_path():
    # round 0 truncated -> escalate; round 1 parses but invalid (bad enum) -> normal repair round 2
    valid = json.dumps(_purpose())
    gen = _gen_meta([(_TRUNCATED, "length"), (json.dumps(_purpose("ML folks")), "stop"), (valid, "stop")])
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", gen, {"E1"})
    assert out.telemetry["converged"]
    assert out.telemetry["truncation_retries"] == 1
    assert len(gen.calls) == 3
    assert gen.calls[1] == ("BASE", 32768)      # truncation escalation
    assert gen.calls[2][1] == 32768             # elevated budget persists (don't risk re-truncation)...
    assert gen.calls[2][0] != "BASE"            # ...with a (longer) repair prompt


# --------------------------------------------- leaked evidence id (BUG 3) ----

def test_leaked_evidence_id_demoted_to_ns():
    leak = _purpose()
    leak["purpose_and_intended_users"]["goal"] = "E01"
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(leak)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["goal"] == "Not specified"
    assert "leaked_evidence_id" in out.flagged.get("purpose_and_intended_users.goal", "")


def test_leaked_evidence_id_bracket_form_demoted():
    leak = _purpose()
    leak["purpose_and_intended_users"]["limitations"] = "[E09]"
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(leak)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["limitations"] == "Not specified"
    assert "leaked_evidence_id" in out.flagged.get("purpose_and_intended_users.limitations", "")


def test_leaked_evidence_id_embedded_in_text_demoted():
    # scalar field whose substantive content is a labelled citation, past the old bare-eid guard
    leak = _purpose()
    leak["purpose_and_intended_users"]["limitations"] = "Provenance: [E07]"
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(leak)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["limitations"] == "Not specified"
    assert "leaked_evidence_id" in out.flagged.get("purpose_and_intended_users.limitations", "")


def test_list_field_drops_only_leaked_element():
    leak = _purpose()
    leak["purpose_and_intended_users"]["tasks"] = ["question answering", "Provenance: [E07]"]
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(leak)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["tasks"] == ["question answering"]
    assert "purpose_and_intended_users.tasks" not in out.flagged


def test_list_field_all_leaks_normalized_to_ns():
    leak = _purpose()
    leak["purpose_and_intended_users"]["tasks"] = ["[E01]", "Evidence: E02"]
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(leak)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["tasks"] == ["Not specified"]


def test_list_field_keeps_legit_id_mention():
    # an id mentioned inside real prose is not a citation-only value -> kept
    leak = _purpose()
    leak["purpose_and_intended_users"]["tasks"] = ["evaluation using the E07 protocol"]
    out = V.run_group_with_repair("purpose", ["purpose_and_intended_users"], SECTION_MODELS,
                                  "BASE", _gen([json.dumps(leak)] * 3), {"E1"})
    assert out.sections["purpose_and_intended_users"]["tasks"] == ["evaluation using the E07 protocol"]


# ----------------------------------------- paper source-label threading (fix c) ----

def _extract_resp(field, quote):
    return json.dumps({"evidence": [{"field": field, "quote": quote, "kind": "stated"}]})


def test_extract_facts_from_paper_default_doc_is_docling(monkeypatch):
    content = "The benchmark contains 1140 hand-written programming problems."
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler(
        _extract_resp("data.size", "1140 hand-written programming problems")))
    recs, _ = C.extract_facts_from_paper(content, "Bench", verify_source=content)
    assert recs and recs[0]["doc"] == "docling"


def test_extract_facts_from_paper_abstract_doc(monkeypatch):
    from auto_benchmarkcard.tools.composer import evidence as ev
    content = "The benchmark contains 1140 hand-written programming problems."
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler(
        _extract_resp("data.size", "1140 hand-written programming problems")))
    recs, _ = C.extract_facts_from_paper(content, "Bench", verify_source=content, doc=ev.DOC_ABSTRACT)
    assert recs and recs[0]["doc"] == "abstract"
