"""CardSchema: the composer reads one schema object; the benchmark schema is the
byte-identical default.

Failure classes pinned here:
  1. Default drift: the benchmark CardSchema must be the SAME objects as the module
     constants, so every consumer that takes schema=None keeps its exact behavior.
  2. Hardwiring: a second schema must drive the extraction prompt, the field-spec
     block, the Stage-B prompt and the validator without any benchmark path leaking.
"""

import json
from typing import List, Optional

from pydantic import BaseModel, Field

from auto_benchmarkcard.tools.composer import composer_tool as C
from auto_benchmarkcard.tools.composer import evidence as E
from auto_benchmarkcard.tools.composer import field_spec as fs
from auto_benchmarkcard.tools.composer import validator as V


class _Identity(BaseModel):
    name: str = Field(..., description="The widget's official name")
    stage: str = Field(..., description="Development stage of the widget")
    provenance: Optional[dict] = None


class _Specs(BaseModel):
    colors: List[str] = Field(..., description="Colors the widget ships in")
    weight: str = Field(default="Not specified", description="Weight of the widget")
    provenance: Optional[dict] = None


def widget_schema():
    return fs.CardSchema(
        name="widget",
        entity_noun="widget",
        card_noun="widget card",
        other_entities_phrase="widgets or gadgets",
        sections=["identity", "specs"],
        curator_fields={
            "identity.name": {"sources": frozenset({"paper"}), "gap_query": None, "hint": "official name"},
            "identity.stage": {"sources": frozenset({"paper", "hf"}), "gap_query": "stage phase", "hint": "stage"},
            "specs.colors": {"sources": frozenset({"hf"}), "gap_query": None, "hint": "colors"},
            "specs.weight": {"sources": frozenset({"paper"}), "gap_query": "weight mass grams", "hint": "weight"},
        },
        field_caps={"specs.weight": "the shipped widget's weight; NOT the packaging weight"},
        enum_registry={"identity.stage": lambda: {"prototype", "production"}},
        enum_multi=set(),
        enum_comma_joined=set(),
        b_authored_enums={"identity.stage"},
        established_enums=set(),
        display_fields=set(),
        established_fields=set(),
        collapse_fields=set(),
        structured_fields=set(),
        evidence_exempt={"identity.name"},
        list_fields={"colors"},
        list_caps={"specs.colors": 2},
        groups=[("all", ["identity", "specs"])],
        stage_b_rules="RULES:\n1. Widgets only.",
        extraction_system="You extract verbatim quotes about one named widget.",
        section_models={"identity": _Identity, "specs": _Specs},
    )


def test_benchmark_schema_is_the_module_constants():
    sch = fs.resolve_schema(None)
    assert sch.name == "benchmark"
    assert sch.enum_registry is fs.ENUM_REGISTRY
    assert sch.field_caps is fs.FIELD_CAPS
    assert sch.list_fields is fs.LIST_FIELDS and sch.list_caps is fs.LIST_CAPS
    assert sch.evidence_exempt is fs.EVIDENCE_EXEMPT
    assert sch.groups is fs.ACTIVE_GROUPS
    assert sch.curator_fields is E.CURATOR_FIELDS
    assert sch.section_models is C.SECTION_MODELS
    assert sch.composed_sections == fs.ALL_SECTIONS
    for path in ("benchmark_details.data_type", "data.size", "data.size_breakdown",
                 "benchmark_details.overview", "benchmark_details.authors"):
        assert sch.field_class(path) == fs.field_class(path)
        assert sch.enum_vocab(path) == fs.enum_vocab(path)


def test_default_prompts_are_unchanged_by_the_schema_parameter():
    fields = E.fields_for_source("paper")
    a = E.build_extraction_prompt("research paper", fields, "FibbleBench", source_text="txt")
    b = E.build_extraction_prompt("research paper", fields, "FibbleBench", source_text="txt",
                                  schema=fs.resolve_schema(None))
    assert a == b
    p1 = C._build_group_prompt("g", ["data"], C.SECTION_MODELS, {}, {}, "")
    p2 = C._build_group_prompt("g", ["data"], C.SECTION_MODELS, {}, {}, "", schema=fs.resolve_schema(None))
    assert p1 == p2


def test_widget_schema_drives_extraction_prompt():
    sch = widget_schema()
    assert E.fields_for_source("hf", sch) == ["identity.stage", "specs.colors"]
    assert E.gap_fields(sch) == {"identity.stage": "stage phase", "specs.weight": "weight mass grams"}
    prompt = E.build_extraction_prompt("spec sheet", E.fields_for_source("paper", sch), "Zorb",
                                       source_text="The Zorb weighs 40 grams.", schema=sch)
    assert prompt.startswith("You extract verbatim quotes about one named widget.")
    assert 'about the widget "Zorb"' in prompt
    assert "Ignore other widgets or gadgets" in prompt
    assert "- specs.weight: weight" in prompt
    assert "benchmark" not in prompt.lower()


def test_widget_schema_verify_items_uses_its_whitelist():
    sch = widget_schema()
    items = [{"field": "specs.weight", "quote": "weighs 40 grams"},
             {"field": "data.size", "quote": "weighs 40 grams"}]
    records, telem = E.verify_items(items, "The Zorb weighs 40 grams.", "hf_readme", schema=sch)
    assert [r["field"] for r in records] == ["specs.weight"]
    assert telem["reject_reasons"]["bad_field"] == 1


def test_widget_schema_drives_stage_b_prompt_and_validator():
    sch = widget_schema()
    prompt = C._build_group_prompt("all", ["identity", "specs"], sch.section_models, {}, {}, "",
                                   schema=sch)
    assert "section(s) of a widget card" in prompt
    assert "RULES:\n1. Widgets only." in prompt
    assert "- stage [enum]: one or more of {production, prototype} or 'other:<text>'" in prompt
    assert "NOT the packaging weight" in prompt
    assert "- colors [prose]" in prompt and "one discrete item per array element" in prompt
    assert "benchmark" not in prompt.lower()

    out = {"identity": {"name": "Zorb", "stage": "beta",
                        "provenance": {"stage": {"evidence_ids": ["E1"]}}},
           "specs": {"colors": ["red", "blue", "green"], "weight": "40 grams",
                     "provenance": {"weight": {"evidence_ids": ["E2"]},
                                    "colors": {"evidence_ids": ["E2"]}}}}
    norm, errors, stats = V.validate_sections(out, ["identity", "specs"], sch.section_models,
                                              {"E1", "E2"}, set(), schema=sch)
    assert norm["specs"]["colors"] == ["red", "blue"]        # the widget cap of 2 applied
    assert stats["list_truncations"] == [{"field": "specs.colors", "original": 3, "kept": 2}]
    assert [e.type for e in errors] == ["enum_not_member"]     # 'beta' is outside the widget vocab
    assert errors[0].path == "identity.stage"


def test_value_fields_reject_sentences_and_the_benchmark_schema_has_none():
    # failure class: Stage B writes a sentence about the value into a bare-value field
    # ("The official display name is OLMo-2-1124-7B-Instruct." as a name, seen on the
    # model-card smoke); the validator must flag it so the repair loop replaces it
    assert fs.resolve_schema(None).value_fields == set()
    sch = widget_schema()
    sch.value_fields = {"identity.name"}
    out = {"identity": {"name": "The widget's official name is Zorb.", "stage": "production",
                        "provenance": {"stage": {"evidence_ids": ["E1"]}}},
           "specs": {"colors": "Not specified", "weight": "Not specified", "provenance": {}}}
    _, errors, _ = V.validate_sections(out, ["identity", "specs"], sch.section_models, {"E1"}, set(), schema=sch)
    assert [(e.path, e.type) for e in errors] == [("identity.name", "sentence_in_value_field")]
    out["identity"]["name"] = "Zorb Mk. II"
    _, errors, _ = V.validate_sections(out, ["identity", "specs"], sch.section_models, {"E1"}, set(), schema=sch)
    assert errors == []
    assert V.looks_like_sentence("The model was released in November 2024.")
    assert not V.looks_like_sentence("November 2024")
    assert not V.looks_like_sentence("7,298,617,344 parameters (safetensors metadata)")
    assert not V.looks_like_sentence("Dec. 2023")
    assert not V.looks_like_sentence("Up to 5 trillion tokens.")     # a value with a trailing period
    # a model that never repairs the sentence ships NS with the flag, never the sentence
    out["identity"]["name"] = "The widget's official name is Zorb."
    outcome = V.run_group_with_repair("all", ["identity", "specs"], sch.section_models, "BASE",
                                      lambda p: json.dumps(out), {"E1"}, schema=sch)
    assert outcome.sections["identity"]["name"] == "Not specified"
    assert outcome.flagged["identity.name"].startswith("[schema_invalid] sentence_in_value_field")


def test_widget_schema_group_loop_ships_with_its_ns_shape():
    sch = widget_schema()
    good = {"identity": {"name": "Zorb", "stage": "production",
                         "provenance": {"stage": {"evidence_ids": ["E1"]}}},
            "specs": {"colors": "Not specified", "weight": "Not specified", "provenance": {}}}
    outcome = V.run_group_with_repair("all", ["identity", "specs"], sch.section_models, "BASE",
                                      lambda p: json.dumps(good), {"E1"}, schema=sch)
    assert outcome.sections["specs"]["colors"] == ["Not specified"]   # list-form NS for the widget list field
    assert outcome.sections["identity"]["stage"] == "production"
    assert outcome.flagged == {}


def test_verify_items_honors_a_per_source_default_referent():
    # A model's family paper is not about the target by default; the benchmark composer
    # never sets a default and keeps the target (the None path is the existing behavior).
    frame = {"target_id": "target", "nodes": [
        {"id": "target", "type": "target", "name": "OLMo-2-1124-7B", "aliases": ["OLMo 2 7B"]},
        {"id": "family:olmo-2", "type": "family", "name": "OLMo 2", "aliases": []},
    ]}
    src = "We pretrain on 4 trillion tokens. OLMo 2 7B reaches 62 on MMLU."
    items = [{"field": "training_context.training_data", "quote": "We pretrain on 4 trillion tokens."},
             {"field": "training_context.training_data", "quote": "OLMo 2 7B reaches 62 on MMLU."}]
    sch = widget_schema()
    sch.curator_fields = {"training_context.training_data": {"sources": frozenset({"paper"}),
                                                              "gap_query": None, "hint": "x"}}
    recs, telem = E.verify_items(items, src, "docling", frame=frame, schema=sch,
                                 default_referent="family:olmo-2")
    assert [r["referent"] for r in recs] == ["family:olmo-2", "target"]
    recs2, _ = E.verify_items(items[:1], src, "docling", frame=frame, schema=sch)
    assert recs2[0]["referent"] == "target"
    # the model's own "target" declaration does not beat the source default for an
    # unnamed sentence (seen on the OLMo-2 Instruct smoke: "this SFT mix contains
    # 939,104 prompts" from the family paper bound as exact_target); a sentence that
    # names the target keeps it, and without a default the declaration still wins
    claimed = [dict(it, referent="target") for it in items]
    recs3, _ = E.verify_items(claimed, src, "docling", frame=frame, schema=sch,
                              default_referent="family:olmo-2")
    assert [r["referent"] for r in recs3] == ["family:olmo-2", "target"]
    recs4, _ = E.verify_items(claimed[:1], src, "docling", frame=frame, schema=sch)
    assert recs4[0]["referent"] == "target"
