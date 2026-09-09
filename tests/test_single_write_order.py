"""Tests for the single write order: the HF corroboration gate runs post-compose in
run_composer, and the run_factreasoner tail writes no new card content (no backfill,
no second override pass, no flag clearing)."""

import inspect

import auto_benchmarkcard.workers as W


def _state(name, hf_json):
    return {
        "query": name,
        "hf_json": hf_json,
        "eee_metadata": {"benchmark_name": name, "metrics": {}},
        "extracted_ids": {},
    }


def _card():
    return {
        "benchmark_details": {
            "name": "mybench",
            "logo": "https://hf.co/otherorg/logo.png",
            "org_url": "https://hf.co/otherorg",
            "overview": "A benchmark for testing.",
        },
        "methodology": {"metrics": ["Accuracy"]},
        "data": {"size": "1000 examples"},
    }


def test_gate_drops_tag_fields_for_uncorroborated_repo():
    # repo basename bears no whole-token run of the benchmark name -> corroboration fails,
    # so the repo-org display fields reset to NS and the verdict lands in state.
    hf_json = {"id": "otherorg/totally_unrelated_collection", "tags": ["language:en"]}
    state = _state("mybench", hf_json)
    card = _card()
    W._apply_hf_corroboration_gate(state, card, provenance=None)
    assert card["benchmark_details"]["logo"] == "Not specified"
    assert card["benchmark_details"]["org_url"] == "Not specified"
    assert state["repo_corroboration"]["repo_id"] == "otherorg/totally_unrelated_collection"
    assert state["repo_corroboration"]["reason"]


def test_gate_leaves_corroborated_repo_untouched():
    hf_json = {"id": "goodorg/mybench", "tags": ["language:en"]}
    state = _state("mybench", hf_json)
    card = _card()
    before = {k: dict(v) for k, v in card.items()}
    W._apply_hf_corroboration_gate(state, card, provenance=None)
    assert card == before
    assert "repo_corroboration" not in state


def test_gate_noop_without_tags_or_repo():
    # mirrors the historical behavior: no tag-derived overrides means no corroboration pass
    state = _state("mybench", {"id": "otherorg/unrelated"})
    card = _card()
    W._apply_hf_corroboration_gate(state, card, provenance=None)
    assert card == _card()
    assert "repo_corroboration" not in state


def test_factreasoner_tail_writes_no_content():
    # the validation worker must not backfill values or re-apply overrides after flagging
    src = inspect.getsource(W.run_factreasoner)
    assert "backfill_from_provenance" not in src
    assert "apply_deterministic_overrides" not in src
    assert "Cleared stale flag" not in src


def test_composer_invokes_corroboration_gate():
    src = inspect.getsource(W.run_composer)
    assert "_apply_hf_corroboration_gate" in src


def test_sub_benchmark_parent_copy_path_is_retired():
    # variants live on the parent card; the parent-copy specialization path is gone
    import auto_benchmarkcard.tools.composer.composer_tool as CT
    for name in ("compose_sub_benchmark_card", "_specialize_card_with_facts",
                 "extract_sub_benchmark_facts", "_EXTRACTOR_SYSTEM"):
        assert not hasattr(CT, name)


def test_always_override_conflict_is_surfaced():
    from auto_benchmarkcard.tools.composer.composer_tool import apply_deterministic_overrides
    card = {"ethical_and_legal_considerations": {"data_licensing": "CC-BY-NC-SA-4.0"}}
    conflicts = []
    apply_deterministic_overrides(
        card, {"ethical_and_legal_considerations.data_licensing": "Apache-2.0"},
        conflicts=conflicts)
    assert card["ethical_and_legal_considerations"]["data_licensing"] == "Apache-2.0"
    assert conflicts and conflicts[0]["field"].endswith("data_licensing")
    assert conflicts[0]["composed"] == "CC-BY-NC-SA-4.0"


def test_always_override_agreement_is_silent():
    from auto_benchmarkcard.tools.composer.composer_tool import apply_deterministic_overrides
    card = {"benchmark_details": {"languages": ["English"]}}
    conflicts = []
    apply_deterministic_overrides(
        card, {"benchmark_details.languages": ["english"]}, conflicts=conflicts)
    assert conflicts == []


def test_version_conflict_flags_minority_field():
    from auto_benchmarkcard.tools.composer.composer_tool import _flag_version_conflicts
    card = {
        "benchmark_details": {"overview": "Arena v2.0 evaluates models on hard prompts.",
                              "name": "arena"},
        "methodology": {"judge_models": ["GPT-4 as used in v0.1 of the pipeline"],
                        "calculation": "Scores follow the v2.0 protocol."},
        "data": {}, "purpose_and_intended_users": {},
        "ethical_and_legal_considerations": {},
    }
    flags = _flag_version_conflicts(card, None)
    assert len(flags) == 1 and flags[0]["field"] == "methodology.judge_models"
    assert "version mismatch" in card["flagged_fields"]["judge_models"]


def test_version_single_token_no_flags():
    from auto_benchmarkcard.tools.composer.composer_tool import _flag_version_conflicts
    card = {"benchmark_details": {"overview": "Uses v2.0 across the board."},
            "methodology": {}, "data": {}, "purpose_and_intended_users": {},
            "ethical_and_legal_considerations": {}}
    assert _flag_version_conflicts(card, None) == []
    assert "flagged_fields" not in card
