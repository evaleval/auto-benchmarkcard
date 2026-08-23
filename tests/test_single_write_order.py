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
