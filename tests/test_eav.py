"""Tests for the Entity-Attribution Verification pass (tools/composer/eav.py).

No network: the LLM side is exercised with fake handler objects whose generate
method replays scripted responses (or raises). Digest items mirror the real
EvidenceItem v2 shape produced by evidence.finalize_evidence.
"""

import copy
import json

from auto_benchmarkcard.tools.composer import eav
from auto_benchmarkcard.tools.composer import evidence


class FakeHandler:
    """Replays one scripted response (str or Exception) per generate call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, response_format=None):
        self.prompts.append(prompt)
        resp = self.responses.pop(0) if self.responses else '{"verdicts": []}'
        if isinstance(resp, Exception):
            raise resp
        return resp


def make_frame():
    return {
        "target_id": "target",
        "nodes": [
            {"id": "target", "type": "target", "name": "FibbleBench",
             "aliases": ["Fibble Bench"], "description": "The target benchmark."},
            {"id": "sibling:wobble", "type": "sibling", "name": "WobbleBench",
             "aliases": [], "description": "Sibling named in the paper."},
            {"id": "subset:hard", "type": "subset", "name": "hard",
             "aliases": [], "description": "HF dataset config."},
        ],
    }


def make_item(eid, field, quote="FibbleBench contains 1,200 items.", referent="target",
              region="data", section_path="Data > Dataset", **overrides):
    item = {
        "evidence_id": eid, "field": field, "quote": quote, "char_start": 0,
        "doc": "docling", "kind": "stated", "note": "", "referent": referent,
        "claim_role": "primary_result", "section_path": section_path,
        "region": region, "table_id": "",
    }
    item.update(overrides)
    return item


def ok_verdicts(*eids):
    return json.dumps({"verdicts": [
        {"id": e, "about_ok": True, "field_ok": True, "referent": "target", "reason": "ok"}
        for e in eids
    ]})


def checked_ids(handler):
    ids = []
    for prompt in handler.prompts:
        for line in prompt.splitlines():
            if line.startswith("- id="):
                ids.append(line.split("- id=")[1].split(" ")[0])
    return ids


def test_selection_priority_and_cap():
    digest = {
        "purpose_and_intended_users.goal": [
            make_item("E01", "purpose_and_intended_users.goal", region="intro")],
        "purpose_and_intended_users.tasks": [
            make_item("E02", "purpose_and_intended_users.tasks",
                      referent="sibling:wobble", region="intro")],
        "purpose_and_intended_users.limitations": [
            make_item("E03", "purpose_and_intended_users.limitations", region="results")],
        "data.size": [make_item("E04", "data.size")],
        "methodology.metrics": [make_item("E05", "methodology.metrics")],
    }
    handler = FakeHandler(ok_verdicts("E04", "E05", "E02"))
    updated, telem = eav.run_eav(digest, make_frame(), handler, max_items=3)

    # high-stakes fields first (digest order), then the non-target referent;
    # the risky-region item is over the cap and the low-risk item never selected
    assert checked_ids(handler) == ["E04", "E05", "E02"]
    assert telem["checked"] == 3
    assert telem["skipped"] == 1
    assert telem["calls"] == 1
    assert updated == digest


def test_judge_paths_and_risky_regions_are_selected():
    digest = {
        "methodology.judge_setup": [make_item("E01", "methodology.judge_setup")],
        "purpose_and_intended_users.goal": [
            make_item("E02", "purpose_and_intended_users.goal", region="ablation"),
            make_item("E03", "purpose_and_intended_users.goal", region="appendix"),
            make_item("E04", "purpose_and_intended_users.goal", region="intro"),
        ],
    }
    handler = FakeHandler(ok_verdicts("E01", "E02", "E03"))
    _, telem = eav.run_eav(digest, make_frame(), handler)
    assert checked_ids(handler) == ["E01", "E02", "E03"]
    assert telem["checked"] == 3
    assert telem["skipped"] == 0


def test_field_mismatch_demotes():
    digest = {"data.size": [make_item("E01", "data.size", quote="We report accuracy."),
                            make_item("E02", "data.size")]}
    handler = FakeHandler(json.dumps({"verdicts": [
        {"id": "E01", "about_ok": True, "field_ok": False,
         "referent": "target", "reason": "metrics not size"},
        {"id": "E02", "about_ok": True, "field_ok": True, "referent": "target"},
    ]}))
    updated, telem = eav.run_eav(digest, make_frame(), handler)

    assert [it["evidence_id"] for it in updated["data.size"]] == ["E02"]
    assert telem["demoted"] == [{"id": "E01", "field": "data.size", "referent": "target",
                                 "reason": "field_mismatch: metrics not size"}]
    # input digest untouched, demoted item still there
    assert [it["evidence_id"] for it in digest["data.size"]] == ["E01", "E02"]


def test_referent_correction():
    digest = {"data.size": [make_item("E01", "data.size",
                                      quote="WobbleBench contains 900 items.")]}
    handler = FakeHandler(json.dumps({"verdicts": [
        {"id": "E01", "about_ok": False, "field_ok": True,
         "referent": "sibling:wobble", "reason": "names the sibling"},
    ]}))
    updated, telem = eav.run_eav(digest, make_frame(), handler)

    assert updated["data.size"][0]["referent"] == "sibling:wobble"
    assert telem["corrected"] == 1
    assert telem["demoted"] == []
    # the quote is never edited and the input item is a different, unmutated object
    assert updated["data.size"][0]["quote"] == "WobbleBench contains 900 items."
    assert digest["data.size"][0]["referent"] == "target"
    assert updated["data.size"][0] is not digest["data.size"][0]


def test_unresolved_referent_demotes():
    digest = {
        "data.size": [make_item("E01", "data.size")],
        "methodology.metrics": [make_item("E02", "methodology.metrics")],
    }
    handler = FakeHandler(json.dumps({"verdicts": [
        # same referent back = no valid correction
        {"id": "E01", "about_ok": False, "field_ok": True, "referent": "target"},
        # unknown node id = no valid correction
        {"id": "E02", "about_ok": False, "field_ok": True, "referent": "sibling:ghost"},
    ]}))
    updated, telem = eav.run_eav(digest, make_frame(), handler)

    assert updated == {}  # emptied fields are dropped
    assert [d["id"] for d in telem["demoted"]] == ["E01", "E02"]
    assert all(d["reason"].startswith("referent_unresolved") for d in telem["demoted"])


def test_malformed_response_leaves_items_untouched():
    digest = {"data.size": [make_item("E01", "data.size")]}
    handler = FakeHandler("I could not decide, sorry!")
    updated, telem = eav.run_eav(digest, make_frame(), handler)
    assert updated == digest
    assert telem["batch_errors"] == 0
    assert telem["demoted"] == [] and telem["corrected"] == 0


def test_malformed_entries_and_unknown_ids_are_ignored():
    digest = {"data.size": [make_item("E01", "data.size"),
                            make_item("E02", "data.size")]}
    handler = FakeHandler(json.dumps({"verdicts": [
        "not an object",
        {"id": "E99", "about_ok": False, "field_ok": False, "reason": "unknown id"},
        {"id": "E01", "about_ok": True, "field_ok": "maybe"},  # non-bool: fail-open
        {"id": "E02", "about_ok": True, "field_ok": False, "reason": "off topic"},
    ]}))
    updated, telem = eav.run_eav(digest, make_frame(), handler)
    assert [it["evidence_id"] for it in updated["data.size"]] == ["E01"]
    assert [d["id"] for d in telem["demoted"]] == ["E02"]
    # unknown ids never add items
    assert sum(len(v) for v in updated.values()) < sum(len(v) for v in digest.values())


def test_fenced_and_bare_list_responses_parse():
    fenced = "```json\n" + json.dumps({"verdicts": [
        {"id": "E01", "about_ok": True, "field_ok": False, "reason": "fenced"}]}) + "\n```"
    digest = {"data.size": [make_item("E01", "data.size")]}
    updated, telem = eav.run_eav(digest, make_frame(), FakeHandler(fenced))
    assert updated == {} and telem["demoted"][0]["id"] == "E01"

    bare = json.dumps([
        {"id": "E01", "about_ok": True, "field_ok": False, "reason": "bare list"}])
    digest = {"data.size": [make_item("E01", "data.size")]}
    updated, telem = eav.run_eav(digest, make_frame(), FakeHandler(bare))
    assert updated == {} and telem["demoted"][0]["id"] == "E01"


def test_batch_error_fail_soft():
    digest = {"data.size": [make_item("E01", "data.size"),
                            make_item("E02", "data.size"),
                            make_item("E03", "data.size")]}
    handler = FakeHandler(
        json.dumps({"verdicts": [
            {"id": "E01", "about_ok": True, "field_ok": False, "reason": "bad"},
            {"id": "E02", "about_ok": True, "field_ok": True, "referent": "target"},
        ]}),
        RuntimeError("transport down"),
    )
    updated, telem = eav.run_eav(digest, make_frame(), handler, batch_size=2)

    assert telem["calls"] == 2
    assert telem["batch_errors"] == 1
    # first batch applied, errored batch untouched
    assert [it["evidence_id"] for it in updated["data.size"]] == ["E02", "E03"]
    assert [d["id"] for d in telem["demoted"]] == ["E01"]


def test_input_digest_never_mutated():
    digest = {
        "data.size": [make_item("E01", "data.size")],
        "methodology.metrics": [make_item("E02", "methodology.metrics")],
        "purpose_and_intended_users.goal": [
            make_item("E03", "purpose_and_intended_users.goal", region="intro")],
    }
    snapshot = copy.deepcopy(digest)
    handler = FakeHandler(json.dumps({"verdicts": [
        {"id": "E01", "about_ok": True, "field_ok": False, "reason": "demote me"},
        {"id": "E02", "about_ok": False, "field_ok": True, "referent": "subset:hard"},
    ]}))
    updated, _ = eav.run_eav(digest, make_frame(), handler)

    assert digest == snapshot
    # untouched items are passed through as the same objects
    assert updated["purpose_and_intended_users.goal"][0] \
        is digest["purpose_and_intended_users.goal"][0]
    assert updated["methodology.metrics"][0]["referent"] == "subset:hard"


def test_prompt_contains_frame_block_and_item_lines():
    frame = make_frame()
    digest = {"data.size": [make_item("E01", "data.size")]}
    handler = FakeHandler(ok_verdicts("E01"))
    eav.run_eav(digest, frame, handler)

    prompt = handler.prompts[0]
    assert evidence.render_frame_block(frame) in prompt
    assert "- sibling:wobble: WobbleBench (sibling)" in prompt
    assert ('- id=E01 | field=data.size | referent=target | section=Dataset '
            '| quote="FibbleBench contains 1,200 items."') in prompt
    assert '"verdicts"' in prompt


def test_noop_without_frame_or_handler_or_items():
    digest = {"data.size": [make_item("E01", "data.size")]}
    for frame, handler in [(None, FakeHandler()),
                           ({"target_id": "target", "nodes": []}, FakeHandler()),
                           (make_frame(), None)]:
        updated, telem = eav.run_eav(digest, frame, handler)
        assert updated == digest
        assert telem["calls"] == 0 and telem["checked"] == 0

    # nothing selected: low-risk items make no LLM call
    handler = FakeHandler()
    digest = {"purpose_and_intended_users.goal": [
        make_item("E01", "purpose_and_intended_users.goal", region="intro")]}
    updated, telem = eav.run_eav(digest, make_frame(), handler)
    assert updated == digest and telem["calls"] == 0 and handler.prompts == []
