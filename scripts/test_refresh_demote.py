#!/usr/bin/env python3
"""Unit test for the guarded demote affordance in the refresh merge.

Covers: a demote spec lowers provably-fake content to "Not specified" (decision
"demoted"); the same spec keeps a genuine value (is_fake False); a non-demote spec
never adopts a blank new (the floor still blocks); and a normal grounded replace
still works. Run: python scripts/test_refresh_demote.py
"""

import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from refresh import changelog as cl
from refresh import merge as mm
from refresh.registry import FieldRegistry, FieldSpec


def _card(metric):
    return {
        "benchmark_card": {
            "methodology": {"metrics": metric, "judge_uses_llm": "Not specified"},
            "missing_fields": [],
        }
    }


CTX = {"slug": "x", "run_dir": "", "b_eligible": True, "eee_metadata": {}, "kwargs": {}}


def _demote_spec():
    # Lowers a synthesized other:<slug>.score metric to honest "Not specified".
    return FieldSpec(
        path="methodology.metrics",
        phase="A",
        recompute=lambda refreshed, kwargs, ctx: ["Not specified"],
        grounding=lambda new, refreshed, ctx: "deterministic-demote",
        decide=lambda old, new, ev, ctx: True,
        is_fake=lambda old, ctx: old == ["other:x.score"],
    )


def test_demote_lowers_fake():
    reg = FieldRegistry()
    reg.register(_demote_spec())
    live = _card(["other:x.score"])
    merged, log = mm.merge(live, None, reg, CTX)
    assert merged["benchmark_card"]["methodology"]["metrics"] == ["Not specified"], merged
    assert log[0]["decision"] == "demoted", log
    assert log[0]["reason"] == "is_fake", log
    print("PASS demote lowers fake -> Not specified, decision=demoted")


def test_demote_keeps_genuine():
    reg = FieldRegistry()
    reg.register(_demote_spec())
    live = _card(["Exact match accuracy"])  # genuine: is_fake returns False
    merged, log = mm.merge(live, None, reg, CTX)
    assert merged["benchmark_card"]["methodology"]["metrics"] == ["Exact match accuracy"], merged
    assert log[0]["decision"] == "kept" and log[0]["reason"] == "floor_blocked", log
    print("PASS demote keeps genuine value (is_fake False -> floor_blocked)")


def test_non_demote_spec_never_adopts_blank():
    # No is_fake: a blank/NS new can never replace, even when decide says True.
    reg = FieldRegistry()
    reg.register(FieldSpec(
        path="methodology.metrics",
        phase="B",
        recompute=lambda refreshed, kwargs, ctx: ["Not specified"],
        grounding=lambda new, refreshed, ctx: "basis",
        decide=lambda old, new, ev, ctx: True,
    ))
    live = _card(["other:x.score"])
    merged, log = mm.merge(live, None, reg, CTX)
    assert merged["benchmark_card"]["methodology"]["metrics"] == ["other:x.score"], merged
    assert log[0]["decision"] == "kept" and log[0]["reason"] == "floor_blocked", log
    print("PASS non-demote spec never demotes (blank new floor_blocked)")


def test_normal_grounded_replace_still_works():
    reg = FieldRegistry()
    reg.register(FieldSpec(
        path="methodology.metrics",
        phase="B",
        recompute=lambda refreshed, kwargs, ctx: ["F1 score"],
        grounding=lambda new, refreshed, ctx: "docling:span",
        decide=lambda old, new, ev, ctx: True,
    ))
    live = _card(["Not specified"])
    merged, log = mm.merge(live, None, reg, CTX)
    assert merged["benchmark_card"]["methodology"]["metrics"] == ["F1 score"], merged
    assert log[0]["decision"] == "replaced", log
    print("PASS normal grounded replace still works (decision=replaced)")


def test_rollup_counts_demote_as_change():
    reg = FieldRegistry()
    reg.register(_demote_spec())
    live = _card(["other:x.score"])
    merged, fields = mm.merge(live, None, reg, CTX)
    cc = cl.card_changelog("x", "", True, fields, [], merged["benchmark_card"]["missing_fields"])
    rollup = cl.build_rollup([cc])
    assert rollup["totals"]["changes"] == 1 and rollup["totals"]["cards_with_change"] == 1, rollup
    assert rollup["per_field"]["methodology.metrics"]["demoted"] == 1, rollup
    print("PASS rollup counts demote as a change (demoted=1)")


def main():
    test_demote_lowers_fake()
    test_demote_keeps_genuine()
    test_non_demote_spec_never_adopts_blank()
    test_normal_grounded_replace_still_works()
    test_rollup_counts_demote_as_change()
    print("\nALL DEMOTE TESTS PASSED")


if __name__ == "__main__":
    main()
