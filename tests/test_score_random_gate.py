"""Regression tests for the paired random-sample scorer (n=40 generalization).

Failure classes pinned here:
  1. Partial-screen undercount: a screen dir missing drawn cards (or carrying extras)
     must fail loudly via check_draw, never score a smaller self-selected n.
  2. Class-movement over-read: per-category deltas must come with a paired bootstrap
     interval; class_pairs must count per card, including cards where a class is absent.
"""

import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from score_random_gate import (  # noqa: E402
    bootstrap_ci,
    by_category,
    check_draw,
    class_pairs,
    needs_fix,
    sign_test,
)


def _row(frozen_cats, new_cats):
    return {"frozen_cats": frozen_cats, "new_cats": new_cats}


def test_check_draw_passes_on_exact_match():
    draw = {"selection_order": ["b", "a", "c"]}
    assert check_draw(["a", "b", "c"], draw) == ["b", "a", "c"]


def test_check_draw_fails_on_missing_card():
    draw = {"selection_order": ["a", "b", "c"]}
    with pytest.raises(SystemExit, match="missing=\\['c'\\]"):
        check_draw(["a", "b"], draw)


def test_check_draw_fails_on_extra_card():
    draw = {"selection_order": ["a", "b"]}
    with pytest.raises(SystemExit, match="extra=\\['z'\\]"):
        check_draw(["a", "b", "z"], draw)


def test_check_draw_fails_on_empty_draw():
    with pytest.raises(SystemExit, match="no selection_order"):
        check_draw(["a"], {})


def test_class_pairs_counts_absent_class_as_zero():
    rows = [
        _row({"splice": 2}, {"splice": 1}),
        _row({}, {"splice": 1}),
        _row({"thin": 1}, {}),
    ]
    assert class_pairs(rows, "splice") == [(2, 1), (0, 1), (0, 0)]
    assert class_pairs(rows, "thin") == [(0, 0), (0, 0), (1, 0)]


def test_bootstrap_ci_degenerate_all_equal_pairs():
    pairs = [(1, 1)] * 10
    lo, hi = bootstrap_ci(pairs)
    assert lo == 0.0 and hi == 0.0


def test_bootstrap_ci_uniform_negative_delta():
    pairs = [(2, 1)] * 10  # every card improves by exactly 1
    lo, hi = bootstrap_ci(pairs)
    assert lo == -1.0 and hi == -1.0


def test_bootstrap_ci_is_seeded_deterministic():
    pairs = [(3, 1), (2, 2), (0, 1), (4, 0), (1, 1)]
    assert bootstrap_ci(pairs) == bootstrap_ci(pairs)


def test_needs_fix_and_by_category():
    verdict = {"findings": [
        {"severity": "needs-fix", "category": "splice"},
        {"severity": "needs-fix", "category": "splice"},
        {"severity": "ok", "category": "splice"},
        {"severity": "needs-fix", "category": "thin"},
    ]}
    nf = needs_fix(verdict)
    assert len(nf) == 3
    cats = by_category(nf)
    assert cats["splice"] == 2 and cats["thin"] == 1


def test_sign_test():
    assert sign_test([(2, 1), (1, 2), (1, 1), (3, 0)]) == (2, 1, 1)
