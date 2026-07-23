#!/usr/bin/env python3
"""Deterministic Phase A pass over the regenerated cards (v3 splice input).

Regen cards are fresh compositions and skip Phase B re-compose (locked decision
2026-07-02); they get the same deterministic Phase A treatment the 516 refresh
cards received in the quality round. This driver reuses the refresh machinery
(run_refresh, registry "a", improve-only merge) but feeds it the regen rundirs
and the regen cards as the live cards:

  - the slug -> run_dir index is built here from the two regen roots and passed
    to run_refresh directly; the shared resolve_index.json cache is never read
    or written,
  - REFRESH_PUB_DIR points at a staged copy of the regen cards and is set
    BEFORE any refresh import binds resolve.PUB_DIR.

Gates (any failure exits nonzero):
  1. null refresh (empty registry) byte-identical to the staged cards
  2. Phase A changes confined to the 4 A-paths + missing_fields
  3. shared resolve_index.json byte-unchanged
  4. no Phase A demote left a flagged_fields entry pointing at the demoted field

Usage:
  python scripts/regen_phase_a.py [--force]

Outputs: output/regen_phase_a/{pub, null_run, phase_a_run}
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(REPO, "output", "regen_phase_a")
STAGE_PUB = os.path.join(OUT_BASE, "pub")
REGEN_ROOTS = [
    os.path.join(REPO, "output", "regen_run", "output"),
    os.path.join(REPO, "output", "regen_run_heavy", "output"),
]
SHARED_INDEX = os.path.join(REPO, "output", "refresh_runs", "resolve_index.json")

EXPECTED_N = 54
EXPECTED_CARDLESS = {"supergpqa"}
IFEVAL_RUNDIR = "ifeval_2026-07-05_02-46"
A_PATHS = {
    "data.format",
    "methodology.metrics",
    "ethical_and_legal_considerations.data_licensing",
    "data.size",
}


def _serialize(card):
    return json.dumps(card, indent=2, ensure_ascii=True)


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _inner(card):
    return card.get("benchmark_card", card)


def discover():
    """{slug: run_dir} for every regen rundir with a card; asserts the splice brief."""
    index, cardless = {}, set()
    for root in REGEN_ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "*_2026-*"))):
            slug = os.path.basename(d).rsplit("_2026-", 1)[0]
            card = os.path.join(d, "benchmarkcard", f"benchmark_card_{slug}.json")
            if not os.path.exists(card):
                cardless.add(slug)
                continue
            assert slug not in index, f"duplicate rundir for {slug}: {index[slug]} and {d}"
            index[slug] = d
    assert len(index) == EXPECTED_N, f"expected {EXPECTED_N} regen cards, found {len(index)}"
    assert cardless == EXPECTED_CARDLESS, f"unexpected cardless rundirs: {sorted(cardless)}"
    assert os.path.basename(index["ifeval"]) == IFEVAL_RUNDIR, index["ifeval"]
    return index


def stage_pub(index):
    """Stage each regen card as pub/<slug>.json in the canonical serialization.

    The rundir cards are byte-stable under the canonical serialization today; the
    assert catches any non-canonical card so the null gate stays byte-consistent."""
    os.makedirs(STAGE_PUB, exist_ok=True)
    for slug, run_dir in sorted(index.items()):
        src = os.path.join(run_dir, "benchmarkcard", f"benchmark_card_{slug}.json")
        with open(src, "rb") as f:
            raw = f.read()
        staged = _serialize(json.loads(raw)).encode()
        assert staged == raw, f"{slug}: rundir card not in canonical serialization"
        with open(os.path.join(STAGE_PUB, f"{slug}.json"), "wb") as f:
            f.write(staged)


def _diff_paths(a, b, prefix=""):
    """Dotted paths at which two nested dicts differ (lists compare as leaves)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else k
            out.extend(_diff_paths(a.get(k), b.get(k), p))
        return out
    return [] if a == b else [prefix]


def gate_scope(index, phase_a_dir):
    """Phase A may only change the 4 registered paths plus missing_fields."""
    allowed = A_PATHS | {"missing_fields"}
    changed_total = {}
    for slug in sorted(index):
        with open(os.path.join(STAGE_PUB, f"{slug}.json")) as f:
            before = _inner(json.load(f))
        with open(os.path.join(phase_a_dir, "cards", f"{slug}.json")) as f:
            after = _inner(json.load(f))
        diff = _diff_paths(before, after)
        illegal = [p for p in diff if p not in allowed]
        assert not illegal, f"{slug}: Phase A touched non-A paths {illegal}"
        if diff:
            changed_total[slug] = diff
    return changed_total


def gate_flags(index, changelogs, phase_a_dir):
    """A metrics demote must not leave a flag pointing at the demoted field."""
    demoted = {
        c["slug"]: [f["field"] for f in c["fields"] if f["decision"] == "demoted"]
        for c in changelogs
    }
    hits = []
    for slug, paths in demoted.items():
        if not paths:
            continue
        with open(os.path.join(phase_a_dir, "cards", f"{slug}.json")) as f:
            flags = _inner(json.load(f)).get("flagged_fields") or {}
        for p in paths:
            for k in flags:
                if k == p or p.startswith(k) or k.startswith(p):
                    hits.append((slug, p, k))
    assert not hits, f"demoted fields still flagged (escalate before splice): {hits}"
    return sum(len(v) for v in demoted.values())


def main():
    ap = argparse.ArgumentParser(description="Phase A only pass over the regen cards.")
    ap.add_argument("--force", action="store_true", help="remove an existing output dir")
    args = ap.parse_args()

    if os.path.exists(OUT_BASE):
        if not args.force:
            sys.exit(f"{OUT_BASE} exists; rerun with --force to redo")
        shutil.rmtree(OUT_BASE)

    index_md5_before = _md5(SHARED_INDEX)
    index = discover()
    stage_pub(index)
    targets = sorted(index)
    print(f"regen index: {len(targets)} cards staged to {STAGE_PUB}")

    os.environ["REFRESH_PUB_DIR"] = STAGE_PUB
    sys.path.insert(0, os.path.join(REPO, "src"))
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from refresh import changelog as cl
    from refresh import metrics as metrics_mod
    from refresh import resolve
    from refresh.registry import build_registry
    import refresh_from_cache as rfc

    assert resolve.PUB_DIR == STAGE_PUB, f"PUB_DIR bound to {resolve.PUB_DIR}"

    # gate 1: null refresh must reproduce the staged cards byte for byte
    null_dir = os.path.join(OUT_BASE, "null_run")
    _, _, mismatches, _ = rfc.run_refresh(
        targets, index, build_registry(""), null_dir, False, True
    )
    if mismatches:
        sys.exit(f"NULL GATE FAILED ({len(mismatches)}): {mismatches}")
    print(f"null gate: {len(targets)}/{len(targets)} byte-identical")

    # the real pass: Phase A registry, compose skipped
    phase_a_dir = os.path.join(OUT_BASE, "phase_a_run")
    pairs, changelogs, _, _ = rfc.run_refresh(
        targets, index, build_registry("a"), phase_a_dir, False, False
    )
    rollup = cl.build_rollup(changelogs)
    cl.write_rollup(phase_a_dir, rollup)
    with open(os.path.join(phase_a_dir, "metrics.json"), "w") as f:
        json.dump(metrics_mod.compute_metrics(pairs), f, indent=2, ensure_ascii=True)

    changed = gate_scope(index, phase_a_dir)
    n_demoted = gate_flags(index, changelogs, phase_a_dir)
    assert _md5(SHARED_INDEX) == index_md5_before, "shared resolve_index.json changed"

    print(f"phase A: cards changed {len(changed)}/{len(targets)}, demotes {n_demoted}")
    for slug, diff in sorted(changed.items()):
        print(f"  {slug}: {diff}")
    print(f"rollup totals: {rollup['totals']}")
    print(f"cards -> {os.path.join(phase_a_dir, 'cards')}")


if __name__ == "__main__":
    main()
