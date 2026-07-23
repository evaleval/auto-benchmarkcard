#!/usr/bin/env python3
"""Assemble the frozen v3 corpus: 516 refresh cards + 54 Phase A'd regen cards.

Builds output/corpus_v3/cards (530 cards), derives the replace/insert/stay-out
lists, runs the freeze pre-checks, and writes a machine-readable splice report:

  - zero-curated: every corpus slug's newest paper-verification sidecar (run-dir
    roots ordered broad -> regen -> regen_heavy, last root wins per slug, the
    corpus_stats semantics) has resolved_from != known_papers; additionally no
    KEPT card's PRODUCING rundir (resolve_index.json) is curated
  - shape parity: every card is {"benchmark_card": {5 sections, possible_risks,
    flagged_fields, missing_fields, card_info}} in canonical serialization
  - parquet preflight: per-section key sets and leaf types of the normalized
    cards vs the frozen v2 Features; deviations are pinned, never fixed
  - attempted universe: unique rundir slugs across the 3 roots (replaces the
    deleted broad backlog draft for corpus_stats --backlogs)
  - corpus fingerprint per the s_sample convention

Usage:
  python scripts/splice_v3.py [--force] [--ref-v2 <v2 staging dir>]

Outputs: output/corpus_v3/{cards, splice_report.json, attempted_universe.txt}
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

ASSEMBLED = os.path.join(REPO, "output", "assembled_v3_fields", "cards")
PHASE_A_CARDS = os.path.join(REPO, "output", "regen_phase_a", "phase_a_run", "cards")
OUT = os.path.join(REPO, "output", "corpus_v3")
CARDS_OUT = os.path.join(OUT, "cards")
SHARED_INDEX = os.path.join(REPO, "output", "refresh_runs", "resolve_index.json")

# order matters: newest_run_dirs lets the LAST root win per slug, so the regen
# roots must come after broad_run for the regen sidecars to shadow the curated ones
RUN_DIR_ROOTS = [
    "output/broad_run/output",
    "output/regen_run/output",
    "output/regen_run_heavy/output",
]

EXPECTED_REPLACE = {
    "acadreason", "anthropic-rlhf-dataset", "arc-agi", "bbq", "best-chatgpt-prompts",
    "boolq", "cnn-dailymail", "covost2", "disinformation", "drop", "echo", "gpqa",
    "hard-problems", "ifeval", "imdb", "koala-test-dataset", "lbpp-v2",
    "livecodebench", "math", "mm-mind2web", "ms-marco-trec", "mt-bench",
    "natural-questions", "omni-math", "openbookqa", "quac", "raft", "screenspot",
    "seal-0", "self-instruct", "studiogan", "terminal-bench", "theory-of-mind",
    "truthfulqa", "vip-bench", "vlue", "winogrande", "wmt-2014", "xsum", "zclawbench",
}
EXPECTED_INSERT = {
    "amc-2022-23", "appworld-test-normal", "big-bench", "cybench", "figqa",
    "helm-capabilities", "helm-lite", "helm-mmlu", "imagemining", "legalbench",
    "mmmu", "mmt-bench", "open-assistant", "swe-bench",
}
EXPECTED_STAY_OUT = {"supergpqa"}
IFEVAL_PAPER = "2311.07911"

INNER_KEYS = [
    "benchmark_details", "purpose_and_intended_users", "data", "methodology",
    "ethical_and_legal_considerations", "possible_risks", "flagged_fields",
    "missing_fields", "card_info",
]

# pinned parquet projection deviations (orchestrator ruling R1: accepted, declared
# in the staged README; anything beyond these fails the build). Kept cards may
# only repeat None-fills the v2 parquet already had (measured 26, all appears_in).
EXPECTED_EXTRA_KEYS = {
    ("helm-capabilities", "benchmark_details", "contains"),
    ("helm-lite", "benchmark_details", "contains"),
}
EXPECTED_REGEN_MISSING_KEYS = {
    ("appworld-test-normal", "benchmark_details", "appears_in"),
    ("helm-capabilities", "benchmark_details", "appears_in"),
    ("helm-lite", "benchmark_details", "appears_in"),
    ("helm-mmlu", "benchmark_details", "appears_in"),
    ("theory-of-mind", "benchmark_details", "appears_in"),
}


def _serialize(card):
    return json.dumps(card, indent=2, ensure_ascii=True)


def _load(p):
    with open(p) as f:
        return json.load(f)


def _file_md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def splice():
    """Copy the 516 verbatim, then overwrite/insert the 54 Phase A outputs."""
    assembled = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ASSEMBLED, "*.json"))}
    regen = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PHASE_A_CARDS, "*.json"))}
    assert len(assembled) == 516, len(assembled)
    assert len(regen) == 54, len(regen)

    replace = regen & assembled
    insert = regen - assembled
    assert replace == EXPECTED_REPLACE, sorted(replace ^ EXPECTED_REPLACE)
    assert insert == EXPECTED_INSERT, sorted(insert ^ EXPECTED_INSERT)

    os.makedirs(CARDS_OUT, exist_ok=True)
    for slug in sorted(assembled):
        shutil.copyfile(os.path.join(ASSEMBLED, f"{slug}.json"),
                        os.path.join(CARDS_OUT, f"{slug}.json"))
    for slug in sorted(regen):
        shutil.copyfile(os.path.join(PHASE_A_CARDS, f"{slug}.json"),
                        os.path.join(CARDS_OUT, f"{slug}.json"))
    corpus = sorted(assembled | regen)
    assert len(corpus) == 530, len(corpus)
    return corpus, sorted(replace), sorted(insert), sorted(regen)


def check_zero_curated(corpus, regen):
    """Newest-sidecar scan (corpus_stats semantics) + kept-producer scan."""
    import corpus_stats as cs

    dirs = cs.newest_run_dirs(RUN_DIR_ROOTS)
    raw, grouped = {}, {}
    curated = []
    for slug in sorted(dirs):
        sidecar = os.path.join(dirs[slug], "tool_output", "paper_resolver",
                               "paper-verification.json")
        if not os.path.exists(sidecar):
            raw["<no sidecar>"] = raw.get("<no sidecar>", 0) + 1
            continue
        rf = _load(sidecar).get("resolved_from")
        raw[str(rf)] = raw.get(str(rf), 0) + 1
        g = cs.group_resolved_from(rf)
        grouped[g] = grouped.get(g, 0) + 1
        if rf == "known_papers":
            curated.append(slug)

    missing = [s for s in corpus if s not in dirs]
    assert not missing, f"corpus slugs without a rundir: {missing}"
    no_sidecar = [s for s in corpus if not os.path.exists(os.path.join(
        dirs[s], "tool_output", "paper_resolver", "paper-verification.json"))]
    assert not no_sidecar, f"corpus slugs without a sidecar: {no_sidecar}"
    assert not curated, f"curated bindings present: {curated}"
    misrouted = [s for s in regen if "regen_run" not in dirs[s]]
    assert not misrouted, f"regen slugs resolving to non-regen rundirs: {misrouted}"

    # stricter angle: the PRODUCING rundir of every kept card is not curated either
    index = _load(SHARED_INDEX)
    kept = [s for s in corpus if s not in set(regen)]
    kept_curated = []
    for slug in kept:
        sc = os.path.join(index[slug], "tool_output", "paper_resolver",
                          "paper-verification.json")
        if os.path.exists(sc) and _load(sc).get("resolved_from") == "known_papers":
            kept_curated.append(slug)
    assert not kept_curated, f"kept cards with curated producers: {kept_curated}"

    return {"n_run_dirs": len(dirs), "resolved_from_raw": dict(sorted(raw.items())),
            "grouped": dict(sorted(grouped.items())),
            "kept_producers_checked": len(kept), "kept_curated": 0}


def check_shape(corpus):
    """Structure + canonical serialization on every spliced card."""
    for slug in corpus:
        p = os.path.join(CARDS_OUT, f"{slug}.json")
        with open(p, "rb") as f:
            raw = f.read()
        card = json.loads(raw)
        assert list(card.keys()) == ["benchmark_card"], f"{slug}: top-level {list(card.keys())}"
        inner_keys = list(card["benchmark_card"].keys())
        assert inner_keys == INNER_KEYS, f"{slug}: inner keys {inner_keys}"
        assert _serialize(card).encode() == raw, f"{slug}: non-canonical bytes"


def _type_ok(feat, val):
    import datasets
    if val is None:
        return True
    if isinstance(feat, dict):
        return isinstance(val, dict)
    if isinstance(feat, (datasets.Sequence, list)):
        return isinstance(val, list)
    if isinstance(feat, datasets.Value):
        if feat.dtype == "string":
            return isinstance(val, str)
        if feat.dtype.startswith("int"):
            return isinstance(val, int) and not isinstance(val, bool)
        if feat.dtype == "bool":
            return isinstance(val, bool)
    return True


def check_parquet_preflight(corpus, regen, ref_v2):
    """Key sets and leaf types of the normalized cards vs the frozen v2 Features."""
    import datasets
    from build_hf_staging import normalize_for_parquet

    v2feat = datasets.Dataset.from_parquet(
        os.path.join(ref_v2, "data", "train-00000-of-00001.parquet")).features
    feat_sections = v2feat["benchmark_card"]

    def scan(cards_dir, slugs):
        missing, extra, type_errors = set(), set(), []
        for slug in slugs:
            c = normalize_for_parquet(_load(os.path.join(cards_dir, f"{slug}.json")))
            for sec, feat in feat_sections.items():
                val = c.get(sec)
                if not isinstance(feat, dict):
                    if not _type_ok(feat, val):
                        type_errors.append((slug, sec, type(val).__name__))
                    continue
                sec_val = val if isinstance(val, dict) else {}
                for k in set(feat) - set(sec_val):
                    missing.add((slug, sec, k))
                for k in set(sec_val) - set(feat):
                    extra.add((slug, sec, k))
                for k in set(feat) & set(sec_val):
                    if not _type_ok(feat[k], sec_val[k]):
                        type_errors.append((slug, f"{sec}.{k}", type(sec_val[k]).__name__))
            for sec in set(c) - set(feat_sections):
                extra.add((slug, sec, "<section>"))
        return missing, extra, type_errors

    missing, extra, type_errors = scan(CARDS_OUT, corpus)
    v2_slugs = [os.path.basename(p)[:-5]
                for p in glob.glob(os.path.join(ref_v2, "cards", "*.json"))]
    v2_missing, v2_extra, _ = scan(os.path.join(ref_v2, "cards"), v2_slugs)
    assert not v2_extra, sorted(v2_extra)

    regen_missing = {t for t in missing if t[0] in set(regen)}
    kept_missing = missing - regen_missing
    new_kept = kept_missing - v2_missing
    assert not new_kept, f"kept cards with None-fills the v2 parquet did not have: {sorted(new_kept)}"
    assert regen_missing == EXPECTED_REGEN_MISSING_KEYS, sorted(
        regen_missing ^ EXPECTED_REGEN_MISSING_KEYS)
    assert extra == EXPECTED_EXTRA_KEYS, sorted(extra ^ EXPECTED_EXTRA_KEYS)
    assert not type_errors, type_errors[:10]
    return {"regen_missing_keys_none_filled": sorted(map(list, regen_missing)),
            "kept_missing_keys_inherited_from_v2": len(kept_missing),
            "v2_parquet_none_fills_baseline": len(v2_missing),
            "extra_keys_dropped_in_parquet": sorted(map(list, extra)),
            "type_errors": []}


def write_attempted_universe():
    slugs = set()
    for root in RUN_DIR_ROOTS:
        for d in glob.glob(os.path.join(REPO, root, "*_2026-*")):
            slugs.add(os.path.basename(d).rsplit("_2026-", 1)[0])
    assert len(slugs) == 531, len(slugs)
    path = os.path.join(OUT, "attempted_universe.txt")
    with open(path, "w") as f:
        f.write("\n".join(sorted(slugs)) + "\n")
    return path, len(slugs), _file_md5(path)


def evidence():
    """Quiescence evidence: ifeval binding, supergpqa failure, gates, staleness."""
    ifeval_dir = glob.glob(os.path.join(REPO, "output/regen_run/output", "ifeval_*"))
    assert [os.path.basename(d) for d in ifeval_dir] == ["ifeval_2026-07-05_02-46"], ifeval_dir
    sidecar = _load(os.path.join(ifeval_dir[0], "tool_output", "paper_resolver",
                                 "paper-verification.json"))
    assert IFEVAL_PAPER in json.dumps(sidecar), "ifeval sidecar does not bind 2311.07911"

    sg = os.path.join(REPO, "output/regen_run_heavy/output", "supergpqa_2026-07-05_00-50")
    sg_cards = os.listdir(os.path.join(sg, "benchmarkcard"))
    assert sg_cards == [], sg_cards
    sg_meta = os.path.exists(os.path.join(
        sg, "tool_output", "composer", "composition_metadata_supergpqa.json"))

    gates = {}
    for name, root in [("regen_run", "output/regen_run/output"),
                       ("regen_run_heavy", "output/regen_run_heavy/output")]:
        g = _load(os.path.join(REPO, root, "gate_metrics.json"))
        v = g.get("verdict", {})
        gates[name] = {"result": v.get("result"),
                       "schema_invalid": v.get("checks", {}).get("schema_invalid", {}).get("count"),
                       "hallucination_rate": v.get("checks", {}).get("hallucination_rate")}
        assert v.get("result") == "GO", (name, v.get("result"))

    fb_path = os.path.join(REPO, "output/regen_run/failed_benchmarks.json")
    stale = [e.get("benchmark") for e in (_load(fb_path) if os.path.exists(fb_path) else [])]

    return {
        "ifeval": {"rundir": "ifeval_2026-07-05_02-46",
                   "resolved_from": sidecar.get("resolved_from"),
                   "binds": IFEVAL_PAPER},
        "supergpqa": {"rundir": "supergpqa_2026-07-05_00-50",
                      "benchmarkcard_empty": True,
                      "composition_metadata_persisted": sg_meta,
                      "note": "composition succeeded, card write failed; stays out; "
                              "post-deadline offline-reassembly candidate"},
        "gate_metrics": gates,
        "failed_benchmarks_staleness": {
            "file": "output/regen_run/failed_benchmarks.json",
            "lists": stale,
            "note": "stale: RAFT succeeded on retry (card in corpus); supergpqa, the "
                    "real failure, is unlisted; regen_run_heavy wrote no such file. "
                    "Report-only per ruling R3; this report is the record."},
    }


def main():
    ap = argparse.ArgumentParser(description="Assemble + pre-check the frozen v3 corpus.")
    ap.add_argument("--force", action="store_true", help="remove an existing output dir")
    published_root = os.environ.get("AUTO_BENCHMARKCARD_MAIN_ROOT", REPO)
    ap.add_argument(
        "--ref-v2",
        default=os.path.join(published_root, "output", "auto-benchmarkcards-v2"),
        help="reference v2 staging directory",
    )
    args = ap.parse_args()

    if os.path.exists(OUT):
        if not args.force:
            sys.exit(f"{OUT} exists; rerun with --force to redo")
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    corpus, replace, insert, regen = splice()
    print(f"spliced: {len(corpus)} cards (replace {len(replace)}, insert {len(insert)}, "
          f"stay-out {sorted(EXPECTED_STAY_OUT)})")

    binding = check_zero_curated(corpus, regen)
    print(f"zero-curated: OK over {binding['n_run_dirs']} rundir slugs; "
          f"grouped {binding['grouped']}")

    check_shape(corpus)
    print("shape parity: 530/530 canonical")

    parquet = check_parquet_preflight(corpus, regen, args.ref_v2)
    print(f"parquet preflight: {len(parquet['regen_missing_keys_none_filled'])} regen none-fills, "
          f"{parquet['kept_missing_keys_inherited_from_v2']} kept none-fills (all in v2 baseline "
          f"{parquet['v2_parquet_none_fills_baseline']}), "
          f"{len(parquet['extra_keys_dropped_in_parquet'])} drops (all pinned)")

    au_path, au_n, au_md5 = write_attempted_universe()
    print(f"attempted universe: {au_n} slugs -> {au_path} (md5 {au_md5})")

    from s_sample import corpus_fingerprint
    pairs = sorted((os.path.basename(p)[:-5], p)
                   for p in glob.glob(os.path.join(CARDS_OUT, "*.json")))
    fp = corpus_fingerprint(pairs)
    print(f"corpus fingerprint: {fp}")

    rollup = _load(os.path.join(REPO, "output/regen_phase_a/phase_a_run/rollup.json"))
    report = {
        "generated_on": date.today().isoformat(),
        "corpus": {"dir": "output/corpus_v3/cards", "n_cards": len(corpus),
                   "fingerprint_md5": fp},
        "splice": {"kept": len(corpus) - len(regen), "replace": replace,
                   "insert": insert, "stay_out": sorted(EXPECTED_STAY_OUT)},
        "completion": {"end_state": f"{len(corpus)}/531 including regen",
                       "single_run_dev_history": "522/529 (broad_run)"},
        "phase_a": {"run": "output/regen_phase_a/phase_a_run",
                    "rollup_totals": rollup["totals"],
                    "per_field": rollup["per_field"]},
        "binding_paths": binding,
        "parquet_preflight": parquet,
        "attempted_universe": {"file": "output/corpus_v3/attempted_universe.txt",
                               "n": au_n, "md5": au_md5},
        "evidence": evidence(),
        "provenance_splice_script": "not landed; deferred post-freeze per handoff 2026-07-04",
    }
    with open(os.path.join(OUT, "splice_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"report -> {os.path.join(OUT, 'splice_report.json')}")


if __name__ == "__main__":
    main()
