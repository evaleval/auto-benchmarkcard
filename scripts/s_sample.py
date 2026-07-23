"""Seeded stratified sampler for the S evaluation sample (paper eval, test set).

Draws n cards from a frozen corpus cards/ dir, stratified by validation flags
(>=1 entry in flagged_fields vs zero), records stratum weights w_h = N_h/n_h
over the resolvable frame, and resolves each card's cached run_dir (the source
bundle the faithfulness judge reads). Regen-vs-refresh provenance is recorded
as a covariate, not a stratum.

Run-dir resolution precedence per slug (first rule with an existing tool_output
wins; the fired rule is recorded per card):
  1. newest output/regen_run/output/<slug>_* or output/regen_run_heavy/output/<slug>_*
  2. output/refresh_runs/resolve_index.json[slug]
  3. newest output/refresh_runs/*/changelog/<slug>.json -> run_dir
  4. newest output/broad_run/output/<slug>_*
  5. none -> excluded from the sampling frame (hard error above --allow-exclusions)

Deterministic: same corpus + same seed -> byte-identical output.

Usage (dry run):
  python scripts/s_sample.py --corpus-dir output/broad_run/cards --seed 20260704 \
      --n 150 --tag dev --out eval/dryrun/sample_dev.json
At freeze only the input changes:
  python scripts/s_sample.py --corpus-dir <frozen v3 cards> --seed 20260704 \
      --n 150 --tag corpus_v3 --out eval/s150/sample.json
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import extract_card  # noqa: E402

from check_frozen import check, file_md5  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

REGEN_GLOBS = ["output/regen_run/output/{slug}_*", "output/regen_run_heavy/output/{slug}_*"]
RESOLVE_INDEX = "output/refresh_runs/resolve_index.json"
REFRESH_RUNS_GLOB = "output/refresh_runs/*"
BROAD_GLOB = "output/broad_run/output/{slug}_*"


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _rel(p):
    return os.path.relpath(os.path.abspath(p), os.path.abspath(REPO))


def _has_tool_output(run_dir_rel):
    return os.path.isdir(os.path.join(REPO, run_dir_rel, "tool_output"))


def _newest(pattern):
    hits = sorted(glob.glob(os.path.join(REPO, pattern)))
    return _rel(hits[-1]) if hits else None


def resolve_run_dir(slug, resolve_index, changelog_index):
    """Return (run_dir_rel, provenance, rule) or (None, None, None)."""
    regen_hits = []
    for g in REGEN_GLOBS:
        regen_hits += glob.glob(os.path.join(REPO, g.format(slug=slug)))
    if regen_hits:
        cand = _rel(sorted(regen_hits)[-1])
        if _has_tool_output(cand):
            return cand, "regen", "regen_glob"
    idx = resolve_index.get(slug)
    if idx:
        cand = _rel(idx)
        if _has_tool_output(cand):
            return cand, "refresh", "resolve_index"
    cl = changelog_index.get(slug)
    if cl:
        cand = _rel(cl)
        if _has_tool_output(cand):
            return cand, "refresh", "refresh_changelog"
    cand = _newest(BROAD_GLOB.format(slug=slug))
    if cand and _has_tool_output(cand):
        return cand, "refresh", "newest_glob"
    return None, None, None


def build_changelog_index():
    """slug -> run_dir from the newest refresh run that has a changelog for it."""
    index = {}
    for run in sorted(glob.glob(os.path.join(REPO, REFRESH_RUNS_GLOB))):
        cl_dir = os.path.join(run, "changelog")
        if not os.path.isdir(cl_dir):
            continue
        for p in glob.glob(os.path.join(cl_dir, "*.json")):
            entry = _load(p) or {}
            rd = entry.get("run_dir")
            if rd:
                index[os.path.splitext(os.path.basename(p))[0]] = rd
    return index


def corpus_fingerprint(card_files):
    h = hashlib.md5()
    for name, path in card_files:
        h.update(f"{name} {file_md5(path)}\n".encode())
    return h.hexdigest()


def gold_fixture(out_path, stage_root):
    """Sample file over the 22 gold cards (dry-run fixture): every card taken,
    strata by the frozen card's flagged_fields, all weights 1."""
    gold = _load(os.path.join(REPO, "gold_set", "manifest.json"))
    cards = []
    for e in gold["cards"]:
        p = os.path.join(REPO, e["card_path"])
        card = extract_card(_load(p) or {})
        stratum = "flagged" if (card.get("flagged_fields") or {}) else "unflagged"
        cards.append({"name": e["name"], "stratum": stratum, "weight": 1.0,
                      "provenance": "gold",
                      "corpus_card": e["card_path"], "corpus_card_md5": file_md5(p),
                      "n_flagged": len(card.get("flagged_fields") or {}),
                      "source_run_dir": os.path.dirname(os.path.dirname(e["card_path"])),
                      "run_dir_resolution": "gold_manifest",
                      "card_path": os.path.join(stage_root, "rundirs", e["name"],
                                                "benchmarkcard",
                                                f"benchmark_card_{e['name']}.json")})
    strata = {}
    for h in ("flagged", "unflagged"):
        n = sum(1 for c in cards if c["stratum"] == h)
        if n:
            strata[h] = {"N": n, "n": n, "weight": 1.0}
    out = {"version": "s150-v1", "tag": "gold_fixture", "created": date.today().isoformat(),
           "seed": None, "n": len(cards),
           "corpus": {"dir": "gold_set/manifest.json", "n_cards": len(cards),
                      "fingerprint_md5": None},
           "frame": {"n": len(cards), "excluded": []},
           "strata": strata, "cards": cards}
    with open(os.path.join(REPO, out_path), "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"gold fixture ({len(cards)} cards, weights 1) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Seeded stratified S sampler.")
    ap.add_argument("--gold-fixture", action="store_true",
                    help="emit the 22-gold-card dry-run fixture instead of sampling")
    ap.add_argument("--corpus-dir", default=None, help="frozen corpus cards dir (repo-relative)")
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--tag", default=None, help="dev | corpus_v3 (recorded; make_numbers keys off it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-root", default=None,
                    help="where s_stage.py will stage rundirs (default: dirname of --out)")
    ap.add_argument("--allow-exclusions", type=int, default=0)
    args = ap.parse_args()
    check()

    if args.gold_fixture:
        gold_fixture(args.out, args.stage_root or os.path.dirname(args.out))
        return
    if not args.corpus_dir or not args.tag:
        sys.exit("--corpus-dir and --tag required (or --gold-fixture)")

    corpus_dir = os.path.join(REPO, args.corpus_dir)
    card_paths = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))
    if not card_paths:
        sys.exit(f"no cards in {args.corpus_dir}")
    card_files = [(os.path.splitext(os.path.basename(p))[0], p) for p in card_paths]

    resolve_index = _load(os.path.join(REPO, RESOLVE_INDEX)) or {}
    changelog_index = build_changelog_index()

    frame, excluded = [], []
    for name, path in card_files:
        card = extract_card(_load(path) or {})
        stratum = "flagged" if (card.get("flagged_fields") or {}) else "unflagged"
        run_dir, provenance, rule = resolve_run_dir(name, resolve_index, changelog_index)
        if run_dir is None:
            excluded.append({"slug": name, "reason": "no_run_dir"})
            continue
        frame.append({"name": name, "stratum": stratum, "provenance": provenance,
                      "corpus_card": _rel(path), "corpus_card_md5": file_md5(path),
                      "n_flagged": len(card.get("flagged_fields") or {}),
                      "source_run_dir": run_dir, "run_dir_resolution": rule})
    if len(excluded) > args.allow_exclusions:
        for e in excluded:
            print(f"excluded: {e['slug']} ({e['reason']})", file=sys.stderr)
        sys.exit(f"{len(excluded)} frame exclusions > --allow-exclusions {args.allow_exclusions}")

    by_stratum = {"flagged": [c for c in frame if c["stratum"] == "flagged"],
                  "unflagged": [c for c in frame if c["stratum"] == "unflagged"]}
    targets = {"flagged": args.n // 2, "unflagged": args.n - args.n // 2}
    # if one stratum is smaller than its target, take it whole and shift the remainder
    for small, other in (("flagged", "unflagged"), ("unflagged", "flagged")):
        if len(by_stratum[small]) < targets[small]:
            shift = targets[small] - len(by_stratum[small])
            targets[small] = len(by_stratum[small])
            targets[other] += shift
    if targets["flagged"] + targets["unflagged"] > len(frame):
        sys.exit(f"frame too small: {len(frame)} < n={args.n}")

    rng = random.Random(args.seed)
    sampled = []
    strata_meta = {}
    for h in ("flagged", "unflagged"):
        pool = sorted(by_stratum[h], key=lambda c: c["name"])
        n_h = targets[h]
        picks = rng.sample(pool, n_h) if n_h < len(pool) else list(pool)
        picks.sort(key=lambda c: c["name"])
        w = round(len(pool) / n_h, 6) if n_h else None
        strata_meta[h] = {"N": len(pool), "n": n_h, "weight": w}
        for c in picks:
            c = dict(c)
            c["weight"] = w
            sampled.append(c)

    stage_root = args.stage_root or os.path.dirname(args.out)
    for c in sampled:
        c["card_path"] = os.path.join(stage_root, "rundirs", c["name"],
                                      "benchmarkcard", f"benchmark_card_{c['name']}.json")

    out = {
        "version": "s150-v1", "tag": args.tag, "created": date.today().isoformat(),
        "seed": args.seed, "n": len(sampled),
        "corpus": {"dir": args.corpus_dir, "n_cards": len(card_files),
                   "fingerprint_md5": corpus_fingerprint(card_files)},
        "frame": {"n": len(frame), "excluded": excluded},
        "strata": strata_meta,
        "cards": sampled,
    }
    os.makedirs(os.path.dirname(os.path.join(REPO, args.out)) or ".", exist_ok=True)
    with open(os.path.join(REPO, args.out), "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    prov = {"regen": 0, "refresh": 0}
    rules = {}
    for c in sampled:
        prov[c["provenance"]] += 1
        rules[c["run_dir_resolution"]] = rules.get(c["run_dir_resolution"], 0) + 1
    print(f"corpus {args.corpus_dir}: {len(card_files)} cards, frame {len(frame)}, "
          f"excluded {len(excluded)}")
    for h, m in strata_meta.items():
        print(f"  {h:9s} N={m['N']:>3d} n={m['n']:>3d} w={m['weight']}")
    print(f"  provenance {prov}  resolution {rules}")
    print(f"sample -> {args.out}")


if __name__ == "__main__":
    main()
