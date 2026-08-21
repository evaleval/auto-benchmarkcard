"""Check regenerated cards against the S=150 confirmed-material findings.

For each record in eval/regression/findings_regression.json, looks up the
regenerated card in --cards-dir and classifies the finding's field:

  UNCHANGED      value equals the recorded eval-time value; defect likely persists
  CHANGED        value differs; needs re-screen
  NOW_NS         value became the canonical Not specified
  FIELD_MISSING  the field path no longer resolves in the card
  CARD_MISSING   no card JSON found for the card name
  VALUE_UNKNOWN  no eval-time value was recoverable; the field's new state is
                 still reported when the locator resolves

--cards-dir supports both a flat layout (<dir>/<name>.json) and run-dir
layouts (<dir>/<name>/benchmarkcard/*.json, or timestamped
<dir>/<name>_*/benchmarkcard/*.json, latest picked).

Usage:
  python scripts/check_regressions.py --cards-dir output/auto-benchmarkcards-v4/cards \
      [--regressions eval/regression/findings_regression.json] \
      [--out eval/regression/check_report.json]
"""

import argparse
import collections
import glob
import importlib.util
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# load card_utils directly; the package __init__ drags in workflow deps we don't need
_spec = importlib.util.spec_from_file_location(
    "card_utils_standalone", os.path.join(REPO_ROOT, "src/auto_benchmarkcard/card_utils.py")
)
_card_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_card_utils)
extract_card = _card_utils.extract_card
is_not_specified = _card_utils.is_not_specified

CLASSIFICATIONS = ["UNCHANGED", "CHANGED", "NOW_NS", "FIELD_MISSING", "CARD_MISSING", "VALUE_UNKNOWN"]

_brs = importlib.util.spec_from_file_location(
    "build_regression_set", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_regression_set.py")
)
_build = importlib.util.module_from_spec(_brs)
_brs.loader.exec_module(_build)
parse_path = _build.parse_path
resolve_path = _build.resolve_path
resolve_locator = _build.resolve_locator


def find_card_file(cards_dir, name):
    """Locate a regenerated card JSON for a benchmark name under cards_dir."""
    flat = os.path.join(cards_dir, f"{name}.json")
    if os.path.isfile(flat):
        return flat
    for run_dir in [os.path.join(cards_dir, name)] + sorted(glob.glob(os.path.join(cards_dir, f"{name}_*"))):
        bc_dir = os.path.join(run_dir, "benchmarkcard")
        if not os.path.isdir(bc_dir):
            continue
        preferred = os.path.join(bc_dir, f"benchmark_card_{name}.json")
        if os.path.isfile(preferred):
            return preferred
        others = sorted(glob.glob(os.path.join(bc_dir, "*.json")))
        if others:
            return others[-1]
    return None


def classify(rec, card):
    """Classify one regression record against a regenerated card dict (or None)."""
    if card is None:
        return "CARD_MISSING", None, False

    if not rec["field_path_resolved"]:
        # no eval-time value; still report what the locator points at now
        path, value = resolve_locator(card, rec["field_locator"])
        return "VALUE_UNKNOWN", value if path is not None else None, path is not None

    parts = parse_path(rec["field_path"])
    found, value = resolve_path(card, parts)
    if not found:
        return "FIELD_MISSING", None, False
    if value == rec["current_value"]:
        return "UNCHANGED", value, True
    if is_not_specified(value):
        return "NOW_NS", value, True
    return "CHANGED", value, True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cards-dir", required=True)
    ap.add_argument("--regressions", default=os.path.join(REPO_ROOT, "eval/regression/findings_regression.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "eval/regression/check_report.json"))
    args = ap.parse_args()

    regressions = json.load(open(args.regressions))
    results = []
    card_cache = {}
    for rec in regressions["findings"]:
        name = rec["card"]
        if name not in card_cache:
            path = find_card_file(args.cards_dir, name)
            card_cache[name] = (path, extract_card(json.load(open(path))) if path else None)
        card_path, card = card_cache[name]
        cls, new_value, new_resolved = classify(rec, card)
        results.append(
            {
                "card": name,
                "finding_index": rec["finding_index"],
                "field_path": rec["field_path"],
                "category": rec["category"],
                "classification": cls,
                "card_file": card_path,
                "new_value": new_value if new_resolved else None,
                "new_value_resolved": new_resolved,
            }
        )

    by_cls = collections.Counter(r["classification"] for r in results)
    by_cat = collections.defaultdict(collections.Counter)
    for r in results:
        by_cat[r["category"]][r["classification"]] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        "cards_dir": args.cards_dir,
        "regressions_file": args.regressions,
        "n_findings": len(results),
        "by_classification": dict(by_cls),
        "by_category": {cat: dict(c) for cat, c in sorted(by_cat.items())},
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"checked {len(results)} findings against {args.cards_dir}")
    cols = [c for c in CLASSIFICATIONS if by_cls.get(c)]
    header = ["category"] + cols + ["total"]
    rows = []
    for cat in sorted(by_cat, key=lambda c: -sum(by_cat[c].values())):
        counts = by_cat[cat]
        rows.append([cat] + [str(counts.get(c, 0)) for c in cols] + [str(sum(counts.values()))])
    rows.append(["total"] + [str(by_cls.get(c, 0)) for c in cols] + [str(len(results))])
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
