"""Build the Phase-0 findings regression set from the S=150 screen + verifier data.

Joins the 636-finding screen output (eval/s150/screen/screen_results.json) with
the per-finding verifier labels (eval/s150/screen_verification_final/
ratings_verifier_v1_raw_copy.csv, keyed by card + finding_index) and keeps only
verifier-confirmed-material findings. The total is asserted against
raw_label_counts in screen_verification_scores_final.json.

current_value is recovered from the frozen evaluated cards. Supply their directory
with --frozen-cards-dir or BENCHMARKCARD_FROZEN_CARDS_DIR; the default is
output/auto-benchmarkcards-v3/cards in this checkout. Cards are read strictly
read-only and every used copy is md5-checked against sample.json.

Usage:
  python scripts/build_regression_set.py \
      [--out eval/regression/findings_regression.json] [--frozen-cards-dir DIR]
"""

import argparse
import collections
import csv
import hashlib
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
SCREEN_RESULTS = os.path.join(REPO_ROOT, "eval/s150/screen/screen_results.json")
RATINGS_CSV = os.path.join(
    REPO_ROOT, "eval/s150/screen_verification_final/ratings_verifier_v1_raw_copy.csv"
)
SCORES_FINAL = os.path.join(
    REPO_ROOT, "eval/s150/screen_verification_final/screen_verification_scores_final.json"
)
SAMPLE = os.path.join(REPO_ROOT, "eval/s150/sample.json")
FROZEN_CARDS_DEFAULT = os.environ.get(
    "BENCHMARKCARD_FROZEN_CARDS_DIR",
    os.path.join(REPO_ROOT, "output", "auto-benchmarkcards-v3", "cards"),
)


def parse_path(path):
    """Parse a dotted field path with optional [0] / ["key"] segments.

    Returns a list of str keys and int indices, or None if the string is not a
    well-formed path (spaces, unbalanced brackets, non-integer bare index).
    """
    parts = []
    token = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if not token:
                return None
            parts.append(token)
            token = ""
            i += 1
        elif c == "[":
            if token:
                parts.append(token)
                token = ""
            j = path.find("]", i)
            if j < 0:
                return None
            inner = path[i + 1 : j].strip()
            if len(inner) >= 2 and inner[0] in "\"'" and inner[-1] == inner[0]:
                parts.append(inner[1:-1])
            else:
                try:
                    parts.append(int(inner))
                except ValueError:
                    return None
            i = j + 1
        elif c.isalnum() or c == "_":
            token += c
            i += 1
        else:
            return None
    if token:
        parts.append(token)
    return parts or None


def render_path(parts):
    """Render parsed path segments back to canonical dotted form."""
    out = ""
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        elif out and p.isidentifier():
            out += f".{p}"
        elif not out:
            out = p
        else:
            out += f'["{p}"]'
    return out


def resolve_path(card, parts):
    """Follow parsed path segments through a card dict. Returns (found, value)."""
    node = card
    for p in parts:
        if isinstance(p, int):
            if isinstance(node, list) and 0 <= p < len(node):
                node = node[p]
            else:
                return False, None
        elif isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return False, None
    return True, node


def resolve_locator(card, locator):
    """Try each ' / ' or ' + ' separated candidate path in a free-text field
    locator against the card. Returns (canonical_path, value) for the first one
    that resolves, else (None, None)."""
    candidates = []
    for chunk in locator.split(" / "):
        candidates.extend(chunk.split(" + "))
    for cand in candidates:
        cand = cand.strip()
        # some locators carry a trailing note, e.g. "benchmark_details.resources (paper link missing)"
        if cand.endswith(")") and " (" in cand:
            cand = cand[: cand.rindex(" (")].strip()
        parts = parse_path(cand)
        if parts is None:
            continue
        found, value = resolve_path(card, parts)
        if found:
            return render_path(parts), value
    return None, None


def load_frozen_cards(sample, frozen_dir):
    """Load the frozen evaluated cards, md5-verified against sample.json.

    Tries the corpus pointer and per-card card_path from sample.json first
    (both absent on this branch), then frozen_dir. Only md5-matching copies
    are used.
    """
    cards = {}
    unverified = []
    for entry in sample["cards"]:
        name = entry["name"]
        candidates = [
            os.path.join(REPO_ROOT, entry["corpus_card"]),
            os.path.join(REPO_ROOT, entry.get("card_path", "")),
            os.path.join(frozen_dir, f"{name}.json"),
        ]
        found = None
        for path in candidates:
            if not path or not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                raw = f.read()
            if hashlib.md5(raw).hexdigest() == entry["corpus_card_md5"]:
                found = extract_card(json.loads(raw))
                break
        if found is None:
            unverified.append(name)
        else:
            cards[name] = found
    return cards, unverified


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "eval/regression/findings_regression.json"))
    ap.add_argument("--frozen-cards-dir", default=FROZEN_CARDS_DEFAULT)
    args = ap.parse_args()

    screen = json.load(open(SCREEN_RESULTS))["per_card"]
    sample = json.load(open(SAMPLE))
    expected = json.load(open(SCORES_FINAL))["finding_level"]["raw_label_counts"]["confirmed-material"]

    with open(RATINGS_CSV) as f:
        ratings = list(csv.DictReader(f))

    frozen, unverified = load_frozen_cards(sample, args.frozen_cards_dir)
    if unverified:
        print(f"warning: no md5-verified frozen card for {len(unverified)} cards: {unverified}")

    records = []
    unresolved = []
    for row in ratings:
        if row["verifier_label"] != "confirmed-material":
            continue
        card, idx = row["card"], int(row["finding_index"])
        finding = screen[card]["findings"][idx]
        # the CSV was generated from the screen findings; make sure the join is sound
        assert finding["category"] == row["category"] and finding["field"] == row["field"], (card, idx)

        field_path, current_value, resolved = row["field"], None, False
        if card in frozen:
            canonical, value = resolve_locator(frozen[card], row["field"])
            if canonical is not None:
                field_path, current_value, resolved = canonical, value, True
        if not resolved:
            unresolved.append((card, idx, row["field"]))

        rec = {
            "card": card,
            "finding_index": idx,
            "field_path": field_path,
            "field_path_resolved": resolved,
            "field_locator": row["field"],
            "category": row["category"],
            "severity": finding["severity"],
            "finding_text": row["issue"],
            "current_value": current_value,
            "verification": "confirmed-material",
        }
        if row.get("evidence_url"):
            rec["evidence_url"] = row["evidence_url"]
        records.append(rec)

    assert len(records) == expected, f"got {len(records)} records, verifier file says {expected}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {
        "source_screen": os.path.relpath(SCREEN_RESULTS, REPO_ROOT),
        "source_ratings": os.path.relpath(RATINGS_CSV, REPO_ROOT),
        "frozen_cards_dir": args.frozen_cards_dir,
        "n_findings": len(records),
        "n_field_path_resolved": sum(r["field_path_resolved"] for r in records),
        "findings": records,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    by_cat = collections.Counter(r["category"] for r in records)
    print(f"wrote {len(records)} confirmed-material findings to {args.out}")
    print("per category:")
    for cat, n in by_cat.most_common():
        print(f"  {cat}: {n}")
    print(f"field path resolved with current_value: {out['n_field_path_resolved']}/{len(records)}")
    if unresolved:
        print("unresolved locators (kept as free text, current_value null):")
        for card, idx, loc in unresolved:
            print(f"  {card}[{idx}]: {loc}")


if __name__ == "__main__":
    main()
