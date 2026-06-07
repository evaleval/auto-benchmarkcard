"""Step 1 of the gold set: select a stratified ~24-benchmark sample from the existing
generated cards, write gold_set/manifest.json, and compute the FREE automatic metrics
(Not-specified rate, flagged-field count) as the first 'before' baseline. No inference.

Truth layers B (source) and C (human-light faithfulness/specificity) get added later.
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auto_benchmarkcard.card_utils import card_field_stats, extract_card  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
GOLD_DIR = os.path.join(REPO, "gold_set")

# Candidate set, tagged by stratum. The finder keeps whatever actually exists.
CANDIDATES = [
    # single + paper (source easy/known, faithfulness baseline)
    ("bigcodebench", "single/paper/known"), ("arc", "single/paper/known"),
    ("bigcodebench-full", "single/paper/known"), ("aider-polyglot", "single/paper"),
    ("beyond-aime", "single/paper"), ("autologi", "single/paper"),
    # composite / suite (composite-validation + hard source)
    ("glue", "composite/suite"), ("superglue", "composite/suite"),
    ("aa-index", "composite/suite"), ("aa-lcr", "composite/suite"),
    # EEE-only / long-tail / acronym (source-finding hard + EEE-validation gap)
    ("agentharm", "eee/longtail"), ("androidworld", "eee/longtail"),
    ("biolp-bench", "eee/longtail"), ("alpacaeval-2.0", "eee/longtail"),
    ("api-bank", "eee/longtail"), ("assistantbench", "eee/longtail"),
    ("bfcl", "eee/longtail/acronym"), ("bbh", "eee/longtail/acronym"),
    # multimodal / other domain (vocab spread)
    ("blink", "multimodal/other"), ("charxiv-d", "multimodal/other"),
    ("bixbench", "multimodal/other"), ("activitynet", "multimodal/other"),
    ("chexpert-cxr", "multimodal/other"), ("bold", "safety/other"),
    ("attaq", "safety/other"),
]


def ts_of(path):
    m = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})", path)
    return m.group(1) if m else ""


def find_latest_card(name):
    """Latest non-trash benchmark_card file whose folder matches the name (case-insensitive)."""
    hits = []
    for f in glob.glob(os.path.join(REPO, "output", "**", "benchmark_card_*.json"), recursive=True):
        if "_trash" in f or "legacy" in f:
            continue
        folder = os.path.basename(os.path.dirname(os.path.dirname(f)))  # <bench>_<ts>
        bench = re.sub(r"_\d{4}-\d{2}-\d{2}.*$", "", folder).lower()
        if bench == name.lower():
            hits.append(f)
    return max(hits, key=ts_of) if hits else None


def card_metrics(card_path):
    s = card_field_stats(extract_card(json.load(open(card_path))))
    total, ns = s["n_fields"], s["n_not_specified"]
    return {"n_fields": total, "n_not_specified": ns,
            "ns_rate": round(ns / total, 3) if total else None, "n_flagged": s["n_flagged"]}


def main():
    os.makedirs(GOLD_DIR, exist_ok=True)
    entries, missing = [], []
    for name, stratum in CANDIDATES:
        card = find_latest_card(name)
        if not card:
            missing.append(name)
            continue
        rel = os.path.relpath(card, REPO)
        entries.append({"name": name, "stratum": stratum, "card_path": rel, **card_metrics(card)})

    manifest = {"n": len(entries), "strata": sorted({e["stratum"] for e in entries}), "cards": entries}
    with open(os.path.join(GOLD_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Found {len(entries)} / {len(CANDIDATES)} candidate cards. Missing: {', '.join(missing) or 'none'}\n")
    print(f"{'benchmark':22s} {'stratum':24s} {'fields':>6s} {'NS':>3s} {'NS%':>5s} {'flag':>4s}")
    import collections
    by_str = collections.Counter()
    rates = []
    for e in sorted(entries, key=lambda x: x["stratum"]):
        by_str[e["stratum"]] += 1
        rates.append(e["ns_rate"] or 0)
        print(f"  {e['name']:20s} {e['stratum']:24s} {e['n_fields']:>6d} {e['n_not_specified']:>3d} "
              f"{100 * (e['ns_rate'] or 0):>4.0f}% {e['n_flagged']:>4d}")
    print(f"\nStratum coverage: {dict(by_str)}")
    print(f"BASELINE (existing v31 cards): mean Not-specified rate = {100 * sum(rates) / len(rates):.1f}%"
          if rates else "no cards")
    print(f"Manifest -> {os.path.relpath(os.path.join(GOLD_DIR, 'manifest.json'), REPO)}")


if __name__ == "__main__":
    main()
