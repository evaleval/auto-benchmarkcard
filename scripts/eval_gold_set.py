"""Step 2 of the gold set: the reusable, free (no-inference) measurement harness.

Reads gold_set/manifest.json + gold_set/truth.json and computes, per card and
aggregated, the metrics that make each Eimer-2 improvement a before/after number:

  - Not-specified rate (overall, plus prose vs deterministic split)
  - Flag rate (count of flagged_fields)
  - Validation coverage (FactReasoner ran with non-trivial output vs skipped)
  - Source-correct (what the card resolved vs the frozen truth paper)
  - Slot for faithfulness/specificity (judge, deferred - left null)

Usage:
  python scripts/eval_gold_set.py --label v31
  python scripts/eval_gold_set.py --label v32 --cards-root output/v32_regen
  python scripts/eval_gold_set.py --compare gold_set/report_v31.json gold_set/report_v32.json
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import card_field_stats, extract_card  # noqa: E402
from auto_benchmarkcard.tools.eee.paper_resolver import KNOWN_PAPERS  # noqa: E402

try:
    from auto_benchmarkcard.tools.eee.paper_resolver import _normalize_name as _norm_name
except Exception:  # pragma: no cover - fallback if the helper is renamed
    def _norm_name(s):
        return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

REPO = os.path.join(os.path.dirname(__file__), "..")
GOLD_DIR = os.path.join(REPO, "gold_set")

_ARXIV_RES = [
    re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I),
    re.compile(r"arxiv:(\d{4}\.\d{4,5})", re.I),
    re.compile(r"^\s*(\d{4}\.\d{4,5})\s*$"),
]
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.I)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def norm_arxiv(s):
    if not isinstance(s, str):
        return None
    for rx in _ARXIV_RES:
        m = rx.search(s)
        if m:
            return m.group(1)
    return None


def norm_doi(s):
    if not isinstance(s, str):
        return None
    m = _DOI_RE.search(s)
    return m.group(1).rstrip(".").lower() if m else None


def run_dir_of(card_path):
    return os.path.join(REPO, os.path.dirname(os.path.dirname(card_path)))


def validated(run_dir):
    """A card counts as validated if FactReasoner produced a non-trivial result
    (num_atoms > 0). Composite/EEE-only cards that skip RAG+FR have no such file."""
    for p in glob.glob(os.path.join(run_dir, "tool_output", "factreasoner", "*.json")):
        fr = _load(p) or {}
        n = (fr.get("results") or {}).get("num_atoms")
        if not n:
            n = len(fr.get("atom_summary") or [])
        if n and n > 0:
            return True
    return False


def resolved_source(name, run_dir, card):
    """What paper the card actually resolved, by descending authority:
    paper_resolver resolved_url -> extractor paper_url -> first arxiv/doi in the
    card resources -> KNOWN_PAPERS. Returns {arxiv, doi, url, via} or None."""
    to = os.path.join(run_dir, "tool_output")

    for p in glob.glob(os.path.join(to, "paper_resolver", "*.json")):
        ru = (_load(p) or {}).get("resolved_url")
        a, d = norm_arxiv(ru), norm_doi(ru)
        if a or d:
            return {"arxiv": a, "doi": d, "url": ru, "via": "paper_resolver"}

    for p in glob.glob(os.path.join(to, "extractor", "*.json")):
        ext = _load(p) or {}
        for key in ("paper_url", "paper_url_from_hf"):
            u = ext.get(key)
            a, d = norm_arxiv(u), norm_doi(u)
            if a or d:
                return {"arxiv": a, "doi": d, "url": u, "via": "extractor"}

    res = card.get("benchmark_details", {}).get("resources")
    if isinstance(res, list):
        for u in res:
            a, d = norm_arxiv(u), norm_doi(u)
            if a or d:
                return {"arxiv": a, "doi": d, "url": u, "via": "resources"}

    kp = KNOWN_PAPERS.get(_norm_name(name)) or KNOWN_PAPERS.get(
        _norm_name(card.get("benchmark_details", {}).get("name") or ""))
    if kp:
        return {"arxiv": norm_arxiv(kp), "doi": norm_doi(kp), "url": kp, "via": "known_papers"}
    return None


def source_status(truth_entry, resolved):
    """Compare resolved source against frozen truth.
    correct / wrong / unresolved for paper truths; abstain_correct / abstain_wrong
    for leaderboard/website/none truths (no introducing paper expected)."""
    has_paper_truth = bool(truth_entry) and truth_entry.get("source_type") in ("arxiv", "doi")
    res_id = (resolved or {}).get("arxiv") or (resolved or {}).get("doi")

    if not has_paper_truth:
        return "abstain_wrong" if res_id else "abstain_correct"
    if not res_id:
        return "unresolved"
    t_arxiv = truth_entry.get("arxiv_id") or norm_arxiv(truth_entry.get("source_url"))
    t_doi = norm_doi(truth_entry.get("doi") or "") or norm_doi(truth_entry.get("source_url") or "")
    r_arxiv = (resolved or {}).get("arxiv")
    r_doi = norm_doi((resolved or {}).get("doi") or "")
    if (t_arxiv and r_arxiv and t_arxiv == r_arxiv) or (t_doi and r_doi and t_doi == r_doi):
        return "correct"
    return "wrong"


def build_report(label, manifest, truth, cards_root=None):
    cards = []
    for entry in manifest["cards"]:
        name = entry["name"]
        if cards_root:
            # re-run mode: pull the latest regenerated card for this name from the root
            hits = glob.glob(os.path.join(REPO, cards_root, "**",
                                          f"benchmark_card_{name}.json"), recursive=True)
            card_path = os.path.relpath(max(hits, key=os.path.getmtime), REPO) if hits else entry["card_path"]
        else:
            card_path = entry["card_path"]

        rd = run_dir_of(card_path)
        card = extract_card(_load(os.path.join(REPO, card_path)) or {})
        s = card_field_stats(card)
        is_val = validated(rd)
        resolved = resolved_source(name, rd, card)
        status = source_status(truth.get(name), resolved)
        cards.append({
            "name": name, "stratum": entry["stratum"], "card_path": card_path,
            "n_fields": s["n_fields"], "n_not_specified": s["n_not_specified"],
            "ns_rate": round(s["n_not_specified"] / s["n_fields"], 3) if s["n_fields"] else None,
            "n_prose": s["n_prose"], "n_prose_ns": s["n_prose_ns"],
            "n_det": s["n_det"], "n_det_ns": s["n_det_ns"],
            "n_flagged": s["n_flagged"],
            "validated": is_val,
            "resolved_source": resolved, "truth_source": truth.get(name, {}).get("source_url"),
            "source_status": status,
            "faithfulness": None,  # deferred: filled by the judge step (FactReasoner v3)
        })

    return {"label": label, "n_cards": len(cards),
            "aggregate": aggregate(cards), "by_stratum": by_stratum(cards), "cards": cards}


def _agg(cards):
    tf = sum(c["n_fields"] for c in cards)
    tns = sum(c["n_not_specified"] for c in cards)
    tp = sum(c["n_prose"] for c in cards)
    tpns = sum(c["n_prose_ns"] for c in cards)
    td = sum(c["n_det"] for c in cards)
    tdns = sum(c["n_det_ns"] for c in cards)
    rates = [c["ns_rate"] for c in cards if c["ns_rate"] is not None]
    status = collections.Counter(c["source_status"] for c in cards)
    paper_truth = status["correct"] + status["wrong"] + status["unresolved"]
    return {
        "n_cards": len(cards),
        "ns_rate_mean": round(sum(rates) / len(rates), 3) if rates else None,
        "ns_rate_pooled": round(tns / tf, 3) if tf else None,
        "ns_rate_prose": round(tpns / tp, 3) if tp else None,
        "ns_rate_deterministic": round(tdns / td, 3) if td else None,
        "n_flagged_total": sum(c["n_flagged"] for c in cards),
        "flag_rate_mean": round(sum(c["n_flagged"] for c in cards) / len(cards), 3) if cards else None,
        "validation": {"validated": sum(c["validated"] for c in cards), "total": len(cards),
                       "frac": round(sum(c["validated"] for c in cards) / len(cards), 3) if cards else None},
        "source": {
            **{k: status[k] for k in ("correct", "wrong", "unresolved", "abstain_correct", "abstain_wrong")},
            "paper_truth_n": paper_truth,
            "accuracy": round(status["correct"] / paper_truth, 3) if paper_truth else None,
        },
    }


def aggregate(cards):
    return _agg(cards)


def by_stratum(cards):
    groups = collections.defaultdict(list)
    for c in cards:
        groups[c["stratum"]].append(c)
    return {st: _agg(cs) for st, cs in sorted(groups.items())}


def print_report(rep):
    print(f"\nGold-set report: {rep['label']}  ({rep['n_cards']} cards)\n")
    hdr = f"{'benchmark':18s} {'stratum':18s} {'NS%':>5s} {'pro':>4s} {'det':>4s} {'flag':>4s} {'val':>4s} {'source'}"
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(rep["cards"], key=lambda x: x["stratum"]):
        print(f"{c['name']:18s} {c['stratum']:18s} {100 * (c['ns_rate'] or 0):>4.0f}% "
              f"{c['n_prose_ns']:>4d} {c['n_det_ns']:>4d} {c['n_flagged']:>4d} "
              f"{('Y' if c['validated'] else '-'):>4s} {c['source_status']}")
    a = rep["aggregate"]
    print("\nAggregate:")
    print(f"  Not-specified  mean={pct(a['ns_rate_mean'])}  pooled={pct(a['ns_rate_pooled'])}  "
          f"prose={pct(a['ns_rate_prose'])}  deterministic={pct(a['ns_rate_deterministic'])}")
    print(f"  Flags          total={a['n_flagged_total']}  mean/card={a['flag_rate_mean']}")
    v = a["validation"]
    note = "  (note: 100% on this set - no composite/EEE-only skips selected)" if v["frac"] == 1.0 else ""
    print(f"  Validation     {v['validated']}/{v['total']} = {pct(v['frac'])}{note}")
    s = a["source"]
    print(f"  Source-correct accuracy={pct(s['accuracy'])} over {s['paper_truth_n']} paper-truths "
          f"(correct={s['correct']} wrong={s['wrong']} unresolved={s['unresolved']}); "
          f"abstain ok={s['abstain_correct']} bad={s['abstain_wrong']}")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_compare(before, after):
    a, b = before["aggregate"], after["aggregate"]
    print(f"\nCompare: {before['label']} -> {after['label']}\n")
    print(f"{'metric':26s} {'before':>9s} {'after':>9s} {'delta':>9s}")
    print("-" * 56)
    rows = [
        ("NS rate (mean)", a["ns_rate_mean"], b["ns_rate_mean"], True),
        ("NS rate (pooled)", a["ns_rate_pooled"], b["ns_rate_pooled"], True),
        ("NS rate (prose)", a["ns_rate_prose"], b["ns_rate_prose"], True),
        ("NS rate (deterministic)", a["ns_rate_deterministic"], b["ns_rate_deterministic"], True),
        ("Validation frac", a["validation"]["frac"], b["validation"]["frac"], True),
        ("Source accuracy", a["source"]["accuracy"], b["source"]["accuracy"], True),
    ]
    for label, x, y, ispct in rows:
        dx = None if (x is None or y is None) else y - x
        fmt = pct if ispct else (lambda v: "n/a" if v is None else str(v))
        ds = "n/a" if dx is None else f"{'+' if dx >= 0 else ''}{100 * dx:.1f}pp"
        print(f"{label:26s} {fmt(x):>9s} {fmt(y):>9s} {ds:>9s}")
    # integer metrics
    for label, x, y in [
        ("Flags (total)", a["n_flagged_total"], b["n_flagged_total"]),
        ("Source correct (n)", a["source"]["correct"], b["source"]["correct"]),
        ("Source wrong (n)", a["source"]["wrong"], b["source"]["wrong"]),
    ]:
        d = y - x
        print(f"{label:26s} {x:>9d} {y:>9d} {('+' if d >= 0 else '') + str(d):>9s}")


def main():
    ap = argparse.ArgumentParser(description="Gold-set measurement harness (no inference).")
    ap.add_argument("--label", default="v31", help="report label, e.g. v31")
    ap.add_argument("--manifest", default=os.path.join(GOLD_DIR, "manifest.json"))
    ap.add_argument("--truth", default=os.path.join(GOLD_DIR, "truth.json"))
    ap.add_argument("--cards-root", default=None,
                    help="dir (relative to repo) to pull regenerated cards from, by name")
    ap.add_argument("--out", default=None, help="output report path")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="print a delta table between two report json files and exit")
    args = ap.parse_args()

    if args.compare:
        before, after = _load(args.compare[0]), _load(args.compare[1])
        if not before or not after:
            sys.exit(f"could not load both reports: {args.compare}")
        print_compare(before, after)
        return

    manifest = _load(args.manifest)
    truth = _load(args.truth) or {}
    if not manifest:
        sys.exit(f"no manifest at {args.manifest}")
    if not truth:
        print(f"warning: no truth at {args.truth} - source-correct will be all abstain/unresolved")

    rep = build_report(args.label, manifest, truth, cards_root=args.cards_root)
    out = args.out or os.path.join(GOLD_DIR, f"report_{args.label}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print_report(rep)
    print(f"\nReport -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
