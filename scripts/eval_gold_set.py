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
from auto_benchmarkcard.validation_policy import UNFLAGGABLE_FIELDS  # noqa: E402

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
    card resources. Returns {arxiv, doi, url, via} or None."""
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


def build_report(label, manifest, truth, cards_root=None, judge=None, amendments=None):
    judge_cards = (judge or {}).get("per_card", {})
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
            "flagged_paths": sorted((card.get("flagged_fields") or {}).keys()),
            "validated": is_val,
            "resolved_source": resolved, "truth_source": truth.get(name, {}).get("source_url"),
            "source_status": status,
            "faithfulness": judge_cards.get(name, {}).get("aggregate"),  # None if no judge run
        })

    agg = aggregate(cards)
    if judge:
        agg["faithfulness"] = judge.get("aggregate")
        agg["flag_quality"] = flag_quality(cards, judge)
        if amendments:
            agg["flag_quality_amended"] = flag_quality(cards, judge, amendments)
    return {"label": label, "n_cards": len(cards),
            "aggregate": agg, "by_stratum": by_stratum(cards), "cards": cards}


def flag_quality(cards, judge, amendments=None):
    """FactReasoner flag precision/recall/F1 against the judge's field verdicts.

    The flagger is a hallucination detector; the judge gives ground truth. Positive
    (should-flag) = judge status 'unsupported'; a second variant also counts
    'partial'. Only fields the judge labeled as filled (supported/partial/
    unsupported) count toward precision; flags on fields the judge did not label are
    reported separately as flagged_unlabeled.

    `amendments` is the gold_set/judge_amendments.json dict ("card:path" keys).
    Fields amended to supported_by_eee_only count as filled-but-not-positive
    (a flag on them is a false positive, and they never count as fn) -- the
    judge could not see the EEE artifacts those values were verified against.
    """
    jcards = (judge or {}).get("per_card", {})
    # Fields the pipeline force-unflags regardless of FactReasoner score
    # (validation_policy.UNFLAGGABLE_FIELDS), plus *.name which the flagger
    # skips. A hallucination in one of these can never be flagged, so they cap
    # recall structurally; we report recall against the flaggable subset too.
    UNFLAGGABLE = UNFLAGGABLE_FIELDS

    def _flaggable(path):
        return path not in UNFLAGGABLE and not path.endswith(".name")

    n_amended = 0
    amended_by_card = collections.defaultdict(dict)
    for key, am in (amendments or {}).items():
        cname, _, path = key.partition(":")
        if cname and path and am.get("amended"):
            amended_by_card[cname][path] = am

    def score(positive):
        nonlocal n_amended
        n_amended = 0
        tp = fp = fn = unlabeled = 0
        pos_total = pos_flaggable = tp_flaggable = 0
        for c in cards:
            verdicts = {
                v["path"]: v.get("status")
                for v in jcards.get(c["name"], {}).get("field_verdicts", [])
            }
            for path, am in amended_by_card.get(c["name"], {}).items():
                if am.get("original") and verdicts.get(path) != am["original"]:
                    print(f"warning: SKIPPING stale amendment {c['name']}:{path} "
                          f"(judge={verdicts.get(path)!r}, amendment expects {am['original']!r})")
                    continue
                verdicts[path] = am["amended"]
                n_amended += 1
            filled = {p: s for p, s in verdicts.items()
                      if s in ("supported", "partial", "unsupported", "supported_by_eee_only")}
            positives = {p for p, s in filled.items() if s in positive}
            flagged = set(c.get("flagged_paths", []))
            for p in flagged:
                if p in filled:
                    tp += 1 if p in positives else 0
                    fp += 0 if p in positives else 1
                else:
                    unlabeled += 1
            fn += len(positives - flagged)
            pos_total += len(positives)
            pos_flaggable += sum(1 for p in positives if _flaggable(p))
            tp_flaggable += sum(1 for p in positives & flagged if _flaggable(p))
        prec = round(tp / (tp + fp), 3) if (tp + fp) else None
        rec = round(tp / (tp + fn), 3) if (tp + fn) else None
        f1 = round(2 * prec * rec / (prec + rec), 3) if (prec and rec) else (
            0.0 if (tp + fp + fn) else None)
        rec_flag = round(tp_flaggable / pos_flaggable, 3) if pos_flaggable else None
        out = {"tp": tp, "fp": fp, "fn": fn, "flagged_unlabeled": unlabeled,
               "precision": prec, "recall": rec, "f1": f1,
               "positives": pos_total, "positives_flaggable": pos_flaggable,
               "recall_flaggable": rec_flag}
        if amendments:
            out["n_amendments_applied"] = n_amended
        return out

    return {"unsupported": score({"unsupported"}),
            "unsupported_or_partial": score({"unsupported", "partial"})}


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
    fa = a.get("faithfulness")
    if fa:
        eee = f" eee-only={fa['supported_by_eee_only']}" if "supported_by_eee_only" in fa else ""
        comp = f"completeness-miss={pct(fa['completeness_miss_rate'])} ({fa['ns_missed']}/{fa['n_ns']})"
        if fa.get("completeness_miss_rate_primary") is not None:
            comp += (f" [primary {pct(fa['completeness_miss_rate_primary'])}, "
                     f"eee-only {fa.get('ns_missed_eee_only', 0)}]")
        print(f"  Faithfulness   support={pct(fa['support_rate'])} unsupported={pct(fa['unsupported_rate'])} "
              f"specific={pct(fa['specific_rate'])} (over {fa['n_filled']} filled{eee}); "
              f"{comp}; risk-grounded={pct(fa['risk_grounded_rate'])}")
    fq = a.get("flag_quality")
    if fq:
        u = fq["unsupported"]
        print(f"  Flag quality   vs unsupported: P={pct(u['precision'])} R={pct(u['recall'])} "
              f"F1={pct(u['f1'])} (tp={u['tp']} fp={u['fp']} fn={u['fn']}, "
              f"{u['flagged_unlabeled']} flags on unlabeled fields)")
        print(f"                 recall vs flaggable={pct(u['recall_flaggable'])} "
              f"({u['positives_flaggable']}/{u['positives']} of unsupported are structurally flaggable)")
    fqa = a.get("flag_quality_amended")
    if fqa:
        u = fqa["unsupported"]
        print(f"  Flag quality (amended, {u.get('n_amendments_applied', 0)} labels corrected): "
              f"P={pct(u['precision'])} R={pct(u['recall'])} F1={pct(u['f1'])} "
              f"(tp={u['tp']} fp={u['fp']} fn={u['fn']})")
        print(f"                 recall vs flaggable={pct(u['recall_flaggable'])} "
              f"({u['positives_flaggable']}/{u['positives']})")


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
    fa, fb = a.get("faithfulness"), b.get("faithfulness")
    if fa and fb:
        rows += [
            ("Faithful: support", fa["support_rate"], fb["support_rate"], True),
            ("Faithful: unsupported", fa["unsupported_rate"], fb["unsupported_rate"], True),
            ("Faithful: specific", fa["specific_rate"], fb["specific_rate"], True),
            ("Completeness miss", fa["completeness_miss_rate"], fb["completeness_miss_rate"], True),
            # v31-comparable variant: a v2-eee judge sees EEE info the v31 judge could not.
            ("Completeness miss (prim)",
             fa.get("completeness_miss_rate_primary", fa["completeness_miss_rate"]),
             fb.get("completeness_miss_rate_primary", fb["completeness_miss_rate"]), True),
            ("Risk grounded", fa["risk_grounded_rate"], fb["risk_grounded_rate"], True),
        ]
    qa, qb = a.get("flag_quality"), b.get("flag_quality")
    if qa and qb:
        rows += [
            ("Flag precision (uns)", qa["unsupported"]["precision"], qb["unsupported"]["precision"], True),
            ("Flag recall (uns)", qa["unsupported"]["recall"], qb["unsupported"]["recall"], True),
            ("Flag F1 (uns)", qa["unsupported"]["f1"], qb["unsupported"]["f1"], True),
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
    ap.add_argument("--judge", default=None,
                    help="gold_set/judge_<label>.json to merge faithfulness into the report")
    ap.add_argument("--amendments", default=None,
                    help="gold_set/judge_amendments.json with label corrections "
                         "(supported_by_eee_only); raw flag_quality is always reported too")
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

    judge = _load(args.judge) if args.judge else None
    if args.judge and not judge:
        print(f"warning: could not load judge file {args.judge} - faithfulness left empty")
    amendments = _load(args.amendments) if args.amendments else None
    if args.amendments and not amendments:
        print(f"warning: could not load amendments file {args.amendments}")
    rep = build_report(args.label, manifest, truth, cards_root=args.cards_root, judge=judge,
                       amendments=amendments)
    out = args.out or os.path.join(GOLD_DIR, f"report_{args.label}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print_report(rep)
    print(f"\nReport -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
