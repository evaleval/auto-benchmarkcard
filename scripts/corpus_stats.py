"""Corpus-level (RQ1) numbers from artifacts only: no model calls, no S sample.

Collects, over a corpus cards/ dir plus its run dirs and logs:
  completion      n cards in the corpus dir (+ regen failed_benchmarks.json)
  field fill      strict slot-level fill via card_utils.card_field_stats
  flags           total flagged fields, cards with >= 1 flag
  binding paths   resolved_from distribution over paper-verification.json
                  sidecars (newest run dir per slug), raw + grouped
  runtime         per-card generation seconds from run_log.txt "INFO OK <name> (<n>s)"
  regen cost      llm_usage_*.jsonl telemetry (cost, tokens, provider/quantization/
                  response_format), measured-on-regen + extrapolated per-card mean

Usage:
  python scripts/corpus_stats.py --corpus-dir output/broad_run/cards \
      --run-dirs output/broad_run/output \
      --logs output/broad_run/run_log.txt \
      --regen-dirs output/regen_run/output output/regen_run_heavy/output \
      --out eval/corpus_stats.json
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import (  # noqa: E402
    GOLD_SECTIONS, card_field_stats, extract_card, is_not_specified,
)

from check_frozen import check  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

OK_RE = re.compile(r"INFO OK (.+?) \((\d+)s\)\s*$")

# keys inside the five sections that are pipeline bookkeeping, not schema fields
_NON_SCHEMA_KEYS = {"provenance", "flagged_fields", "missing_fields"}


def _is_empty(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def strict_field_fill(card):
    """THE paper fill convention (orchestrator ruling 2026-07-05): every schema
    field present in the five sections is a slot; filled = non-empty AND not a
    Not-specified sentinel. Unlike card_field_stats this keeps appears_in,
    benchmark_type, size_breakdown, judge_*, authors, logo, org_url in the
    universe and counts empty values as unfilled."""
    n_slots = n_filled = n_ns = n_empty = 0
    for sec in GOLD_SECTIONS:
        fields = card.get(sec, {})
        if not isinstance(fields, dict):
            continue
        for k, v in fields.items():
            if k in _NON_SCHEMA_KEYS:
                continue
            n_slots += 1
            if is_not_specified(v):
                n_ns += 1
            elif _is_empty(v):
                n_empty += 1
            else:
                n_filled += 1
    return {"n_slots": n_slots, "n_filled": n_filled,
            "n_ns_sentinel": n_ns, "n_empty": n_empty}


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def group_resolved_from(v):
    if v is None:
        return "no_paper_honest"
    if v.startswith("binding_verified"):
        return "pre_set_gate_verified"
    if v == "known_papers":
        return "curated_known_papers"
    if "fallback" in v:
        return "deterministic_fallback"
    if "batch_verification" in v or "llm" in v:
        return "search_llm_verified"
    return f"other:{v}"


def newest_run_dirs(run_dir_roots):
    dirs = {}
    for root in run_dir_roots:
        for d in sorted(glob.glob(os.path.join(REPO, root, "*_2026-*"))):
            slug = os.path.basename(d).rsplit("_2026-", 1)[0]
            dirs[slug] = d
    return dirs


def main():
    ap = argparse.ArgumentParser(description="Corpus/RQ1 stats from artifacts.")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="run-dir roots for binding sidecars (newest per slug wins)")
    ap.add_argument("--logs", nargs="+", default=[], help="run_log.txt files for runtime")
    ap.add_argument("--backlogs", nargs="*", default=[],
                    help="backlog files; unique non-empty lines = benchmarks attempted")
    ap.add_argument("--regen-dirs", nargs="*", default=[],
                    help="regen output roots for llm_usage cost telemetry")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    check()

    # cards: fill + flags
    card_paths = sorted(glob.glob(os.path.join(REPO, args.corpus_dir, "*.json")))
    n_slots = n_ns = n_flags = cards_flagged = 0
    st = {"n_slots": 0, "n_filled": 0, "n_ns_sentinel": 0, "n_empty": 0}
    for p in card_paths:
        card = extract_card(_load(p) or {})
        s = card_field_stats(card)
        n_slots += s["n_fields"]
        n_ns += s["n_not_specified"]
        for k, v in strict_field_fill(card).items():
            st[k] += v
        nf = len(card.get("flagged_fields") or {})
        n_flags += nf
        cards_flagged += 1 if nf else 0

    # attempted benchmarks (backlog lines)
    attempted = None
    if args.backlogs:
        names = set()
        for bp in args.backlogs:
            with open(os.path.join(REPO, bp)) as f:
                names |= {ln.strip() for ln in f if ln.strip()}
        attempted = len(names)

    # binding paths + subject-veto actions
    raw, grouped, subject_actions = {}, {}, {}
    dirs = newest_run_dirs(args.run_dirs)
    n_sidecars = 0
    for slug, d in dirs.items():
        hits = glob.glob(os.path.join(d, "tool_output", "paper_resolver",
                                      "paper-verification.json"))
        if not hits:
            raw["<no sidecar>"] = raw.get("<no sidecar>", 0) + 1
            continue
        v = _load(hits[0]) or {}
        n_sidecars += 1
        rf = v.get("resolved_from")
        raw[str(rf)] = raw.get(str(rf), 0) + 1
        g = group_resolved_from(rf)
        grouped[g] = grouped.get(g, 0) + 1
        so = v.get("subject_overlap")
        if isinstance(so, dict) and so.get("action"):
            a = so["action"]
            subject_actions[a] = subject_actions.get(a, 0) + 1

    # runtime
    runtimes = []
    for lp in args.logs:
        with open(os.path.join(REPO, lp)) as f:
            for line in f:
                m = OK_RE.search(line)
                if m:
                    runtimes.append(int(m.group(2)))
    rt = None
    if runtimes:
        arr = np.array(runtimes, float)
        rt = {"n_generations": len(runtimes), "median_s": float(np.median(arr)),
              "mean_s": float(arr.mean()), "p90_s": float(np.percentile(arr, 90)),
              "logs": args.logs}

    # regen cost telemetry
    cost = None
    if args.regen_dirs:
        per_card = {}
        providers, quants, formats, models = set(), set(), set(), set()
        failed = []
        for root in args.regen_dirs:
            for fb in glob.glob(os.path.join(REPO, root, "..", "failed_benchmarks.json")):
                fl = _load(fb)
                if isinstance(fl, list):
                    failed += fl
            for jp in sorted(glob.glob(os.path.join(
                    REPO, root, "*_2026-*", "tool_output", "llm_usage", "llm_usage_*.jsonl"))):
                slug = os.path.basename(os.path.dirname(os.path.dirname(
                    os.path.dirname(jp))))
                slug = slug.rsplit("_2026-", 1)[0]
                c = per_card.setdefault(slug, {"cost": 0.0, "input_tokens": 0,
                                               "output_tokens": 0, "calls": 0})
                with open(jp) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        c["calls"] += 1
                        c["cost"] += r.get("cost") or 0.0
                        c["input_tokens"] += r.get("input_tokens") or 0
                        c["output_tokens"] += r.get("output_tokens") or 0
                        if r.get("provider"):
                            providers.add(r["provider"])
                        if r.get("quantization"):
                            quants.add(r["quantization"])
                        if r.get("response_format"):
                            formats.add(r["response_format"])
                        if r.get("model"):
                            models.add(r["model"])
        if per_card:
            costs = np.array([c["cost"] for c in per_card.values()])
            cost = {
                "provenance": "measured-on-regen",
                "cost_scope": "OpenRouter-served composer calls only; RITS calls "
                              "(flagger, light model) are logged with cost 0",
                "n_cards_measured": len(per_card),
                "total_cost_usd": float(costs.sum()),
                "mean_cost_per_card_usd": float(costs.mean()),
                "median_cost_per_card_usd": float(np.median(costs)),
                "max_cost_per_card_usd": float(costs.max()),
                "total_calls": sum(c["calls"] for c in per_card.values()),
                "total_input_tokens": sum(c["input_tokens"] for c in per_card.values()),
                "total_output_tokens": sum(c["output_tokens"] for c in per_card.values()),
                "providers": sorted(providers), "quantizations": sorted(quants),
                "response_formats": sorted(formats), "models": sorted(models),
                "regen_failed_benchmarks": failed,
                "extrapolated": {
                    "provenance": "extrapolated",
                    "note": "mean regen cost per card x corpus size; broad_run itself "
                            "had no cost instrumentation",
                    "corpus_n": len(card_paths),
                    "est_corpus_cost_usd": float(costs.mean()) * len(card_paths),
                },
                "per_card": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                                 for kk, vv in v.items()}
                             for k, v in sorted(per_card.items())},
            }

    out = {
        "generated_on": date.today().isoformat(),
        "corpus": {"dir": args.corpus_dir, "n_cards": len(card_paths),
                   "attempted": attempted,
                   "completion_rate": round(len(card_paths) / attempted, 4)
                   if attempted else None},
        "field_fill": {
            "convention": "strict: all schema fields in the five sections; "
                          "filled = non-empty AND not a Not-specified sentinel "
                          "(THE paper number, orchestrator ruling 2026-07-05)",
            "n_slots": st["n_slots"], "n_filled": st["n_filled"],
            "n_ns_sentinel": st["n_ns_sentinel"], "n_empty": st["n_empty"],
            "fill_rate": round(st["n_filled"] / st["n_slots"], 4) if st["n_slots"] else None,
        },
        "field_fill_cardfieldstats": {
            "convention": "card_field_stats: excludes appears_in, benchmark_type, "
                          "size_breakdown, judge_uses_llm, judge_num, judge_models, "
                          "authors, logo, org_url; empty values count as filled "
                          "(secondary, NOT the paper number)",
            "n_slots": n_slots, "n_not_specified": n_ns,
            "n_filled": n_slots - n_ns,
            "fill_rate": round((n_slots - n_ns) / n_slots, 4) if n_slots else None,
        },
        "flags": {"n_flagged_fields": n_flags, "cards_with_flags": cards_flagged,
                  "cards_total": len(card_paths)},
        "binding_paths": {"n_run_dirs": len(dirs), "n_sidecars": n_sidecars,
                          "resolved_from_raw": dict(sorted(raw.items())),
                          "grouped": dict(sorted(grouped.items())),
                          "subject_actions": dict(sorted(subject_actions.items()))},
        "runtime": rt,
        "regen_cost": cost,
    }
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"cards {len(card_paths)}, strict fill {out['field_fill']['fill_rate']} "
          f"({st['n_filled']}/{st['n_slots']}; ns {st['n_ns_sentinel']}, empty "
          f"{st['n_empty']}), cfs fill {out['field_fill_cardfieldstats']['fill_rate']}, "
          f"flags {n_flags} on {cards_flagged} cards")
    print(f"binding grouped: {out['binding_paths']['grouped']}")
    if rt:
        print(f"runtime n={rt['n_generations']} median {rt['median_s']:.0f}s "
              f"mean {rt['mean_s']:.0f}s p90 {rt['p90_s']:.0f}s")
    if cost:
        print(f"regen cost: {cost['n_cards_measured']} cards, total "
              f"${cost['total_cost_usd']:.2f}, mean ${cost['mean_cost_per_card_usd']:.3f}"
              f"/card ({', '.join(cost['providers'])} {', '.join(cost['quantizations'])})")
    print(f"stats -> {args.out}")


if __name__ == "__main__":
    main()
