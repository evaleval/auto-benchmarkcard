"""Collate and gate the web-screen verdicts written by s_screen_review.js agents.

Checks completeness against the sample file (missing cards -> rerun_list),
validates every verdict against the screen schema, and writes:
  screen_results.json  per-card verdicts + stratum-weighted aggregates
  screen_meta.json     script/sample/config md5s, model, date, rerun_list
  check_export.json    (--export-check) Aris's human-check worklist: all
                       needs-fix + 10 seeded random clean, severity-triaged

Usage:
  python scripts/collect_s_screen.py --sample eval/s150/sample.json \
      --screen-dir eval/s150/screen [--workflow-results r.json] [--export-check]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import date

import jsonschema

from check_frozen import check, file_md5

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

VERDICTS = ("clean", "minor", "needs-fix")
SCHEMA = {
    "type": "object",
    "required": ["card", "real_benchmark", "identity_correct", "paper_assessment",
                 "hf_repo_assessment", "content_accurate", "fabricated_fact",
                 "verdict", "findings", "citations", "reviewing_model", "summary"],
    "properties": {
        "card": {"type": "string"},
        "real_benchmark": {"type": "string"},
        "identity_correct": {"type": ["boolean", "null"]},
        "paper_assessment": {"enum": ["correct", "wrong", "plausible-unverified",
                                      "correctly-none", "missing-should-exist"]},
        "hf_repo_assessment": {"enum": ["correct-kept", "correctly-rejected",
                                        "wrong-kept-CONTAMINATION", "wrongly-rejected-lost-data",
                                        "no-repo-ok", "no-repo-but-one-exists"]},
        "content_accurate": {"type": ["boolean", "null"]},
        "fabricated_fact": {"type": ["boolean", "null"]},
        "verdict": {"enum": list(VERDICTS)},
        "findings": {"type": "array", "items": {
            "type": "object",
            "required": ["severity", "category", "field", "issue", "ground_truth"],
            "properties": {
                "severity": {"enum": ["needs-fix", "minor", "note"]},
                "category": {"enum": ["wrong-identity", "wrong-paper", "fabricated-fact",
                                      "wrong-section-splice", "thin", "other"]},
                "field": {"type": "string"},
                "issue": {"type": "string"},
                "ground_truth": {"type": "string"}}}},
        "citations": {"type": "array", "items": {"type": "string"}},
        "reviewing_model": {"type": "string"},
        "summary": {"type": "string"},
    },
}

TRIAGE_ORDER = {"wrong-identity": 0, "wrong-paper": 1, "fabricated-fact": 2,
                "wrong-section-splice": 3, "thin": 4, "other": 5}


def _load(p):
    with open(p) as f:
        return json.load(f)


def weighted_rate(cards, sample_cards, predicate):
    """Stratified proportion: sum_h w_h * count_h / sum_h w_h * n_h."""
    num = den = 0.0
    for name, v in cards.items():
        w = sample_cards[name]["weight"]
        num += w * (1 if predicate(v) else 0)
        den += w
    return round(num / den, 4) if den else None


def main():
    ap = argparse.ArgumentParser(description="Screen verdict collator/gate.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--workflow-results", default=None,
                    help="optional workflow return value JSON for reconciliation")
    ap.add_argument("--export-check", action="store_true")
    ap.add_argument("--check-seed", type=int, default=20260704)
    ap.add_argument("--expect", default=None,
                    help="comma-separated subset expected (default: whole sample)")
    args = ap.parse_args()
    check()

    sample = _load(os.path.join(REPO, args.sample))
    sample_cards = {c["name"]: c for c in sample["cards"]}
    expected = ([n.strip() for n in args.expect.split(",")] if args.expect
                else list(sample_cards))
    screen_dir = os.path.join(REPO, args.screen_dir)
    vdir = os.path.join(screen_dir, "verdicts")

    verdicts, problems, missing = {}, {}, []
    for name in expected:
        p = os.path.join(vdir, f"{name}.json")
        if not os.path.exists(p):
            missing.append(name)
            continue
        try:
            v = _load(p)
        except Exception as e:
            problems[name] = [f"unparseable: {e}"]
            continue
        errs = []
        try:
            jsonschema.validate(v, SCHEMA)
        except jsonschema.ValidationError as e:
            errs.append(f"schema: {e.message[:200]}")
        if v.get("card") != name:
            errs.append(f"card field {v.get('card')!r} != filename {name!r}")
        if v.get("verdict") != "clean" and not v.get("citations"):
            errs.append("no citations on a non-clean verdict")
        elif not v.get("citations"):
            print(f"warning: {name}: clean verdict with no citations")
        if errs:
            problems[name] = errs
        else:
            verdicts[name] = v

    if args.workflow_results:
        wf = _load(os.path.join(REPO, args.workflow_results))
        wf_by_card = {r.get("card"): r for r in wf if isinstance(r, dict)}
        for name, r in wf_by_card.items():
            if name in verdicts and r.get("verdict") not in (None, "ERROR") \
                    and r.get("verdict") != verdicts[name].get("verdict"):
                print(f"warning: {name}: workflow verdict {r['verdict']!r} != "
                      f"file verdict {verdicts[name]['verdict']!r} (file wins)")
            if name not in verdicts and name not in problems and name in expected \
                    and r.get("verdict") not in (None, "ERROR"):
                print(f"warning: {name}: workflow returned a verdict but no file written")

    rerun = sorted(missing + list(problems))
    agg = None
    scored = {n: v for n, v in verdicts.items()}
    if scored:
        agg = {
            "n_scored": len(scored),
            "needs_fix_rate": weighted_rate(scored, sample_cards,
                                            lambda v: v["verdict"] == "needs-fix"),
            "clean_rate": weighted_rate(scored, sample_cards,
                                        lambda v: v["verdict"] == "clean"),
            "fabricated_fact_rate": weighted_rate(scored, sample_cards,
                                                  lambda v: v["fabricated_fact"] is True),
            "identity_wrong_rate": weighted_rate(scored, sample_cards,
                                                 lambda v: v["identity_correct"] is False),
            "wrong_paper_rate": weighted_rate(scored, sample_cards,
                                              lambda v: v["paper_assessment"] == "wrong"),
            "contamination_count_raw": sum(
                v["hf_repo_assessment"] == "wrong-kept-CONTAMINATION" for v in scored.values()),
            "verdict_counts_raw": {k: sum(v["verdict"] == k for v in scored.values())
                                   for k in VERDICTS},
            "note": "rates are stratum-weighted over the scored subset; raw counts unweighted",
        }

    results = {"sample_file": args.sample, "per_card": verdicts, "aggregate": agg,
               "problems": problems, "missing": missing}
    rp = os.path.join(screen_dir, "screen_results.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cfg_path = os.path.join(screen_dir, "screen_config.json")
    cfg = _load(cfg_path) if os.path.exists(cfg_path) else {}
    meta = {
        "date": date.today().isoformat(),
        "model_id": cfg.get("model_id"),
        "model_alias": cfg.get("model_alias"),
        "script_md5": file_md5(os.path.join(REPO, "s_screen_review.js")),
        "sample_md5": file_md5(os.path.join(REPO, args.sample)),
        "config_md5": file_md5(cfg_path) if os.path.exists(cfg_path) else None,
        "n_expected": len(expected), "n_scored": len(verdicts),
        "n_missing": len(missing), "n_invalid": len(problems),
        "rerun_list": rerun,
        "models_reported": sorted({v.get("reviewing_model") for v in verdicts.values()}),
    }
    with open(os.path.join(screen_dir, "screen_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.export_check and scored:
        needs_fix = sorted([n for n, v in scored.items() if v["verdict"] == "needs-fix"],
                           key=lambda n: min([TRIAGE_ORDER.get(f["category"], 9)
                                              for f in scored[n]["findings"]] or [9]))
        clean = sorted(n for n, v in scored.items() if v["verdict"] == "clean")
        rng = random.Random(args.check_seed)
        clean_pick = sorted(rng.sample(clean, min(10, len(clean))))
        export = {"instructions": "re-check every needs-fix card (triage order: wrong-identity, "
                                  "wrong-paper, fabrication first), then the 10 random clean; "
                                  "record agree/disagree + note per card",
                  "needs_fix": [{"card": n, "verdict": scored[n]["verdict"],
                                 "findings": scored[n]["findings"],
                                 "citations": scored[n]["citations"],
                                 "summary": scored[n]["summary"],
                                 "human_agrees": None, "human_note": ""} for n in needs_fix],
                  "random_clean": [{"card": n, "summary": scored[n]["summary"],
                                    "citations": scored[n]["citations"],
                                    "human_agrees": None, "human_note": ""}
                                   for n in clean_pick],
                  "check_seed": args.check_seed}
        ep = os.path.join(screen_dir, "check_export.json")
        with open(ep, "w") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"check export -> {os.path.relpath(ep, REPO)} "
              f"({len(needs_fix)} needs-fix + {len(clean_pick)} clean)")

    print(f"scored {len(verdicts)}/{len(expected)}, missing {len(missing)}, "
          f"invalid {len(problems)}")
    if agg:
        print(f"  weighted needs-fix {agg['needs_fix_rate']}, fabricated "
              f"{agg['fabricated_fact_rate']}, identity-wrong {agg['identity_wrong_rate']}, "
              f"wrong-paper {agg['wrong_paper_rate']}")
    if rerun:
        print(f"  rerun_list: {rerun}")
        sys.exit(2)


if __name__ == "__main__":
    main()
