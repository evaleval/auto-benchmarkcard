"""Stratum-weighted analysis over the S sample (judge + flags + screen).

Every reported number is stratum-weighted (combined ratio estimator over
card-level counts); raw pooled S rates are never emitted. CIs are card-
clustered stratified bootstrap with a shared replicate tensor, with an
effective-sample-size Wilson interval as the rare-event fallback (see
s_stats.py). Flag-vs-judge join semantics are lifted from
eval_gold_set.flag_quality (filled = judge-labeled supported/partial/
unsupported/supported_by_eee_only; flags on judge-unlabeled fields are
outside the universe; flaggable = not UNFLAGGABLE_FIELDS and not *.name;
amendments applied with the same stale-guard).

Usage:
  python scripts/analyze_s.py --sample eval/s150/sample.json \
      --judge eval/s150/judge_s150_source_parity_analysis.json --screen eval/s150/screen/screen_results.json \
      --out eval/s150/analysis_s150.json [--amendments f.json] [--seed 20260704] \
      [--B 5000] [--allow-partial] [--flags-from-report report.json] [--skip-card-md5]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import extract_card  # noqa: E402
from auto_benchmarkcard.validation_policy import UNFLAGGABLE_FIELDS  # noqa: E402

import s_stats  # noqa: E402
from check_frozen import check, file_md5  # noqa: E402
from judge_analysis_guard import validate_judge_analysis_frame  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

FILLED_STATUSES = ("supported", "partial", "unsupported", "supported_by_eee_only")
MISSED_INFO = ("yes_primary", "yes_eee_only", "yes")  # "yes" = legacy v31 label


def _load(p):
    with open(p) as f:
        return json.load(f)


def _flaggable(path):
    return path not in UNFLAGGABLE_FIELDS and not path.endswith(".name")


def build_records(sample, judge, screen_per_card, amendments, flags_by_card,
                  allow_partial, skip_card_md5):
    jcards = judge.get("per_card", {})
    amended_by_card = {}
    for key, am in (amendments or {}).items():
        cname, _, path = key.partition(":")
        if cname and path and am.get("amended"):
            amended_by_card.setdefault(cname, {})[path] = am

    records, missing_judge = [], []
    n_amended = 0
    for c in sample["cards"]:
        name = c["name"]
        jc = jcards.get(name)
        if not jc:
            missing_judge.append(name)
            continue
        verdicts = {v["path"]: dict(v) for v in jc["field_verdicts"]}
        for path, am in amended_by_card.get(name, {}).items():
            cur = verdicts.get(path, {}).get("status")
            if am.get("original") and cur != am["original"]:
                print(f"warning: SKIPPING stale amendment {name}:{path} "
                      f"(judge={cur!r}, amendment expects {am['original']!r})")
                continue
            if path in verdicts:
                verdicts[path]["status"] = am["amended"]
                n_amended += 1

        filled = {p: v for p, v in verdicts.items() if v["status"] in FILLED_STATUSES}
        ns = {p: v for p, v in verdicts.items() if v["status"] == "not_specified"}

        if name in flags_by_card:
            flagged = set(flags_by_card[name])
        else:
            card_p = os.path.join(REPO, c["corpus_card"])
            if not skip_card_md5 and file_md5(card_p) != c["corpus_card_md5"]:
                sys.exit(f"{name}: corpus card md5 mismatch vs sample file "
                         f"(corpus changed after sampling?)")
            card = extract_card(_load(card_p))
            flagged = set((card.get("flagged_fields") or {}).keys())

        pos_s = {p for p, v in filled.items() if v["status"] == "unsupported"}
        pos_l = {p for p, v in filled.items() if v["status"] in ("unsupported", "partial")}
        flagged_labeled = flagged & set(filled)
        rec = {
            "name": name, "stratum": c["stratum"], "provenance": c.get("provenance"),
            "filled": len(filled), "ns": len(ns),
            "supported": sum(v["status"] == "supported" for v in filled.values()),
            "eee_only": sum(v["status"] == "supported_by_eee_only" for v in filled.values()),
            "partial": sum(v["status"] == "partial" for v in filled.values()),
            "unsupported": len(pos_s),
            "specific": sum(v.get("specificity") == "specific" for v in filled.values()),
            "ns_missed": sum(v.get("info_in_source") in MISSED_INFO for v in ns.values()),
            "ns_missed_primary": sum(v.get("info_in_source") in ("yes_primary", "yes")
                                     for v in ns.values()),
            "risks": len(jc.get("risk_verdicts", [])),
            "risks_grounded": sum(r.get("relevant_and_grounded") == "yes"
                                  for r in jc.get("risk_verdicts", [])),
            "n_flags": len(flagged),
            "flagged_labeled": len(flagged_labeled),
            "flags_outside": len(flagged - set(filled)),
            "tp_strict": len(flagged_labeled & pos_s),
            "tp_lenient": len(flagged_labeled & pos_l),
            "pos_strict": len(pos_s),
            "pos_lenient": len(pos_l),
            "pos_strict_flaggable": sum(1 for p in pos_s if _flaggable(p)),
            "tp_strict_flaggable": sum(1 for p in flagged_labeled & pos_s if _flaggable(p)),
            "pos_lenient_flaggable": sum(1 for p in pos_l if _flaggable(p)),
            "tp_lenient_flaggable": sum(1 for p in flagged_labeled & pos_l if _flaggable(p)),
        }
        sc = screen_per_card.get(name)
        rec["has_screen"] = sc is not None
        if sc:
            rec["screen_verdict"] = sc["verdict"]
            rec["faithful"] = rec["unsupported"] == 0
            rec["right_source"] = bool(sc.get("identity_correct")) and \
                sc.get("paper_assessment") in ("correct", "correctly-none")
            rec["right_source_lenient"] = bool(sc.get("identity_correct")) and \
                sc.get("paper_assessment") in ("correct", "correctly-none",
                                               "plausible-unverified")
            rec["fabricated"] = sc.get("fabricated_fact") is True
            rec["identity_wrong"] = sc.get("identity_correct") is False
            rec["paper_wrong"] = sc.get("paper_assessment") == "wrong"
            rec["contaminated"] = sc.get("hf_repo_assessment") == "wrong-kept-CONTAMINATION"
            rec["recall_gap"] = sc.get("paper_assessment") == "missing-should-exist"
        records.append(rec)

    if missing_judge and not allow_partial:
        sys.exit(f"{len(missing_judge)} sampled cards missing from the judge file: "
                 f"{missing_judge}")
    return records, missing_judge, n_amended


def col(records, key):
    return np.array([r.get(key, 0) or 0 for r in records], float)


def main():
    ap = argparse.ArgumentParser(description="Stratum-weighted S analysis.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--screen", default=None)
    ap.add_argument("--amendments", default=None)
    ap.add_argument("--flags-from-report", default=None,
                    help="take flagged_paths per card from an eval_gold_set report "
                         "instead of the corpus cards (replay/reapply experiments)")
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--skip-card-md5", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    check()

    sample = _load(os.path.join(REPO, args.sample))
    judge = _load(os.path.join(REPO, args.judge))
    try:
        validate_judge_analysis_frame(judge)
    except ValueError as exc:
        sys.exit(f"invalid judge analysis artifact: {exc}")
    screen_per_card = {}
    if args.screen:
        screen_per_card = _load(os.path.join(REPO, args.screen)).get("per_card", {})
    amendments = _load(os.path.join(REPO, args.amendments)) if args.amendments else None
    if amendments:
        amendments = {k: v for k, v in amendments.items() if not k.startswith("_")}
    flags_by_card = {}
    if args.flags_from_report:
        rep = _load(os.path.join(REPO, args.flags_from_report))
        flags_by_card = {c["name"]: c.get("flagged_paths", []) for c in rep["cards"]}

    records, missing_judge, n_amended = build_records(
        sample, judge, screen_per_card, amendments, flags_by_card,
        args.allow_partial, args.skip_card_md5)
    if not records:
        sys.exit("no cards to analyze")

    weights = {h: m["weight"] for h, m in sample["strata"].items()}
    strata = s_stats.make_strata([r["stratum"] for r in records], weights)
    reps = s_stats.make_replicates(strata, B=args.B, seed=args.seed)

    m = {}

    def add(key, num_k, den_k, kind="field_ratio", records_=records, mask=None):
        num, den = col(records_, num_k), col(records_, den_k)
        if mask is not None:
            num, den = s_stats.restrict(num, mask), s_stats.restrict(den, mask)
        m[key] = s_stats.make_metric(key, num, den, strata, reps, kind=kind)

    ones = np.ones(len(records))
    for r in records:
        r["one"] = 1

    add("judge.unsupported_rate", "unsupported", "filled")
    add("judge.support_rate", "supported", "filled")
    m["judge.support_rate_incl_eee"] = s_stats.make_metric(
        "judge.support_rate_incl_eee", col(records, "supported") + col(records, "eee_only"),
        col(records, "filled"), strata, reps, kind="field_ratio")
    add("judge.partial_rate", "partial", "filled")
    add("judge.specific_rate", "specific", "filled")
    add("judge.completeness_miss_rate", "ns_missed", "ns")
    add("judge.completeness_miss_rate_primary", "ns_missed_primary", "ns")
    add("judge.risk_grounded_rate", "risks_grounded", "risks")
    m["judge.ns_rate_judged_fields"] = s_stats.make_metric(
        "judge.ns_rate_judged_fields", col(records, "ns"),
        col(records, "filled") + col(records, "ns"), strata, reps, kind="field_ratio")

    for variant in ("strict", "lenient"):
        tp, pos = f"tp_{variant}", f"pos_{variant}"
        add(f"flags.precision_{variant}", tp, "flagged_labeled")
        add(f"flags.recall_{variant}", tp, pos)
        add(f"flags.recall_flaggable_{variant}", f"tp_{variant}_flaggable",
            f"pos_{variant}_flaggable")
        fo_num = col(records, pos) - col(records, tp)
        fo_den = col(records, "filled") - col(records, "flagged_labeled")
        m[f"flags.false_omission_{variant}"] = s_stats.make_metric(
            f"flags.false_omission_{variant}", fo_num, fo_den, strata, reps,
            kind="field_ratio")
        p_boot = s_stats.bootstrap_ratio(col(records, tp), col(records, "flagged_labeled"),
                                         strata, reps)
        r_boot = s_stats.bootstrap_ratio(col(records, tp), col(records, pos), strata, reps)
        p_pt = m[f"flags.precision_{variant}"]["value"]
        r_pt = m[f"flags.recall_{variant}"]["value"]
        m[f"flags.f1_{variant}"] = s_stats.derived_metric(
            f"flags.f1_{variant}", lambda p, r: 2 * p * r / (p + r),
            (p_boot, r_boot),
            (p_pt if p_pt is not None else np.nan, r_pt if r_pt is not None else np.nan))
    # share of detected errors that sit on unflagged fields (the honest residual)
    m["flags.miss_share_strict"] = s_stats.make_metric(
        "flags.miss_share_strict",
        col(records, "pos_strict") - col(records, "tp_strict"),
        col(records, "pos_strict"), strata, reps, kind="field_ratio")
    m["flags.n_flags_outside_universe_raw"] = {
        "name": "flags.n_flags_outside_universe_raw",
        "value": float(col(records, "flags_outside").sum()),
        "estimator": "raw-count", "provenance": "S-weighted",
        "note": "flags on fields the judge did not label; excluded from the "
                "budget-match universe", "ci95": None, "ci_method": None}

    # budget-matched flags-vs-random over judge-labeled filled fields
    B_b = s_stats.bootstrap_totals(col(records, "flagged_labeled"), strata, reps)
    M_b = s_stats.bootstrap_totals(col(records, "filled"), strata, reps)
    U_b = s_stats.bootstrap_totals(col(records, "pos_strict"), strata, reps)
    C_b = s_stats.bootstrap_totals(col(records, "tp_strict"), strata, reps)
    B_p = sum(weights[r["stratum"]] * r["flagged_labeled"] for r in records)
    M_p = sum(weights[r["stratum"]] * r["filled"] for r in records)
    U_p = sum(weights[r["stratum"]] * r["pos_strict"] for r in records)
    C_p = sum(weights[r["stratum"]] * r["tp_strict"] for r in records)
    bm = {"universe": "judge-labeled filled fields",
          "budget_weighted_flagged_fields": B_p,
          "total_fields_weighted": M_p, "total_errors_weighted_strict": U_p,
          "caught_flag_weighted": C_p,
          "caught_random_expected_weighted": (B_p * U_p / M_p) if M_p else None}
    bm["catch_ratio_strict"] = s_stats.derived_metric(
        "budget.catch_ratio_strict",
        lambda b, mm, u, cc: (cc / b) / (u / mm),
        (B_b, M_b, U_b, C_b), (B_p, M_p, U_p, C_p))
    bm["catch_diff_weighted_strict"] = s_stats.derived_metric(
        "budget.catch_diff_weighted_strict",
        lambda b, mm, u, cc: cc - b * u / mm,
        (B_b, M_b, U_b, C_b), (B_p, M_p, U_p, C_p))
    bm["recall_flag_strict"] = s_stats.derived_metric(
        "budget.recall_flag_strict", lambda u, cc: cc / u, (U_b, C_b), (U_p, C_p))
    bm["budget_fraction"] = s_stats.derived_metric(
        "budget.budget_fraction", lambda b, mm: b / mm, (B_b, M_b), (B_p, M_p))

    # budget table (paper rows A/B/C): flag-guided review spends the budget on
    # flagged fields first (random flagged subset below the natural point, random
    # unflagged spillover above it); random review catches budget x error share
    # and leaves the remainder rate unchanged
    def policy_rows():
        if not (M_p and U_p and B_p):
            return None
        f0 = B_p / M_p
        u = U_p / M_p
        prec = C_p / B_p
        rows = {}
        for label, b in (("A", 0.01), ("B", 0.05), ("C", f0)):
            reviewed = b * M_p
            if b <= f0:
                caught = reviewed * prec
            else:
                caught = C_p + (reviewed - B_p) * (U_p - C_p) / (M_p - B_p)
            caught = min(caught, U_p)
            rows[label] = {
                "budget_fraction": b,
                "flags_catch_share": caught / U_p,
                "random_catch_share": b,
                "flags_for_remainder": (U_p - caught) / (M_p - reviewed)
                if M_p > reviewed else None,
                "random_for_remainder": u,
            }
        return rows

    bm["budget_table"] = policy_rows()
    m["budget_match"] = bm

    screened = [r for r in records if r["has_screen"]]
    if screened:
        mask = np.array([1.0 if r["has_screen"] else 0.0 for r in records])
        for key, pred in (("screen.needs_fix_rate", lambda r: r["screen_verdict"] == "needs-fix"),
                          ("screen.clean_rate", lambda r: r["screen_verdict"] == "clean"),
                          ("screen.minor_rate", lambda r: r["screen_verdict"] == "minor"),
                          ("screen.fabricated_rate", lambda r: r.get("fabricated") is True),
                          ("screen.wrong_identity_rate", lambda r: r.get("identity_wrong") is True),
                          ("screen.wrong_paper_rate", lambda r: r.get("paper_wrong") is True),
                          ("screen.contamination_rate", lambda r: r.get("contaminated") is True),
                          ("screen.recall_gap_rate", lambda r: r.get("recall_gap") is True),
                          ("joint.faithful_rate", lambda r: r.get("faithful") is True),
                          ("joint.right_source_rate", lambda r: r.get("right_source") is True),
                          ("joint.right_source_and_faithful",
                           lambda r: r.get("right_source") is True and r.get("faithful") is True),
                          ("joint.right_source_lenient_and_faithful",
                           lambda r: r.get("right_source_lenient") is True
                           and r.get("faithful") is True)):
            num = np.array([1.0 if (r["has_screen"] and pred(r)) else 0.0 for r in records])
            m[key] = s_stats.make_metric(key, num, mask, strata, reps, kind="proportion")
            m[key]["n_screened"] = len(screened)

    by_prov = {}
    for prov in sorted({r["provenance"] for r in records if r.get("provenance")}):
        mask = np.array([1.0 if r["provenance"] == prov else 0.0 for r in records])
        sub = {}
        for key, num_k, den_k in (("judge.unsupported_rate", "unsupported", "filled"),
                                  ("judge.support_rate", "supported", "filled"),
                                  ("judge.completeness_miss_rate", "ns_missed", "ns")):
            sub[key] = s_stats.make_metric(
                f"{key}[{prov}]", s_stats.restrict(col(records, num_k), mask),
                s_stats.restrict(col(records, den_k), mask), strata, reps,
                kind="field_ratio")
        sub["n_cards"] = int(mask.sum())
        by_prov[prov] = sub

    out = {
        "schema_version": 1,
        "label": os.path.splitext(os.path.basename(args.out))[0],
        "design": {
            "sample_file": args.sample, "sample_md5": file_md5(os.path.join(REPO, args.sample)),
            "judge_file": args.judge, "screen_file": args.screen,
            "tag": sample.get("tag"), "n_sampled": sample.get("n"),
            "n_analyzed": len(records), "n_missing_judge": len(missing_judge),
            "missing_judge": missing_judge,
            "n_screened": len(screened),
            "strata": sample["strata"], "seed_sample": sample.get("seed"),
            "seed_bootstrap": args.seed, "B": args.B,
            "amendments_applied": n_amended,
            "flags_source": args.flags_from_report or "corpus cards (flagged_fields)",
            "fpc_ignored": True,
            "notes": "combined ratio estimator; CIs card-clustered stratified bootstrap "
                     "with shared replicates; wilson-neff fallback for rare events; "
                     "FPC ignored (conservative at ~0.3 sampling fractions)",
        },
        "metrics": m,
        "by_provenance": by_prov,
    }
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    def show(key):
        mm = m.get(key)
        if not mm or mm.get("value") is None:
            return f"  {key:44s} n/a"
        ci = mm.get("ci95") or [None, None]
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci and ci[0] is not None else "[-]"
        return f"  {key:44s} {mm['value']:.4f} {ci_s} ({mm.get('ci_method')})"

    print(f"analyzed {len(records)} cards "
          f"({len(missing_judge)} missing judge, {len(screened)} screened)")
    for k in ("judge.unsupported_rate", "judge.support_rate", "flags.precision_strict",
              "flags.recall_strict", "flags.false_omission_strict",
              "joint.right_source_and_faithful"):
        print(show(k))
    cr = m["budget_match"]["catch_ratio_strict"]
    if cr.get("value") is not None:
        print(f"  budget.catch_ratio_strict                    {cr['value']:.2f} "
              f"ci {cr['ci95']}")
    print(f"analysis -> {args.out}")


if __name__ == "__main__":
    main()
