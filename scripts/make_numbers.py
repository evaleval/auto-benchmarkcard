"""Emit every paper number per the numbers_manifest.json contract.

The manifest (paper/fragments/numbers_manifest.json, maintained by the paper
chat) is THE contract: this script emits numbers.tex with one
\\defnum{key}{value} line per manifest key, sorted by key, trailing comment =
provenance | source | date. Formatting lives in the value (percent keys
include \\%, dollar keys \\$, compound P/R/F1 is one key, CIs one bracketed
value). Values not yet measurable are emitted as the literal n/a plus a
comment, never omitted.

Checks enforced here:
  coverage      every status=pending manifest key is emitted; in --freeze mode
                a pending key without a value is a build error
  curated       hard-assert the corpus sidecars record ZERO curated bindings
                (resolved_from never known_papers); build fails in --freeze
  dev-set lint  manifest keys tagged dev-set must not be used_in the abstract
                or intro contributions (headline positions)

No recomputation: values come from the artifact files that computed them.

Usage:
  python scripts/make_numbers.py --out eval/numbers.tex [--freeze]
      --manifest <path> [--analysis ...] [--calibration ...]
      [--corpus-stats ...] [--json eval/paper_numbers.json]
"""

import argparse
import json
import os
import sys
from datetime import date

from check_frozen import check, file_md5

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEV_JUDGE = "gold_set/judge_v31b.json"
DEV_REPORT_SHIP = "gold_set/report_fr2_nofill_v31b.json"
HF_PUBLISHED = "eval/corpus/hf_published.json"

HEADLINE_FILES = {"00-abstract.tex", "01-intro-p3-contributions.tex"}

DESIGN_LOCKED = {
    "schemaFields": ("40", "schema v2"),
    "schemaSections": ("5", "schema v2"),
    "designSampleSize": ("150", "eval design point 3 (2026-07-04)"),
    "designStratumFlagged": ("75", "eval design point 3"),
    "designStratumUnflagged": ("75", "eval design point 3"),
    "designRaterCount": ("3", "pre-annotation amendment 2026-07-17 (locked)"),
    "designGoldSetSize": ("22", "gold_set/manifest.json"),
}


def _load(rel_or_abs):
    p = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(REPO, rel_or_abs)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pct(v, nd=1):
    return None if v is None else f"{100 * v:.{nd}f}\\%"


def ci_pct(ci):
    if not ci or ci[0] is None:
        return None
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def ci_num(ci, nd=2):
    if not ci or ci[0] is None:
        return None
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def usd(v, nd=2):
    return None if v is None else f"\\${v:,.{nd}f}"


def num(v, nd=2):
    return None if v is None else f"{v:.{nd}f}"


def intc(v):
    return None if v is None else f"{int(round(v)):,}"


def build_values(analysis, calibration, cs, hf_pub):
    """manifest key -> (formatted value or None, source string, extra note)."""
    v = {}

    for key, (value, src) in DESIGN_LOCKED.items():
        v[key] = (value, src, None)

    judge = _load(DEV_JUDGE)
    if judge:
        a = judge["aggregate"]
        v["devUnsupportedRate"] = (pct(a["unsupported_rate"]), DEV_JUDGE,
                                   f"{a['unsupported']}/{a['n_filled']} judged prose fields")
        v["devSupportedRate"] = (pct(a["support_rate"]), DEV_JUDGE, None)
    ship = _load(DEV_REPORT_SHIP)
    if ship:
        q = ship["aggregate"]["flag_quality"]["unsupported"]
        v["devFlagPrf"] = (f"{q['precision']:.2f}/{q['recall']:.2f}/{q['f1']:.2f}",
                           DEV_REPORT_SHIP, f"ship config, {q['positives']} positives")

    if cs:
        src = "eval/corpus/corpus_stats.json"
        corp, ff, fl, bp = cs["corpus"], cs["field_fill"], cs["flags"], cs["binding_paths"]
        v["corpusBenchmarksAttempted"] = (intc(corp.get("attempted")), src, None)
        v["corpusCardsCompleted"] = (intc(corp["n_cards"]), src, None)
        v["corpusCompletionRate"] = (pct(corp.get("completion_rate")), src, None)
        v["corpusFieldFillRate"] = (pct(ff["fill_rate"], 0), src,
                                    "strict convention: non-empty AND not Not-specified")
        v["corpusFilledFieldsTotal"] = (intc(ff["n_filled"]), src, None)
        v["corpusFieldSlots"] = (intc(ff["n_slots"]), src, None)
        rt = cs.get("runtime")
        if rt:
            v["corpusRuntimeMedianMin"] = (num(rt["median_s"] / 60, 1), src,
                                           f"n={rt['n_generations']} generations")
            v["corpusRuntimeP90Min"] = (num(rt["p90_s"] / 60, 1), src, None)
        v["corpusFlaggedFields"] = (intc(fl["n_flagged_fields"]), src, None)
        v["corpusFlaggedCards"] = (intc(fl["cards_with_flags"]), src, None)
        v["corpusFlaggedFieldShare"] = (
            pct(fl["n_flagged_fields"] / ff["n_filled"]) if ff["n_filled"] else None,
            src, "flags over strict-filled fields")
        v["corpusZeroFlagCardShare"] = (
            pct((fl["cards_total"] - fl["cards_with_flags"]) / fl["cards_total"], 0)
            if fl["cards_total"] else None, src, None)

        n_bind = bp["n_run_dirs"]
        g = bp["grouped"]

        def bind(cnt_key, share_key, group):
            n = g.get(group, 0)
            v[cnt_key] = (intc(n), src, None)
            v[share_key] = (pct(n / n_bind) if n_bind else None, src,
                            f"over {n_bind} attempted bindings")

        bind("rq1BindSearchVerifiedCount", "rq1BindSearchVerifiedShare",
             "search_llm_verified")
        bind("rq1BindPresetVerifiedCount", "rq1BindPresetVerifiedShare",
             "pre_set_gate_verified")
        bind("rq1BindFallbackCount", "rq1BindFallbackShare", "deterministic_fallback")
        bind("rq1BindHonestNoneCount", "rq1BindHonestNoneShare", "no_paper_honest")
        sa = bp.get("subject_actions", {})
        v["rq1BindVetoCount"] = (intc(sum(n for a, n in sa.items() if "veto" in a)),
                                 src, "subject_overlap.action, veto-era sidecars only")
        v["rq1BindAbstainCount"] = (intc(sum(n for a, n in sa.items() if "abstain" in a)),
                                    src, "subject_overlap.action, veto-era sidecars only")

        rc = cs.get("regen_cost")
        if rc:
            v["costRegenCards"] = (intc(rc["n_cards_measured"]), src, None)
            v["costRegenPerCardUsd"] = (usd(rc["mean_cost_per_card_usd"]), src,
                                        f"{'/'.join(rc['providers'])} "
                                        f"{'/'.join(rc['quantizations'])}, composer serving only")
            v["costCorpusExtrapolatedUsd"] = (
                usd(rc["extrapolated"]["est_corpus_cost_usd"], 0), src,
                "EXTRAPOLATED: regen mean x corpus size; broad_run uninstrumented")

    v["corpusCardsPublished"] = ((intc(hf_pub["n_cards"]), HF_PUBLISHED,
                                  f"HF API checked {hf_pub.get('checked_on')}")
                                 if hf_pub else (None, HF_PUBLISHED,
                                                 "written after the HF v3 push"))

    if analysis:
        src = "analysis"
        m = analysis["metrics"]

        def mt(key):
            return m.get(key) or {}

        v["rq2SampleCards"] = (intc(analysis["design"]["n_analyzed"]), src, None)
        v["rq2JudgedFields"] = (intc(mt("judge.ns_rate_judged_fields")
                                     .get("counts", {}).get("den")), src,
                                "filled + not-specified judged fields")
        v["rq2SupportedRate"] = (pct(mt("judge.support_rate").get("value")), src, None)
        v["rq2SupportedRateCI"] = (ci_pct(mt("judge.support_rate").get("ci95")), src, None)
        v["rq2UnsupportedRate"] = (pct(mt("judge.unsupported_rate").get("value")), src, None)
        v["rq2UnsupportedRateCI"] = (ci_pct(mt("judge.unsupported_rate").get("ci95")),
                                     src, None)
        v["rq2RightSourceRate"] = (pct(mt("joint.right_source_rate").get("value")), src, None)
        v["rq2RightSourceRateCI"] = (ci_pct(mt("joint.right_source_rate").get("ci95")),
                                     src, None)
        v["rq2JointRightSourceFaithfulRate"] = (
            pct(mt("joint.right_source_and_faithful").get("value")), src, None)
        v["rq2JointRightSourceFaithfulRateCI"] = (
            ci_pct(mt("joint.right_source_and_faithful").get("ci95")), src, None)
        v["rq2WrongIdentityRate"] = (pct(mt("screen.wrong_identity_rate").get("value")),
                                     src, None)
        v["rq2WrongPaperRate"] = (pct(mt("screen.wrong_paper_rate").get("value")), src, None)
        v["rq2FabricatedRate"] = (pct(mt("screen.fabricated_rate").get("value")), src, None)
        v["rq2ContaminationRate"] = (pct(mt("screen.contamination_rate").get("value")),
                                     src, None)
        v["rq2RecallGapRate"] = (pct(mt("screen.recall_gap_rate").get("value")), src, None)
        v["rq3UnsupportedGivenFlagged"] = (pct(mt("flags.precision_strict").get("value"), 0),
                                           src, "equals strict flag precision")
        v["rq3UnsupportedGivenFlaggedCI"] = (
            ci_pct(mt("flags.precision_strict").get("ci95")), src, None)
        v["rq3FalseOmissionUnflagged"] = (
            pct(mt("flags.false_omission_strict").get("value")), src, None)
        v["rq3FalseOmissionUnflaggedCI"] = (
            ci_pct(mt("flags.false_omission_strict").get("ci95")), src, None)
        v["rq3ResidualErrorShare"] = (pct(mt("flags.miss_share_strict").get("value"), 0),
                                      src, "share of detected errors on unflagged fields")
        bt = (m.get("budget_match") or {}).get("budget_table") or {}
        for row in ("A", "B", "C"):
            r = bt.get(row) or {}
            v[f"rq3Budget{row}"] = (pct(r.get("budget_fraction"),
                                        1 if row == "C" else 0), src,
                                    "natural flag-rate operating point" if row == "C"
                                    else "picked budget row")
            v[f"rq3FlagsCatch{row}"] = (pct(r.get("flags_catch_share"), 0), src, None)
            v[f"rq3RandomCatch{row}"] = (pct(r.get("random_catch_share"), 0), src, None)
            v[f"rq3FlagsFor{row}"] = (pct(r.get("flags_for_remainder")), src, None)
            v[f"rq3RandomFor{row}"] = (pct(r.get("random_for_remainder")), src, None)

    if calibration:
        src = "eval/s150/human_validation/scores.json"
        v["rq2CalibrationFields"] = (
            intc(calibration.get("n_unique_items", calibration.get("n_items"))),
            src, "unique displayed fields; dual-arm items counted once")
        probability = calibration.get("probability_arm_corpus_weighted") or {}
        filled = probability.get("filled") or {}
        ns = probability.get("not_specified") or {}
        v["rq2JudgeHumanAgreement"] = (
            pct(filled.get("weighted_agreement"), 0), src,
            "probability arm only; filled fields; Hájek ratio estimate")
        v["rq2JudgeHumanAgreementCI"] = (
            ci_pct(filled.get(
                "weighted_agreement_card_bootstrap_sensitivity_interval95")), src,
            "low-cluster whole-card bootstrap sensitivity interval, conditional "
            "on the realized field draw")
        v["rq2JudgeHumanKappa"] = (
            num(filled.get("cohens_kappa")), src,
            "probability arm only; corpus-weighted filled-field labels")
        v["rq2JudgeHumanKappaCI"] = (
            ci_num(filled.get(
                "cohens_kappa_card_bootstrap_sensitivity_interval95")), src,
            "low-cluster whole-card bootstrap sensitivity interval, conditional "
            "on the realized field draw")
        v["rq2JudgeHumanAgreementNotSpecified"] = (
            pct(ns.get("weighted_agreement"), 0), src,
            "probability arm only; Not specified reverse-checks")
        v["rq2JudgeHumanKappaNotSpecified"] = (
            num(ns.get("cohens_kappa")), src,
            "probability arm only; Not specified reverse-checks")

        ia = (((calibration.get("inter_rater_reliability") or {})
               .get("probability_arm") or {}).get("filled") or {})
        v["rq2RaterIaa"] = (
            num(ia.get("fleiss_kappa")), src,
            "three-rater Fleiss kappa; filled probability-arm fields")
        v["rq2RaterAlpha"] = (
            num(ia.get("krippendorff_alpha_nominal")), src,
            "nominal alpha; filled probability-arm fields")

        error_raw = (((calibration.get("judge_vs_human_reference") or {})
                      .get("error_arm_raw") or {}).get("filled") or {})
        v["rq2JudgeHumanAgreementErrorArm"] = (
            pct(error_raw.get("raw_agreement"), 0), src,
            "conditional error-arm sample")
        v["rq2JudgeHumanKappaErrorArm"] = (
            num(error_raw.get("cohens_kappa")), src,
            "conditional error-arm sample")
        confirmations = calibration.get("error_arm_conditional_confirmation") or {}
        v["rq2UnsupportedCallConfirmation"] = (
            pct((confirmations.get("unsupported") or {})
                .get("design_weighted_confirmation_rate"), 0), src,
            "conditional on judge-unsupported")
        v["rq2PartialCallConfirmation"] = (
            pct((confirmations.get("partial") or {})
                .get("design_weighted_confirmation_rate"), 0), src,
            "conditional on judge-partial")

    return v


def main():
    ap = argparse.ArgumentParser(description="Emit numbers.tex per the manifest contract.")
    ap.add_argument(
        "--manifest",
        required=True,
        help="paper numbers manifest maintained with the manuscript",
    )
    ap.add_argument("--analysis", default="eval/s150/judge/summary.json")
    ap.add_argument("--calibration",
                    default="eval/s150/human_validation/scores.json")
    ap.add_argument("--corpus-stats", default="eval/corpus/corpus_stats.json")
    ap.add_argument("--out", default="eval/numbers.tex")
    ap.add_argument("--json", default="eval/paper_numbers.json")
    ap.add_argument("--freeze", action="store_true",
                    help="freeze mode: missing pending values and curated bindings "
                         "become build errors")
    args = ap.parse_args()
    check()

    manifest = _load(args.manifest)
    if not manifest:
        sys.exit(f"no manifest at {args.manifest}")
    mkeys = {k["key"]: k for k in manifest["keys"]}

    # dev-set provenance lint (headline positions)
    lint_bad = [k for k, mk in mkeys.items()
                if mk["provenance"] == "dev-set" and HEADLINE_FILES & set(mk["used_in"])]
    if lint_bad:
        sys.exit(f"provenance lint: dev-set keys used in headline files: {lint_bad}")

    analysis = _load(args.analysis)
    dry_analysis = bool(analysis) and (analysis.get("design", {}).get("tag") != "corpus_v3")
    if args.freeze and dry_analysis:
        sys.exit(f"--freeze but the analysis tag is "
                 f"{analysis.get('design', {}).get('tag')!r}, not corpus_v3")
    calibration = _load(args.calibration)
    cs = _load(args.corpus_stats)
    hf_pub = _load(HF_PUBLISHED)

    # curated-binding assert (the paper's zero-curated claim is measured)
    if cs:
        curated = cs["binding_paths"]["grouped"].get("curated_known_papers", 0)
        if curated:
            msg = (f"curated bindings present in corpus sidecars: {curated} "
                   f"(resolved_from=known_papers)")
            if args.freeze:
                sys.exit("BUILD FAILED: " + msg)
            print(f"warning: {msg} - dry-run corpus stand-in only", file=sys.stderr)

    # every manifest key is emitted; absent artifacts leave n/a defaults behind
    produced = build_values(analysis, calibration, cs, hf_pub)
    values = {k: (None, mk["source"], None) for k, mk in mkeys.items()}
    orphans = [k for k in produced if k not in mkeys]
    values.update({k: pv for k, pv in produced.items() if k in mkeys})

    # coverage: pending keys need values at freeze
    unfilled_pending = [k for k, mk in mkeys.items()
                        if mk["status"] == "pending" and values[k][0] is None]
    if args.freeze and unfilled_pending:
        sys.exit(f"BUILD FAILED: pending manifest keys without values at freeze: "
                 f"{sorted(unfilled_pending)}")
    for k in sorted(orphans):
        print(f"warning: orphan emitted key not in manifest: {k}", file=sys.stderr)

    today = date.today().isoformat()
    lines = []
    if not args.freeze:
        lines += ["% DRY RUN (no --freeze): corpus/S values are stand-ins or n/a.",
                  "% Do NOT \\input this file into the paper.", ""]
    for key in sorted(mkeys):
        mk = mkeys[key]
        val, src, note = values[key]
        comment = f"{mk['provenance']} | {src or mk['source']} | {today}"
        if mk["provenance"] == "dev-set":
            comment += " | DEV-SET, never a headline number"
        if val is None:
            val = "n/a"
            comment += " | pending" if not note else f" | pending: {note}"
        elif note:
            comment += f" | {note}"
        lines.append(f"\\defnum{{{key}}}{{{val}}} % {comment}")
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    record = {"generated_on": today, "freeze": args.freeze,
              "manifest": args.manifest,
              "manifest_md5": file_md5(args.manifest if os.path.isabs(args.manifest)
                                       else os.path.join(REPO, args.manifest)),
              "numbers": {k: {"value": values[k][0], "source": values[k][1],
                              "note": values[k][2],
                              "provenance": mkeys[k]["provenance"],
                              "status": "ok" if values[k][0] is not None else "pending"}
                          for k in sorted(mkeys)}}
    with open(os.path.join(REPO, args.json), "w") as f:
        json.dump(record, f, indent=2)

    n_ok = sum(1 for k in mkeys if values[k][0] is not None)
    print(f"{n_ok}/{len(mkeys)} keys with values -> {args.out} "
          f"({'FREEZE' if args.freeze else 'dry run'})")
    if unfilled_pending:
        print(f"  n/a pending ({len(unfilled_pending)}): "
              f"{', '.join(sorted(unfilled_pending)[:8])}"
              + (" ..." if len(unfilled_pending) > 8 else ""))


if __name__ == "__main__":
    main()
