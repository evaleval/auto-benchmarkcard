"""Calibrate per-field acceptance thresholds from the S150 judge frame.

Offline instrument, pure computation, no LLM. Joins the verifier-confirmed
material findings (eval/regression/findings_regression.json) onto the S150
per-field judge verdicts (eval/s150/judge_s150.json) by card+field and
estimates, for each judged content field, the design-weighted probability
that a filled value carries a material error.

Estimator: the S150 sample is flag-stratified (75 of 152 flagged cards,
75 of 378 unflagged), so every card carries its sampling weight N_s/n_s.
Per field f the material rate is the Hajek ratio estimator

    sum_c w_c * material_cf / sum_c w_c * filled_cf

over cards c, where filled_cf = 1 when the judge status is anything but
not_specified and material_cf = 1 when the pair is filled and carries at
least one confirmed-material finding (multiple findings on one pair count
once). The unsupported+partial rate uses judge status in {unsupported,
partial} as the numerator over the same denominator. Errors cluster within
cards, so confidence intervals come from a stratified card-cluster
bootstrap: within each stratum resample n_s cards with replacement, keep
the weights, recompute the ratio, take the 2.5/97.5 percentiles over the
replicates. Seedable and deterministic (random.Random, default seed
20260824, default 1000 replicates).

The field vocabulary is the locked 23-field content frame of the human
validation design (build_calibration_packet.py, design v7), restated
literally here per the same lock philosophy: benchmark_details.benchmark_type
and benchmark_details.contains stay excluded as structural fields. Findings
on fields outside this frame have no judged denominator; they are counted in
join diagnostics, never in the rates.

Caveats: calibration runs on the FROZEN pipeline's outputs; thresholds are
advisory until recomputed on v4 outputs; S=150 is development-visible;
conformal guarantees only transfer after recalibration on the new pipeline.
"""

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_FINDINGS = os.path.join(REPO, "eval", "regression", "findings_regression.json")
DEFAULT_JUDGE = os.path.join(REPO, "eval", "s150", "judge_s150.json")
DEFAULT_SAMPLE = os.path.join(REPO, "eval", "s150", "sample.json")
DEFAULT_OUT_DIR = os.path.join(REPO, "eval", "calibration")
DEFAULT_SEED = 20260824
DEFAULT_REPLICATES = 1000
ALPHAS = (0.05, 0.10, 0.20)

# Literal and locked: the 23-field content frame of human-validation design
# v7 (build_calibration_packet.py). Restated, not imported, so this
# instrument stays frozen even if field accounting elsewhere moves.
JUDGED_FIELD_PATHS = (
    "benchmark_details.domains",
    "benchmark_details.name",
    "benchmark_details.overview",
    "benchmark_details.similar_benchmarks",
    "data.annotation",
    "data.contamination_controls",
    "data.source",
    "ethical_and_legal_considerations.compliance_with_regulations",
    "ethical_and_legal_considerations.consent_procedures",
    "ethical_and_legal_considerations.privacy_and_anonymity",
    "methodology.baseline_results",
    "methodology.calculation",
    "methodology.human_baseline",
    "methodology.interpretation",
    "methodology.judge_score_consolidation",
    "methodology.methods",
    "methodology.validation",
    "methodology.validity_justification",
    "purpose_and_intended_users.audience",
    "purpose_and_intended_users.goal",
    "purpose_and_intended_users.limitations",
    "purpose_and_intended_users.out_of_scope_uses",
    "purpose_and_intended_users.tasks",
)

STRUCTURAL_EXCLUSIONS = (
    "benchmark_details.benchmark_type",
    "benchmark_details.contains",
)

FILLED_STATUSES = ("supported", "supported_by_eee_only", "partial", "unsupported")
RISKY_STATUSES = ("unsupported", "partial")
VALID_STATUSES = FILLED_STATUSES + ("not_specified",)
# Worst status wins when a card carries duplicate verdicts for one field.
STATUS_SEVERITY = {
    "unsupported": 4,
    "partial": 3,
    "supported_by_eee_only": 2,
    "supported": 1,
    "not_specified": 0,
}

CAVEATS = (
    "Calibration is computed on the FROZEN pipeline's outputs (corpus_v3 "
    "rundirs, S=150 judge frame). The v4 composer changes the error "
    "distribution, so every rate and flag below is advisory until "
    "recomputed on v4 outputs.",
    "S=150 is development-visible. It steered pipeline development, so "
    "these are not held-out estimates; treat them as optimistic.",
    "Conformal guarantees only transfer after recalibration on the new "
    "pipeline. Split-conformal / conformal risk control validity assumes "
    "calibration and deployment draws are exchangeable, and a pipeline "
    "change breaks that exchangeability.",
    "Most confirmed-material findings fall on fields OUTSIDE the 23-field "
    "judged frame (see join diagnostics), so this table does not bound "
    "whole-card risk.",
    "Zero-event fields get a degenerate bootstrap CI of [0%, 0%]. Read "
    "those with a rule-of-three upper bound (about 3/n_filled), not as "
    "evidence of zero risk.",
)

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def normalize_field_path(path):
    """Map a finding field_path onto the section.field vocabulary.

    Strips bracketed indices/keys and truncates to the first two dotted
    segments, so data.size_breakdown.train and benchmark_details.resources[0]
    land on their parent field. Paths with fewer than two segments stay as-is
    and simply never match a judged field.
    """
    cleaned = _BRACKET_RE.sub("", path).strip()
    segments = [s for s in cleaned.split(".") if s]
    return ".".join(segments[:2])


def load_json(path, label):
    if not os.path.exists(path):
        raise SystemExit(f"calibrate_thresholds: missing {label} file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_frame(judge, sample, findings):
    """Join judge verdicts, weights, and findings into a per-card frame.

    Returns (cards, diagnostics). Each card is a dict with name, stratum,
    weight, and per judged field the filled / risky / material indicators.
    """
    sample_cards = {c["name"]: c for c in sample["cards"]}
    judge_cards = judge["per_card"]
    if set(sample_cards) != set(judge_cards):
        only_sample = sorted(set(sample_cards) - set(judge_cards))[:5]
        only_judge = sorted(set(judge_cards) - set(sample_cards))[:5]
        raise SystemExit(
            "calibrate_thresholds: sample and judge card sets differ "
            f"(sample-only {only_sample}, judge-only {only_judge})"
        )

    field_set = set(JUDGED_FIELD_PATHS)

    material_pairs = {}
    diagnostics = {
        "n_findings_total": len(findings["findings"]),
        "n_findings_joined": 0,
        "n_findings_nonjudged_field": 0,
        "n_findings_on_unfilled_pair": 0,
        "n_material_pairs": 0,
        "nonjudged_field_paths": {},
        "findings_on_unfilled_pairs": [],
        "duplicate_verdicts": [],
    }
    for finding in findings["findings"]:
        field = normalize_field_path(finding["field_path"])
        card = finding["card"]
        if card not in judge_cards:
            raise SystemExit(
                f"calibrate_thresholds: finding card not in judge frame: {card}"
            )
        if field not in field_set:
            diagnostics["n_findings_nonjudged_field"] += 1
            bucket = diagnostics["nonjudged_field_paths"]
            bucket[field] = bucket.get(field, 0) + 1
            continue
        material_pairs.setdefault((card, field), 0)
        material_pairs[(card, field)] += 1

    cards = []
    for name in sorted(judge_cards):
        entry = judge_cards[name]
        weight = sample_cards[name]["weight"]
        stratum = sample_cards[name]["stratum"]
        statuses = {}
        for verdict in entry["field_verdicts"]:
            path = verdict["path"]
            if path not in field_set:
                continue
            status = verdict["status"]
            if status not in VALID_STATUSES:
                raise SystemExit(
                    f"calibrate_thresholds: unknown judge status {status!r} "
                    f"on {name}:{path}"
                )
            prev = statuses.get(path)
            if prev is not None:
                diagnostics["duplicate_verdicts"].append(
                    {"card": name, "field": path, "statuses": [prev, status]}
                )
                if STATUS_SEVERITY[status] <= STATUS_SEVERITY[prev]:
                    continue
            statuses[path] = status

        filled = {}
        risky = {}
        material = {}
        for field in JUDGED_FIELD_PATHS:
            status = statuses.get(field, "not_specified")
            is_filled = status in FILLED_STATUSES
            filled[field] = is_filled
            risky[field] = status in RISKY_STATUSES
            has_finding = (name, field) in material_pairs
            if has_finding and not is_filled:
                diagnostics["n_findings_on_unfilled_pair"] += material_pairs[
                    (name, field)
                ]
                diagnostics["findings_on_unfilled_pairs"].append(
                    {"card": name, "field": field}
                )
                has_finding = False
            material[field] = has_finding
        cards.append(
            {
                "name": name,
                "stratum": stratum,
                "weight": weight,
                "filled": filled,
                "risky": risky,
                "material": material,
            }
        )

    diagnostics["n_findings_joined"] = (
        sum(material_pairs.values()) - diagnostics["n_findings_on_unfilled_pair"]
    )
    diagnostics["n_material_pairs"] = sum(
        1 for card in cards for field in JUDGED_FIELD_PATHS if card["material"][field]
    )
    return cards, diagnostics


def weighted_rates(cards, fields=JUDGED_FIELD_PATHS):
    """Hajek point estimates per field.

    Returns {field: {material_rate, unsupported_partial_rate, n_filled,
    n_filled_weighted, n_material_pairs}} with None rates when the field is
    filled nowhere.
    """
    out = {}
    for field in fields:
        den = 0.0
        mat = 0.0
        risky = 0.0
        n_filled = 0
        n_material = 0
        for card in cards:
            if not card["filled"][field]:
                continue
            w = card["weight"]
            den += w
            n_filled += 1
            if card["material"][field]:
                mat += w
                n_material += 1
            if card["risky"][field]:
                risky += w
        out[field] = {
            "material_rate": (mat / den) if den else None,
            "unsupported_partial_rate": (risky / den) if den else None,
            "n_filled": n_filled,
            "n_filled_weighted": den,
            "n_material_pairs": n_material,
        }
    return out


def _percentile(sorted_values, q):
    """Linear-interpolation percentile on an already sorted list."""
    if not sorted_values:
        return None
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_cis(cards, fields=JUDGED_FIELD_PATHS, replicates=DEFAULT_REPLICATES,
                  seed=DEFAULT_SEED):
    """Stratified card-cluster bootstrap 95% CIs for both rates.

    Resamples cards with replacement within each stratum (cluster = card, so
    within-card error correlation survives the resample), keeps the design
    weights, recomputes the Hajek ratio per replicate. Replicates where a
    field's denominator is empty are dropped for that field and counted.
    """
    strata = {}
    for index, card in enumerate(cards):
        strata.setdefault(card["stratum"], []).append(index)
    strata = {name: indices for name, indices in sorted(strata.items())}

    samples = {field: {"material": [], "risky": []} for field in fields}
    empty = {field: 0 for field in fields}
    rng = random.Random(seed)
    for _ in range(replicates):
        drawn = []
        for indices in strata.values():
            drawn.extend(rng.choices(indices, k=len(indices)))
        acc = {field: [0.0, 0.0, 0.0] for field in fields}
        for index in drawn:
            card = cards[index]
            w = card["weight"]
            for field in fields:
                if not card["filled"][field]:
                    continue
                slot = acc[field]
                slot[0] += w
                if card["material"][field]:
                    slot[1] += w
                if card["risky"][field]:
                    slot[2] += w
        for field in fields:
            den, mat, risky = acc[field]
            if den == 0:
                empty[field] += 1
                continue
            samples[field]["material"].append(mat / den)
            samples[field]["risky"].append(risky / den)

    out = {}
    for field in fields:
        mat = sorted(samples[field]["material"])
        risky = sorted(samples[field]["risky"])
        out[field] = {
            "material_ci95": [_percentile(mat, 0.025), _percentile(mat, 0.975)],
            "unsupported_partial_ci95": [
                _percentile(risky, 0.025),
                _percentile(risky, 0.975),
            ],
            "n_empty_replicates": empty[field],
        }
    return out


def _round(value, digits=6):
    if value is None:
        return None
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return round(value, digits)


def calibrate(judge, sample, findings, seed=DEFAULT_SEED,
              replicates=DEFAULT_REPLICATES, now=None):
    cards, diagnostics = build_frame(judge, sample, findings)
    points = weighted_rates(cards)
    cis = bootstrap_cis(cards, replicates=replicates, seed=seed)

    per_field = {}
    for field in JUDGED_FIELD_PATHS:
        point = points[field]
        ci = cis[field]
        rate = point["material_rate"]
        ci_lo = ci["material_ci95"][0]
        per_field[field] = {
            "material_rate": _round(rate),
            "ci95": _round(ci["material_ci95"]),
            "unsupported_partial_rate": _round(point["unsupported_partial_rate"]),
            "unsupported_partial_ci95": _round(ci["unsupported_partial_ci95"]),
            "n_filled": point["n_filled"],
            "n_filled_weighted": _round(point["n_filled_weighted"]),
            "n_material_pairs": point["n_material_pairs"],
            "exceeds_alpha": {
                f"{alpha:.2f}": (rate is not None and rate > alpha)
                for alpha in ALPHAS
            },
            "ci_lower_exceeds_alpha": {
                f"{alpha:.2f}": (ci_lo is not None and ci_lo > alpha)
                for alpha in ALPHAS
            },
            "n_empty_replicates": ci["n_empty_replicates"],
        }

    generated_at = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = {
        "generated_at": generated_at,
        "design": {
            "sample": "eval/s150/sample.json",
            "judge": "eval/s150/judge_s150.json",
            "findings": "eval/regression/findings_regression.json",
            "strata": sample["strata"],
            "estimator": (
                "Hajek weighted ratio per field: sum(w_c * indicator) / "
                "sum(w_c * filled), weights = stratum N_s/n_s from "
                "sample.json. 95% CIs from stratified card-cluster "
                "bootstrap (resample cards with replacement within each "
                "stratum, percentile method)."
            ),
            "field_frame": (
                "23-field content frame of human-validation design v7; "
                "structural exclusions: " + ", ".join(STRUCTURAL_EXCLUSIONS)
            ),
            "seed": seed,
            "replicates": replicates,
            "alphas": list(ALPHAS),
        },
        "per_field": per_field,
        "join_diagnostics": diagnostics,
        "notes": list(CAVEATS),
    }
    return result


def ranked_fields(result):
    def sort_key(item):
        field, stats = item
        rate = stats["material_rate"]
        risky = stats["unsupported_partial_rate"]
        return (
            -(rate if rate is not None else -1.0),
            -(risky if risky is not None else -1.0),
            field,
        )

    return sorted(result["per_field"].items(), key=sort_key)


def format_table(result):
    lines = [
        "| rank | field | material rate | 95% CI | unsup+partial | n filled "
        "| >0.05 | >0.10 | >0.20 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, (field, stats) in enumerate(ranked_fields(result), start=1):
        rate = stats["material_rate"]
        ci = stats["ci95"]
        risky = stats["unsupported_partial_rate"]
        exceeds = stats["exceeds_alpha"]

        def pct(value):
            return "n/a" if value is None else f"{100 * value:.1f}%"

        ci_text = (
            "n/a"
            if ci[0] is None
            else f"[{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"
        )
        lines.append(
            f"| {rank} | {field} | {pct(rate)} | {ci_text} | {pct(risky)} "
            f"| {stats['n_filled']} "
            f"| {'yes' if exceeds['0.05'] else 'no'} "
            f"| {'yes' if exceeds['0.10'] else 'no'} "
            f"| {'yes' if exceeds['0.20'] else 'no'} |"
        )
    return "\n".join(lines)


def write_report(result, path):
    diag = result["join_diagnostics"]
    caveats = "\n".join(f"- {c}" for c in result["notes"])
    nonjudged = sorted(
        diag["nonjudged_field_paths"].items(), key=lambda kv: (-kv[1], kv[0])
    )
    nonjudged_text = "\n".join(f"- `{p or '(top level)'}`: {n}" for p, n in nonjudged)
    body = f"""# Per-field acceptance risk calibration (S150, frozen pipeline)

Generated {result['generated_at']} by scripts/calibrate_thresholds.py
(seed {result['design']['seed']}, {result['design']['replicates']} bootstrap
replicates).

## Caveats, read these first

{caveats}

## Estimator

{result['design']['estimator']}

Field frame: {result['design']['field_frame']}. A filled value is any judge
status other than not_specified; a material pair is a filled card-field with
at least one verifier-confirmed material finding (multiple findings on one
pair count once). Sampling design: flagged {result['design']['strata']['flagged']['n']}/{result['design']['strata']['flagged']['N']} (weight {result['design']['strata']['flagged']['weight']}),
unflagged {result['design']['strata']['unflagged']['n']}/{result['design']['strata']['unflagged']['N']} (weight {result['design']['strata']['unflagged']['weight']}).

## Ranked field risk

Fields whose material rate exceeds alpha are where filling was empirically
riskier than the target family error rate; the pipeline should prefer
abstention or flagging there when evidence is thin.

{format_table(result)}

The unsupported+partial column is the judge-based upper funnel (weak or
missing support, not necessarily a confirmed error); the material rate is
the verifier-confirmed floor.

## Join diagnostics

- findings total: {diag['n_findings_total']}
- joined onto judged filled pairs: {diag['n_findings_joined']} ({diag['n_material_pairs']} distinct card-field pairs)
- on fields outside the judged frame (no denominator, excluded): {diag['n_findings_nonjudged_field']}
- on judged but unfilled pairs (excluded, listed in thresholds.json): {diag['n_findings_on_unfilled_pair']}

Non-judged finding fields:

{nonjudged_text}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate per-field acceptance thresholds (offline)."
    )
    parser.add_argument("--findings", default=DEFAULT_FINDINGS)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--now", default=None, help="Override generated_at (ISO timestamp)."
    )
    args = parser.parse_args(argv)

    findings = load_json(args.findings, "findings")
    judge = load_json(args.judge, "judge")
    sample = load_json(args.sample, "sample")

    result = calibrate(
        judge,
        sample,
        findings,
        seed=args.seed,
        replicates=args.replicates,
        now=args.now,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    thresholds_path = os.path.join(args.out_dir, "thresholds.json")
    report_path = os.path.join(args.out_dir, "REPORT.md")
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_report(result, report_path)

    print(f"wrote {thresholds_path}")
    print(f"wrote {report_path}")
    print()
    print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
