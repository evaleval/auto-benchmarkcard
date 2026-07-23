"""Score the three-rater human validation of the S150 field judge.

The packet has two deliberately different sampling arms:

* ``error`` estimates confirmation conditional on the automatic judge having
  called a field unsupported or partial.
* ``probability`` is the only arm used for an overall, corpus-weighted
  judge-versus-human estimate.

An item can belong to both arms.  It is rated once and contributes to each
arm's estimand.  This script intentionally does not construct a combined-arm
Horvitz--Thompson estimate.

Three raters are required.  A human reference label is the majority label
when at least two raters agree.  The only cases sent to blind adjudication are
true three-way splits.  Until all such splits have been adjudicated, the
script writes the reliability results and blind worklist but suppresses every
judge-versus-human result so a changing denominator cannot become an analysis
choice.

Canonical per-item design fields in key.json:

``arm_membership``
    ``error``, ``probability``, or pipe-delimited ``error|probability``.
``error_status_stratum``, ``error_row_pi_within_pool``
    Required for error-arm rows.
``prob_flag_stratum``, ``prob_card_pi_within_s``,
``prob_field_pi_given_card``, ``prob_row_pi_within_s``,
``s_to_corpus_weight``
    Required for probability-arm rows.  The corpus analysis weight is
    ``s_to_corpus_weight / prob_row_pi_within_s``.  The recorded, per-row
    field probability is used; it is never assumed to be constant.

Usage:

  python scripts/score_calibration.py \
    --ratings R1.csv R2.csv R3.csv --key key.json \
    --out calibration_scores.json

After a three-way split worklist has been completed, rerun with:

  --adjudicated calibration_adjudication_worklist.csv
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

from check_frozen import check


REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

FILLED_LABELS = ["supported", "supported_by_eee_only", "partial", "unsupported"]
NS_LABELS = ["yes_primary", "yes_eee_only", "no"]
HUMAN_LABEL_ALIASES = {
    # Current participant-facing wording.
    "supported_with_registry": "supported_by_eee_only",
    "yes_source": "yes_primary",
    "yes_registry": "yes_eee_only",
    # Accepted only so already-created local drafts remain readable.
    "supported_registry_only": "supported_by_eee_only",
    "yes_outside_registry": "yes_primary",
    "yes_registry_only": "yes_eee_only",
}
PUBLIC_COLUMNS = ("item_id", "card", "field_path", "kind", "field_value")
ARMS = {"error", "probability"}
B_DEFAULT = 5000


def _path(path):
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def normalize_human_label(label):
    """Map participant-facing labels to the frozen internal judge space."""
    label = (label or "").strip()
    return HUMAN_LABEL_ALIASES.get(label, label)


def read_ratings(path):
    """Compatibility helper returning non-empty normalized labels by item id."""
    out = {}
    with open(_path(path), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = normalize_human_label(row.get("human_label"))
            if label:
                out[row["item_id"]] = label
    return out


def read_rating_table(path):
    """Read one completed rating CSV and reject duplicate item identifiers."""
    rows = {}
    with open(_path(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = set(PUBLIC_COLUMNS) | {"human_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")
        for line_no, raw in enumerate(reader, 2):
            item_id = (raw.get("item_id") or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{line_no}: empty item_id")
            if item_id in rows:
                raise ValueError(f"{path}:{line_no}: duplicate item_id {item_id}")
            rows[item_id] = {
                "label": normalize_human_label(raw.get("human_label")),
                "public": {c: (raw.get(c) or "") for c in PUBLIC_COLUMNS},
            }
    return rows


def require_distinct_rating_files(paths):
    """Reject accidentally supplying one returned file for multiple raters."""
    resolved = [os.path.realpath(_path(path)) for path in paths]
    if len(resolved) != 3 or len(set(resolved)) != 3:
        raise ValueError("ratings must identify three distinct returned files")


def read_adjudications(path):
    """Read item_id plus any supported adjudication-label column."""
    if not path:
        return {}
    out = {}
    with open(_path(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        label_col = next((c for c in ("adjudicated_label", "label", "human_label")
                          if c in fields), None)
        if "item_id" not in fields or label_col is None:
            raise ValueError(f"{path}: expected item_id and adjudicated_label columns")
        for line_no, row in enumerate(reader, 2):
            item_id = (row.get("item_id") or "").strip()
            label = normalize_human_label(row.get(label_col))
            if not item_id or not label:
                continue
            if item_id in out:
                raise ValueError(f"{path}:{line_no}: duplicate item_id {item_id}")
            out[item_id] = label
    return out


def cohens_kappa(pairs, labels):
    """Return (kappa, observed agreement) for unweighted label pairs."""
    n = len(pairs)
    if not n:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if pe >= 1.0:
        return None, po
    return (po - pe) / (1 - pe), po


def fleiss_kappa(rating_sets, labels):
    """Fleiss' kappa for complete, fixed-size nominal ratings."""
    if not rating_sets:
        return None
    n_raters = len(rating_sets[0])
    if n_raters < 2 or any(len(x) != n_raters for x in rating_sets):
        raise ValueError("Fleiss kappa requires a fixed number of complete ratings")
    counts = Counter(label for item in rating_sets for label in item)
    n_items = len(rating_sets)
    p_bar = sum((sum(Counter(item)[label] ** 2 for label in labels) - n_raters)
                / (n_raters * (n_raters - 1)) for item in rating_sets) / n_items
    p_e = sum((counts[label] / (n_items * n_raters)) ** 2 for label in labels)
    return None if p_e >= 1.0 else (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(rating_sets, labels):
    """Krippendorff's alpha with nominal distance and complete ratings."""
    if not rating_sets:
        return None
    n_raters = len(rating_sets[0])
    if n_raters < 2 or any(len(x) != n_raters for x in rating_sets):
        raise ValueError("Krippendorff alpha requires complete fixed-size ratings")
    label_set = set(labels)
    if any(label not in label_set for item in rating_sets for label in item):
        raise ValueError("rating outside supplied Krippendorff label space")
    n_items = len(rating_sets)
    observed_disagree = 0
    for item in rating_sets:
        c = Counter(item)
        observed_disagree += n_raters * n_raters - sum(v * v for v in c.values())
    d_o = observed_disagree / (n_items * n_raters * (n_raters - 1))
    all_counts = Counter(label for item in rating_sets for label in item)
    n_total = n_items * n_raters
    if n_total < 2:
        return None
    expected_disagree = sum(all_counts[label] * (n_total - all_counts[label])
                            for label in labels)
    d_e = expected_disagree / (n_total * (n_total - 1))
    return None if d_e <= 0 else 1 - d_o / d_e


def _labels_for_kind(kind):
    if kind == "filled":
        return FILLED_LABELS
    if kind == "not_specified":
        return NS_LABELS
    raise ValueError(f"unknown item kind {kind!r}")


def parse_arm_membership(row):
    """Parse canonical arm_membership, accepting a few migration aliases."""
    raw = row.get("arm_membership", row.get("arms", row.get("arm", row.get("sample"))))
    if isinstance(raw, dict):
        values = [name for name, included in raw.items() if included]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif raw is None:
        values = []
    else:
        text = str(raw).strip()
        for separator in ("|", ",", ";"):
            text = text.replace(separator, " ")
        values = text.split()
    aliases = {"risk": "error", "clean": "probability", "random": "probability"}
    arms = {aliases.get(str(v).strip(), str(v).strip()) for v in values if str(v).strip()}
    invalid = arms - ARMS
    if invalid or not arms:
        raise ValueError(f"invalid or missing arm_membership for {row.get('item_id')}: {raw!r}")
    return arms


def _positive_float(row, name, aliases=(), probability=False):
    """Get a required numeric design value, including nested design aliases."""
    containers = [row]
    for key in ("design", "probability_design", "error_design"):
        if isinstance(row.get(key), dict):
            containers.append(row[key])
    names = (name,) + tuple(aliases)
    value = None
    for container in containers:
        for candidate in names:
            if container.get(candidate) is not None:
                value = container[candidate]
                break
        if value is not None:
            break
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{row.get('item_id')}: missing/non-numeric {name}") from None
    if not math.isfinite(value) or value <= 0 or (probability and value > 1):
        bound = "in (0, 1]" if probability else "> 0"
        raise ValueError(f"{row.get('item_id')}: {name} must be {bound}, got {value}")
    return value


def validate_key(key):
    """Validate hidden labels and all arm-specific sampling metadata."""
    if key.get("n_raters") not in (None, 3):
        raise ValueError(f"key declares {key.get('n_raters')} raters; exactly three required")
    raw_rows = key.get("key")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("key.json must contain a non-empty key list")
    seen = set()
    seen_fields = {}
    rows = []
    probability_card_design = {}
    for raw in raw_rows:
        row = dict(raw)
        item_id = (row.get("item_id") or "").strip()
        if not item_id or item_id in seen:
            raise ValueError(f"empty or duplicate key item_id: {item_id!r}")
        seen.add(item_id)
        labels = _labels_for_kind(row.get("kind"))
        if row.get("judge_label") not in labels:
            raise ValueError(f"{item_id}: invalid judge label {row.get('judge_label')!r}")
        if not row.get("card") or not row.get("field_path"):
            raise ValueError(f"{item_id}: key must include card and field_path")
        value_hash = row.get("field_value_sha256")
        value_chars = row.get("field_value_chars")
        if value_hash is not None or value_chars is not None:
            if not isinstance(value_hash, str) or len(value_hash) != 64 or any(
                    char not in "0123456789abcdef" for char in value_hash.lower()):
                raise ValueError(f"{item_id}: invalid field_value_sha256")
            if isinstance(value_chars, bool) or not isinstance(value_chars, int) or value_chars < 0:
                raise ValueError(f"{item_id}: invalid field_value_chars")
        field_key = (row["card"], row["field_path"])
        if field_key in seen_fields:
            raise ValueError(
                f"{item_id}: duplicate card/field pair already present as "
                f"{seen_fields[field_key]}; merge arm memberships instead")
        seen_fields[field_key] = item_id
        row["arms"] = parse_arm_membership(row)
        if "error" in row["arms"]:
            status = row.get("error_status_stratum")
            if status not in {"unsupported", "partial"}:
                raise ValueError(f"{item_id}: error_status_stratum must be unsupported/partial")
            if row["judge_label"] != status:
                raise ValueError(f"{item_id}: error stratum and judge label disagree")
            row["error_pi"] = _positive_float(
                row, "error_row_pi_within_pool", ("error_inclusion_prob",), True)
        if "probability" in row["arms"]:
            stratum = row.get("prob_flag_stratum", row.get("outer_stratum"))
            if not stratum:
                raise ValueError(f"{item_id}: missing prob_flag_stratum")
            row["prob_stratum"] = str(stratum)
            row["prob_card_pi"] = _positive_float(
                row, "prob_card_pi_within_s", ("card_inclusion_prob", "pi_card"), True)
            row["prob_field_pi"] = _positive_float(
                row, "prob_field_pi_given_card", ("field_inclusion_prob", "pi_field"), True)
            row["prob_row_pi"] = _positive_float(
                row, "prob_row_pi_within_s", ("probability_inclusion_prob", "row_inclusion_prob"), True)
            row["outer_weight"] = _positive_float(
                row, "s_to_corpus_weight", ("outer_weight", "s_outer_weight"))
            product = row["prob_card_pi"] * row["prob_field_pi"]
            # Builders may serialize probabilities to six decimal places.  The
            # point estimate still uses the recorded row probability; this
            # tolerance only distinguishes harmless serialization from a
            # genuinely inconsistent design record.
            if not math.isclose(row["prob_row_pi"], product, rel_tol=1e-6, abs_tol=5e-7):
                raise ValueError(
                    f"{item_id}: prob_row_pi_within_s={row['prob_row_pi']} does not equal "
                    f"card_pi*field_pi={product}")
            row["analysis_weight"] = row["outer_weight"] / row["prob_row_pi"]
            card_key = row["card"]
            card_design = (row["prob_stratum"], row["prob_card_pi"], row["outer_weight"])
            if card_key in probability_card_design and probability_card_design[card_key] != card_design:
                raise ValueError(f"{item_id}: inconsistent probability design within card {card_key}")
            probability_card_design[card_key] = card_design
        rows.append(row)
    return rows


def validate_and_join(rows, rating_tables, adjudications=None):
    """Validate three identical complete rating tables and join hidden fields."""
    if len(rating_tables) != 3:
        raise ValueError(f"exactly three rating CSVs are required, got {len(rating_tables)}")
    key_by_id = {row["item_id"]: row for row in rows}
    expected = set(key_by_id)
    for index, table in enumerate(rating_tables, 1):
        ids = set(table)
        if ids != expected:
            missing, extra = sorted(expected - ids), sorted(ids - expected)
            raise ValueError(f"rater {index} item set mismatch; missing={missing}, extra={extra}")
    base = rating_tables[0]
    joined = []
    for item_id in sorted(expected):
        public = base[item_id]["public"]
        for index, table in enumerate(rating_tables[1:], 2):
            if table[item_id]["public"] != public:
                raise ValueError(f"rater {index} public row differs for {item_id}")
        key_row = key_by_id[item_id]
        if public["item_id"] != item_id:
            raise ValueError(f"{item_id}: rating item_id differs from its row key")
        for column, key_name in (("card", "card"), ("field_path", "field_path"),
                                 ("kind", "kind")):
            if public[column] != str(key_row[key_name]):
                raise ValueError(f"{item_id}: rating {column} disagrees with hidden key")
        expected_hash = key_row.get("field_value_sha256")
        expected_chars = key_row.get("field_value_chars")
        if expected_hash is not None:
            visible_value = public["field_value"]
            actual_hash = hashlib.sha256(visible_value.encode("utf-8")).hexdigest()
            if len(visible_value) != expected_chars or actual_hash != expected_hash:
                raise ValueError(
                    f"{item_id}: returned field_value differs from the frozen packet"
                )
        labels = [table[item_id]["label"] for table in rating_tables]
        valid = set(_labels_for_kind(key_row["kind"]))
        for index, label in enumerate(labels, 1):
            if not label:
                raise ValueError(f"{item_id}: rater {index} label is blank")
            if label not in valid:
                raise ValueError(f"{item_id}: invalid rater {index} label {label!r}")
        counts = Counter(labels)
        majority = next((label for label, count in counts.items() if count >= 2), None)
        true_split = majority is None
        row = dict(key_row)
        row.update({"public": public, "ratings": labels, "majority": majority,
                    "true_three_way_split": true_split})
        joined.append(row)

    adjudications = adjudications or {}
    split_ids = {row["item_id"] for row in joined if row["true_three_way_split"]}
    extras = set(adjudications) - split_ids
    if extras:
        raise ValueError(f"adjudication is allowed only for true three-way splits: {sorted(extras)}")
    for row in joined:
        adjudicated = adjudications.get(row["item_id"])
        if adjudicated and adjudicated not in _labels_for_kind(row["kind"]):
            raise ValueError(f"{row['item_id']}: invalid adjudicated label {adjudicated!r}")
        row["adjudicated"] = adjudicated
        row["human_reference"] = row["majority"] or adjudicated
    return joined


def reliability_block(rows):
    """Three-rater nominal reliability, kept separate by label space."""
    out = {}
    for kind in ("filled", "not_specified"):
        selected = [row for row in rows if row["kind"] == kind]
        sets = [row["ratings"] for row in selected]
        out[kind] = {
            "n_items": len(selected),
            "unanimous": sum(len(set(x)) == 1 for x in sets),
            "majority_2_of_3": sum(len(set(x)) == 2 for x in sets),
            "true_three_way_splits": sum(len(set(x)) == 3 for x in sets),
            "fleiss_kappa": fleiss_kappa(sets, _labels_for_kind(kind)),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(
                sets, _labels_for_kind(kind)),
        }
    return out


def _percentile_ci(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    lo = values[max(0, int(0.025 * len(values)))]
    hi = values[min(len(values) - 1, max(0, math.ceil(0.975 * len(values)) - 1))]
    return [lo, hi]


def _cluster_replicates(rows, stratum_key, B, seed):
    """Yield whole-card bootstrap samples, resampled within design strata."""
    by_stratum = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_stratum[stratum_key(row)][row["card"]].append(row)
    rng = random.Random(seed)
    for _ in range(B):
        replicate = []
        for stratum in sorted(by_stratum, key=str):
            cards = sorted(by_stratum[stratum])
            for _ in cards:
                sampled_card = rng.choice(cards)
                replicate.extend(by_stratum[stratum][sampled_card])
        yield replicate


def _unweighted_point(rows, kind):
    selected = [row for row in rows if row["kind"] == kind]
    pairs = [(row["judge_label"], row["human_reference"]) for row in selected]
    kappa, agreement = cohens_kappa(pairs, _labels_for_kind(kind))
    return len(selected), agreement, kappa


def unweighted_agreement_block(rows, stratum_key, B, seed):
    """Raw judge/reference agreement with stratified card-cluster CIs."""
    out = {}
    replicates = list(_cluster_replicates(rows, stratum_key, B, seed)) if rows and B else []
    for kind in ("filled", "not_specified"):
        n, agreement, kappa = _unweighted_point(rows, kind)
        boot_points = [_unweighted_point(rep, kind) for rep in replicates]
        out[kind] = {
            "n": n,
            "raw_agreement": agreement,
            "raw_agreement_ci95_card_clustered": _percentile_ci(
                [x[1] for x in boot_points]),
            "cohens_kappa": kappa,
            "cohens_kappa_ci95_card_clustered": _percentile_ci(
                [x[2] for x in boot_points]),
        }
    return out


def weighted_confusion(rows, kind):
    """Corpus-weighted confusion matrix and agreement for probability rows."""
    labels = _labels_for_kind(kind)
    matrix = {judge: {human: 0.0 for human in labels} for judge in labels}
    for row in rows:
        if row["kind"] != kind:
            continue
        matrix[row["judge_label"]][row["human_reference"]] += row["analysis_weight"]
    total = sum(sum(line.values()) for line in matrix.values())
    if total <= 0:
        return {"weighted_total": 0.0, "confusion": matrix,
                "weighted_agreement": None, "cohens_kappa": None}
    observed = sum(matrix[label][label] for label in labels) / total
    judge_marginal = {label: sum(matrix[label].values()) for label in labels}
    human_marginal = {label: sum(matrix[j][label] for j in labels) for label in labels}
    expected = sum((judge_marginal[label] / total) * (human_marginal[label] / total)
                   for label in labels)
    kappa = None if expected >= 1.0 else (observed - expected) / (1 - expected)
    return {"weighted_total": total, "confusion": matrix,
            "weighted_agreement": observed, "cohens_kappa": kappa}


def probability_weighted_block(rows, B, seed):
    """Overall probability-arm results; never called on either arm combined."""
    clusters_by_stratum = Counter()
    for stratum in {row["prob_stratum"] for row in rows}:
        clusters_by_stratum[stratum] = len({
            row["card"] for row in rows if row["prob_stratum"] == stratum
        })
    out = {
        "estimator": "two-stage inverse-probability ratio, probability arm only",
        "weight": "s_to_corpus_weight / prob_row_pi_within_s",
        "uncertainty": (
            "stratified whole-card percentile-bootstrap sensitivity interval"),
        "uncertainty_scope": (
            "Conditional on the realized second-stage field draw. This is not an "
            "exact two-stage design-based interval and does not propagate a new "
            "within-card field draw in each replicate."),
        "sampled_card_clusters_by_stratum": dict(sorted(clusters_by_stratum.items())),
        "low_cluster_caution": (
            "The probability arm has few sampled cards per stratum; percentile "
            "whole-card bootstrap results are low-cluster sensitivity intervals, "
            "conditional on the realized second-stage field draw, and must not be "
            "presented as exact two-stage design-based or high-precision uncertainty."
            if clusters_by_stratum and min(clusters_by_stratum.values()) < 10 else None
        ),
    }
    replicates = list(_cluster_replicates(
        rows, lambda row: row["prob_stratum"], B, seed)) if rows and B else []
    for kind in ("filled", "not_specified"):
        point = weighted_confusion(rows, kind)
        boot = [weighted_confusion(rep, kind) for rep in replicates]
        point["weighted_agreement_card_bootstrap_sensitivity_interval95"] = _percentile_ci(
            [x["weighted_agreement"] for x in boot])
        point["cohens_kappa_card_bootstrap_sensitivity_interval95"] = _percentile_ci(
            [x["cohens_kappa"] for x in boot])
        point["n_sampled_rows"] = sum(row["kind"] == kind for row in rows)
        point["n_sampled_cards"] = len({row["card"] for row in rows if row["kind"] == kind})
        out[kind] = point
    return out


def error_confirmation_block(rows, B, seed):
    """Conditional error-arm confirmation; unsupported and partial stay separate."""
    out = {}
    for offset, status in enumerate(("unsupported", "partial")):
        selected = [row for row in rows if row["error_status_stratum"] == status]
        labels = FILLED_LABELS

        def point(sample):
            weights = [1.0 / row["error_pi"] for row in sample]
            total = sum(weights)
            confirm = sum(w for row, w in zip(sample, weights)
                          if row["human_reference"] == status)
            distribution = {label: sum(w for row, w in zip(sample, weights)
                                       if row["human_reference"] == label)
                            for label in labels}
            return (confirm / total if total else None), distribution

        value, distribution = point(selected)
        boot = [point(rep)[0] for rep in _cluster_replicates(
            selected, lambda row: row["error_status_stratum"], B, seed + offset)
                ] if selected and B else []
        out[status] = {
            "n": len(selected),
            "n_cards": len({row["card"] for row in selected}),
            "raw_confirms": sum(row["human_reference"] == status for row in selected),
            "raw_confirmation_rate": (sum(row["human_reference"] == status
                                           for row in selected) / len(selected)
                                      if selected else None),
            "design_weighted_confirmation_rate": value,
            "card_cluster_bootstrap_sensitivity_interval95": _percentile_ci(boot),
            "weighted_human_label_distribution": distribution,
            "weight": "1 / error_row_pi_within_pool",
            "uncertainty_note": (
                "The point estimate follows the field-SRS error design. The whole-card "
                "bootstrap is a clustering sensitivity analysis, not an exact "
                "design-based variance estimator for that field sample."),
        }
    return out


def score_calibration(key, rating_tables, adjudications=None, B=B_DEFAULT, seed=11):
    """Validate and score already-loaded key and rating tables."""
    key_rows = validate_key(key)
    rows = validate_and_join(key_rows, rating_tables, adjudications)
    error_rows = [row for row in rows if "error" in row["arms"]]
    probability_rows = [row for row in rows if "probability" in row["arms"]]
    split_rows = [row for row in rows if row["true_three_way_split"]]
    unresolved = [row for row in split_rows if row["human_reference"] is None]
    result = {
        "analysis_policy": {
            "overall_population_estimate": "probability arm only",
            "error_arm_estimate": (
                "conditional confirmation within the S judge-error pool, "
                "reported separately by judge error status"),
            "combined_arm_estimate": None,
            "no_post_result_metric_selection": True,
        },
        "uncertainty_policy": {
            "bootstrap_replicates": B,
            "seed": seed,
            "unit": "whole sampled card",
            "probability_strata": "prob_flag_stratum",
            "finite_population_correction": False,
            "probability_interval_role": (
                "low-cluster sensitivity interval conditional on the realized "
                "second-stage field draw; not an exact two-stage design-based interval"),
            "error_interval_role": (
                "card-clustering sensitivity interval for a field-SRS point estimate; "
                "not an exact design-based interval"),
        },
        "n_unique_items": len(rows),
        "n_error_arm_rows": len(error_rows),
        "n_probability_arm_rows": len(probability_rows),
        "n_dual_arm_rows": sum(row["arms"] == {"error", "probability"} for row in rows),
        "n_true_three_way_splits": len(split_rows),
        "n_unresolved_three_way_splits": len(unresolved),
        "complete_for_judge_comparison": not unresolved,
        "inter_rater_reliability": {
            "scope": "unweighted and conditional on the displayed packet/arm",
            "all_unique_items": reliability_block(rows),
            "error_arm": reliability_block(error_rows),
            "probability_arm": reliability_block(probability_rows),
        },
    }
    if unresolved:
        result["judge_vs_human_reference"] = None
        result["error_arm_conditional_confirmation"] = None
        result["probability_arm_corpus_weighted"] = None
        result["incomplete_note"] = (
            "Judge comparisons are suppressed until every true three-way split "
            "has a blind adjudicated label.")
        return result, rows

    error_raw = unweighted_agreement_block(
        error_rows, lambda row: row["error_status_stratum"], B, seed)
    error_raw["scope"] = (
        "unweighted and conditional on the fixed error-arm allocation; not an "
        "overall population accuracy estimate")
    probability_raw = unweighted_agreement_block(
        probability_rows, lambda row: row["prob_stratum"], B, seed + 10)
    probability_raw["scope"] = (
        "unweighted probability-arm sample description; use the separately "
        "reported weighted block for corpus inference")
    result["judge_vs_human_reference"] = {
        "error_arm_raw": error_raw,
        "probability_arm_raw": probability_raw,
    }
    result["error_arm_conditional_confirmation"] = error_confirmation_block(
        error_rows, B, seed + 20)
    result["probability_arm_corpus_weighted"] = probability_weighted_block(
        probability_rows, B, seed + 30)
    return result, rows


def write_adjudication_worklist(path, rows):
    """Write only true three-way splits, with all judge/rater/design data hidden."""
    fields = list(PUBLIC_COLUMNS) + ["adjudicated_label", "adjudication_note"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["true_three_way_split"]:
                continue
            public = dict(row["public"])
            public.update({"adjudicated_label": row.get("adjudicated") or "",
                           "adjudication_note": ""})
            writer.writerow(public)


def main():
    ap = argparse.ArgumentParser(description="Three-rater calibration scorer")
    ap.add_argument("--ratings", nargs=3, required=True, metavar=("R1", "R2", "R3"))
    ap.add_argument("--key", required=True)
    ap.add_argument("--adjudicated", default=None,
                    help="completed blind worklist; accepted only for true 3-way splits")
    ap.add_argument("--adjudication-out", default=None,
                    help="blind split worklist (default: next to --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--bootstrap", type=int, default=B_DEFAULT,
                    help="number of stratified whole-card sensitivity replicates")
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap must be non-negative")
    check()

    try:
        with open(_path(args.key), encoding="utf-8") as f:
            key = json.load(f)
        require_distinct_rating_files(args.ratings)
        tables = [read_rating_table(path) for path in args.ratings]
        adjudications = read_adjudications(args.adjudicated)
        result, rows = score_calibration(
            key, tables, adjudications, B=args.bootstrap, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"calibration scoring failed: {exc}")

    out_path = _path(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    worklist_path = (_path(args.adjudication_out) if args.adjudication_out else
                     os.path.splitext(out_path)[0] + "_adjudication_worklist.csv")
    write_adjudication_worklist(worklist_path, rows)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"scores -> {args.out}")
    print(f"blind adjudication worklist -> {worklist_path}")
    print(f"items={result['n_unique_items']}, true three-way splits="
          f"{result['n_true_three_way_splits']}, unresolved="
          f"{result['n_unresolved_three_way_splits']}")
    if not result["complete_for_judge_comparison"]:
        print("judge comparisons suppressed until the blind worklist is complete")


if __name__ == "__main__":
    main()
