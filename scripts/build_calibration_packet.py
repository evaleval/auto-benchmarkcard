"""Build the locked two-arm human validation packet.

The error arm is an SRS of 30 judge-unsupported and 13 judge-partial
content fields.  Independently, the probability arm samples eight cards from
each original S150 stratum and two unrestricted eligible fields per selected
card.  Arms may overlap: an overlapping row is shown once and records both
memberships.

This consumes the validated derived judge frame, not the raw judge output.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from numbers import Real

from check_frozen import check


REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MASTER_SEED = 7
DESIGN_VERSION = "human-validation-v7-two-arm-16-cluster"
SEED_NAMESPACES = {
    # Preserve the already-audited streams: the v4 five-card/three-field draws
    # are extended or truncated by prefix under the locked allocation.
    "error": "human-validation-v3",
    "probability_cards": "human-validation-v4",
    "probability_fields": "human-validation-v4",
}

N_ERROR_UNSUPPORTED = 30
N_ERROR_PARTIAL = 13
N_PROBABILITY_V4_CARD_PREFIX = 5
N_PROBABILITY_V4_FIELDS_PER_CARD = 3
N_PROBABILITY_CARDS_PER_STRATUM = 8
N_PROBABILITY_FIELDS_PER_CARD = 2
EXPECTED_N_CARDS = 150
EXPECTED_N_FRAME_ROWS = 3450
OUTER_STRATA = ("flagged", "unflagged")

# Literal and locked: never import this from production field accounting.
ELIGIBLE_FIELD_PATHS = (
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

STRUCTURAL_FIELD_EXCLUSIONS = (
    {
        "field_path": "benchmark_details.benchmark_type",
        "reason": "deterministic_registry_enum_outside_content_faithfulness_construct",
    },
    {
        "field_path": "benchmark_details.contains",
        "reason": "structured_composite_membership_outside_content_faithfulness_construct",
    },
)
EXPECTED_DECLARED_EXCLUSIONS = ()
VALID_STATUSES = {
    "supported", "supported_by_eee_only", "not_specified", "partial", "unsupported"
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _file_fingerprint(path):
    with open(path, "rb") as f:
        data = f.read()
    return {"bytes": len(data), "sha256": _sha256(data)}


def _canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _fraction(numerator, denominator):
    if denominator <= 0:
        raise ValueError(f"fraction denominator must be positive, got {denominator}")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
    }


def derive_subseed(master_seed, label):
    """Derive a platform-stable numeric sub-seed from master seed plus label."""
    if label.startswith("error:"):
        namespace_key = "error"
    elif label.startswith("probability:cards:"):
        namespace_key = "probability_cards"
    elif label.startswith("probability:fields:"):
        namespace_key = "probability_fields"
    else:
        raise ValueError(f"unknown deterministic stream label: {label!r}")
    namespace = SEED_NAMESPACES[namespace_key]
    material = f"{namespace}\0master={master_seed}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _sample_srs(rows, n, seed, description):
    if len(rows) < n:
        raise ValueError(
            f"{description} pool has {len(rows)} rows, fewer than required {n}"
        )
    ordered = sorted(rows, key=lambda row: (row["card"], row["field_path"]))
    return random.Random(seed).sample(ordered, n)


def _extend_v4_card_sample(card_pool, seed):
    """Extend the frozen v4 five-card SRS to an equivalent eight-card SRS.

    The first stage is exactly the v4 call, including its RNG consumption.  A
    second SRS of three from the remaining cards extends it without redraws or
    quality filtering.  Uniform-5 followed by uniform-3 from the remainder is
    an unordered uniform sample of eight, while preserving the audited v4 set.
    """
    ordered = sorted(card_pool)
    rng = random.Random(seed)
    v4_prefix = rng.sample(ordered, N_PROBABILITY_V4_CARD_PREFIX)
    prefix_set = set(v4_prefix)
    remaining = [card for card in ordered if card not in prefix_set]
    extension = rng.sample(
        remaining, N_PROBABILITY_CARDS_PER_STRATUM - N_PROBABILITY_V4_CARD_PREFIX
    )
    return v4_prefix + extension, v4_prefix, extension


def _truncate_v4_field_sample(field_pool, seed):
    """Return the first two positions of the audited v4 three-field draw."""
    ordered = sorted(field_pool, key=lambda row: row["field_path"])
    v4_draw = random.Random(seed).sample(ordered, N_PROBABILITY_V4_FIELDS_PER_CARD)
    return v4_draw[:N_PROBABILITY_FIELDS_PER_CARD], v4_draw


def load_sample_design(sample_path):
    sample = _load(sample_path)
    cards = sample.get("cards")
    if not isinstance(cards, list):
        raise ValueError("sample artifact must contain a cards list")
    strata_meta = sample.get("strata")
    if not isinstance(strata_meta, dict) or set(strata_meta) != set(OUTER_STRATA):
        raise ValueError("sample artifact must define flagged and unflagged strata")
    stratum_weights = {}
    for stratum in OUTER_STRATA:
        weight = strata_meta[stratum].get("weight")
        if isinstance(weight, bool) or not isinstance(weight, Real) or weight <= 0:
            raise ValueError(f"invalid {stratum} sample-to-corpus weight: {weight!r}")
        stratum_weights[stratum] = weight

    outer_strata, card_weights = {}, {}
    for row in cards:
        name, stratum = row.get("name"), row.get("stratum")
        if not name or stratum not in OUTER_STRATA:
            raise ValueError(f"invalid sample card/stratum row: {row!r}")
        if name in outer_strata:
            raise ValueError(f"duplicate card in sample artifact: {name}")
        card_weight = row.get("weight")
        if card_weight != stratum_weights[stratum]:
            raise ValueError(
                f"{name}: card weight {card_weight!r} does not equal "
                f"sample.strata[{stratum!r}].weight {stratum_weights[stratum]!r}"
            )
        outer_strata[name] = stratum
        card_weights[name] = card_weight
    if len(outer_strata) != EXPECTED_N_CARDS:
        raise ValueError(
            f"sample has {len(outer_strata)} cards; expected {EXPECTED_N_CARDS}"
        )
    counts = Counter(outer_strata.values())
    expected = Counter({"flagged": 75, "unflagged": 75})
    if counts != expected:
        raise ValueError(f"unexpected S150 stratum counts: {dict(counts)}")
    return outer_strata, card_weights


def load_outer_strata(sample_path):
    """Compatibility helper for frame-only validation and focused tests."""
    return load_sample_design(sample_path)[0]


def validate_and_build_frame(judge, outer_strata):
    """Fail closed on any undeclared missing, duplicate, or extra verdict."""
    analysis_frame = judge.get("analysis_frame")
    if not isinstance(analysis_frame, dict):
        raise ValueError("judge artifact is missing top-level analysis_frame")
    if analysis_frame.get("allowlisted_paths") != list(ELIGIBLE_FIELD_PATHS):
        raise ValueError(
            "analysis_frame.allowlisted_paths does not exactly match the locked "
            "23-field allowlist"
        )
    exclusions = analysis_frame.get("declared_exclusions")
    if exclusions != list(EXPECTED_DECLARED_EXCLUSIONS):
        raise ValueError(
            "analysis_frame.declared_exclusions does not match the locked frame"
        )
    if analysis_frame.get("n_cards") != EXPECTED_N_CARDS:
        raise ValueError(f"analysis_frame.n_cards must be {EXPECTED_N_CARDS}")
    if analysis_frame.get("n_rows") != EXPECTED_N_FRAME_ROWS:
        raise ValueError(f"analysis_frame.n_rows must be {EXPECTED_N_FRAME_ROWS}")

    per_card = judge.get("per_card")
    if not isinstance(per_card, dict):
        raise ValueError("judge artifact must contain a per_card mapping")
    if set(per_card) != set(outer_strata):
        missing = sorted(set(outer_strata) - set(per_card))
        extra = sorted(set(per_card) - set(outer_strata))
        raise ValueError(f"judge/sample card mismatch; missing={missing}, extra={extra}")

    excluded_keys = {(row["card"], row["field_path"]) for row in exclusions}
    allowlisted = set(ELIGIBLE_FIELD_PATHS)
    frame = []
    for card in sorted(per_card):
        card_record = per_card[card]
        stratum = outer_strata[card]
        if card_record.get("stratum") != stratum:
            raise ValueError(f"outer stratum mismatch for {card}")
        verdicts = card_record.get("field_verdicts")
        if not isinstance(verdicts, list):
            raise ValueError(f"{card}: field_verdicts must be a list")
        by_path = defaultdict(list)
        for verdict in verdicts:
            path = verdict.get("path")
            if path not in allowlisted:
                raise ValueError(f"{card}: non-allowlisted verdict remains: {path!r}")
            by_path[path].append(verdict)

        for path in ELIGIBLE_FIELD_PATHS:
            verdicts_for_path = by_path.get(path, [])
            expected_count = 0 if (card, path) in excluded_keys else 1
            if len(verdicts_for_path) != expected_count:
                raise ValueError(
                    f"{card}/{path}: expected {expected_count} verdict(s), "
                    f"found {len(verdicts_for_path)}"
                )
            if not verdicts_for_path:
                continue
            verdict = verdicts_for_path[0]
            status = verdict.get("status")
            if status not in VALID_STATUSES:
                raise ValueError(f"{card}/{path}: invalid judge status {status!r}")
            frame.append({
                "card": card,
                "outer_stratum": stratum,
                "field_path": path,
                "judge_status": status,
                "info_in_source": verdict.get("info_in_source"),
                "specificity": verdict.get("specificity"),
                "fv": verdict,
            })
    if len(frame) != EXPECTED_N_FRAME_ROWS:
        raise ValueError(
            f"validated frame has {len(frame)} rows; expected {EXPECTED_N_FRAME_ROWS}"
        )
    return frame


def frame_fingerprint(frame):
    records = [{
        "card": row["card"],
        "outer_stratum": row["outer_stratum"],
        "field_path": row["field_path"],
        "judge_status": row["judge_status"],
        "info_in_source": row["info_in_source"],
        "specificity": row["specificity"],
        "s_to_corpus_weight": row.get("s_to_corpus_weight"),
    } for row in sorted(frame, key=lambda row: (row["card"], row["field_path"]))]
    return {
        "sha256": _sha256(_canonical_json_bytes(records)),
        "n_rows": len(records),
        "canonicalization": (
            "UTF-8 compact JSON with sorted keys; rows sorted by card and field_path"
        ),
    }


def select_two_arm_sample(frame, master_seed=MASTER_SEED):
    """Select both independent arms and retain every exact membership."""
    by_status, by_card, cards_by_stratum = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in frame:
        by_status[row["judge_status"]].append(row)
        by_card[row["card"]].append(row)
        cards_by_stratum[row["outer_stratum"]].add(row["card"])

    subseeds = {
        "error_unsupported": derive_subseed(master_seed, "error:unsupported"),
        "error_partial": derive_subseed(master_seed, "error:partial"),
        "probability_cards": {
            stratum: derive_subseed(master_seed, f"probability:cards:{stratum}")
            for stratum in OUTER_STRATA
        },
        "probability_fields": {},
    }
    memberships, selected_by_key = defaultdict(list), {}

    error_meta = {}
    for status, n, seed_key in (
        ("unsupported", N_ERROR_UNSUPPORTED, "error_unsupported"),
        ("partial", N_ERROR_PARTIAL, "error_partial"),
    ):
        pool = by_status[status]
        inclusion = _fraction(n, len(pool))
        draw = _sample_srs(pool, n, subseeds[seed_key], f"error-{status}")
        for row in draw:
            key = (row["card"], row["field_path"])
            selected_by_key[key] = row
            memberships[key].append({
                "arm": "error",
                "category": status,
                "inclusion_fraction": inclusion,
            })
        error_meta[status] = {
            "sampling": "simple_random_sample_without_replacement",
            "pool_size": len(pool),
            "n_selected": n,
            "inclusion_fraction": inclusion,
            "subseed": subseeds[seed_key],
        }

    probability_meta = {
        "sampling": (
            "stratified eight-card SRS extension preserving the v4 five-card "
            "set, then unrestricted two-field SRS within each card"
        ),
        "independent_of_error_arm": True,
        "field_kind_allocation": (
            "unrestricted; filled/not_specified mix is accepted as realized"
        ),
        "overlap_policy": (
            "no redraw; dual-arm fields are emitted once with both memberships"
        ),
        "cards_by_outer_stratum": {},
        "fields_by_card": {},
    }
    for stratum in OUTER_STRATA:
        card_pool = sorted(cards_by_stratum[stratum])
        card_inclusion = _fraction(N_PROBABILITY_CARDS_PER_STRATUM, len(card_pool))
        selected_in_draw_order, v4_prefix, extension = _extend_v4_card_sample(
            card_pool, subseeds["probability_cards"][stratum]
        )
        selected_cards = sorted(selected_in_draw_order)
        probability_meta["cards_by_outer_stratum"][stratum] = {
            "sampling": (
                "v4 SRS-5 followed by SRS-3 from remaining cards; equivalent SRS-8"
            ),
            "pool_size": len(card_pool),
            "n_selected": len(selected_cards),
            "selected_cards": selected_cards,
            "selected_cards_draw_order": selected_in_draw_order,
            "preserved_v4_cards_draw_order": v4_prefix,
            "preserved_v4_cards": sorted(v4_prefix),
            "v5_extension_cards_draw_order": extension,
            "v5_extension_cards": sorted(extension),
            "inclusion_fraction": card_inclusion,
            "subseed": subseeds["probability_cards"][stratum],
        }
        for card in selected_cards:
            field_pool = sorted(by_card[card], key=lambda row: row["field_path"])
            field_seed = derive_subseed(master_seed, f"probability:fields:{card}")
            subseeds["probability_fields"][card] = field_seed
            field_inclusion = _fraction(N_PROBABILITY_FIELDS_PER_CARD, len(field_pool))
            row_inclusion = _fraction(
                card_inclusion["numerator"] * field_inclusion["numerator"],
                card_inclusion["denominator"] * field_inclusion["denominator"],
            )
            field_draw_order, v4_field_draw_order = _truncate_v4_field_sample(
                field_pool, field_seed
            )
            field_draw = sorted(field_draw_order, key=lambda row: row["field_path"])
            selected_rows = [
                {
                    "field_path": row["field_path"],
                    "kind": (
                        "not_specified" if row["judge_status"] == "not_specified"
                        else "filled"
                    ),
                }
                for row in field_draw
            ]
            probability_meta["fields_by_card"][card] = {
                "outer_stratum": stratum,
                "pool_size": len(field_pool),
                "n_selected": len(field_draw),
                "selected_field_paths": [row["field_path"] for row in field_draw],
                "selected_field_paths_draw_order": [
                    row["field_path"] for row in field_draw_order
                ],
                "preserved_v4_three_field_draw_order": [
                    row["field_path"] for row in v4_field_draw_order
                ],
                "selected_rows": selected_rows,
                "field_conditional_inclusion_fraction": field_inclusion,
                "row_inclusion_fraction": row_inclusion,
                "subseed": field_seed,
            }
            for row in field_draw:
                key = (row["card"], row["field_path"])
                selected_by_key[key] = row
                memberships[key].append({
                    "arm": "probability",
                    "outer_stratum": stratum,
                    "selected_kind": (
                        "not_specified" if row["judge_status"] == "not_specified"
                        else "filled"
                    ),
                    "card_inclusion_fraction": card_inclusion,
                    "field_conditional_inclusion_fraction": field_inclusion,
                    "row_inclusion_fraction": row_inclusion,
                })

    selected = []
    for key in sorted(selected_by_key):
        row = dict(selected_by_key[key])
        row["arm_memberships"] = memberships[key]
        selected.append(row)
    membership_counts = Counter(
        f"error_{membership['category']}" if membership["arm"] == "error"
        else "probability"
        for member_list in memberships.values()
        for membership in member_list
    )
    probability_kind_counts = Counter(
        membership["selected_kind"]
        for member_list in memberships.values()
        for membership in member_list
        if membership["arm"] == "probability"
    )
    meta = {
        "master_seed": master_seed,
        "seed_derivation": {
            "version": 1,
            "namespaces": dict(SEED_NAMESPACES),
            "algorithm": (
                "uint64_be(first_8_bytes(SHA256(namespace + NUL + "
                "'master=<seed>' + NUL + stream_label)))"
            ),
            "subseeds": subseeds,
        },
        "error_arm": error_meta,
        "probability_arm": probability_meta,
        "n_unique_rows": len(selected),
        "n_dual_membership_rows": sum(
            1 for member_list in memberships.values() if len(member_list) > 1
        ),
        "membership_counts": dict(sorted(membership_counts.items())),
        "probability_kind_counts": dict(sorted(probability_kind_counts.items())),
    }
    return selected, meta


def _field_value_for_csv(value):
    """Losslessly render the complete judge-input value for a human rater."""
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "), allow_nan=False
    )


def _input_for_card(inputs_dir, card):
    path = os.path.join(inputs_dir, f"{card}.json")
    if not os.path.isfile(path):
        raise ValueError(f"missing prepared judge input: {path}")
    input_fingerprint = _file_fingerprint(path)
    inp = _load(path)
    source_text = inp.get("source_text")
    if not isinstance(source_text, str):
        raise ValueError(f"{card}: source_text must be a string")
    fields = inp.get("fields")
    if not isinstance(fields, list):
        raise ValueError(f"{card}: fields must be a list")
    by_path = defaultdict(list)
    for field in fields:
        by_path[field.get("path")].append(field)
    return by_path, input_fingerprint, source_text.encode("utf-8")


def _clear_generated_outputs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for path in glob.glob(os.path.join(out_dir, "ratings_rater*.csv")):
        os.remove(path)
    sources_dir = os.path.join(out_dir, "sources")
    os.makedirs(sources_dir, exist_ok=True)
    for path in glob.glob(os.path.join(sources_dir, "*.txt")):
        os.remove(path)


def _flat_arm_fields(arm_memberships):
    error = next(
        (membership for membership in arm_memberships if membership["arm"] == "error"),
        None,
    )
    probability = next(
        (membership for membership in arm_memberships
         if membership["arm"] == "probability"),
        None,
    )
    tokens = []
    if error:
        tokens.append("error")
    if probability:
        tokens.append("probability")
    return {
        "arm_membership": "|".join(tokens),
        "error_status_stratum": error["category"] if error else None,
        "error_row_pi_within_pool": (
            error["inclusion_fraction"]["value"] if error else None
        ),
        "prob_flag_stratum": probability["outer_stratum"] if probability else None,
        "prob_card_pi_within_s": (
            probability["card_inclusion_fraction"]["value"] if probability else None
        ),
        "prob_field_pi_given_card": (
            probability["field_conditional_inclusion_fraction"]["value"]
            if probability else None
        ),
        "prob_row_pi_within_s": (
            probability["row_inclusion_fraction"]["value"] if probability else None
        ),
    }


def write_packet(selected, selection_meta, frame, judge_path, sample_path,
                 inputs_dir, out_dir, raters, card_weights):
    if raters != 3:
        raise ValueError("the locked design requires exactly three raters")
    _clear_generated_outputs(out_dir)
    sources_dir = os.path.join(out_dir, "sources")
    inputs_cache, source_records = {}, {}

    def get_input(card):
        if card not in inputs_cache:
            inputs_cache[card] = _input_for_card(inputs_dir, card)
        return inputs_cache[card]

    header = [
        "item_id", "card", "field_path", "kind", "field_value",
        "human_label", "human_note",
    ]
    key_rows, csv_rows = [], []
    for index, row in enumerate(selected):
        card, field_path = row["card"], row["field_path"]
        input_fields, input_fingerprint, source_bytes = get_input(card)
        matching_fields = input_fields.get(field_path, [])
        if len(matching_fields) != 1:
            raise ValueError(
                f"{card}/{field_path}: prepared input must contain exactly one field; "
                f"found {len(matching_fields)}"
            )
        rendered_value = _field_value_for_csv(matching_fields[0].get("value", ""))
        kind = "not_specified" if row["judge_status"] == "not_specified" else "filled"
        item_id = f"c{index:03d}"
        csv_rows.append([
            item_id, card, field_path, kind, rendered_value, "", "",
        ])
        source_sha256 = _sha256(source_bytes)
        key_row = {
            "item_id": item_id,
            "card": card,
            "field_path": field_path,
            "kind": kind,
            "judge_label": (
                row["info_in_source"] if kind == "not_specified"
                else row["judge_status"]
            ),
            "specificity": row["specificity"],
            "outer_stratum": row["outer_stratum"],
            "arm_memberships": row["arm_memberships"],
            "s_to_corpus_weight": card_weights[card],
            "source_sha256": source_sha256,
            "source_bytes": len(source_bytes),
            "field_value_sha256": _sha256(rendered_value.encode("utf-8")),
            "field_value_chars": len(rendered_value),
        }
        key_row.update(_flat_arm_fields(row["arm_memberships"]))
        key_rows.append(key_row)
        if card not in source_records:
            source_path = os.path.join(sources_dir, f"{card}.txt")
            with open(source_path, "wb") as f:
                f.write(source_bytes)
            source_records[card] = {
                "file": f"sources/{card}.txt",
                "bytes": len(source_bytes),
                "sha256": source_sha256,
                "prepared_input_file": os.path.join(inputs_dir, f"{card}.json"),
                "prepared_input_bytes": input_fingerprint["bytes"],
                "prepared_input_sha256": input_fingerprint["sha256"],
            }

    for rater_number in range(1, raters + 1):
        path = os.path.join(out_dir, f"ratings_rater{rater_number}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_rows)

    status_counts = Counter(row["judge_status"] for row in frame)
    key_meta = {
        "design_version": DESIGN_VERSION,
        "judge_artifact": {"file": judge_path, **_file_fingerprint(judge_path)},
        "sample_artifact": {"file": sample_path, **_file_fingerprint(sample_path)},
        "eligible_frame": {
            "allowlisted_paths": list(ELIGIBLE_FIELD_PATHS),
            "structural_field_exclusions": list(STRUCTURAL_FIELD_EXCLUSIONS),
            "declared_mechanical_exclusions": list(EXPECTED_DECLARED_EXCLUSIONS),
            "n_cards": EXPECTED_N_CARDS,
            "n_rows": len(frame),
            "pool_sizes_by_judge_status": dict(sorted(status_counts.items())),
            "frame_hash": frame_fingerprint(frame),
        },
        "sampling": selection_meta,
        "n_raters": raters,
        "n_unique_items": len(key_rows),
        "source_files": dict(sorted(source_records.items())),
        "rubric": (
            "filled: supported / supported_by_eee_only / partial / unsupported; "
            "not_specified: info_in_source yes_primary / yes_eee_only / no"
        ),
        "key": key_rows,
    }
    with open(os.path.join(out_dir, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key_meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    readme_lines = [
        "CALIBRATION RATING PACKET",
        "",
        f"This packet contains {len(key_rows)} fields from {len(source_records)} cards.",
        "",
        "For each CSV row, use only the matching sources/<card>.txt file.",
        "An EEE evaluation registry section counts as supplied evidence.",
        "",
        "For kind=filled: supported, supported_with_registry, partial, unsupported.",
        "For kind=not_specified: yes_source, yes_registry, no.",
        "",
        "A clear paraphrase or a conclusion entailed by the evidence can be supported.",
        "A plausible guess, generic default, or topic-based inference is not support.",
        "Evidence must answer the named field, not merely contain similar words.",
        "",
        "Strict category rules:",
        "- audience: the source must state or clearly establish the intended users; do not infer generic groups from the topic.",
        "- goal and tasks: require a stated or clearly established evaluation objective or task; do not infer one only from the title, topic, or modality.",
        "- limitations: require a stated caveat, weakness, or coverage boundary; low scores, task difficulty, and future work do not count by themselves.",
        "- out_of_scope_uses: require an excluded, discouraged, or inappropriate use; absence from intended scope is not enough.",
        "- baseline_results: require a reported performance result for a named baseline; saying that baselines were evaluated is not enough.",
        "- human_baseline: require a human-performance study, setup, or result; human annotation or expert review is not a baseline.",
        "- consent_procedures: require how people agreed, could decline, or could withdraw; recruitment, ethics approval, privacy, or workflow alone is not consent.",
        "- contamination_controls: require a measure or test for leakage, overlap, or memorization; a train/test split or generic data cleaning alone is not a control.",
        "- compliance_with_regulations: require compliance with a law, regulation, policy, or institutional rule; licensing alone is not regulatory compliance.",
        "- judge_score_consolidation: require a rule for combining multiple judge or grader outputs into one judgment or score; normal metric, task, or dataset aggregation does not count.",
        "- interpretation: derived statements are allowed only when they follow from supplied facts, such as a metric definition or score direction.",
        "",
        "Write the label in human_label. Notes are optional. Do not use an AI assistant.",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))
    return key_meta


def build_packet(judge_path, sample_path, inputs_dir, out_dir,
                 master_seed=MASTER_SEED, raters=3):
    if master_seed != MASTER_SEED:
        raise ValueError(f"locked master seed is {MASTER_SEED}, got {master_seed}")
    outer_strata, card_weights = load_sample_design(sample_path)
    judge = _load(judge_path)
    frame = validate_and_build_frame(judge, outer_strata)
    for row in frame:
        row["s_to_corpus_weight"] = card_weights[row["card"]]
    selected, selection_meta = select_two_arm_sample(frame, master_seed)
    return write_packet(
        selected, selection_meta, frame, judge_path, sample_path,
        inputs_dir, out_dir, raters, card_weights,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build the locked blind two-arm human validation packet."
    )
    parser.add_argument("--judge", required=True, help="derived validated judge artifact")
    parser.add_argument("--sample", required=True, help="original S150 sample artifact")
    parser.add_argument("--inputs", required=True, help="prepared judge inputs directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--raters", type=int, default=3)
    args = parser.parse_args()
    check()
    if args.seed != MASTER_SEED:
        parser.error(f"the locked master seed is {MASTER_SEED}")
    if args.raters != 3:
        parser.error("the locked design requires exactly three raters")

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(REPO, path)

    meta = build_packet(
        resolve(args.judge), resolve(args.sample), resolve(args.inputs), resolve(args.out),
        master_seed=args.seed, raters=args.raters,
    )
    sampling = meta["sampling"]
    print(
        f"packet: {meta['n_unique_items']} unique rows; "
        f"43 error memberships + 32 probability memberships; "
        f"{sampling['n_dual_membership_rows']} dual-arm rows; "
        f"{len(meta['source_files'])} sources -> {args.out}"
    )


if __name__ == "__main__":
    main()
