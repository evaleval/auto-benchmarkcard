"""Reproduce the field-level analyses used in the paper revision.

This script derives five related outputs from the frozen corpus-v3 evaluation:

1. A five-state decomposition over all 3,450 source-judge field slots.
2. Coverage of the three source-checkable prose fields in the Ethical and Legal
   Considerations section, compared with the other 20 judged fields.
3. The field distribution of human-confirmed judge-unsupported decisions.
4. The exact-field overlap between verifier-confirmed material findings and the
   source-judge verdicts.
5. A complete 23-field outcome matrix for supplementary reporting.

The source-judge estimates use the same combined-ratio estimator and shared
5,000-replicate stratified whole-card bootstrap as ``analyze_s.py``. Aggregate
intervals use that bootstrap; rare per-field events use the frozen
Wilson-effective-sample-size fallback. Full-corpus ``Not specified`` rates are
census descriptions of generated card output. They do not establish that
information is absent from all public sources or that a field applies to every
benchmark.

Usage:
  python scripts/analyze_paper_extensions.py \
      --corpus-cards /path/to/output/auto-benchmarkcards-v3/cards \
      --out eval/s150/paper_extension_analysis.json \
      --matrix-out eval/s150/paper_extension_field_matrix.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from build_judge_analysis_view import ALLOWLISTED_PATHS  # noqa: E402
from check_frozen import check  # noqa: E402
from judge_analysis_guard import validate_judge_analysis_frame  # noqa: E402
from score_calibration import (  # noqa: E402
    read_adjudications,
    read_rating_table,
    require_distinct_rating_files,
    validate_and_join,
    validate_key,
)
from score_screen_verification import (  # noqa: E402
    validate_sample_design,
    validate_lock,
    validate_verifier_return,
)
import s_stats  # noqa: E402


NOT_SPECIFIED_SENTINELS = frozenset(
    {
        "not specified",
        "not specified.",
        "no information found",
        "no information found.",
    }
)
FULL_SUPPORT = frozenset({"supported", "supported_by_eee_only"})
FILLED = frozenset({"supported", "supported_by_eee_only", "partial", "unsupported"})
MISSED_INFO = frozenset({"yes_primary", "yes_eee_only", "yes"})
ETHICAL_LEGAL_PATHS = (
    "ethical_and_legal_considerations.compliance_with_regulations",
    "ethical_and_legal_considerations.consent_procedures",
    "ethical_and_legal_considerations.privacy_and_anonymity",
)
FIVE_STATES = (
    "filled_fully_supported",
    "filled_partially_supported",
    "filled_unsupported",
    "not_specified_information_available",
    "not_specified_no_information",
)


def extract_card(obj: dict) -> dict:
    """Unwrap the public card envelope without importing pipeline dependencies."""
    return obj.get("benchmark_card", obj) if isinstance(obj, dict) else obj


def is_not_specified(value) -> bool:
    """Match the frozen corpus convention from ``card_utils.is_not_specified``."""
    if isinstance(value, str):
        return value.strip().lower() in NOT_SPECIFIED_SENTINELS
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0].strip().lower() in NOT_SPECIFIED_SENTINELS
    return False


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _display_path(path: str | Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _load_json(path: str | Path):
    with _resolve(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _state(verdict: dict) -> str:
    status = verdict["status"]
    if status in FULL_SUPPORT:
        return "filled_fully_supported"
    if status == "partial":
        return "filled_partially_supported"
    if status == "unsupported":
        return "filled_unsupported"
    if status != "not_specified":
        raise ValueError(f"unexpected source-judge status: {status!r}")
    if verdict.get("info_in_source") in MISSED_INFO:
        return "not_specified_information_available"
    if verdict.get("info_in_source") == "no":
        return "not_specified_no_information"
    raise ValueError(
        "not_specified verdict has an unclassified info_in_source value: "
        f"{verdict.get('info_in_source')!r}"
    )


def _build_card_records(sample: dict, judge: dict) -> list[dict]:
    by_name = {card["name"]: card for card in sample["cards"]}
    if set(by_name) != set(judge["per_card"]):
        missing = sorted(set(by_name) - set(judge["per_card"]))
        extra = sorted(set(judge["per_card"]) - set(by_name))
        raise ValueError(f"sample and judge card sets differ: missing={missing}, extra={extra}")

    expected_paths = set(ALLOWLISTED_PATHS)
    records = []
    for name in sorted(by_name):
        card_meta = by_name[name]
        verdicts = {
            verdict["path"]: verdict
            for verdict in judge["per_card"][name]["field_verdicts"]
        }
        if set(verdicts) != expected_paths or len(verdicts) != len(ALLOWLISTED_PATHS):
            raise ValueError(f"{name}: source-judge paths do not match the 23-field frame")
        states = {path: _state(verdict) for path, verdict in verdicts.items()}
        counts = Counter(states.values())
        records.append(
            {
                "name": name,
                "stratum": card_meta["stratum"],
                "verdicts": verdicts,
                "states": states,
                "counts": counts,
            }
        )
    return records


def _arrays(records: list[dict], key: str) -> np.ndarray:
    return np.array([record.get(key, 0) or 0 for record in records], dtype=float)


def _metric(
    name: str,
    numerator: np.ndarray,
    denominator: np.ndarray,
    strata: dict,
    replicates: dict,
    kind: str = "field_ratio",
) -> dict:
    return s_stats.make_metric(
        name,
        numerator,
        denominator,
        strata,
        replicates,
        kind=kind,
    )


def _difference_metric(
    name: str,
    a_num: np.ndarray,
    a_den: np.ndarray,
    b_num: np.ndarray,
    b_den: np.ndarray,
    strata: dict,
    replicates: dict,
) -> dict:
    a_boot = s_stats.bootstrap_ratio(a_num, a_den, strata, replicates)
    b_boot = s_stats.bootstrap_ratio(b_num, b_den, strata, replicates)
    a_point = s_stats.ratio_estimate(a_num, a_den, strata)
    b_point = s_stats.ratio_estimate(b_num, b_den, strata)
    return s_stats.derived_metric(
        name,
        lambda left, right: left - right,
        (a_boot, b_boot),
        (a_point, b_point),
    )


def _sample_analysis(
    sample: dict,
    records: list[dict],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[dict, dict, list[dict]]:
    weights = {name: meta["weight"] for name, meta in sample["strata"].items()}
    strata = s_stats.make_strata([record["stratum"] for record in records], weights)
    replicates = s_stats.make_replicates(strata, B=bootstrap_replicates, seed=seed)
    ones = np.ones(len(records), dtype=float)
    all_slots = np.full(len(records), len(ALLOWLISTED_PATHS), dtype=float)

    state_arrays = {
        state: np.array([record["counts"][state] for record in records], dtype=float)
        for state in FIVE_STATES
    }
    five_state = {
        state: _metric(
            f"five_state.{state}",
            state_arrays[state],
            all_slots,
            strata,
            replicates,
        )
        for state in FIVE_STATES
    }
    state_sum = sum(metric["value"] for metric in five_state.values())
    if not np.isclose(state_sum, 1.0, atol=1e-12):
        raise ValueError(f"five-state estimates do not sum to one: {state_sum}")

    filled = sum(state_arrays[state] for state in FIVE_STATES[:3])
    full = state_arrays["filled_fully_supported"]
    partial = state_arrays["filled_partially_supported"]
    unsupported = state_arrays["filled_unsupported"]
    not_specified = (
        state_arrays["not_specified_information_available"]
        + state_arrays["not_specified_no_information"]
    )
    missed = state_arrays["not_specified_information_available"]
    conditional = {
        "filled_fully_supported": _metric(
            "conditional.filled_fully_supported", full, filled, strata, replicates
        ),
        "filled_partially_supported": _metric(
            "conditional.filled_partially_supported", partial, filled, strata, replicates
        ),
        "filled_unsupported": _metric(
            "conditional.filled_unsupported", unsupported, filled, strata, replicates
        ),
        "not_specified_rate": _metric(
            "conditional.not_specified_rate",
            not_specified,
            all_slots,
            strata,
            replicates,
        ),
        "information_available_given_not_specified": _metric(
            "conditional.information_available_given_not_specified",
            missed,
            not_specified,
            strata,
            replicates,
        ),
    }

    ethical_set = set(ETHICAL_LEGAL_PATHS)
    other_paths = tuple(path for path in ALLOWLISTED_PATHS if path not in ethical_set)
    if len(other_paths) != 20:
        raise ValueError("ethical/legal comparison no longer has exactly 20 other paths")

    def count_paths(record: dict, paths: tuple[str, ...], states: set[str]) -> int:
        return sum(record["states"][path] in states for path in paths)

    ns_states = {
        "not_specified_information_available",
        "not_specified_no_information",
    }
    ethical_arrays = {
        "not_specified": np.array(
            [count_paths(record, ETHICAL_LEGAL_PATHS, ns_states) for record in records],
            dtype=float,
        ),
        "no_information": np.array(
            [
                count_paths(
                    record,
                    ETHICAL_LEGAL_PATHS,
                    {"not_specified_no_information"},
                )
                for record in records
            ],
            dtype=float,
        ),
        "information_available": np.array(
            [
                count_paths(
                    record,
                    ETHICAL_LEGAL_PATHS,
                    {"not_specified_information_available"},
                )
                for record in records
            ],
            dtype=float,
        ),
    }
    other_arrays = {
        "not_specified": np.array(
            [count_paths(record, other_paths, ns_states) for record in records],
            dtype=float,
        ),
        "no_information": np.array(
            [
                count_paths(record, other_paths, {"not_specified_no_information"})
                for record in records
            ],
            dtype=float,
        ),
        "information_available": np.array(
            [
                count_paths(
                    record,
                    other_paths,
                    {"not_specified_information_available"},
                )
                for record in records
            ],
            dtype=float,
        ),
    }
    ethical_den = np.full(len(records), len(ETHICAL_LEGAL_PATHS), dtype=float)
    other_den = np.full(len(records), len(other_paths), dtype=float)
    ethical_metrics = {}
    other_metrics = {}
    differences = {}
    for label in ("not_specified", "no_information", "information_available"):
        ethical_metrics[label] = _metric(
            f"ethical_legal.{label}",
            ethical_arrays[label],
            ethical_den,
            strata,
            replicates,
        )
        other_metrics[label] = _metric(
            f"other_fields.{label}",
            other_arrays[label],
            other_den,
            strata,
            replicates,
        )
        differences[label] = _difference_metric(
            f"ethical_legal_minus_other.{label}",
            ethical_arrays[label],
            ethical_den,
            other_arrays[label],
            other_den,
            strata,
            replicates,
        )

    all_three_ns = np.array(
        [
            all(record["states"][path] in ns_states for path in ETHICAL_LEGAL_PATHS)
            for record in records
        ],
        dtype=float,
    )
    all_three_no_info = np.array(
        [
            all(
                record["states"][path] == "not_specified_no_information"
                for path in ETHICAL_LEGAL_PATHS
            )
            for record in records
        ],
        dtype=float,
    )
    ethical_legal = {
        "paths": list(ETHICAL_LEGAL_PATHS),
        "comparison_paths": list(other_paths),
        "held_out": {
            "ethical_legal_fields": ethical_metrics,
            "other_20_fields": other_metrics,
            "paired_differences": differences,
            "cards_all_three_not_specified": _metric(
                "ethical_legal.cards_all_three_not_specified",
                all_three_ns,
                ones,
                strata,
                replicates,
                kind="proportion",
            ),
            "cards_all_three_not_specified_no_information": _metric(
                "ethical_legal.cards_all_three_not_specified_no_information",
                all_three_no_info,
                ones,
                strata,
                replicates,
                kind="proportion",
            ),
        },
    }

    matrix = []
    for path in ALLOWLISTED_PATHS:
        row = {"path": path, "states": {}}
        for state in FIVE_STATES:
            num = np.array(
                [record["states"][path] == state for record in records],
                dtype=float,
            )
            row["states"][state] = _metric(
                f"field.{path}.{state}",
                num,
                ones,
                strata,
                replicates,
                kind="proportion",
            )
        matrix.append(row)

    return (
        {"five_state": five_state, "conditional": conditional},
        ethical_legal,
        matrix,
    )


def _full_corpus_analysis(
    corpus_cards: str | Path,
    corpus_manifest_path: str | Path,
) -> tuple[dict, dict[str, dict]]:
    corpus_dir = _resolve(corpus_cards)
    card_paths = sorted(corpus_dir.glob("*.json"))
    corpus_manifest = _load_json(corpus_manifest_path)
    manifest_cards = corpus_manifest.get("cards") or {}
    if corpus_manifest.get("n_cards") != 530 or len(manifest_cards) != 530:
        raise ValueError("corpus manifest does not declare exactly 530 cards")
    actual_names = {path.stem for path in card_paths}
    if actual_names != set(manifest_cards):
        raise ValueError(
            "corpus card filenames differ from the published-corpus manifest"
        )
    for path in card_paths:
        expected = manifest_cards[path.stem]
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"{path.name}: bytes or SHA-256 differ from corpus manifest")

    cards = [(path, extract_card(_load_json(path))) for path in card_paths]
    expected_fields = tuple(ALLOWLISTED_PATHS)
    expected_field_set = set(expected_fields)

    field_counts: Counter[str] = Counter()
    absent_counts: Counter[str] = Counter()
    missing_field_mismatches = []
    ethical_field_mismatches = []
    all_three = 0
    for path, card in cards:
        literal_missing = set()
        for dotted in expected_fields:
            section, field = dotted.split(".", 1)
            if field not in card.get(section, {}):
                absent_counts[dotted] += 1
            elif is_not_specified(card[section][field]):
                field_counts[dotted] += 1
                literal_missing.add(dotted)

        stored_missing = set(card.get("missing_fields") or []) & expected_field_set
        if literal_missing != stored_missing:
            missing_field_mismatches.append(
                {
                    "card": path.stem,
                    "literal_only": sorted(literal_missing - stored_missing),
                    "stored_only": sorted(stored_missing - literal_missing),
                }
            )
        if (
            literal_missing & set(ETHICAL_LEGAL_PATHS)
            != stored_missing & set(ETHICAL_LEGAL_PATHS)
        ):
            ethical_field_mismatches.append(path.stem)
        if all(field in literal_missing for field in ETHICAL_LEGAL_PATHS):
            all_three += 1

    if ethical_field_mismatches:
        raise ValueError(
            "ethical/legal literal Not specified values disagree with stored "
            f"missing_fields for {len(ethical_field_mismatches)} cards"
        )

    per_field = {
        field: {
            "not_specified_count": field_counts[field],
            "absent_count": absent_counts[field],
            "denominator": len(card_paths),
            "not_specified_rate": field_counts[field] / len(card_paths),
        }
        for field in expected_fields
    }
    ethical_total = sum(field_counts[field] for field in ETHICAL_LEGAL_PATHS)
    ethical_slots = len(card_paths) * len(ETHICAL_LEGAL_PATHS)
    summary = {
        "n_cards": len(card_paths),
        "n_census_paths": len(expected_fields),
        "published_dataset": corpus_manifest.get("dataset"),
        "published_revision": corpus_manifest.get("revision"),
        "published_corpus_fingerprint_md5": corpus_manifest.get(
            "published_corpus_fingerprint_md5"
        ),
        "corpus_sha256": _corpus_digest(card_paths),
        "ethical_legal_not_specified_count": ethical_total,
        "ethical_legal_slots": ethical_slots,
        "ethical_legal_not_specified_rate": ethical_total / ethical_slots,
        "cards_all_three_not_specified_count": all_three,
        "cards_all_three_not_specified_rate": all_three / len(card_paths),
        "stored_missing_fields_mismatch_cards_across_23_paths": len(
            missing_field_mismatches
        ),
        "stored_missing_fields_mismatch_cards_for_ethical_legal_paths": len(
            ethical_field_mismatches
        ),
        "interpretation": (
            "Census of generated-card output. Not specified does not establish "
            "non-applicability, author non-disclosure, or absence from all public sources."
        ),
    }
    return summary, per_field


def _confirmed_unsupported(
    human_key_path: str | Path,
    ratings_paths: list[str | Path],
    adjudication_path: str | Path,
) -> dict:
    key = _load_json(human_key_path)
    key_rows = validate_key(key)
    rating_paths_resolved = [str(_resolve(path)) for path in ratings_paths]
    require_distinct_rating_files(rating_paths_resolved)
    rating_tables = [read_rating_table(path) for path in rating_paths_resolved]
    adjudications = read_adjudications(str(_resolve(adjudication_path)))
    joined = validate_and_join(key_rows, rating_tables, adjudications)

    unsupported_items = [
        item
        for item in joined
        if item.get("error_status_stratum") == "unsupported"
        and item.get("judge_label") == "unsupported"
    ]
    if len(unsupported_items) != 30:
        raise ValueError(
            f"expected the 30-item judge-unsupported census, found {len(unsupported_items)}"
        )

    rows = []
    for item in unsupported_items:
        reference = item["human_reference"]
        if not reference:
            raise ValueError(f"{item['item_id']}: no majority or adjudicated reference")
        rows.append(
            {
                "item_id": item["item_id"],
                "card": item["card"],
                "path": item["field_path"],
                "human_reference": reference,
                "confirmed": reference == "unsupported",
            }
        )
    confirmed = [row for row in rows if row["confirmed"]]
    by_path = Counter(row["path"] for row in confirmed)
    if len(confirmed) != 27:
        raise ValueError(f"expected 27 human-confirmed unsupported calls, found {len(confirmed)}")
    return {
        "judge_unsupported_census_size": len(rows),
        "human_confirmed_unsupported": len(confirmed),
        "by_path": dict(sorted(by_path.items(), key=lambda item: (-item[1], item[0]))),
        "items": rows,
        "scope_note": (
            "Distribution among human-confirmed items selected because the source judge "
            "classified them as unsupported. This is not a corpus-wide human error taxonomy."
        ),
    }


def matched_judged_paths(field_locator: str) -> list[str]:
    """Return exact judged field paths named in a free-text screen locator."""
    matched = []
    for path in ALLOWLISTED_PATHS:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(path)}(?![A-Za-z0-9_])"
        if re.search(pattern, field_locator or ""):
            matched.append(path)
    return matched


def _cross_instrument_overlap(
    verifier_rows: list[dict],
    records: list[dict],
) -> dict:
    verdicts_by_card = {record["name"]: record["verdicts"] for record in records}
    material = [
        row for row in verifier_rows if row.get("verifier_label") == "confirmed-material"
    ]

    mapping = []
    matched_findings = 0
    matched_cards = set()
    for row in material:
        paths = matched_judged_paths(row.get("field", ""))
        if paths:
            matched_findings += 1
            matched_cards.add(row["card"])
        for path in paths:
            if row["card"] not in verdicts_by_card:
                raise ValueError(f"{row['card']}: verifier card not present in judge sample")
            status = verdicts_by_card[row["card"]][path]["status"]
            mapping.append(
                {
                    "row_id": int(row["row_id"]),
                    "card": row["card"],
                    "screen_category": row["category"],
                    "screen_field_locator": row["field"],
                    "matched_path": path,
                    "source_judge_status": status,
                }
            )

    counts = Counter(row["source_judge_status"] for row in mapping)
    fully_supported = counts["supported"] + counts["supported_by_eee_only"]
    expected_counts = {
        "supported": 20,
        "supported_by_eee_only": 9,
        "partial": 17,
        "unsupported": 6,
    }
    if (
        len(material) != 111
        or matched_findings != 43
        or len(mapping) != 52
        or dict(counts) != expected_counts
        or len(matched_cards) != 35
    ):
        raise ValueError(
            "cross-instrument freeze check failed: "
            f"material={len(material)}, matched_findings={matched_findings}, "
            f"checks={len(mapping)}, cards={len(matched_cards)}, counts={dict(counts)}"
        )
    return {
        "confirmed_material_findings": len(material),
        "findings_without_exact_judged_path_match": len(material) - matched_findings,
        "findings_naming_at_least_one_exact_judged_path": matched_findings,
        "matched_field_checks": len(mapping),
        "cards_with_matched_checks": len(matched_cards),
        "source_judge_status_counts": expected_counts,
        "fully_supported_checks": fully_supported,
        "mapping": mapping,
        "scope_note": (
            "Counts among exact canonical judged paths named by screen-selected, "
            "verifier-confirmed material findings. No exact match does not prove that "
            "a finding is conceptually outside the 23-field universe. Findings may "
            "name multiple paths, checks are not independent, and these counts are "
            "not a population error rate."
        ),
    }


def _reproduction_checks(derived: dict, prior_summary: dict) -> dict:
    expected = {
        "filled_fully_supported": "judge.support_rate_incl_eee",
        "filled_partially_supported": "judge.partial_rate",
        "filled_unsupported": "judge.unsupported_rate",
        "not_specified_rate": "judge.ns_rate_judged_fields",
        "information_available_given_not_specified": "judge.completeness_miss_rate",
    }
    checks = {}
    metrics = prior_summary["metrics"]
    for derived_key, prior_key in expected.items():
        current = derived["conditional"][derived_key]["value"]
        prior = metrics[prior_key]["value"]
        delta = current - prior
        passed = bool(np.isclose(current, prior, atol=1e-12))
        checks[derived_key] = {
            "derived_value": current,
            "prior_metric": prior_key,
            "prior_value": prior,
            "difference": delta,
            "passed": passed,
        }
        if not passed:
            raise ValueError(f"reproduction check failed for {derived_key}: {current} vs {prior}")
    return checks


def _write_matrix(path: str | Path, matrix: list[dict], full_corpus: dict[str, dict]):
    output_path = _resolve(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "path",
        "corpus_not_specified_count",
        "corpus_absent_count",
        "corpus_denominator",
        "corpus_not_specified_rate",
    ]
    for state in FIVE_STATES:
        columns.extend(
            [
                f"{state}_estimate",
                f"{state}_ci95_low",
                f"{state}_ci95_high",
                f"{state}_ci_method",
                f"{state}_raw_count",
                f"{state}_raw_denominator",
                f"{state}_by_stratum_counts",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in matrix:
            corpus = full_corpus[entry["path"]]
            row = {
                "path": entry["path"],
                "corpus_not_specified_count": corpus["not_specified_count"],
                "corpus_absent_count": corpus["absent_count"],
                "corpus_denominator": corpus["denominator"],
                "corpus_not_specified_rate": corpus["not_specified_rate"],
            }
            for state in FIVE_STATES:
                metric = entry["states"][state]
                row[f"{state}_estimate"] = metric["value"]
                row[f"{state}_ci95_low"] = metric["ci95"][0]
                row[f"{state}_ci95_high"] = metric["ci95"][1]
                row[f"{state}_ci_method"] = metric["ci_method"]
                row[f"{state}_raw_count"] = int(metric["counts"]["num"])
                row[f"{state}_raw_denominator"] = int(metric["counts"]["den"])
                row[f"{state}_by_stratum_counts"] = json.dumps(
                    metric["counts"]["by_stratum"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            writer.writerow(row)


def run_analysis(
    *,
    sample_path: str | Path,
    judge_path: str | Path,
    prior_summary_path: str | Path,
    verifier_path: str | Path,
    screen_lock_path: str | Path,
    human_key_path: str | Path,
    ratings_paths: list[str | Path],
    adjudication_path: str | Path,
    corpus_cards: str | Path,
    corpus_manifest_path: str | Path,
    output_path: str | Path,
    matrix_output_path: str | Path,
    bootstrap_replicates: int = 5000,
    seed: int = 20260704,
) -> dict:
    """Validate the frozen inputs, write both derived artifacts, and return JSON."""
    check()
    sample = _load_json(sample_path)
    validate_sample_design(sample)
    judge = _load_json(judge_path)
    try:
        validate_judge_analysis_frame(judge)
    except ValueError as exc:
        raise ValueError(f"invalid source-judge analysis frame: {exc}") from exc
    records = _build_card_records(sample, judge)
    derived, ethical_legal, matrix = _sample_analysis(
        sample,
        records,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    corpus_summary, corpus_per_field = _full_corpus_analysis(
        corpus_cards,
        corpus_manifest_path,
    )
    ethical_legal["full_corpus"] = {
        **corpus_summary,
        "fields": {path: corpus_per_field[path] for path in ETHICAL_LEGAL_PATHS},
    }
    for row in matrix:
        row["full_corpus"] = corpus_per_field[row["path"]]

    confirmed = _confirmed_unsupported(
        human_key_path,
        ratings_paths,
        adjudication_path,
    )
    screen_lock = _load_json(screen_lock_path)
    validate_lock(screen_lock)
    verifier_rows = validate_verifier_return(_resolve(verifier_path), screen_lock)
    overlap = _cross_instrument_overlap(verifier_rows, records)
    reproduction = _reproduction_checks(derived, _load_json(prior_summary_path))

    inputs = {
        "sample": {"sha256": _sha256(sample_path)},
        "judge": {"sha256": _sha256(judge_path)},
        "prior_summary": {"sha256": _sha256(prior_summary_path)},
        "verifier": {"sha256": _sha256(verifier_path)},
        "screen_lock": {"sha256": _sha256(screen_lock_path)},
        "human_key": {"sha256": _sha256(human_key_path)},
        "ratings": [{"sha256": _sha256(path)} for path in ratings_paths],
        "adjudication": {"sha256": _sha256(adjudication_path)},
        "corpus_cards": {
            "n_cards": corpus_summary["n_cards"],
            "aggregate_sha256": corpus_summary["corpus_sha256"],
        },
        "corpus_manifest": {"sha256": _sha256(corpus_manifest_path)},
    }
    output = {
        "schema_version": 1,
        "analysis_version": "paper-extensions-v1-2026-07-25",
        "design": {
            "sample_tag": sample.get("tag"),
            "n_cards": len(records),
            "n_judged_paths": len(ALLOWLISTED_PATHS),
            "n_field_slots": len(records) * len(ALLOWLISTED_PATHS),
            "strata": sample["strata"],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "estimator": "stratified combined ratio over card-level counts",
            "aggregate_interval": "stratified whole-card percentile bootstrap",
            "rare_event_fallback": (
                "Wilson interval at bootstrap-implied effective sample size when "
                "the raw event count is below 10 or an eligible stratum has no events"
            ),
            "field_matrix_ci_method_recorded_per_cell": True,
            "fpc_ignored": True,
        },
        "inputs": inputs,
        "reproduction_checks": reproduction,
        "field_slot_outcomes": derived,
        "ethical_legal_coverage": ethical_legal,
        "human_confirmed_unsupported": confirmed,
        "cross_instrument_overlap": overlap,
        "field_matrix": matrix,
        "claim_guards": [
            "Source-judge outcomes are relative to the evidence supplied to the judge.",
            "Not specified is a card-output state, not proof of public non-disclosure.",
            "The ethical/legal grouping does not establish regulatory noncompliance.",
            "The overlap is selected-candidate evidence, not a population error rate.",
            "The two automated instruments used the same model family.",
        ],
    }

    resolved_output = _resolve(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_matrix(matrix_output_path, matrix, corpus_per_field)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="eval/s150/sample.json")
    parser.add_argument("--judge", default="eval/s150/judge/analysis_frame.json")
    parser.add_argument("--prior-summary", default="eval/s150/judge/summary.json")
    parser.add_argument("--verifier", default="eval/s150/screen/verifier_ratings.csv")
    parser.add_argument("--screen-lock", default="eval/s150/screen/scoring_lock.json")
    parser.add_argument("--human-key", default="eval/s150/human_validation/key.json")
    parser.add_argument(
        "--ratings",
        nargs=3,
        default=[
            "eval/s150/human_validation/ratings_r1.csv",
            "eval/s150/human_validation/ratings_r2.csv",
            "eval/s150/human_validation/ratings_r3.csv",
        ],
    )
    parser.add_argument(
        "--adjudication",
        default="eval/s150/human_validation/adjudication.csv",
    )
    parser.add_argument("--corpus-cards", required=True)
    parser.add_argument("--corpus-manifest", default="eval/corpus/manifest.json")
    parser.add_argument(
        "--out",
        default="eval/s150/paper_extension_analysis.json",
    )
    parser.add_argument(
        "--matrix-out",
        default="eval/s150/paper_extension_field_matrix.csv",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    output = run_analysis(
        sample_path=args.sample,
        judge_path=args.judge,
        prior_summary_path=args.prior_summary,
        verifier_path=args.verifier,
        screen_lock_path=args.screen_lock,
        human_key_path=args.human_key,
        ratings_paths=args.ratings,
        adjudication_path=args.adjudication,
        corpus_cards=args.corpus_cards,
        corpus_manifest_path=args.corpus_manifest,
        output_path=args.out,
        matrix_output_path=args.matrix_out,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    print(f"analysis -> {_display_path(args.out)}")
    print(f"field matrix -> {_display_path(args.matrix_out)}")
    print("five-state estimates:")
    for state in FIVE_STATES:
        metric = output["field_slot_outcomes"]["five_state"][state]
        print(f"  {state:46s} {100 * metric['value']:.2f}% "
              f"[{100 * metric['ci95'][0]:.2f}, {100 * metric['ci95'][1]:.2f}]")
    full = output["ethical_legal_coverage"]["full_corpus"]
    print(
        "ethical/legal full-corpus Not specified: "
        f"{full['ethical_legal_not_specified_count']}/"
        f"{full['ethical_legal_slots']} "
        f"({100 * full['ethical_legal_not_specified_rate']:.2f}%)"
    )
    print(
        "cross-instrument overlap: "
        f"{output['cross_instrument_overlap']['fully_supported_checks']}/"
        f"{output['cross_instrument_overlap']['matched_field_checks']} fully supported"
    )


if __name__ == "__main__":
    main()
