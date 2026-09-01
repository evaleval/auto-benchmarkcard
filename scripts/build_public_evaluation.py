"""Build the public evaluation snapshot from the private frozen workspace.

The private workspace contains raw participant returns, copied source text,
request identifiers, and local paths. Those files must not be copied into a
public release. This script uses an explicit allowlist and produces the
documented public projections.

Participant labels and evidence URLs are preserved. Optional participant
notes are removed uniformly because they are not used by either scorer.
Author-side contamination notes are retained because they are part of the
published screen score record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analyze_corpus_schema_fill import analyze as analyze_corpus_schema_fill
from analyze_paper_extensions import run_analysis


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CORPUS_REVISION = "0a86cea5b55d6070bd7f1f020f01281e1631adba"
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
RATING_COLUMNS = (
    "item_id",
    "card",
    "field_path",
    "kind",
    "field_value",
    "human_label",
    "human_note",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_release_static_files(out_root: Path) -> None:
    """Copy reviewed public-only documents and frozen exposure inputs.

    The two source-complexity exposure files cannot be regenerated without the
    withheld source-run snapshots. They contain no copied source text, local
    paths, participant identities, or outcomes and are therefore kept as
    explicit release inputs. The statistical join is regenerated below from
    the public projections.
    """
    relative_paths = (
        "eval/README.md",
        "eval/s150/PAPER_EXTENSION_ANALYSIS.md",
        "eval/s150/source_complexity/README.md",
        "eval/s150/source_complexity/exposure_reconstruction_audit.json",
        "eval/s150/source_complexity/exposures_outcome_free.csv",
    )
    for relative in relative_paths:
        source = REPO_ROOT / relative
        destination = out_root / Path(relative).relative_to("eval")
        copy_file(source, destination)


def redact_public_text(value: Any) -> Any:
    if isinstance(value, str):
        return EMAIL_RE.sub("[redacted email]", value)
    if isinstance(value, list):
        return [redact_public_text(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_public_text(item) for key, item in value.items()}
    return value


def project_csv(
    source: Path,
    destination: Path,
    *,
    blank_columns: tuple[str, ...] = (),
    expected_rows: int | None = None,
    expected_columns: tuple[str, ...] | None = None,
) -> None:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)

    if expected_columns is not None and columns != expected_columns:
        raise ValueError(
            f"{source}: columns differ from the public contract: "
            f"{columns!r} != {expected_columns!r}"
        )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"{source}: row count differs from the public contract: "
            f"{len(rows)} != {expected_rows}"
        )
    for column in blank_columns:
        if column not in columns:
            raise ValueError(f"{source}: missing note column {column!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            projected = {
                column: EMAIL_RE.sub("[redacted email]", row.get(column, ""))
                for column in columns
            }
            for column in blank_columns:
                projected[column] = ""
            writer.writerow(projected)


def sanitize_human_key(source: Path, destination: Path) -> None:
    key = load_json(source)
    key["judge_artifact"]["file"] = "eval/s150/judge/analysis_frame.json"
    key["sample_artifact"]["file"] = "eval/s150/sample.json"
    key.pop("source_files", None)
    key["public_projection"] = {
        "raw_source_snapshots_included": False,
        "reason": (
            "Third-party source snapshots and local preparation paths are not "
            "part of the public release."
        ),
    }
    write_json(destination, key)


def sanitize_screen_lock(source: Path, destination: Path) -> None:
    lock = redact_public_text(load_json(source))
    packet_rows = lock["packet"]["rows"]
    lock["packet"]["protected_rows_sha256"] = canonical_sha256(packet_rows)

    rows_by_id = {row["row_id"]: row for row in packet_rows}
    overlap_rows = [
        rows_by_id[row_id] for row_id in lock["author_overlap"]["row_ids"]
    ]
    lock["author_overlap"]["protected_rows_sha256"] = canonical_sha256(overlap_rows)
    contamination_rows = lock["contamination_check"]["rows"]
    lock["contamination_check"]["protected_rows_sha256"] = canonical_sha256(
        contamination_rows
    )
    lock["public_projection"] = {
        "email_redaction": (
            "One public academic email address in a protected finding was "
            "replaced with [redacted email]. Internal protected-row hashes were "
            "recomputed over this public projection."
        ),
        "raw_returns_included": False,
    }
    write_json(destination, lock)


def corpus_fingerprint(card_files: list[tuple[str, Path]]) -> str:
    digest = hashlib.md5()
    for name, path in card_files:
        digest.update(f"{name} {md5_file(path)}\n".encode())
    return digest.hexdigest()


def build_corpus_manifest(source_root: Path, out_root: Path, sample: dict) -> dict:
    cards_dir = source_root / "output/auto-benchmarkcards-v3/cards"
    card_files = sorted((path.stem, path) for path in cards_dir.glob("*.json"))
    if len(card_files) != 530:
        raise ValueError(f"expected 530 frozen cards, found {len(card_files)}")

    sample_mismatches = []
    for row in sample["cards"]:
        path = cards_dir / f"{row['name']}.json"
        actual = md5_file(path)
        if actual != row["corpus_card_md5"]:
            sample_mismatches.append(
                {
                    "card": row["name"],
                    "sample_md5": row["corpus_card_md5"],
                    "final_md5": actual,
                }
            )
    if sample_mismatches:
        raise ValueError(f"sample card mismatch: {sample_mismatches[:3]}")

    strata: dict[str, list[str]] = {"flagged": [], "unflagged": []}
    for path_name, path in card_files:
        card = load_json(path)
        if "benchmark_card" in card and isinstance(card["benchmark_card"], dict):
            card = card["benchmark_card"]
        stratum = "flagged" if (card.get("flagged_fields") or {}) else "unflagged"
        strata[stratum].append(path_name)

    import random

    rng = random.Random(sample["seed"])
    reproduced = []
    for stratum in ("flagged", "unflagged"):
        reproduced.extend(sorted(rng.sample(sorted(strata[stratum]), 75)))
    selected = sorted(row["name"] for row in sample["cards"])
    if sorted(reproduced) != selected:
        raise ValueError("seeded sample does not reproduce against the final corpus")

    manifest = {
        "schema_version": 1,
        "dataset": "evaleval/auto-benchmarkcards",
        "revision": HF_CORPUS_REVISION,
        "url": (
            "https://huggingface.co/datasets/evaleval/auto-benchmarkcards/"
            f"tree/{HF_CORPUS_REVISION}"
        ),
        "n_cards": len(card_files),
        "published_corpus_fingerprint_md5": corpus_fingerprint(card_files),
        "sample_recorded_corpus_fingerprint_md5": sample["corpus"][
            "fingerprint_md5"
        ],
        "verification": {
            "sampled_cards_md5_match_final": len(sample["cards"]),
            "sampled_cards_md5_mismatched": 0,
            "seeded_selection_reproduces": True,
            "final_strata": {
                "flagged": len(strata["flagged"]),
                "unflagged": len(strata["unflagged"]),
            },
        },
        "cards": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in card_files
        },
    }
    write_json(out_root / "corpus/manifest.json", manifest)
    return manifest


def build_run_summaries(source_root: Path, out_root: Path) -> None:
    judge = load_json(
        source_root / "eval/s150/judge_source_parity_merged/run_meta.json"
    )
    merge = judge["merge"]
    judge_summary = {
        "model": judge["model"],
        "transport": judge["transport"],
        "execution": judge["execution"],
        "input_mode": judge["input_mode"],
        "prompt_version": judge["prompt_version"],
        "prompt_md5": judge["prompt_md5"],
        "schema_md5": judge["schema_md5"],
        "inline_appendix_md5": judge["inline_appendix_md5"],
        "request_params": judge["request_params"],
        "sdk_version": judge["sdk_version"],
        "total_cost_est_usd": judge["total_cost_est_usd"],
        "n_cards": merge["n_cards"],
        "n_retained": merge["n_retained"],
        "n_corrected": merge["n_corrected"],
        "n_source_corrected": merge["n_source_corrected"],
        "n_schema_corrected": merge["n_schema_corrected"],
        "component_cost_est_usd": merge["component_cost_est_usd"],
        "created_at": merge["created_at"],
        "request_ids_included": False,
    }
    write_json(out_root / "s150/judge/run_summary.json", judge_summary)

    screen = load_json(source_root / "eval/s150/screen/run_meta.json")
    screen_summary = {
        "model": screen["model"],
        "transport": screen["transport"],
        "prompt_prefix_md5": screen["prompt_prefix_md5"],
        "config_md5": screen["config_md5"],
        "request_params": screen["request_params"],
        "sdk_version": screen["sdk_version"],
        "total_cost_est_usd": screen["total_cost_est_usd"],
        "total_usage": screen["total_usage"],
        "total_web_searches": screen["total_web_searches"],
        "web_search_cost_usd_per_request": screen[
            "web_search_cost_usd_per_request"
        ],
        "request_ids_included": False,
    }
    write_json(out_root / "s150/screen/run_summary.json", screen_summary)


def build_paper_extension_analysis(source_root: Path, out_root: Path) -> dict:
    """Derive the paper extension outputs from the sanitized public inputs."""
    human = out_root / "s150/human_validation"
    screen = out_root / "s150/screen"
    return run_analysis(
        sample_path=out_root / "s150/sample.json",
        judge_path=out_root / "s150/judge/analysis_frame.json",
        prior_summary_path=out_root / "s150/judge/summary.json",
        verifier_path=screen / "verifier_ratings.csv",
        screen_lock_path=screen / "scoring_lock.json",
        human_key_path=human / "key.json",
        ratings_paths=[
            human / "ratings_r1.csv",
            human / "ratings_r2.csv",
            human / "ratings_r3.csv",
        ],
        adjudication_path=human / "adjudication.csv",
        corpus_cards=source_root / "output/auto-benchmarkcards-v3/cards",
        corpus_manifest_path=out_root / "corpus/manifest.json",
        output_path=out_root / "s150/paper_extension_analysis.json",
        matrix_output_path=out_root / "s150/paper_extension_field_matrix.csv",
    )


def build_corpus_schema_fill(source_root: Path, out_root: Path) -> None:
    """Recompute the fixed 40-field fill summary from the frozen card files."""
    result = analyze_corpus_schema_fill(
        source_root / "output/auto-benchmarkcards-v3/cards"
    )
    write_json(out_root / "corpus/schema_fill_summary.json", result)


def build_source_complexity_analysis(source_root: Path, out_root: Path) -> None:
    """Replay the public source-complexity join from frozen exposure inputs."""
    output_dir = out_root / "s150/source_complexity"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/analyze_source_complexity.py"),
        "--exposures",
        str(output_dir / "exposures_outcome_free.csv"),
        "--sample",
        str(out_root / "s150/sample.json"),
        "--judge",
        str(out_root / "s150/judge/analysis_frame.json"),
        "--verifier",
        str(out_root / "s150/screen/verifier_ratings.csv"),
        "--corpus-cards",
        str(source_root / "output/auto-benchmarkcards-v3/cards"),
        "--out-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def project_metric(metric: dict) -> dict:
    """Keep the concise result, interval, and raw design counts."""
    return {
        key: metric[key]
        for key in ("value", "ci95", "ci_method", "counts")
        if key in metric
    }


def build_results_summary(out_root: Path, corpus_manifest: dict) -> None:
    corpus = load_json(out_root / "corpus/corpus_stats.json")
    judge = load_json(out_root / "s150/judge/summary.json")
    human = load_json(out_root / "s150/human_validation/scores.json")
    screen = load_json(out_root / "s150/screen/verification_scores.json")
    extensions = load_json(out_root / "s150/paper_extension_analysis.json")
    metrics = judge["metrics"]
    five_state = extensions["field_slot_outcomes"]["five_state"]
    ethical = extensions["ethical_legal_coverage"]
    confirmed = extensions["human_confirmed_unsupported"]
    overlap = extensions["cross_instrument_overlap"]
    risk_metric = metrics["judge.risk_grounded_rate"]
    risk_grounded = int(risk_metric["counts"]["num"])
    risk_total = int(risk_metric["counts"]["den"])

    summary = {
        "schema_version": 3,
        "frozen_corpus": {
            "attempted": corpus["corpus"]["attempted"],
            "published": corpus_manifest["n_cards"],
            "completion_rate": corpus["corpus"]["completion_rate"],
            "hf_revision": corpus_manifest["revision"],
            "fingerprint_md5": corpus_manifest[
                "published_corpus_fingerprint_md5"
            ],
            "binding_paths": corpus["binding_paths"]["grouped"],
        },
        "sample": {
            "n_cards": judge["design"]["n_analyzed"],
            "seed": judge["design"]["seed_sample"],
            "flagged": judge["design"]["strata"]["flagged"],
            "unflagged": judge["design"]["strata"]["unflagged"],
        },
        "source_bounded_judge": {
            "scope": (
                "23 content fields per sampled card; rates are S-weighted over "
                "filled fields unless stated otherwise."
            ),
            "field_rows": metrics["judge.ns_rate_judged_fields"]["counts"]["den"],
            "filled_rows": metrics["judge.unsupported_rate"]["counts"]["den"],
            "not_specified_rows": metrics["judge.ns_rate_judged_fields"]["counts"][
                "num"
            ],
            "supported_including_eee": metrics[
                "judge.support_rate_incl_eee"
            ]["value"],
            "partial": metrics["judge.partial_rate"]["value"],
            "unsupported": metrics["judge.unsupported_rate"]["value"],
            "common_denominator_five_state": {
                state: project_metric(metric)
                for state, metric in five_state.items()
            },
            "ethical_legal_comparison": {
                "analysis_status": "post_hoc_schema_defined",
                "paths": ethical["paths"],
                "comparison_path_count": len(ethical["comparison_paths"]),
                "ethical_legal_fields": {
                    key: project_metric(metric)
                    for key, metric in ethical["held_out"][
                        "ethical_legal_fields"
                    ].items()
                },
                "other_20_fields": {
                    key: project_metric(metric)
                    for key, metric in ethical["held_out"]["other_20_fields"].items()
                },
                "paired_differences": {
                    key: project_metric(metric)
                    for key, metric in ethical["held_out"][
                        "paired_differences"
                    ].items()
                },
                "scope_guard": (
                    "No information means no fillable information in the evidence "
                    "supplied to the source judge. This comparison does not establish "
                    "public non-disclosure, non-applicability, or noncompliance."
                ),
            },
            "full_corpus_ethical_legal_coverage": {
                key: ethical["full_corpus"][key]
                for key in (
                    "n_cards",
                    "ethical_legal_not_specified_count",
                    "ethical_legal_slots",
                    "ethical_legal_not_specified_rate",
                    "cards_all_three_not_specified_count",
                    "cards_all_three_not_specified_rate",
                    "fields",
                    "interpretation",
                )
            },
        },
        "candidate_risk_source_judge": {
            "scope": (
                "Candidate annotations in the 150 sampled cards; separate from "
                "the fixed 40-field BenchmarkCard schema."
            ),
            "sample_counts": {
                "total": risk_total,
                "relevant_and_grounded": risk_grounded,
                "not_relevant_or_not_grounded": risk_total - risk_grounded,
            },
            "s_weighted_grounded_rate": project_metric(risk_metric),
            "human_validated": False,
            "headline_result": False,
            "interpretation": (
                "The paper reports the unweighted sample counts because these "
                "risk judgements were made by the automated source judge and were "
                "not checked by the human raters. Treat every candidate risk as a "
                "prompt for human review, not as a verified benchmark property."
            ),
        },
        "human_validation": {
            "n_raters": 3,
            "n_unique_items": human["n_unique_items"],
            "n_cards": 49,
            "n_adjudicated_three_way_splits": human["n_true_three_way_splits"],
            "probability_arm_filled": {
                "n_rows": human["probability_arm_corpus_weighted"]["filled"][
                    "n_sampled_rows"
                ],
                "weighted_agreement": human["probability_arm_corpus_weighted"][
                    "filled"
                ]["weighted_agreement"],
                "cohens_kappa": human["probability_arm_corpus_weighted"]["filled"][
                    "cohens_kappa"
                ],
            },
            "probability_arm_not_specified": {
                "n_rows": human["probability_arm_corpus_weighted"][
                    "not_specified"
                ]["n_sampled_rows"],
                "weighted_agreement": human["probability_arm_corpus_weighted"][
                    "not_specified"
                ]["weighted_agreement"],
            },
            "judge_error_call_confirmation": {
                "unsupported": human["error_arm_conditional_confirmation"][
                    "unsupported"
                ]["raw_confirms"],
                "unsupported_denominator": human[
                    "error_arm_conditional_confirmation"
                ]["unsupported"]["n"],
                "partial": human["error_arm_conditional_confirmation"]["partial"][
                    "raw_confirms"
                ],
                "partial_denominator": human[
                    "error_arm_conditional_confirmation"
                ]["partial"]["n"],
            },
            "confirmed_unsupported_by_field": {
                "judge_unsupported_census_size": confirmed[
                    "judge_unsupported_census_size"
                ],
                "human_confirmed_unsupported": confirmed[
                    "human_confirmed_unsupported"
                ],
                "by_path": confirmed["by_path"],
                "scope_note": confirmed["scope_note"],
            },
        },
        "public_source_screen": {
            "scope": screen["card_level"]["name"],
            "inference_guard": screen["card_level"]["inference_guard"],
            "n_findings": screen["finding_level"]["n_findings"],
            "raw_label_counts": screen["finding_level"]["raw_label_counts"],
            "n_cards_with_confirmed_material_finding_raw": screen["card_level"][
                "n_screen_detected_verifier_confirmed_material_cards_raw"
            ],
            "weighted_rate": screen["card_level"]["rate"],
            "approximate_design_interval95": [
                screen["card_level"]["approximate_design_interval95"]["lower"],
                screen["card_level"]["approximate_design_interval95"]["upper"],
            ],
            "cross_instrument_overlap": {
                key: overlap[key]
                for key in (
                    "confirmed_material_findings",
                    "findings_without_exact_judged_path_match",
                    "findings_naming_at_least_one_exact_judged_path",
                    "matched_field_checks",
                    "cards_with_matched_checks",
                    "source_judge_status_counts",
                    "fully_supported_checks",
                    "scope_note",
                )
            },
        },
        "validation_flags": {
            "scope": "overlapping 23-field filled-field judge universe",
            "n_flagged_fields_raw": metrics["flags.precision_strict"]["counts"]["den"],
            "n_unsupported_fields_raw": metrics["flags.recall_strict"]["counts"]["den"],
            "n_overlap_raw": metrics["flags.precision_strict"]["counts"]["num"],
            "weighted_precision": metrics["flags.precision_strict"]["value"],
            "weighted_recall": metrics["flags.recall_strict"]["value"],
            "weighted_miss_share": metrics["flags.miss_share_strict"]["value"],
        },
    }
    write_json(out_root / "results_summary.json", summary)


def build_provenance(source_root: Path, out_root: Path) -> None:
    source_files = {
        "human_design_artifact": "eval/s150/human_validation_v7/key.json",
        "ratings_r1": "eval/s150/human_validation_final/ratings_rater1.csv",
        "ratings_r2": "eval/s150/human_validation_final/ratings_rater2.csv",
        "ratings_r3": "eval/s150/human_validation_final/ratings_rater3.csv",
        "human_adjudication": (
            "eval/s150/human_validation_final/"
            "calibration_adjudication_completed.csv"
        ),
        "human_scores": (
            "eval/s150/human_validation_final/calibration_scores_final.json"
        ),
        "screen_lock": "eval/s150/screen_verification/scoring_lock.json",
        "screen_verifier_raw": (
            "eval/s150/screen_verification_final/"
            "ratings_verifier_v1_raw_copy.csv"
        ),
        "screen_author_overlap": (
            "eval/s150/screen_verification_final/author_overlap_completed.csv"
        ),
        "screen_contamination": (
            "eval/s150/screen_verification_final/contamination_completed.csv"
        ),
        "screen_scores": (
            "eval/s150/screen_verification_final/"
            "screen_verification_scores_final.json"
        ),
    }
    hashes = {
        name: sha256_file(source_root / relative)
        for name, relative in source_files.items()
    }
    provenance = {
        "schema_version": 1,
        "release_projection": "public-sanitized-v1",
        "raw_participant_returns_public": False,
        "participant_labels_changed": False,
        "participant_evidence_urls_changed": False,
        "participant_notes": (
            "Removed uniformly from R1-R3 and V1 public tables. Notes are not "
            "used by the final scorers."
        ),
        "r1_normalization": {
            "canonical_columns": list(RATING_COLUMNS),
            "rows": 75,
            "label_rule": (
                "Use final_label when non-empty; otherwise use auto_label. This "
                "normalization was confirmed with the participant before release."
            ),
            "normalized_private_input_sha256": hashes["ratings_r1"],
        },
        "private_input_sha256": hashes,
        "derived_outputs": {
            "producer_script": "scripts/analyze_paper_extensions.py",
            "script_sha256": sha256_file(
                Path(__file__).resolve().parent / "analyze_paper_extensions.py"
            ),
            "corpus": {
                "dataset": "evaleval/auto-benchmarkcards",
                "revision": HF_CORPUS_REVISION,
                "manifest_sha256": sha256_file(out_root / "corpus/manifest.json"),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "paper_extension_analysis": {
                "path": "s150/paper_extension_analysis.json",
                "sha256": sha256_file(
                    out_root / "s150/paper_extension_analysis.json"
                ),
            },
            "paper_extension_field_matrix": {
                "path": "s150/paper_extension_field_matrix.csv",
                "sha256": sha256_file(
                    out_root / "s150/paper_extension_field_matrix.csv"
                ),
            },
        },
    }
    write_json(out_root / "provenance.json", provenance)


def write_checksums(out_root: Path) -> None:
    checksum_path = out_root / "SHA256SUMS.txt"
    files = sorted(
        path
        for path in out_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(out_root).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Private integration worktree containing the frozen artifacts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval"),
        help="Public evaluation output directory.",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    out_root = args.out.resolve()
    copy_release_static_files(out_root)

    direct_copies = {
        "output/corpus_v3/corpus_stats_precheck.json": "corpus/corpus_stats.json",
        "output/corpus_v3/attempted_universe.txt": "corpus/attempted_universe.txt",
        "eval/hf_published.json": "corpus/hf_published.json",
        "eval/frozen_md5.json": "corpus/frozen_code_hashes_md5.json",
        "eval/s150/sample.json": "s150/sample.json",
        "eval/s150/manifest.json": "s150/manifest.json",
        "eval/s150/staging_report.json": "s150/staging_report.json",
        (
            "eval/s150/judge_s150_source_parity_analysis.json"
        ): "s150/judge/analysis_frame.json",
        (
            "eval/s150/analysis_s150_source_parity_judge_only.json"
        ): "s150/judge/summary.json",
        (
            "eval/s150/judge_source_parity_merged/results_s150.json"
        ): "s150/judge/results.json",
        (
            "eval/s150/judge_source_parity_merged/inputs_md5.json"
        ): "s150/judge/input_manifest.json",
        (
            "eval/s150/judge_source_parity_merged/verdicts_sha256.json"
        ): "s150/judge/verdict_manifest.json",
        (
            "eval/s150/human_validation_final/calibration_scores_final.json"
        ): "s150/human_validation/scores.json",
        (
            "eval/s150/screen_verification_final/"
            "screen_verification_scores_final.json"
        ): "s150/screen/verification_scores.json",
    }
    for source, destination in direct_copies.items():
        copy_file(source_root / source, out_root / destination)

    sample = load_json(out_root / "s150/sample.json")
    corpus_manifest = build_corpus_manifest(source_root, out_root, sample)
    build_corpus_schema_fill(source_root, out_root)
    build_run_summaries(source_root, out_root)

    screen_results = redact_public_text(
        load_json(source_root / "eval/s150/screen/screen_results.json")
    )
    write_json(out_root / "s150/screen/screen_results.json", screen_results)

    sanitize_human_key(
        source_root / "eval/s150/human_validation_v7/key.json",
        out_root / "s150/human_validation/key.json",
    )
    human_final = source_root / "eval/s150/human_validation_final"
    for index in (1, 2, 3):
        project_csv(
            human_final / f"ratings_rater{index}.csv",
            out_root / f"s150/human_validation/ratings_r{index}.csv",
            blank_columns=("human_note",),
            expected_rows=75,
            expected_columns=RATING_COLUMNS,
        )
    project_csv(
        human_final / "calibration_adjudication_completed.csv",
        out_root / "s150/human_validation/adjudication.csv",
        expected_rows=1,
    )

    screen_base = source_root / "eval/s150"
    sanitize_screen_lock(
        screen_base / "screen_verification/scoring_lock.json",
        out_root / "s150/screen/scoring_lock.json",
    )
    project_csv(
        screen_base
        / "screen_verification_final/ratings_verifier_v1_raw_copy.csv",
        out_root / "s150/screen/verifier_ratings.csv",
        blank_columns=("notes",),
        expected_rows=154,
    )
    project_csv(
        screen_base / "screen_verification_final/author_overlap_completed.csv",
        out_root / "s150/screen/author_overlap.csv",
        blank_columns=("author_notes",),
        expected_rows=23,
    )
    project_csv(
        screen_base / "screen_verification_final/contamination_completed.csv",
        out_root / "s150/screen/contamination.csv",
        expected_rows=3,
    )

    build_paper_extension_analysis(source_root, out_root)
    build_source_complexity_analysis(source_root, out_root)
    build_results_summary(out_root, corpus_manifest)
    build_provenance(source_root, out_root)
    write_checksums(out_root)
    print(f"public evaluation snapshot written to {out_root}")


if __name__ == "__main__":
    main()
