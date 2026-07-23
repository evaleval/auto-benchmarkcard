"""Build the analysis-safe view of the S faithfulness-judge results.

The original judge results are immutable evidence.  This script reads them,
checks them against the sampled cards, and writes a derived artifact containing
only the 23 predeclared evidence-assessable content fields. Deterministic,
display-only, and other structured bookkeeping fields can never enter the
derived view.

Every allowlisted field and every expected risk category must occur exactly
once. Historical malformed verdicts must be corrected upstream and retained in
the merge lineage; this projection never chooses between duplicate results.

Usage:
  python scripts/build_judge_analysis_view.py \
      --raw-results eval/s150/judge/results_s150.json \
      --verdict-dir eval/s150/judge/verdicts \
      --run-meta eval/s150/judge/run_meta.json \
      --manifest eval/s150/manifest.json \
      --label s150 \
      --out eval/s150/judge_s150_analysis.json

Omit ``--out`` for a read-only validation run.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile


REPO = Path(__file__).resolve().parents[1]
EXPECTED_N_CARDS = 150
EXPECTED_N_ROWS = 3450

# Positive, ordered analysis frame.  These are the substantive content fields
# authored for every schema-v2 card in S.  Do not replace this with a negative
# skip list: newly added structured fields must be reviewed before they can
# become part of the measurement construct.
ALLOWLISTED_PATHS = (
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

# Exact non-content paths the historical judge may have emitted.  They are
# accepted as raw provenance but always removed.  Any other path outside the
# positive frame is an error, which makes the projection fail closed.
KNOWN_NONCONTENT_PATHS = frozenset({
    "benchmark_details.authors",
    "benchmark_details.logo",
    "benchmark_details.org_url",
    "benchmark_details.languages",
    "benchmark_details.data_type",
    "benchmark_details.resources",
    "benchmark_details.appears_in",
    "benchmark_details.benchmark_type",
    "benchmark_details.contains",
    "data.size",
    "data.format",
    "data.size_breakdown",
    "data.collection_date",
    "methodology.metrics",
    "methodology.judge_uses_llm",
    "methodology.judge_num",
    "methodology.judge_models",
    "ethical_and_legal_considerations.data_licensing",
})

DECLARED_EXCLUSIONS = ()

FIELD_STATUSES = frozenset({
    "supported", "supported_by_eee_only", "partial", "unsupported",
    "not_specified",
})
SPECIFICITY_VALUES = frozenset({"specific", "generic", "na"})
INFO_VALUES = frozenset({"yes_primary", "yes_eee_only", "yes", "no", "na"})


class FrameValidationError(ValueError):
    """The raw artifact cannot be projected into the declared analysis frame."""


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _verify_verdict_dir(raw_results, verdict_dir):
    """Cross-check the combined raw file against its per-card source files."""
    verdict_dir = Path(verdict_dir)
    if not verdict_dir.is_dir():
        raise FrameValidationError(f"verdict directory does not exist: {verdict_dir}")
    entries = sorted(verdict_dir.iterdir(), key=lambda path: path.name)
    if any(not path.is_file() or path.suffix != ".json" for path in entries):
        bad = [path.name for path in entries if not path.is_file() or path.suffix != ".json"]
        raise FrameValidationError(
            f"verdict directory contains non-JSON entries: {', '.join(bad)}"
        )
    expected_files = {f"{name}.json" for name in raw_results}
    actual_files = {path.name for path in entries}
    if actual_files != expected_files:
        raise FrameValidationError(
            "verdict directory/results mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    files = []
    for path in entries:
        name = path.stem
        verdict = _load(path)
        if verdict != raw_results[name]:
            raise FrameValidationError(
                f"{name}: per-card verdict differs from combined raw results"
            )
        files.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {
        "n_files": len(files),
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
        "files": files,
    }


def _judge_info_from_run_meta(run_meta, metadata=None):
    if not isinstance(run_meta, dict):
        raise FrameValidationError("run metadata is not a JSON object")
    required = ("model", "prompt_version", "transport", "execution")
    missing = [key for key in required
               if not isinstance(run_meta.get(key), str) or not run_meta[key]]
    if missing:
        raise FrameValidationError(
            f"run metadata is missing required values: {', '.join(missing)}"
        )
    params = run_meta.get("request_params")
    if not isinstance(params, dict):
        raise FrameValidationError("run metadata has no request_params object")
    if params.get("model") != run_meta["model"]:
        raise FrameValidationError(
            "run metadata model conflicts with request_params.model"
        )
    temperature = params.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise FrameValidationError("run metadata has no numeric request temperature")

    batches = run_meta.get("batches")
    if not isinstance(batches, list) or not batches:
        raise FrameValidationError("run metadata has no batch timestamps")
    judged_dates = set()
    for index, batch in enumerate(batches):
        started = batch.get("started") if isinstance(batch, dict) else None
        if not isinstance(started, str):
            raise FrameValidationError(f"run metadata batch {index} has no started time")
        try:
            judged_dates.add(datetime.fromisoformat(started.replace("Z", "+00:00")).date().isoformat())
        except ValueError as exc:
            raise FrameValidationError(
                f"run metadata batch {index} has invalid started time"
            ) from exc
    judged_dates = sorted(judged_dates)

    judge_info = {
        "model": run_meta["model"],
        "prompt_version": run_meta["prompt_version"],
        "transport": run_meta["transport"],
        "execution": run_meta["execution"],
        "temperature": temperature,
        "judged_on": judged_dates[-1],
    }
    if len(judged_dates) > 1:
        judge_info["judged_on_dates"] = judged_dates
    old_info = metadata.get("judge_info") if isinstance(metadata, dict) else None
    if old_info is not None:
        if not isinstance(old_info, dict):
            raise FrameValidationError("metadata_from judge_info is not an object")
        conflicts = {
            key: {"metadata_from": old_info[key], "run_meta": value}
            for key, value in judge_info.items()
            if key in old_info and old_info[key] is not None and old_info[key] != value
        }
        if conflicts:
            raise FrameValidationError(
                f"metadata_from conflicts with authoritative run metadata: {conflicts}"
            )
    return judge_info


def _resolve(path, repo=REPO):
    p = Path(path)
    return p if p.is_absolute() else Path(repo) / p


def _unwrap_results(raw):
    if isinstance(raw, dict) and isinstance(raw.get("result"), dict):
        raw = raw["result"]
    if isinstance(raw, list):
        out = {}
        for verdict in raw:
            if not isinstance(verdict, dict) or not verdict.get("name"):
                raise FrameValidationError("results list contains a verdict without a name")
            name = verdict["name"]
            if name in out:
                raise FrameValidationError(f"duplicate card result: {name}")
            out[name] = verdict
        raw = out
    if not isinstance(raw, dict):
        raise FrameValidationError("raw results must be a name-to-verdict object")
    return raw


def _unwrap_card(raw):
    if not isinstance(raw, dict):
        raise FrameValidationError("card artifact is not a JSON object")
    card = raw.get("benchmark_card", raw)
    if not isinstance(card, dict):
        raise FrameValidationError("benchmark_card is not a JSON object")
    return card


def _card_contract(card, name):
    missing = []
    for path in ALLOWLISTED_PATHS:
        section, field = path.split(".", 1)
        value = card.get(section)
        if not isinstance(value, dict) or field not in value:
            missing.append(path)
    if missing:
        raise FrameValidationError(
            f"{name}: card is missing allowlisted fields: {', '.join(missing)}"
        )

    risks = card.get("possible_risks", [])
    if not isinstance(risks, list):
        raise FrameValidationError(f"{name}: possible_risks is not a list")
    categories = []
    for index, risk in enumerate(risks):
        if not isinstance(risk, dict) or not isinstance(risk.get("category"), str) \
                or not risk["category"]:
            raise FrameValidationError(
                f"{name}: possible_risks[{index}] has no non-empty category"
            )
        categories.append(risk["category"])
    duplicates = sorted(k for k, n in Counter(categories).items() if n != 1)
    if duplicates:
        raise FrameValidationError(
            f"{name}: duplicate risk categories in card: {', '.join(duplicates)}"
        )
    return categories


def _validate_field_verdict(name, verdict, index):
    if not isinstance(verdict, dict):
        raise FrameValidationError(f"{name}: field_verdicts[{index}] is not an object")
    path = verdict.get("path")
    status = verdict.get("status")
    specificity = verdict.get("specificity")
    info = verdict.get("info_in_source")
    if not isinstance(path, str) or not path:
        raise FrameValidationError(f"{name}: field_verdicts[{index}] has no path")
    if not isinstance(verdict.get("note"), str):
        raise FrameValidationError(f"{name}:{path}: note is not a string")
    if status not in FIELD_STATUSES:
        raise FrameValidationError(f"{name}:{path}: invalid status {status!r}")
    if specificity not in SPECIFICITY_VALUES:
        raise FrameValidationError(
            f"{name}:{path}: invalid specificity {specificity!r}"
        )
    if info not in INFO_VALUES:
        raise FrameValidationError(f"{name}:{path}: invalid info_in_source {info!r}")
    if status == "not_specified":
        if specificity != "na" or info not in {"yes_primary", "yes_eee_only", "yes", "no"}:
            raise FrameValidationError(
                f"{name}:{path}: inconsistent not_specified verdict"
            )
    elif specificity not in {"specific", "generic"} or info != "na":
        raise FrameValidationError(f"{name}:{path}: inconsistent filled verdict")
    return path


def _validate_risk_verdict(name, verdict, index):
    if not isinstance(verdict, dict):
        raise FrameValidationError(f"{name}: risk_verdicts[{index}] is not an object")
    category = verdict.get("category")
    if not isinstance(category, str) or not category:
        raise FrameValidationError(f"{name}: risk_verdicts[{index}] has no category")
    if not isinstance(verdict.get("note"), str):
        raise FrameValidationError(f"{name}:{category}: note is not a string")
    if verdict.get("relevant_and_grounded") not in {"yes", "no"}:
        raise FrameValidationError(
            f"{name}:{category}: invalid relevant_and_grounded value"
        )
    return category


def _aggregate(verdicts):
    fields = [field for verdict in verdicts for field in verdict["field_verdicts"]]
    filled = [field for field in fields if field["status"] != "not_specified"]
    ns = [field for field in fields if field["status"] == "not_specified"]
    risks = [risk for verdict in verdicts for risk in verdict["risk_verdicts"]]

    def fraction(numerator, denominator):
        return round(numerator / denominator, 3) if denominator else None

    supported = sum(field["status"] == "supported" for field in filled)
    eee_only = sum(field["status"] == "supported_by_eee_only" for field in filled)
    partial = sum(field["status"] == "partial" for field in filled)
    unsupported = sum(field["status"] == "unsupported" for field in filled)
    specific = sum(field["specificity"] == "specific" for field in filled)
    missed_primary = sum(
        field["info_in_source"] in {"yes_primary", "yes"} for field in ns
    )
    missed_eee = sum(field["info_in_source"] == "yes_eee_only" for field in ns)
    missed = missed_primary + missed_eee
    grounded = sum(risk["relevant_and_grounded"] == "yes" for risk in risks)
    return {
        "n_filled": len(filled),
        "n_ns": len(ns),
        "n_risks": len(risks),
        "supported": supported,
        "supported_by_eee_only": eee_only,
        "partial": partial,
        "unsupported": unsupported,
        "support_rate": fraction(supported, len(filled)),
        "unsupported_rate": fraction(unsupported, len(filled)),
        "specific_rate": fraction(specific, len(filled)),
        "ns_missed": missed,
        "ns_missed_primary": missed_primary,
        "ns_missed_eee_only": missed_eee,
        "ns_correct_abstain": len(ns) - missed,
        "completeness_miss_rate": fraction(missed, len(ns)),
        "completeness_miss_rate_primary": fraction(missed_primary, len(ns)),
        "risk_grounded_rate": fraction(grounded, len(risks)),
    }


def build_analysis_view(raw_results, manifest, run_meta, repo=REPO, metadata=None,
                        label=None, source_artifacts=None):
    """Validate raw results and return the cleaned, analysis-shaped artifact."""
    raw_results = _unwrap_results(raw_results)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cards"), list):
        raise FrameValidationError("manifest must contain a cards list")

    entries = manifest["cards"]
    if manifest.get("n") is not None and manifest["n"] != len(entries):
        raise FrameValidationError(
            f"manifest n={manifest['n']} does not match {len(entries)} card entries"
        )
    names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
    if len(names) != len(entries) or any(not isinstance(name, str) or not name for name in names):
        raise FrameValidationError("manifest contains an entry without a name")
    duplicate_names = sorted(k for k, n in Counter(names).items() if n != 1)
    if duplicate_names:
        raise FrameValidationError(
            f"duplicate card names in manifest: {', '.join(duplicate_names)}"
        )
    missing_cards = sorted(set(names) - set(raw_results))
    extra_cards = sorted(set(raw_results) - set(names))
    if missing_cards or extra_cards:
        raise FrameValidationError(
            f"results/manifest card mismatch; missing={missing_cards}, extra={extra_cards}"
        )

    allowed = frozenset(ALLOWLISTED_PATHS)
    if len(allowed) != 23 or len(allowed) != len(ALLOWLISTED_PATHS):
        raise FrameValidationError("analysis allowlist must contain 23 unique paths")
    overlap = sorted(allowed & KNOWN_NONCONTENT_PATHS)
    if overlap:
        raise FrameValidationError(
            f"allowlist overlaps non-content paths: {', '.join(overlap)}"
        )

    exclusion_lookup = {
        (item["card"], item["field_path"]): item for item in DECLARED_EXCLUSIONS
    }
    if len(exclusion_lookup) != len(DECLARED_EXCLUSIONS):
        raise FrameValidationError("duplicate declared analysis exclusion")
    for card, path in exclusion_lookup:
        if card not in names or path not in allowed:
            raise FrameValidationError(
                f"invalid declared exclusion: {card}:{path}"
            )

    per_card = {}
    excluded_noncontent = Counter()
    excluded_raw_rows = 0
    for entry in entries:
        name = entry["name"]
        card_path = entry.get("card_path")
        if not isinstance(card_path, str) or not card_path:
            raise FrameValidationError(f"{name}: manifest entry has no card_path")
        card = _unwrap_card(_load(_resolve(card_path, repo)))
        expected_risks = _card_contract(card, name)

        verdict = raw_results[name]
        if not isinstance(verdict, dict):
            raise FrameValidationError(f"{name}: result is not an object")
        if verdict.get("name") != name:
            raise FrameValidationError(
                f"{name}: embedded result name is {verdict.get('name')!r}"
            )
        raw_fields = verdict.get("field_verdicts")
        raw_risks = verdict.get("risk_verdicts")
        if not isinstance(raw_fields, list) or not isinstance(raw_risks, list):
            raise FrameValidationError(
                f"{name}: field_verdicts and risk_verdicts must be lists"
            )

        by_path = {}
        for index, field in enumerate(raw_fields):
            path = _validate_field_verdict(name, field, index)
            by_path.setdefault(path, []).append(field)

        unknown = sorted(set(by_path) - allowed - KNOWN_NONCONTENT_PATHS)
        if unknown:
            raise FrameValidationError(
                f"{name}: unknown field paths outside analysis frame: {', '.join(unknown)}"
            )
        for path in set(by_path) & KNOWN_NONCONTENT_PATHS:
            excluded_noncontent[path] += len(by_path[path])
            excluded_raw_rows += len(by_path[path])

        cleaned_fields = []
        for path in ALLOWLISTED_PATHS:
            occurrences = by_path.get(path, [])
            declared = exclusion_lookup.get((name, path))
            if declared:
                signatures = {
                    (field["status"], field["specificity"], field["info_in_source"])
                    for field in occurrences
                }
                if len(occurrences) != 2 or len(signatures) != 2:
                    raise FrameValidationError(
                        f"{name}:{path}: declared duplicate_conflicting_verdict "
                        f"requires exactly two conflicting raw verdicts"
                    )
                excluded_raw_rows += len(occurrences)
                continue
            if len(occurrences) != 1:
                raise FrameValidationError(
                    f"{name}:{path}: expected exactly one verdict, found {len(occurrences)}"
                )
            cleaned_fields.append(dict(occurrences[0]))

        risk_categories = [
            _validate_risk_verdict(name, risk, index)
            for index, risk in enumerate(raw_risks)
        ]
        if Counter(risk_categories) != Counter(expected_risks):
            raise FrameValidationError(
                f"{name}: risk coverage mismatch; expected={expected_risks}, "
                f"found={risk_categories}"
            )
        cleaned_risks = [dict(risk) for risk in raw_risks]
        card_view = {
            "stratum": entry.get("stratum"),
            "field_verdicts": cleaned_fields,
            "risk_verdicts": cleaned_risks,
        }
        card_view["aggregate"] = _aggregate([card_view])
        per_card[name] = card_view

    rows = sum(len(card["field_verdicts"]) for card in per_card.values())
    expected_rows = len(entries) * len(ALLOWLISTED_PATHS) - len(DECLARED_EXCLUSIONS)
    if rows != expected_rows:
        raise FrameValidationError(
            f"derived row-count mismatch: expected {expected_rows}, found {rows}"
        )

    metadata = metadata if isinstance(metadata, dict) else {}
    judge_info = _judge_info_from_run_meta(run_meta, metadata)
    output = {
        "label": label or metadata.get("label") or "s150-analysis",
        "n_cards": len(per_card),
        "judge_info": dict(judge_info),
        "analysis_frame": {
            "allowlisted_paths": list(ALLOWLISTED_PATHS),
            "declared_exclusions": [dict(item) for item in DECLARED_EXCLUSIONS],
            "n_rows": rows,
            "n_cards": len(per_card),
        },
        "source_audit": {
            "excluded_noncontent_paths": [
                {"field_path": path, "n_raw_rows": count}
                for path, count in sorted(excluded_noncontent.items())
            ],
            "n_excluded_raw_rows": excluded_raw_rows,
        },
        "aggregate": _aggregate(list(per_card.values())),
        "per_card": per_card,
    }
    if source_artifacts:
        output["source_artifacts"] = source_artifacts
    return output


def _write_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Validate raw S judge results and build the 23-field analysis view."
    )
    parser.add_argument("--raw-results", required=True)
    parser.add_argument(
        "--verdict-dir", required=True,
        help="directory of per-card raw verdict JSON files",
    )
    parser.add_argument(
        "--run-meta", required=True,
        help="authoritative judge execution metadata",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--metadata-from",
        help="optional prior artifact; label is reused and judge_info is cross-checked",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--out", default=None,
        help="derived output path; omit for read-only validation",
    )
    args = parser.parse_args()

    raw_path = _resolve(args.raw_results)
    verdict_dir = _resolve(args.verdict_dir)
    run_meta_path = _resolve(args.run_meta)
    manifest_path = _resolve(args.manifest)
    metadata_path = _resolve(args.metadata_from) if args.metadata_from else None
    out_path = _resolve(args.out) if args.out else None
    protected = {raw_path.resolve(), run_meta_path.resolve(), manifest_path.resolve()}
    if metadata_path:
        protected.add(metadata_path.resolve())
    if out_path:
        out_resolved = out_path.resolve()
        if out_resolved in protected or out_resolved.is_relative_to(verdict_dir.resolve()):
            raise SystemExit(
                "--out must not overwrite a raw, verdict, manifest, or metadata artifact"
            )

    metadata = _load(metadata_path) if metadata_path else None
    raw_results = _unwrap_results(_load(raw_path))
    verdict_manifest = _verify_verdict_dir(raw_results, verdict_dir)
    source_artifacts = {
        "raw_results": {"path": args.raw_results, "sha256": _sha256(raw_path)},
        "verdict_dir": {
            "path": args.verdict_dir,
            **verdict_manifest,
        },
        "run_meta": {"path": args.run_meta, "sha256": _sha256(run_meta_path)},
        "manifest": {"path": args.manifest, "sha256": _sha256(manifest_path)},
    }
    if metadata_path:
        source_artifacts["metadata_from"] = {
            "path": args.metadata_from,
            "sha256": _sha256(metadata_path),
        }
    view = build_analysis_view(
        raw_results,
        _load(manifest_path),
        _load(run_meta_path),
        repo=REPO,
        metadata=metadata,
        label=args.label,
        source_artifacts=source_artifacts,
    )
    frame = view["analysis_frame"]
    if frame["n_cards"] != EXPECTED_N_CARDS or frame["n_rows"] != EXPECTED_N_ROWS:
        raise SystemExit(
            f"unexpected S analysis frame: {frame['n_cards']} cards, "
            f"{frame['n_rows']} rows; expected {EXPECTED_N_CARDS}, {EXPECTED_N_ROWS}"
        )
    print(
        f"validated {frame['n_cards']} cards, {frame['n_rows']} analysis rows; "
        f"declared exclusions={len(frame['declared_exclusions'])}"
    )
    if out_path:
        _write_atomic(out_path, view)
        print(f"derived judge view -> {out_path}")
    else:
        print("validation only; no file written")


if __name__ == "__main__":
    main()
