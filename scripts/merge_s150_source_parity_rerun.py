"""Validate and merge the targeted S150 source-parity judge rerun.

The merge is deliberately separate from both judge runs.  It retains verdicts
for inputs that remained byte-identical and replaces verdicts only for the
exact input-difference set recorded by the source-parity repair audit.

Nothing is written until every input, manifest, verdict, and run-metadata gate
passes.  The destination must not exist and must not overlap any source path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from check_frozen import check
from run_s_judge import validate_verdict


REPO = Path(__file__).resolve().parent.parent
MERGE_VERSION = "s150-source-parity-merge-v2"


class MergeError(RuntimeError):
    """The corrected and retained judge artifacts are not safe to merge."""


def _load_json(path: Path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "md5": _digest(path, "md5"),
        "sha256": _digest(path, "sha256"),
    }


def _resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _path_ref(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def _json_file_map(directory: Path, label: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise MergeError(f"{label} is not a directory: {directory}")
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise MergeError(f"{label} contains a symlink: {path.name}")
        if path.is_file() and path.suffix == ".json":
            if path.stem in files:
                raise MergeError(f"{label} has duplicate JSON stem {path.stem}")
            files[path.stem] = path
    return files


def _require_exact_set(actual, expected, label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise MergeError(f"{label} set mismatch; missing={missing}, extra={extra}")


def _load_frozen_schema(repo: Path) -> dict:
    """Read the literal schema without importing the application package."""
    source = repo / "scripts" / "judge_gold_set.py"
    try:
        module = ast.parse(source.read_text())
    except (OSError, SyntaxError) as exc:
        raise MergeError(f"cannot parse frozen judge source: {exc}") from exc
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "JUDGE_SCHEMA" in names:
            try:
                schema = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise MergeError("frozen JUDGE_SCHEMA is no longer a literal") from exc
            if not isinstance(schema, dict):
                break
            return schema
    raise MergeError("frozen JUDGE_SCHEMA assignment not found")


def _manifest_cards(manifest: dict, label: str) -> dict:
    cards = manifest.get("cards") if isinstance(manifest, dict) else None
    if not isinstance(cards, dict) or not cards:
        raise MergeError(f"{label} has no non-empty cards object")
    for name, record in cards.items():
        if not isinstance(name, str) or not name or not isinstance(record, dict):
            raise MergeError(f"{label} contains an invalid card record")
        if not isinstance(record.get("md5"), str) or not isinstance(record.get("bytes"), int):
            raise MergeError(f"{label} has invalid hash/size metadata for {name}")
    return cards


def _verify_input_tree(
    directory: Path,
    cards: dict,
    label: str,
) -> tuple[dict[str, Path], dict[str, dict]]:
    files = _json_file_map(directory, label)
    _require_exact_set(files, cards, label)
    loaded: dict[str, dict] = {}
    for name in sorted(cards):
        path = files[name]
        record = _file_record(path)
        expected = cards[name]
        if record["md5"] != expected["md5"] or record["bytes"] != expected["bytes"]:
            raise MergeError(f"{label} hash/size mismatch for {name}")
        value = _load_json(path)
        if not isinstance(value, dict) or value.get("name") != name:
            raise MergeError(f"{label} embedded name mismatch for {name}")
        loaded[name] = value
    return files, loaded


def _validate_audit_paths(
    audit: dict,
    repo: Path,
    original_inputs: Path,
    original_manifest: Path,
    repaired_inputs: Path,
    repaired_manifest: Path,
) -> None:
    expected = {
        "original_inputs_dir": original_inputs,
        "original_manifest": original_manifest,
        "output_inputs_dir": repaired_inputs,
        "repaired_manifest": repaired_manifest,
    }
    for key, path in expected.items():
        value = audit.get(key)
        if not isinstance(value, str) or _resolve(repo, value) != path:
            raise MergeError(f"repair audit {key} does not identify the supplied artifact")


def _verify_run_meta(
    run_meta: dict,
    expected_names: set[str],
    label: str,
) -> None:
    if not isinstance(run_meta, dict) or not isinstance(run_meta.get("batches"), list):
        raise MergeError(f"{label} has no batches list")
    if run_meta.get("aborted"):
        raise MergeError(f"{label} records an aborted run: {run_meta['aborted']}")
    attempted: set[str] = set()
    succeeded: set[str] = set()
    for batch_index, batch in enumerate(run_meta["batches"]):
        cards = batch.get("cards") if isinstance(batch, dict) else None
        if not isinstance(cards, list):
            raise MergeError(f"{label} batch {batch_index} has no cards list")
        for record in cards:
            name = record.get("name") if isinstance(record, dict) else None
            if not isinstance(name, str):
                raise MergeError(f"{label} batch {batch_index} has an unnamed card record")
            attempted.add(name)
            if record.get("ok") is True:
                succeeded.add(name)
    extra_attempts = sorted(attempted - expected_names)
    if extra_attempts:
        raise MergeError(f"{label} attempted cards outside its allowed set: {extra_attempts}")
    _require_exact_set(succeeded, expected_names, f"{label} successful-card")


def _verify_compatible_runs(original: dict, rerun: dict) -> None:
    scalar_keys = (
        "model",
        "transport",
        "prompt_version",
        "prompt_md5",
        "schema_md5",
        "input_mode",
    )
    for key in scalar_keys:
        if original.get(key) != rerun.get(key):
            raise MergeError(
                f"judge run incompatibility for {key}: "
                f"original={original.get(key)!r}, rerun={rerun.get(key)!r}"
            )
    if original.get("request_params") != rerun.get("request_params"):
        raise MergeError("judge run incompatibility: request_params differ")


def _validate_verdict_file(
    path: Path,
    input_json: dict,
    schema: dict,
    label: str,
) -> dict:
    verdict = _load_json(path)
    problems = validate_verdict(verdict, input_json, schema)
    if problems:
        raise MergeError(f"{label} invalid for {input_json['name']}: {'; '.join(problems)}")
    return verdict


def _ensure_separate_output(output: Path, sources: list[Path]) -> None:
    if output.exists():
        raise MergeError(f"destination already exists; refusing to overwrite: {output}")
    for source in sources:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise MergeError(f"destination overlaps source artifact: {source}")


def merge_source_parity_verdicts(
    *,
    repo: str | Path,
    original_inputs: str | Path,
    original_manifest: str | Path,
    original_verdicts: str | Path,
    original_run_meta: str | Path,
    repaired_inputs: str | Path,
    repaired_manifest: str | Path,
    repair_audit: str | Path,
    rerun_verdicts: str | Path,
    rerun_run_meta: str | Path,
    schema_repair_verdicts: str | Path | None = None,
    schema_repair_run_meta: str | Path | None = None,
    schema_repair_names: list[str] | tuple[str, ...] | set[str] | None = None,
    output: str | Path,
) -> dict:
    """Validate all merge inputs and atomically create a new merged artifact."""
    repo_path = Path(repo).resolve()
    paths = {
        "original_inputs": _resolve(repo_path, original_inputs),
        "original_manifest": _resolve(repo_path, original_manifest),
        "original_verdicts": _resolve(repo_path, original_verdicts),
        "original_run_meta": _resolve(repo_path, original_run_meta),
        "repaired_inputs": _resolve(repo_path, repaired_inputs),
        "repaired_manifest": _resolve(repo_path, repaired_manifest),
        "repair_audit": _resolve(repo_path, repair_audit),
        "rerun_verdicts": _resolve(repo_path, rerun_verdicts),
        "rerun_run_meta": _resolve(repo_path, rerun_run_meta),
        "output": _resolve(repo_path, output),
    }
    schema_names = set(schema_repair_names or ())
    supplied_schema_paths = (schema_repair_verdicts is not None, schema_repair_run_meta is not None)
    if any(supplied_schema_paths) or schema_names:
        if not all(supplied_schema_paths) or not schema_names:
            raise MergeError(
                "schema repair requires verdict directory, run metadata, and at least one card name"
            )
        paths["schema_repair_verdicts"] = _resolve(repo_path, schema_repair_verdicts)
        paths["schema_repair_run_meta"] = _resolve(repo_path, schema_repair_run_meta)
    _ensure_separate_output(paths["output"], [path for key, path in paths.items() if key != "output"])

    original_manifest_json = _load_json(paths["original_manifest"])
    repaired_manifest_json = _load_json(paths["repaired_manifest"])
    audit = _load_json(paths["repair_audit"])
    original_meta = _load_json(paths["original_run_meta"])
    rerun_meta = _load_json(paths["rerun_run_meta"])
    schema_repair_meta = (
        _load_json(paths["schema_repair_run_meta"]) if schema_names else None
    )
    original_cards = _manifest_cards(original_manifest_json, "original input manifest")
    repaired_cards = _manifest_cards(repaired_manifest_json, "repaired input manifest")
    _require_exact_set(repaired_cards, original_cards, "input-manifest card")

    if audit.get("parity_gate", {}).get("passed") is not True:
        raise MergeError("repair audit parity gate did not pass")
    _validate_audit_paths(
        audit,
        repo_path,
        paths["original_inputs"],
        paths["original_manifest"],
        paths["repaired_inputs"],
        paths["repaired_manifest"],
    )
    parent_sha = repaired_manifest_json.get("parent_manifest_sha256")
    if parent_sha != _digest(paths["original_manifest"], "sha256"):
        raise MergeError("repaired manifest parent hash does not match original manifest")
    parent_ref = repaired_manifest_json.get("parent_manifest")
    if not isinstance(parent_ref, str) or _resolve(repo_path, parent_ref) != paths["original_manifest"]:
        raise MergeError("repaired manifest parent path does not match original manifest")

    changed_list = audit.get("changed_card_names")
    if not isinstance(changed_list, list) or any(not isinstance(name, str) for name in changed_list):
        raise MergeError("repair audit has no valid changed_card_names list")
    if len(changed_list) != len(set(changed_list)):
        raise MergeError("repair audit changed_card_names contains duplicates")
    changed = set(changed_list)
    all_names = set(original_cards)
    if not schema_names <= all_names:
        raise MergeError(
            f"schema-repair names outside the sample: {sorted(schema_names - all_names)}"
        )
    if schema_names & changed:
        raise MergeError(
            "schema-repair names overlap source-repaired cards: "
            f"{sorted(schema_names & changed)}"
        )
    if audit.get("n_sample_cards") != len(all_names):
        raise MergeError("repair audit n_sample_cards does not match input manifests")
    if audit.get("n_changed_cards") != len(changed):
        raise MergeError("repair audit n_changed_cards does not match changed_card_names")
    if audit.get("n_unchanged_cards_byte_identical") != len(all_names - changed):
        raise MergeError("repair audit unchanged count does not match changed_card_names")

    original_files, original_inputs_json = _verify_input_tree(
        paths["original_inputs"], original_cards, "original input directory"
    )
    repaired_files, repaired_inputs_json = _verify_input_tree(
        paths["repaired_inputs"], repaired_cards, "repaired input directory"
    )
    actual_changed = {
        name
        for name in all_names
        if original_files[name].read_bytes() != repaired_files[name].read_bytes()
    }
    _require_exact_set(actual_changed, changed, "repair-audit changed-card")
    for name in sorted(changed):
        for key in ("name", "fields", "risks"):
            if original_inputs_json[name].get(key) != repaired_inputs_json[name].get(key):
                raise MergeError(f"repair changed {key} for {name}; only source evidence may change")

    audit_records = audit.get("changed_cards")
    if not isinstance(audit_records, list):
        raise MergeError("repair audit has no changed_cards records")
    by_name = {
        record.get("name"): record
        for record in audit_records
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    if len(by_name) != len(audit_records):
        raise MergeError("repair audit changed_cards contains duplicate or unnamed records")
    _require_exact_set(by_name, changed, "repair-audit changed-card record")
    for name in sorted(changed):
        record = by_name[name]
        original_record = _file_record(original_files[name])
        repaired_record = _file_record(repaired_files[name])
        expected_pairs = {
            "original_input_md5": original_record["md5"],
            "original_input_sha256": original_record["sha256"],
            "repaired_input_md5": repaired_record["md5"],
            "repaired_input_sha256": repaired_record["sha256"],
        }
        for key, expected in expected_pairs.items():
            if record.get(key) != expected:
                raise MergeError(f"repair audit {key} mismatch for {name}")

    corrected = changed | schema_names
    retained = all_names - corrected
    original_verdict_files = _json_file_map(paths["original_verdicts"], "original verdict directory")
    rerun_verdict_files = _json_file_map(paths["rerun_verdicts"], "rerun verdict directory")
    _require_exact_set(original_verdict_files, all_names, "original verdict")
    _require_exact_set(rerun_verdict_files, changed, "rerun verdict")
    _verify_run_meta(original_meta, all_names, "original run metadata")
    _verify_run_meta(rerun_meta, changed, "rerun metadata")
    _verify_compatible_runs(original_meta, rerun_meta)
    schema_repair_verdict_files: dict[str, Path] = {}
    if schema_names:
        schema_repair_verdict_files = _json_file_map(
            paths["schema_repair_verdicts"], "schema-repair verdict directory"
        )
        _require_exact_set(
            schema_repair_verdict_files, schema_names, "schema-repair verdict"
        )
        _verify_run_meta(schema_repair_meta, schema_names, "schema-repair run metadata")
        _verify_compatible_runs(original_meta, schema_repair_meta)

    schema = _load_frozen_schema(repo_path)
    selected_verdicts: dict[str, dict] = {}
    selected_paths: dict[str, Path] = {}
    for name in sorted(retained):
        selected_verdicts[name] = _validate_verdict_file(
            original_verdict_files[name], original_inputs_json[name], schema, "retained verdict"
        )
        selected_paths[name] = original_verdict_files[name]
    for name in sorted(changed):
        selected_verdicts[name] = _validate_verdict_file(
            rerun_verdict_files[name], repaired_inputs_json[name], schema, "corrected verdict"
        )
        selected_paths[name] = rerun_verdict_files[name]
    for name in sorted(schema_names):
        original_problems = validate_verdict(
            _load_json(original_verdict_files[name]), original_inputs_json[name], schema
        )
        if not original_problems:
            raise MergeError(f"schema repair requested for already-valid original verdict: {name}")
        selected_verdicts[name] = _validate_verdict_file(
            schema_repair_verdict_files[name],
            original_inputs_json[name],
            schema,
            "schema-repaired verdict",
        )
        selected_paths[name] = schema_repair_verdict_files[name]

    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        verdict_output = stage / "verdicts"
        verdict_output.mkdir()
        verdict_manifest_cards = {}
        merged_input_cards = {}
        for name in sorted(all_names):
            destination = verdict_output / f"{name}.json"
            shutil.copyfile(selected_paths[name], destination)
            verdict_record = _file_record(destination)
            if name in changed:
                source = "source_corrected_rerun"
            elif name in schema_names:
                source = "schema_corrected_rerun"
            else:
                source = "retained_original"
            input_record = repaired_cards[name] if name in changed else original_cards[name]
            verdict_manifest_cards[name] = {
                **verdict_record,
                "source": source,
                "input_md5": input_record["md5"],
            }
            merged_input_cards[name] = {
                "bytes": input_record["bytes"],
                "md5": input_record["md5"],
                "source": "repaired_manifest" if name in changed else "original_manifest",
            }

        results_path = stage / "results_s150.json"
        _write_json(results_path, selected_verdicts)
        input_manifest_path = stage / "inputs_md5.json"
        _write_json(
            input_manifest_path,
            {
                "version": MERGE_VERSION,
                "n_cards": len(all_names),
                "n_corrected": len(corrected),
                "n_source_corrected": len(changed),
                "n_schema_corrected": len(schema_names),
                "n_retained": len(retained),
                "cards": merged_input_cards,
            },
        )
        verdict_manifest_path = stage / "verdicts_sha256.json"
        _write_json(
            verdict_manifest_path,
            {
                "version": MERGE_VERSION,
                "n_cards": len(all_names),
                "n_corrected": len(corrected),
                "n_source_corrected": len(changed),
                "n_schema_corrected": len(schema_names),
                "n_retained": len(retained),
                "cards": verdict_manifest_cards,
            },
        )

        merged_meta = {
            key: rerun_meta[key]
            for key in (
                "model",
                "transport",
                "execution",
                "sdk_version",
                "request_params",
                "pricing_usd_per_mtok",
                "inline_appendix_md5",
                "prompt_md5",
                "schema_md5",
                "prompt_version",
                "input_mode",
            )
            if key in rerun_meta
        }
        merged_meta["batches"] = list(original_meta["batches"]) + list(rerun_meta["batches"])
        if schema_repair_meta:
            merged_meta["batches"].extend(schema_repair_meta["batches"])
        source_meta_keys = (
            "original_manifest",
            "original_run_meta",
            "repaired_manifest",
            "repair_audit",
            "rerun_run_meta",
        ) + (("schema_repair_run_meta",) if schema_names else ())
        merged_meta["merge"] = {
            "version": MERGE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_cards": len(all_names),
            "n_corrected": len(corrected),
            "n_source_corrected": len(changed),
            "n_schema_corrected": len(schema_names),
            "n_retained": len(retained),
            "changed_card_names": sorted(changed),
            "schema_repaired_card_names": sorted(schema_names),
            "source_artifacts": {
                key: {
                    "path": _path_ref(repo_path, paths[key]),
                    **_file_record(paths[key]),
                }
                for key in source_meta_keys
            },
            "output_artifacts": {
                "results_s150.json": _file_record(results_path),
                "inputs_md5.json": _file_record(input_manifest_path),
                "verdicts_sha256.json": _file_record(verdict_manifest_path),
            },
        }
        original_cost = original_meta.get("total_cost_est_usd")
        rerun_cost = rerun_meta.get("total_cost_est_usd")
        schema_repair_cost = (
            schema_repair_meta.get("total_cost_est_usd") if schema_repair_meta else 0
        )
        if (
            isinstance(original_cost, (int, float))
            and isinstance(rerun_cost, (int, float))
            and isinstance(schema_repair_cost, (int, float))
        ):
            merged_meta["total_cost_est_usd"] = round(
                original_cost + rerun_cost + schema_repair_cost, 4
            )
            merged_meta["merge"]["component_cost_est_usd"] = {
                "original_full_run": original_cost,
                "source_corrected_targeted_rerun": rerun_cost,
                "schema_corrected_targeted_rerun": schema_repair_cost,
            }
        run_meta_path = stage / "run_meta.json"
        _write_json(run_meta_path, merged_meta)

        if output.exists():
            raise MergeError(f"destination appeared during validation: {output}")
        stage.rename(output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return {
        "output": str(output),
        "n_cards": len(all_names),
        "n_corrected": len(corrected),
        "n_source_corrected": len(changed),
        "n_schema_corrected": len(schema_names),
        "n_retained": len(retained),
        "changed_card_names": sorted(changed),
        "schema_repaired_card_names": sorted(schema_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly validate and merge the S150 source-parity rerun."
    )
    parser.add_argument("--original-inputs", required=True)
    parser.add_argument("--original-manifest", required=True)
    parser.add_argument("--original-verdicts", required=True)
    parser.add_argument("--original-run-meta", required=True)
    parser.add_argument("--repaired-inputs", required=True)
    parser.add_argument("--repaired-manifest", required=True)
    parser.add_argument("--repair-audit", required=True)
    parser.add_argument("--rerun-verdicts", required=True)
    parser.add_argument("--rerun-run-meta", required=True)
    parser.add_argument("--schema-repair-verdicts")
    parser.add_argument("--schema-repair-run-meta")
    parser.add_argument(
        "--schema-repair-names",
        help="Comma-separated original verdicts replaced only to correct invalid schema",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    check()
    try:
        summary = merge_source_parity_verdicts(
            repo=REPO,
            original_inputs=args.original_inputs,
            original_manifest=args.original_manifest,
            original_verdicts=args.original_verdicts,
            original_run_meta=args.original_run_meta,
            repaired_inputs=args.repaired_inputs,
            repaired_manifest=args.repaired_manifest,
            repair_audit=args.repair_audit,
            rerun_verdicts=args.rerun_verdicts,
            rerun_run_meta=args.rerun_run_meta,
            schema_repair_verdicts=args.schema_repair_verdicts,
            schema_repair_run_meta=args.schema_repair_run_meta,
            schema_repair_names=(
                [name.strip() for name in args.schema_repair_names.split(",") if name.strip()]
                if args.schema_repair_names
                else None
            ),
            output=args.out,
        )
    except MergeError as exc:
        parser.exit(1, f"source-parity merge failed: {exc}\n")
    print(
        f"merged {summary['n_cards']} verdicts: {summary['n_corrected']} corrected "
        f"({summary['n_source_corrected']} source, {summary['n_schema_corrected']} schema), "
        f"{summary['n_retained']} retained -> {summary['output']}"
    )


if __name__ == "__main__":
    main()
