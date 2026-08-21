"""Repair persisted-source parity in the frozen S150 judge inputs.

The original S150 input builder assembles paper, webpage, Hugging Face README,
EEE, and retrieved-context artifacts.  It omitted two complete sources that
the Composer could use: the authoritative paper abstract on every
``degraded_to_abstract_only`` run, and the persisted GitHub README on cards
whose judged fields cite GitHub provenance.  Field snippets are insufficient
for Not-specified and risk checks.  This script therefore requires a hashed
manifest of the complete abstracts and reads the complete GitHub artifacts.

No model is called.  Original inputs, verdicts, prompt, schema, and the frozen
``scripts/judge_gold_set.py`` are never modified. Three cards whose complete
direct-source assembly exceeded the original cap are rebuilt from their full
persisted direct sources under a separate, explicit repair cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
ORIGINAL_SOURCE_CAP = 280_000  # Frozen S150 preparer cap, in characters.
REPAIRED_SOURCE_CAP = 600_000  # Holds every complete direct-source repair.
REPAIR_VERSION = "s150-source-parity-v1"

# These source classes are already assembled by judge_gold_set.py.  Presence is
# checked by the builder's unambiguous section marker, not fuzzy text matching.
ASSEMBLED_SOURCE_MARKERS = {
    "paper": "[PAPER / DOCUMENT]\n",
    "html": "[WEBPAGE]\n",
    "hf": "[HF README]\n",
    "eee": "[EEE EVALUATION DATA]\n",
}

# These are persisted primary-source classes that the frozen assembler has no
# dedicated reader for.  Provenance routes the audit; snippets are never used
# as repair payloads.
SUPPLEMENTAL_SOURCE_LABELS = {"abstract", "github"}

# Composer-internal provenance labels are audited but are not source documents
# and must not be promoted into primary-source evidence.  Keep this explicit:
# a new label causes a hard failure and must be classified deliberately.
INTERNAL_SOURCE_LABELS = {
    "Already established",
    "Not specified",
    "already established",
    "benchmark_details",
    "benchmark_details.overview",
    "deterministic",
    "established",
    "inferred from benchmark content",
    "inferred from overview",
    "inferred from problem source (math competitions are typically in English)",
    "no evidence",
    "none",
}

SOURCE_ORDER = ("abstract", "github")
BLOCK_HEADERS = {
    "abstract": "[PRIMARY SOURCE: COMPLETE PAPER TITLE AND ABSTRACT]",
    "github": "[PRIMARY SOURCE: COMPLETE PERSISTED GITHUB README]",
}


class RepairError(RuntimeError):
    """Raised when a fail-closed repair precondition is not met."""


@dataclass(frozen=True)
class ProvenanceRecord:
    path: str
    source: str
    evidence: Any
    evidence_ids: tuple[str, ...]


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RepairError(f"missing required file: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"cannot read valid JSON from {path}: {exc}") from exc


def _digest(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def _file_digest(path: Path, algorithm: str) -> str:
    try:
        return _digest(path.read_bytes(), algorithm)
    except OSError as exc:
        raise RepairError(f"cannot hash {path}: {exc}") from exc


def _display_path(path: Path, repo: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_under_repo(repo: Path, value: str, *, what: str) -> Path:
    candidate = (repo / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise RepairError(f"{what} escapes repository root: {value!r}") from exc
    return candidate


def _iter_provenance_records(value: Any, path: str = "") -> Iterable[ProvenanceRecord]:
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, str) and "evidence" in value:
            raw_ids = value.get("evidence_ids", [])
            if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
                raise RepairError(f"invalid evidence_ids at provenance path {path!r}")
            yield ProvenanceRecord(path, source, value.get("evidence"), tuple(raw_ids))
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_provenance_records(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_provenance_records(child, f"{path}[{index}]")


def _source_class(label: str) -> str:
    if label in SUPPLEMENTAL_SOURCE_LABELS:
        return "supplemental_primary"
    if label in ASSEMBLED_SOURCE_MARKERS:
        return "assembled"
    if label in INTERNAL_SOURCE_LABELS:
        return "internal_or_derived"
    raise RepairError(
        f"unclassified provenance source label {label!r}; classify it before repairing"
    )


def _evidence_sha256(evidence: str) -> str:
    return _digest(evidence.encode("utf-8"), "sha256")


def _normalize_evidence(value: str) -> str:
    return " ".join(value.casefold().split())


def _abstract_evidence_present(
    *, path: str, evidence: str, paper_title: str, paper_abstract: str
) -> bool:
    normalized_evidence = _normalize_evidence(evidence)
    normalized_source = _normalize_evidence(paper_title + " " + paper_abstract)
    if normalized_evidence in normalized_source:
        return True
    fragments = [
        _normalize_evidence(fragment)
        for fragment in re.split(
            r"\s*;\s*|\n+|(?<=[.!?])\s+(?=[A-Z0-9])",
            evidence,
        )
        if fragment.strip()
    ]
    if len(fragments) > 1 and all(fragment in normalized_source for fragment in fragments):
        return True
    if path == "benchmark_details.name":
        without_year = re.sub(r"\s*\(\d{4}\)$", "", normalized_evidence)
        return without_year == _normalize_evidence(paper_title)
    return False


def _build_block(source: str, payload: str) -> str:
    description = {
        "abstract": "Complete authoritative paper title and abstract supplied by the hashed manifest:",
        "github": "Complete README text persisted by the card-generation run:",
    }[source]
    return "\n".join((BLOCK_HEADERS[source], description, "", payload))


def _source_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        return value.get("text") or value.get("content") or value.get("markdown") or ""
    return str(value)


def _complete_direct_sources(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """Mirror the frozen direct-source order without truncation or RAG."""
    tool_output = run_dir / "tool_output"
    parts: list[str] = []
    artifacts: list[dict[str, Any]] = []

    specs = (
        ("paper", "docling", "[PAPER / DOCUMENT]"),
        ("html", "html", "[WEBPAGE]"),
    )
    for source, directory, marker in specs:
        paths = sorted((tool_output / directory).glob("*.json"))
        if len(paths) > 1:
            raise RepairError(f"multiple {directory} artifacts in {run_dir}")
        if not paths:
            continue
        value = _json_load(paths[0])
        text = _source_text(value)
        if text:
            parts.append(marker + "\n" + text)
            artifacts.append(
                {
                    "source": source,
                    "path": paths[0],
                    "sha256": _file_digest(paths[0], "sha256"),
                    "chars": len(text),
                }
            )

    hf_paths = sorted((tool_output / "hf").glob("*.json"))
    if len(hf_paths) > 1:
        raise RepairError(f"multiple hf artifacts in {run_dir}")
    if hf_paths:
        hf = _json_load(hf_paths[0])
        readme = (hf.get("readme_markdown") or hf.get("readme") or "") if isinstance(hf, dict) else ""
        if readme:
            parts.append("[HF README]\n" + readme)
            artifacts.append(
                {
                    "source": "hf",
                    "path": hf_paths[0],
                    "sha256": _file_digest(hf_paths[0], "sha256"),
                    "chars": len(readme),
                }
            )

    eee_paths = sorted((tool_output / "eee").glob("*.json"))
    if len(eee_paths) > 1:
        raise RepairError(f"multiple eee artifacts in {run_dir}")
    if eee_paths:
        eee = _json_load(eee_paths[0])
        eee_text = json.dumps(eee, indent=2, ensure_ascii=False)
        parts.append("[EEE EVALUATION DATA]\n" + eee_text)
        artifacts.append(
            {
                "source": "eee",
                "path": eee_paths[0],
                "sha256": _file_digest(eee_paths[0], "sha256"),
                "chars": len(eee_text),
            }
        )

    return "\n\n".join(parts), artifacts


def _load_abstract_manifest(
    path: Path, *, expected_cards: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load exact abstracts; refuse missing, extra, unhashed, or misbound entries."""
    manifest = _json_load(path)
    if not isinstance(manifest, dict) or manifest.get("version") != "s150-paper-abstracts-v1":
        raise RepairError("abstract manifest version must be 's150-paper-abstracts-v1'")
    cards = manifest.get("cards")
    if not isinstance(cards, dict) or set(cards) != set(expected_cards):
        missing = sorted(set(expected_cards) - set(cards or {})) if isinstance(cards, dict) else []
        extra = sorted(set(cards or {}) - set(expected_cards)) if isinstance(cards, dict) else []
        raise RepairError(
            "abstract manifest card set must exactly match degraded-to-abstract-only telemetry "
            f"(missing={missing}, extra={extra})"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_cards):
        record = cards[name]
        if not isinstance(record, dict):
            raise RepairError(f"abstract manifest record is not an object for {name}")
        abstract = record.get("paper_abstract")
        expected_hash = record.get("paper_abstract_sha256")
        if not isinstance(abstract, str) or not abstract.strip():
            raise RepairError(f"abstract manifest has no complete paper_abstract for {name}")
        actual_hash = _evidence_sha256(abstract)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise RepairError(f"abstract manifest SHA-256 mismatch for {name}")
        if record.get("source_run_dir") != expected_cards[name]:
            raise RepairError(f"abstract manifest source_run_dir mismatch for {name}")
        required_text = (
            "paper_title",
            "recovery_source_url",
            "recovered_at",
            "recovery_method",
        )
        for key in required_text:
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise RepairError(f"abstract manifest has no {key} for {name}")
        normalized[name] = {
            "paper_abstract": abstract,
            "paper_abstract_sha256": actual_hash,
            "source_run_dir": expected_cards[name],
            "paper_title": record["paper_title"],
            "recovery_source_url": record["recovery_source_url"],
            "recovered_at": record["recovered_at"],
            "recovery_method": record["recovery_method"],
        }
    return normalized, _file_digest(path, "sha256")


def _validate_input(
    *, name: str, raw: bytes, expected: dict[str, Any], path: Path
) -> dict[str, Any]:
    expected_md5 = expected.get("md5")
    expected_bytes = expected.get("bytes")
    if not isinstance(expected_md5, str) or not isinstance(expected_bytes, int):
        raise RepairError(f"invalid original manifest record for {name}")
    if len(raw) != expected_bytes or _digest(raw, "md5") != expected_md5:
        raise RepairError(f"original input guard failed for {name}: {path}")
    try:
        input_json = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"invalid prepared input JSON for {name}: {exc}") from exc
    if not isinstance(input_json, dict) or set(input_json) != {
        "name", "fields", "risks", "source_text"
    }:
        raise RepairError(f"unexpected prepared-input shape for {name}")
    if input_json["name"] != name:
        raise RepairError(f"prepared-input name mismatch for {name}")
    if not isinstance(input_json["fields"], list) or not isinstance(input_json["risks"], list):
        raise RepairError(f"fields/risks must be lists for {name}")
    if not isinstance(input_json["source_text"], str):
        raise RepairError(f"source_text must be a string for {name}")
    if len(input_json["source_text"]) > ORIGINAL_SOURCE_CAP:
        raise RepairError(f"original source_text exceeds ORIGINAL_SOURCE_CAP for {name}")
    field_paths = [field.get("path") for field in input_json["fields"] if isinstance(field, dict)]
    if len(field_paths) != len(input_json["fields"]) or not all(
        isinstance(field_path, str) for field_path in field_paths
    ):
        raise RepairError(f"invalid field record for {name}")
    if len(field_paths) != len(set(field_paths)):
        raise RepairError(f"duplicate judged field path for {name}")
    # The frozen preparer emitted this canonical one-line representation.  This
    # guard means serializing a changed input can only alter source_text.
    if json.dumps(input_json).encode("utf-8") != raw:
        raise RepairError(f"prepared input is not in the expected canonical encoding: {path}")
    return input_json


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repair_inputs(
    *,
    repo: Path,
    sample_path: Path,
    inputs_dir: Path,
    original_manifest_path: Path,
    abstract_manifest_path: Path,
    output_dir: Path,
    repaired_manifest_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Build repaired inputs and return the deterministic audit object."""
    repo = repo.resolve()
    sample_path = sample_path.resolve()
    inputs_dir = inputs_dir.resolve()
    original_manifest_path = original_manifest_path.resolve()
    abstract_manifest_path = abstract_manifest_path.resolve()
    output_dir = output_dir.resolve()
    repaired_manifest_path = repaired_manifest_path.resolve()
    audit_path = audit_path.resolve()

    if output_dir == inputs_dir:
        raise RepairError("output directory must differ from original inputs directory")
    for destination in (output_dir, repaired_manifest_path, audit_path):
        if destination.exists():
            raise RepairError(f"refusing to overwrite existing repair output: {destination}")

    sample = _json_load(sample_path)
    original_manifest = _json_load(original_manifest_path)
    if not isinstance(sample, dict) or not isinstance(sample.get("cards"), list):
        raise RepairError("sample JSON must contain a cards list")
    if sample.get("n") != len(sample["cards"]):
        raise RepairError("sample n does not match the cards list length")
    manifest_cards = original_manifest.get("cards") if isinstance(original_manifest, dict) else None
    if not isinstance(manifest_cards, dict):
        raise RepairError("original inputs manifest must contain a cards object")

    name_re = re.compile(r"^[A-Za-z0-9._-]+$")
    entries: dict[str, dict[str, Any]] = {}
    for entry in sample["cards"]:
        if not isinstance(entry, dict):
            raise RepairError("sample card entry is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name_re.fullmatch(name) or name in {".", ".."}:
            raise RepairError(f"unsafe or invalid sample card name: {name!r}")
        if name in entries:
            raise RepairError(f"duplicate sample card name: {name}")
        if not isinstance(entry.get("source_run_dir"), str):
            raise RepairError(f"missing source_run_dir for {name}")
        entries[name] = entry

    names = sorted(entries)
    if set(manifest_cards) != set(names):
        raise RepairError("sample names and original input manifest names differ")
    if not inputs_dir.is_dir():
        raise RepairError(f"missing original inputs directory: {inputs_dir}")
    actual_input_names = {path.stem for path in inputs_dir.glob("*.json") if path.is_file()}
    if actual_input_names != set(names):
        raise RepairError("original input directory and sample card names differ")

    run_dirs: dict[str, Path] = {}
    docling_telemetry_paths: dict[str, Path] = {}
    abstract_expected: dict[str, str] = {}
    for name in names:
        source_run_dir = entries[name]["source_run_dir"]
        run_dir = _resolve_under_repo(
            repo, source_run_dir, what=f"source_run_dir for {name}"
        )
        run_dirs[name] = run_dir
        telemetry_paths = sorted(
            (run_dir / "tool_output" / "composer").glob("docling_telemetry_*.json")
        )
        if len(telemetry_paths) != 1:
            raise RepairError(
                f"expected exactly one docling_telemetry_*.json for {name}, "
                f"found {len(telemetry_paths)}"
            )
        telemetry = _json_load(telemetry_paths[0])
        if not isinstance(telemetry, dict) or not isinstance(
            telemetry.get("degraded_to_abstract_only"), bool
        ):
            raise RepairError(f"invalid Docling telemetry for {name}")
        docling_telemetry_paths[name] = telemetry_paths[0]
        if telemetry["degraded_to_abstract_only"]:
            abstract_expected[name] = source_run_dir

    abstracts, abstract_manifest_sha256 = _load_abstract_manifest(
        abstract_manifest_path, expected_cards=abstract_expected
    )

    planned: list[dict[str, Any]] = []
    source_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "classification": None,
            "records_total": 0,
            "records_on_judged_fields": 0,
            "judged_evidence_exactly_present": 0,
            "judged_evidence_missing": 0,
            "cards": set(),
            "cards_with_missing_judged_evidence": set(),
            "cards_with_assembled_marker_absent": set(),
        }
    )

    abstract_required_cards: set[str] = set()
    github_required_cards: set[str] = set()

    for name in names:
        input_path = inputs_dir / f"{name}.json"
        try:
            raw = input_path.read_bytes()
        except OSError as exc:
            raise RepairError(f"cannot read original input {input_path}: {exc}") from exc
        input_json = _validate_input(
            name=name, raw=raw, expected=manifest_cards[name], path=input_path
        )
        judged_fields = {field["path"]: field for field in input_json["fields"]}
        judged_paths = set(judged_fields)
        source_text = input_json["source_text"]

        run_dir = run_dirs[name]
        complete_direct_source, direct_source_artifacts = _complete_direct_sources(run_dir)
        direct_sources_exceeded_original_cap = (
            len(complete_direct_source) > ORIGINAL_SOURCE_CAP
        )
        base_source = (
            complete_direct_source if direct_sources_exceeded_original_cap else source_text
        )
        provenance_paths = sorted((run_dir / "tool_output" / "composer").glob("provenance_*.json"))
        if len(provenance_paths) != 1:
            raise RepairError(
                f"expected exactly one provenance_*.json for {name}, found {len(provenance_paths)}"
            )
        provenance_path = provenance_paths[0]
        provenance = _json_load(provenance_path)
        records = list(_iter_provenance_records(provenance))
        if not records:
            raise RepairError(f"no provenance records found for {name}")

        abstract_fields: set[str] = set()
        github_fields: set[str] = set()
        abstract_evidence: list[tuple[str, str]] = []
        github_evidence: list[tuple[str, str]] = []
        for record in records:
            classification = _source_class(record.source)
            summary = source_summary[record.source]
            summary["classification"] = classification
            summary["records_total"] += 1
            summary["cards"].add(name)
            if record.path not in judged_paths:
                continue
            summary["records_on_judged_fields"] += 1

            if classification == "internal_or_derived":
                continue
            if record.source == "eee":
                if ASSEMBLED_SOURCE_MARKERS["eee"] not in source_text:
                    raise RepairError(f"EEE provenance has no EEE source block for {name}")
                continue
            if record.source == "abstract":
                abstract_fields.add(record.path)
            elif record.source == "github":
                github_fields.add(record.path)
            if not isinstance(record.evidence, str) or not record.evidence:
                if judged_fields[record.path].get("is_ns") is True:
                    continue
                raise RepairError(
                    f"empty/non-string document evidence for {name} {record.path} "
                    f"source={record.source!r}"
                )

            exact_present = record.evidence in base_source
            if exact_present:
                summary["judged_evidence_exactly_present"] += 1
            else:
                summary["judged_evidence_missing"] += 1
                summary["cards_with_missing_judged_evidence"].add(name)

            if classification == "assembled":
                marker = ASSEMBLED_SOURCE_MARKERS[record.source]
                if marker not in base_source:
                    summary["cards_with_assembled_marker_absent"].add(name)
                    if not exact_present:
                        raise RepairError(
                            f"assembled-source parity failure for {name} {record.path}: "
                            f"source={record.source!r}, marker and exact evidence both absent"
                        )
            elif record.source == "abstract":
                abstract_evidence.append((record.path, record.evidence))
            elif record.source == "github":
                github_evidence.append((record.path, record.evidence))

        blocks: list[str] = []
        supplement_audit: list[dict[str, Any]] = []
        required_payloads: list[tuple[str, str]] = []

        if name in abstracts:
            abstract_required_cards.add(name)
            abstract = abstracts[name]["paper_abstract"]
            paper_title = abstracts[name]["paper_title"]
            abstract_payload = f"Title: {paper_title}\n\nAbstract:\n{abstract}"
            missing_from_abstract = sorted(
                path
                for path, evidence in abstract_evidence
                if not _abstract_evidence_present(
                    path=path,
                    evidence=evidence,
                    paper_title=paper_title,
                    paper_abstract=abstract,
                )
            )
            matched_in_abstract = sorted(
                path for path, _ in abstract_evidence if path not in missing_from_abstract
            )
            required_payloads.append(("abstract", abstract_payload))
            if abstract_payload not in base_source:
                blocks.append(_build_block("abstract", abstract_payload))
                supplement_audit.append(
                    {
                        "source": "abstract",
                        "reason": "degraded_to_abstract_only_full_source_missing",
                        "fields_with_abstract_provenance": sorted(abstract_fields),
                        "provenance_evidence_matched_in_recovered_source": matched_in_abstract,
                        "provenance_evidence_nonliteral_or_missing": missing_from_abstract,
                        "abstract_sha256": abstracts[name]["paper_abstract_sha256"],
                        "payload_sha256": _evidence_sha256(abstract_payload),
                        "payload_chars": len(abstract_payload),
                        "source_manifest": _display_path(abstract_manifest_path, repo),
                        "paper_title": abstracts[name]["paper_title"],
                        "recovery_source_url": abstracts[name]["recovery_source_url"],
                        "recovered_at": abstracts[name]["recovered_at"],
                        "recovery_method": abstracts[name]["recovery_method"],
                        "docling_telemetry": _display_path(docling_telemetry_paths[name], repo),
                        "docling_telemetry_sha256": _file_digest(
                            docling_telemetry_paths[name], "sha256"
                        ),
                    }
                )

        if github_fields:
            github_required_cards.add(name)
            github_paths = sorted((run_dir / "tool_output" / "github").glob("*.json"))
            if len(github_paths) != 1:
                raise RepairError(
                    f"expected exactly one persisted GitHub artifact for {name}, "
                    f"found {len(github_paths)}"
                )
            github_artifact = _json_load(github_paths[0])
            github_text = github_artifact.get("text") if isinstance(github_artifact, dict) else None
            if not isinstance(github_artifact, dict) or github_artifact.get("success") is not True:
                raise RepairError(f"invalid persisted GitHub artifact for {name}")
            if not isinstance(github_text, str) or not github_text.strip():
                raise RepairError(f"empty persisted GitHub README for {name}")
            normalized_github = _normalize_evidence(github_text)
            missing_from_github = sorted(
                path
                for path, evidence in github_evidence
                if _normalize_evidence(evidence) not in normalized_github
            )
            matched_in_github = sorted(
                path for path, _ in github_evidence if path not in missing_from_github
            )
            required_payloads.append(("github", github_text))
            if github_text not in base_source:
                blocks.append(_build_block("github", github_text))
                supplement_audit.append(
                    {
                        "source": "github",
                        "reason": "judged_field_uses_github_full_readme_missing",
                        "fields_with_github_provenance": sorted(github_fields),
                        "provenance_evidence_matched_in_persisted_source": matched_in_github,
                        "provenance_evidence_nonliteral_or_missing": missing_from_github,
                        "payload_sha256": _evidence_sha256(github_text),
                        "payload_chars": len(github_text),
                        "artifact_path": _display_path(github_paths[0], repo),
                        "artifact_sha256": _file_digest(github_paths[0], "sha256"),
                        "url": github_artifact.get("url"),
                    }
                )

        blocks.sort(
            key=lambda block: SOURCE_ORDER.index(
                "abstract" if block.startswith(BLOCK_HEADERS["abstract"]) else "github"
            )
        )

        repaired_source = base_source + (("\n\n" + "\n\n".join(blocks)) if blocks else "")
        if len(repaired_source) > REPAIRED_SOURCE_CAP:
            raise RepairError(
                f"repair would exceed REPAIRED_SOURCE_CAP for {name}: "
                f"{len(base_source)} + {len(repaired_source) - len(base_source)} "
                f"> {REPAIRED_SOURCE_CAP}; "
                "pre-existing source will not be dropped"
            )
        for source, payload in required_payloads:
            if payload not in repaired_source:
                raise RepairError(f"{source} full-source parity check failed for {name}")
        repaired_json = dict(input_json)
        repaired_json["source_text"] = repaired_source
        repaired_raw = (
            json.dumps(repaired_json).encode("utf-8")
            if blocks or direct_sources_exceeded_original_cap
            else raw
        )
        roundtrip = json.loads(repaired_raw.decode("utf-8"))
        if roundtrip["fields"] != input_json["fields"] or roundtrip["risks"] != input_json["risks"]:
            raise RepairError(f"fields/risks changed during repair for {name}")
        if not blocks and not direct_sources_exceeded_original_cap and repaired_raw != raw:
            raise RepairError(f"unchanged card bytes drifted for {name}")

        planned.append(
            {
                "name": name,
                "raw": raw,
                "repaired_raw": repaired_raw,
                "original_source_chars": len(source_text),
                "repaired_source_chars": len(repaired_source),
                "direct_sources_rebuilt_without_truncation": direct_sources_exceeded_original_cap,
                "complete_direct_source_chars": len(complete_direct_source),
                "direct_source_artifacts": direct_source_artifacts,
                "provenance_path": provenance_path,
                "provenance_sha256": _file_digest(provenance_path, "sha256"),
                "supplements": supplement_audit,
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    repaired_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        repaired_cards: dict[str, dict[str, Any]] = {}
        changed_cards: list[dict[str, Any]] = []
        for item in planned:
            destination = staging / f"{item['name']}.json"
            destination.write_bytes(item["repaired_raw"])
            repaired_cards[item["name"]] = {
                "bytes": len(item["repaired_raw"]),
                "md5": _digest(item["repaired_raw"], "md5"),
            }
            if item["repaired_raw"] != item["raw"]:
                changed_cards.append(
                    {
                        "name": item["name"],
                        "original_input_md5": _digest(item["raw"], "md5"),
                        "repaired_input_md5": _digest(item["repaired_raw"], "md5"),
                        "original_input_sha256": _digest(item["raw"], "sha256"),
                        "repaired_input_sha256": _digest(item["repaired_raw"], "sha256"),
                        "original_source_chars": item["original_source_chars"],
                        "repaired_source_chars": item["repaired_source_chars"],
                        "appended_source_chars": (
                            item["repaired_source_chars"] - item["original_source_chars"]
                        ),
                        "provenance_path": _display_path(item["provenance_path"], repo),
                        "provenance_sha256": item["provenance_sha256"],
                        "direct_sources_rebuilt_without_truncation": item[
                            "direct_sources_rebuilt_without_truncation"
                        ],
                        "complete_direct_source_chars": item["complete_direct_source_chars"],
                        "direct_source_artifacts": [
                            {
                                **{key: value for key, value in artifact.items() if key != "path"},
                                "path": _display_path(artifact["path"], repo),
                            }
                            for artifact in item["direct_source_artifacts"]
                        ],
                        "supplements": item["supplements"],
                    }
                )

        repaired_manifest = {
            "version": REPAIR_VERSION,
            "original_source_cap_chars": ORIGINAL_SOURCE_CAP,
            "repaired_source_cap_chars": REPAIRED_SOURCE_CAP,
            "parent_manifest": _display_path(original_manifest_path, repo),
            "parent_manifest_sha256": _file_digest(original_manifest_path, "sha256"),
            "cards": repaired_cards,
        }
        source_labels = {
            label: {
                **{key: value for key, value in summary.items() if not isinstance(value, set)},
                "cards": sorted(summary["cards"]),
                "cards_with_missing_judged_evidence": sorted(
                    summary["cards_with_missing_judged_evidence"]
                ),
                "cards_with_assembled_marker_absent": sorted(
                    summary["cards_with_assembled_marker_absent"]
                ),
            }
            for label, summary in sorted(source_summary.items())
        }
        abstract_supplemented = sorted(
            item["name"]
            for item in planned
            if any(supplement["source"] == "abstract" for supplement in item["supplements"])
        )
        github_supplemented = sorted(
            item["name"]
            for item in planned
            if any(supplement["source"] == "github" for supplement in item["supplements"])
        )
        direct_sources_rebuilt = sorted(
            item["name"]
            for item in planned
            if item["direct_sources_rebuilt_without_truncation"]
        )
        audit = {
            "version": REPAIR_VERSION,
            "original_source_cap_chars": ORIGINAL_SOURCE_CAP,
            "repaired_source_cap_chars": REPAIRED_SOURCE_CAP,
            "sample": _display_path(sample_path, repo),
            "sample_sha256": _file_digest(sample_path, "sha256"),
            "original_inputs_dir": _display_path(inputs_dir, repo),
            "original_manifest": _display_path(original_manifest_path, repo),
            "abstract_manifest": _display_path(abstract_manifest_path, repo),
            "abstract_manifest_sha256": abstract_manifest_sha256,
            "output_inputs_dir": _display_path(output_dir, repo),
            "repaired_manifest": _display_path(repaired_manifest_path, repo),
            "n_sample_cards": len(names),
            "n_changed_cards": len(changed_cards),
            "n_unchanged_cards_byte_identical": len(names) - len(changed_cards),
            "changed_card_names": [item["name"] for item in changed_cards],
            "parity_gate": {
                "passed": True,
                "repair_payload_policy": "complete_sources_only_no_provenance_snippets",
                "abstract_required_cards": sorted(abstract_required_cards),
                "abstract_supplemented_cards": abstract_supplemented,
                "github_required_cards": sorted(github_required_cards),
                "github_supplemented_cards": github_supplemented,
                "direct_sources_rebuilt_without_truncation": direct_sources_rebuilt,
                "all_required_full_payloads_present_after_repair": True,
                "unclassified_source_labels": [],
            },
            "source_label_audit": source_labels,
            "changed_cards": changed_cards,
        }

        manifest_tmp = repaired_manifest_path.with_name(repaired_manifest_path.name + ".tmp")
        audit_tmp = audit_path.with_name(audit_path.name + ".tmp")
        for temporary in (manifest_tmp, audit_tmp):
            if temporary.exists():
                raise RepairError(f"refusing to overwrite temporary output: {temporary}")
        _write_json(manifest_tmp, repaired_manifest)
        _write_json(audit_tmp, audit)
        os.replace(staging, output_dir)
        os.replace(manifest_tmp, repaired_manifest_path)
        os.replace(audit_tmp, audit_path)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _path_arg(value: str, repo: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-parity-repaired copies of the frozen S150 judge inputs."
    )
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--sample", default="eval/s150/sample.json")
    parser.add_argument("--inputs", default="/tmp/gold/judge_inputs_s150")
    parser.add_argument("--original-manifest", default="eval/s150/judge/inputs_md5.json")
    parser.add_argument(
        "--abstract-manifest",
        required=True,
        help="hashed s150-paper-abstracts-v1 manifest for all abstract-only runs",
    )
    parser.add_argument("--output", default="/tmp/gold/judge_inputs_s150_source_parity_repaired")
    parser.add_argument(
        "--repaired-manifest",
        default="eval/s150/judge_source_parity_repair/inputs_md5.json",
    )
    parser.add_argument("--audit", default="eval/s150/judge_source_parity_repair/audit.json")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    try:
        audit = repair_inputs(
            repo=repo,
            sample_path=_path_arg(args.sample, repo),
            inputs_dir=_path_arg(args.inputs, repo),
            original_manifest_path=_path_arg(args.original_manifest, repo),
            abstract_manifest_path=_path_arg(args.abstract_manifest, repo),
            output_dir=_path_arg(args.output, repo),
            repaired_manifest_path=_path_arg(args.repaired_manifest, repo),
            audit_path=_path_arg(args.audit, repo),
        )
    except RepairError as exc:
        parser.exit(1, f"source-parity repair failed: {exc}\n")
    print(
        f"wrote {audit['n_sample_cards']} repaired inputs; "
        f"changed={audit['n_changed_cards']} "
        f"unchanged_byte_identical={audit['n_unchanged_cards_byte_identical']}"
    )
    print("changed cards: " + ", ".join(audit["changed_card_names"]))


if __name__ == "__main__":
    main()
