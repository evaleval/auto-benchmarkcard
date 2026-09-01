#!/usr/bin/env python3
"""Build an outcome-free source-complexity table for the frozen S150 sample.

This script deliberately reads no judge verdicts, screen results, or verifier
ratings. It reconstructs only documentary inputs available to the Stage A
source extractors from the sample's recorded source run directories.

Paper correspondence: this builds the documentary-source-count exposure used
by the exploratory analysis discussed in the main paper's Discussion and
records its replay boundary for Supplement Section K (Reproducibility).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import median
from typing import Any


RELEASE = Path(__file__).resolve().parents[1]

PAPER_BUDGET = 25_000
DOCUMENT_BUDGET = 15_000
DOCLING_RUNON_RE = re.compile(r"([.!?;:,)\]])([A-Z][a-z])")


class ExposureError(RuntimeError):
    """Raised when source lineage cannot be reconstructed unambiguously."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExposureError(f"missing required file: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExposureError(f"cannot read JSON from {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def only_json(directory: Path, *, required: bool = False) -> Path | None:
    paths = sorted(directory.glob("*.json"))
    if len(paths) > 1:
        raise ExposureError(f"multiple JSON artifacts in {directory}: {paths}")
    if required and not paths:
        raise ExposureError(f"no JSON artifact in {directory}")
    return paths[0] if paths else None


def get_hf_readme(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    readme = value.get("readme_markdown") or value.get("readme") or ""
    if isinstance(readme, str) and readme:
        return readme
    for child in value.values():
        if isinstance(child, dict):
            readme = child.get("readme_markdown") or child.get("readme") or ""
            if isinstance(readme, str) and readme:
                return readme
    return ""


def normalize_docling(text: Any) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return DOCLING_RUNON_RE.sub(r"\1 \2", text)


def source_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("text", "content", "markdown"):
        text = value.get(key)
        if isinstance(text, str) and text:
            return text
    return ""


def selected_paper_year(run_dir: Path, paper_title: str) -> str:
    verification = run_dir / "tool_output" / "paper_resolver" / "paper-verification.json"
    if not verification.exists():
        return ""
    value = load_json(verification)
    if not isinstance(value, dict):
        return ""
    metadata = value.get("metadata")
    if isinstance(metadata, dict) and metadata.get("year"):
        return str(metadata["year"])
    for candidate in value.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("title") == paper_title and candidate.get("year"):
            return str(candidate["year"])
    return ""


def build_abstract_text(run_dir: Path, record: dict[str, Any]) -> str:
    title = record.get("paper_title")
    abstract = record.get("paper_abstract")
    if not isinstance(title, str) or not title.strip():
        raise ExposureError(f"abstract manifest lacks paper title for {run_dir}")
    if not isinstance(abstract, str) or not abstract.strip():
        raise ExposureError(f"abstract manifest lacks abstract for {run_dir}")
    year = selected_paper_year(run_dir, title)
    header = f"Paper: {title}" + (f" ({year})" if year else "")
    return f"{header}\n\nAbstract: {abstract}"


def reconstruct_card(
    sample_card: dict[str, Any],
    abstract_cards: dict[str, dict[str, Any]],
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = sample_card["name"]
    source_run_dir = sample_card["source_run_dir"]
    run_dir = (run_root / source_run_dir).resolve()
    try:
        run_dir.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ExposureError(f"source run escapes --run-root for {name}") from exc
    if not run_dir.is_dir():
        raise ExposureError(f"missing source run for {name}: {run_dir}")

    tool_output = run_dir / "tool_output"
    audit: list[dict[str, Any]] = []
    channels: list[tuple[str, int, int]] = []

    telemetry_paths = sorted((tool_output / "composer").glob("docling_telemetry_*.json"))
    if len(telemetry_paths) != 1:
        raise ExposureError(
            f"expected one Docling telemetry artifact for {name}, found {len(telemetry_paths)}"
        )
    telemetry_path = telemetry_paths[0]
    telemetry = load_json(telemetry_path)
    if not isinstance(telemetry, dict):
        raise ExposureError(f"invalid Docling telemetry for {name}")

    docling_path = only_json(tool_output / "docling")
    docling = load_json(docling_path) if docling_path else {}
    full_paper_text = ""
    if isinstance(docling, dict) and docling.get("success") is True:
        full_paper_text = normalize_docling(docling.get("filtered_text") or "")

    has_paper = bool(full_paper_text)
    has_abstract = False
    paper_raw_chars = 0
    paper_stage_a_chars = 0
    paper_mode = "none"

    if has_paper:
        paper_raw_chars = len(full_paper_text)
        paper_stage_a_chars = min(paper_raw_chars, PAPER_BUDGET)
        paper_mode = "full"
        channels.append(("paper", paper_raw_chars, paper_stage_a_chars))
        audit.append(
            {
                "channel": "paper",
                "artifact": str(docling_path.relative_to(run_root.resolve())),
                "sha256": sha256(docling_path),
            }
        )
    elif telemetry.get("degraded_to_abstract_only") is True:
        abstract_record = abstract_cards.get(name)
        if abstract_record is None:
            raise ExposureError(f"missing recovered abstract for degraded card {name}")
        abstract_text = build_abstract_text(run_dir, abstract_record)
        has_abstract = True
        paper_raw_chars = len(abstract_text)
        paper_stage_a_chars = paper_raw_chars
        paper_mode = "abstract"
        channels.append(("paper_abstract", paper_raw_chars, paper_stage_a_chars))
        audit.append(
            {
                "channel": "paper_abstract",
                "paper_abstract_sha256": abstract_record["paper_abstract_sha256"],
                "recovery_source_url": abstract_record.get("recovery_source_url"),
            }
        )

    html_path = only_json(tool_output / "html")
    html = load_json(html_path) if html_path else {}
    html_text = (
        source_text(html)
        if isinstance(html, dict) and html.get("success") is True
        else ""
    )
    has_html = len(html_text) >= 50
    html_raw_chars = len(html_text) if has_html else 0
    html_stage_a_chars = min(html_raw_chars, DOCUMENT_BUDGET)
    if has_html:
        channels.append(("html", html_raw_chars, html_stage_a_chars))
        audit.append(
            {
                "channel": "html",
                "artifact": str(html_path.relative_to(run_root.resolve())),
                "sha256": sha256(html_path),
            }
        )

    github_path = only_json(tool_output / "github")
    github = load_json(github_path) if github_path else {}
    github_text = (
        source_text(github)
        if isinstance(github, dict) and github.get("success") is True
        else ""
    )
    has_github = len(github_text) >= 50
    github_raw_chars = len(github_text) if has_github else 0
    github_stage_a_chars = min(github_raw_chars, DOCUMENT_BUDGET)
    if has_github:
        channels.append(("github", github_raw_chars, github_stage_a_chars))
        audit.append(
            {
                "channel": "github",
                "artifact": str(github_path.relative_to(run_root.resolve())),
                "sha256": sha256(github_path),
            }
        )

    hf_path = only_json(tool_output / "hf")
    hf = load_json(hf_path) if hf_path else {}
    hf_readme = get_hf_readme(hf)
    has_hf_readme = len(hf_readme) >= 100
    hf_raw_chars = len(hf_readme) if has_hf_readme else 0
    hf_stage_a_chars = min(hf_raw_chars, DOCUMENT_BUDGET)
    if has_hf_readme:
        channels.append(("hf_readme", hf_raw_chars, hf_stage_a_chars))
        audit.append(
            {
                "channel": "hf_readme",
                "artifact": str(hf_path.relative_to(run_root.resolve())),
                "sha256": sha256(hf_path),
            }
        )

    visible = [item[2] for item in channels]
    raw = [item[1] for item in channels]
    row = {
        "name": name,
        "stratum": sample_card["stratum"],
        "weight": sample_card["weight"],
        "generation_lineage": sample_card["provenance"],
        "run_dir_resolution": sample_card["run_dir_resolution"],
        "source_run_dir": source_run_dir,
        "paper_mode": paper_mode,
        "has_paper": int(has_paper),
        "has_paper_abstract": int(has_abstract),
        "has_html": int(has_html),
        "has_github_readme": int(has_github),
        "has_hf_readme": int(has_hf_readme),
        "documentary_channels": "|".join(item[0] for item in channels),
        "documentary_channel_count": len(channels),
        "paper_raw_chars": paper_raw_chars,
        "paper_stage_a_chars": paper_stage_a_chars,
        "html_raw_chars": html_raw_chars,
        "html_stage_a_chars": html_stage_a_chars,
        "github_raw_chars": github_raw_chars,
        "github_stage_a_chars": github_stage_a_chars,
        "hf_raw_chars": hf_raw_chars,
        "hf_stage_a_chars": hf_stage_a_chars,
        "total_raw_documentary_chars": sum(raw),
        "total_stage_a_documentary_chars": sum(visible),
        "median_stage_a_chars_per_channel": float(median(visible)) if visible else 0.0,
        "paper_budget_reached": int(paper_mode == "full" and paper_raw_chars >= PAPER_BUDGET),
        "html_budget_reached": int(html_raw_chars >= DOCUMENT_BUDGET),
        "github_budget_reached": int(github_raw_chars >= DOCUMENT_BUDGET),
        "hf_budget_reached": int(hf_raw_chars >= DOCUMENT_BUDGET),
        "abstract_length_is_recovered_proxy": int(has_abstract),
        "reconstruction_status": "complete",
    }
    return row, audit


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=Path,
        default=RELEASE / "eval" / "s150" / "sample.json",
    )
    parser.add_argument(
        "--abstract-manifest",
        type=Path,
        required=True,
        help=(
            "Recovered-abstract manifest used for runs that degraded to "
            "abstract-only evidence."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help=(
            "Root directory against which sample.json source_run_dir values "
            "are resolved. The raw source-run tree is not redistributed."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RELEASE / "eval" / "s150" / "source_complexity",
    )
    args = parser.parse_args()

    sample = load_json(args.sample)
    abstract_manifest = load_json(args.abstract_manifest)
    if not isinstance(sample, dict) or len(sample.get("cards") or []) != 150:
        raise ExposureError("sample must contain exactly 150 cards")
    if not isinstance(abstract_manifest, dict):
        raise ExposureError("abstract manifest is not an object")
    abstract_cards = abstract_manifest.get("cards")
    if not isinstance(abstract_cards, dict):
        raise ExposureError("abstract manifest has no cards object")

    rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for card in sample["cards"]:
        row, audit = reconstruct_card(card, abstract_cards, args.run_root)
        rows.append(row)
        source_audit[row["name"]] = audit

    if len({row["name"] for row in rows}) != 150:
        raise ExposureError("exposure table does not contain 150 unique card names")
    if any(row["reconstruction_status"] != "complete" for row in rows):
        raise ExposureError("incomplete source reconstruction")

    rows.sort(key=lambda row: row["name"])
    csv_path = args.out_dir / "exposures_outcome_free.csv"
    write_csv(csv_path, rows)

    stratum_counts: dict[str, int] = {}
    for row in rows:
        stratum_counts[row["stratum"]] = stratum_counts.get(row["stratum"], 0) + 1
    channel_histogram: dict[str, int] = {}
    for row in rows:
        key = str(row["documentary_channel_count"])
        channel_histogram[key] = channel_histogram.get(key, 0) + 1

    audit_path = args.out_dir / "exposure_reconstruction_audit.json"
    audit_payload = {
        "version": "source-complexity-exposures-v1",
        "outcome_free": True,
        "definitions": {
            "documentary_channels": [
                "paper_or_abstract",
                "html",
                "github_readme",
                "hf_readme",
            ],
            "excluded_structured_inputs": ["eee", "hf_basic"],
            "paper_budget_chars": PAPER_BUDGET,
            "other_document_budget_chars": DOCUMENT_BUDGET,
            "long_paper_measure": (
                "min(normalized persisted full text, paper budget); a ceiling on "
                "retrieved source content, not exact prompt length"
            ),
            "abstract_measure": (
                "reconstructed title-and-abstract payload from the later "
                "quote-validated recovery manifest; not proof of byte identity "
                "with the original in-memory source"
            ),
        },
        "inputs": {
            "sample": str(args.sample),
            "sample_sha256": sha256(args.sample),
            "abstract_manifest": "[withheld-source-snapshot]/recovered_abstracts.json",
            "abstract_manifest_sha256": sha256(args.abstract_manifest),
            "run_root": "[withheld-source-snapshot]/source_runs",
            "builder": "scripts/build_source_complexity_exposures.py",
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
        "n_cards": len(rows),
        "n_complete": sum(row["reconstruction_status"] == "complete" for row in rows),
        "stratum_counts": stratum_counts,
        "documentary_channel_count_histogram": channel_histogram,
        "cards_without_documentary_channels": [
            row["name"] for row in rows if row["documentary_channel_count"] == 0
        ],
        "cards_at_paper_budget": [
            row["name"] for row in rows if row["paper_budget_reached"] == 1
        ],
        "source_artifacts": source_audit,
        "output": str(csv_path),
        "output_sha256": sha256(csv_path),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "n_cards": len(rows),
                "n_complete": audit_payload["n_complete"],
                "strata": stratum_counts,
                "channel_histogram": channel_histogram,
                "cards_without_documentary_channels": len(
                    audit_payload["cards_without_documentary_channels"]
                ),
                "csv": str(csv_path),
                "csv_sha256": audit_payload["output_sha256"],
                "audit": str(audit_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
