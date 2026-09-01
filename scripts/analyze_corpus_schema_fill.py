"""Reproduce the five section-level fill rates reported for the 530-card corpus.

The denominator is the fixed 40-field BenchmarkCard schema, not the keys that
happen to be present in a JSON object. A field is unfilled when it is absent,
empty, or contains only the literal abstention ``Not specified``.

Paper mapping:
  - Deployment and Corpus, Figure 2: section-level fill rates across the
    published corpus.

Usage:
  python scripts/analyze_corpus_schema_fill.py \
      --corpus-cards eval/corpus/cards \
      --out eval/corpus/schema_fill_summary.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
EXPECTED_CARD_COUNT = 530

SCHEMA = {
    "benchmark_details": (
        "name",
        "overview",
        "data_type",
        "domains",
        "languages",
        "similar_benchmarks",
        "resources",
        "authors",
        "org_url",
        "logo",
        "benchmark_type",
        "appears_in",
    ),
    "purpose_and_intended_users": (
        "goal",
        "audience",
        "tasks",
        "limitations",
        "out_of_scope_uses",
    ),
    "data": (
        "source",
        "size",
        "format",
        "annotation",
        "size_breakdown",
        "collection_date",
        "contamination_controls",
    ),
    "methodology": (
        "methods",
        "metrics",
        "calculation",
        "interpretation",
        "baseline_results",
        "validation",
        "human_baseline",
        "judge_uses_llm",
        "judge_num",
        "judge_models",
        "judge_score_consolidation",
        "validity_justification",
    ),
    "ethical_and_legal_considerations": (
        "privacy_and_anonymity",
        "data_licensing",
        "consent_procedures",
        "compliance_with_regulations",
    ),
}

SECTION_LABELS = {
    "benchmark_details": "Benchmark Details",
    "purpose_and_intended_users": "Purpose and Intended Users",
    "data": "Data",
    "methodology": "Methodology",
    "ethical_and_legal_considerations": "Ethical and Legal Considerations",
}

NOT_SPECIFIED = frozenset({"not specified", "not specified."})


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _load_card(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    card = payload.get("benchmark_card", payload)
    if not isinstance(card, dict):
        raise ValueError(f"{path}: card payload is not an object")
    return card


def _has_content(value) -> bool:
    """Return whether a schema value contains anything beyond an abstention."""
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized) and normalized not in NOT_SPECIFIED
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_content(item) for item in value)
    return True


def _corpus_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def analyze(cards_dir: Path) -> dict:
    card_paths = sorted(cards_dir.glob("*.json"), key=lambda path: path.name)
    if not card_paths:
        raise ValueError(f"{cards_dir}: no JSON cards found")
    if len(card_paths) != EXPECTED_CARD_COUNT:
        raise ValueError(
            f"{cards_dir}: expected {EXPECTED_CARD_COUNT} cards, found {len(card_paths)}"
        )

    filled = Counter()
    missing = Counter()
    ignored_extra = Counter()

    for path in card_paths:
        card = _load_card(path)
        for section, fields in SCHEMA.items():
            section_value = card.get(section)
            if section_value is None:
                section_value = {}
            if not isinstance(section_value, dict):
                raise ValueError(f"{path}: section {section!r} is not an object")

            for field in fields:
                if field not in section_value:
                    missing[f"{section}.{field}"] += 1
                if _has_content(section_value.get(field)):
                    filled[section] += 1

            for field in set(section_value) - set(fields):
                ignored_extra[f"{section}.{field}"] += 1

    card_count = len(card_paths)
    sections = {}
    total_filled = 0
    total_slots = 0
    for section, fields in SCHEMA.items():
        slots = card_count * len(fields)
        section_filled = filled[section]
        total_filled += section_filled
        total_slots += slots
        sections[section] = {
            "label": SECTION_LABELS[section],
            "field_count": len(fields),
            "fields": list(fields),
            "filled": section_filled,
            "unfilled": slots - section_filled,
            "slots": slots,
            "fill_rate": section_filled / slots,
            "fill_rate_percent": round(100 * section_filled / slots, 2),
        }

    return {
        "schema_version": 1,
        "paper_mapping": "Deployment and Corpus, Figure 2",
        "definition": (
            "Fixed 40-field schema. Absent, empty, and literal Not specified "
            "values are unfilled; other values are filled."
        ),
        "card_count": card_count,
        "schema_field_count": sum(len(fields) for fields in SCHEMA.values()),
        "corpus_sha256": _corpus_digest(card_paths),
        "sections": sections,
        "overall": {
            "filled": total_filled,
            "unfilled": total_slots - total_filled,
            "slots": total_slots,
            "fill_rate": total_filled / total_slots,
            "fill_rate_percent": round(100 * total_filled / total_slots, 2),
        },
        "missing_schema_keys": dict(sorted(missing.items())),
        "ignored_extra_fields": dict(sorted(ignored_extra.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute fixed-schema section fill rates for public cards."
    )
    parser.add_argument("--corpus-cards", default="eval/corpus/cards")
    parser.add_argument("--out", default="eval/corpus/schema_fill_summary.json")
    args = parser.parse_args()

    cards_dir = _resolve(args.corpus_cards)
    output = analyze(cards_dir)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print(f"analyzed {output['card_count']} cards")
    for section, result in output["sections"].items():
        print(
            f"  {section}: {result['filled']}/{result['slots']} "
            f"({result['fill_rate_percent']:.2f}%)"
        )
    print(f"analysis -> {args.out}")


if __name__ == "__main__":
    main()
