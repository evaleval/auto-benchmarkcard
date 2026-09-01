"""Build and validate the V1 web-screen finding worklist from frozen raw output.

The participant worklist is a deterministic view of ``screen_results.json``.
Every finding whose own severity is ``needs-fix`` is included, irrespective of
the enclosing card-level verdict.  The screen output itself is never modified.

Ordering preserves the original V1 triage rule: candidate cards are ordered by
the most urgent category among all raw findings on the card, with the frozen
S150 sample order as the tie-breaker; selected findings retain raw array order
within a card.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_CARDS = REPO / "eval" / "corpus" / "cards"
DEFAULT_CARD_REVISION = "WITHHELD_REVISION"
CARD_URL_TEMPLATE = "eval/corpus/cards/{card}.json"
TRIAGE_ORDER = {
    "wrong-identity": 0,
    "wrong-paper": 1,
    "fabricated-fact": 2,
    "wrong-section-splice": 3,
    "thin": 4,
    "other": 5,
}
WORKLIST_FIELDS = (
    "row_id",
    "card",
    "card_reference",
    "finding_index",
    "category",
    "field",
    "issue",
    "screen_claimed_ground_truth",
    "verifier_label",
    "evidence_url",
    "notes",
)
PROTECTED_FIELDS = WORKLIST_FIELDS[:-3]
RESPONSE_FIELDS = WORKLIST_FIELDS[-3:]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5()  # noqa: S324 - compatibility fingerprint, not security
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_worklist(
    screen_results,
    sample,
    card_revision=DEFAULT_CARD_REVISION,
    corpus_cards=DEFAULT_CORPUS_CARDS,
):
    """Return participant rows plus source records derived from raw findings."""
    if not card_revision or any(ch.isspace() for ch in card_revision):
        raise ValueError("card revision must be a non-empty, whitespace-free value")

    per_card = screen_results.get("per_card")
    if not isinstance(per_card, dict):
        raise ValueError("screen results must contain a per_card object")
    if screen_results.get("missing"):
        raise ValueError(f"screen results have missing cards: {screen_results['missing']}")
    if screen_results.get("problems"):
        raise ValueError(f"screen results have invalid cards: {screen_results['problems']}")

    sample_rows = sample.get("cards")
    if not isinstance(sample_rows, list) or not sample_rows:
        raise ValueError("sample must contain a non-empty cards list")
    sample_names = [row.get("name") for row in sample_rows]
    if any(not name for name in sample_names):
        raise ValueError("sample contains a card with no name")
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("sample contains duplicate card names")
    corpus_cards = resolve(corpus_cards)
    for sample_row in sample_rows:
        corpus_card = sample_row.get("corpus_card")
        if not corpus_card:
            continue
        recorded_path = resolve(corpus_card)
        public_path = corpus_cards / f"{sample_row['name']}.json"
        card_path = recorded_path if recorded_path.is_file() else public_path
        if not card_path.is_file():
            raise ValueError(
                "sample corpus card is missing from both the recorded and "
                f"public paths: recorded={recorded_path}, public={public_path}"
            )
        expected_md5 = sample_row.get("corpus_card_md5")
        if expected_md5 and md5(card_path) != expected_md5:
            raise ValueError(f"sample corpus card MD5 differs: {sample_row['name']}")
    missing_cards = set(sample_names) - set(per_card)
    extra_cards = set(per_card) - set(sample_names)
    if missing_cards or extra_cards:
        raise ValueError(
            "screen/sample card mismatch: "
            f"missing={sorted(missing_cards)}, extra={sorted(extra_cards)}"
        )

    sample_position = {name: index for index, name in enumerate(sample_names)}
    candidate_cards = []
    severe_by_card = {}
    for card in sample_names:
        verdict = per_card[card]
        if verdict.get("card") != card:
            raise ValueError(
                f"screen card binding mismatch: key={card!r}, payload={verdict.get('card')!r}"
            )
        findings = verdict.get("findings")
        if not isinstance(findings, list):
            raise ValueError(f"{card}: findings must be a list")
        severe = []
        for finding_index, finding in enumerate(findings):
            if not isinstance(finding, dict):
                raise ValueError(f"{card} finding {finding_index}: finding must be an object")
            if finding.get("category") not in TRIAGE_ORDER:
                raise ValueError(
                    f"{card} finding {finding_index}: unknown category "
                    f"{finding.get('category')!r}"
                )
            if finding.get("severity") != "needs-fix":
                continue
            for field in ("category", "field", "issue", "ground_truth"):
                if not isinstance(finding.get(field), str) or not finding[field].strip():
                    raise ValueError(
                        f"{card} finding {finding_index}: missing/non-string {field}"
                    )
            for field in ("field", "issue", "ground_truth"):
                if finding[field].lstrip().startswith(("=", "+", "-", "@")):
                    raise ValueError(
                        f"{card} finding {finding_index}: spreadsheet formula-like "
                        f"prefix in {field}"
                    )
            severe.append((finding_index, finding))
        if severe:
            severe_by_card[card] = severe
            priority = min(TRIAGE_ORDER[finding["category"]] for finding in findings)
            candidate_cards.append((priority, sample_position[card], card))

    rows = []
    source_records = []
    seen_findings = set()
    for _, _, card in sorted(candidate_cards):
        verdict = per_card[card]
        citations = verdict.get("citations")
        if not isinstance(citations, list) or not citations or any(
            not isinstance(citation, str) or not citation.strip() for citation in citations
        ):
            raise ValueError(f"{card}: candidate card must have non-empty string citations")
        finding_indices = []
        for finding_index, finding in severe_by_card[card]:
            identity = (card, finding_index)
            if identity in seen_findings:
                raise ValueError(f"duplicate raw finding identity: {identity}")
            seen_findings.add(identity)
            finding_indices.append(finding_index)
            rows.append(
                {
                    "row_id": str(len(rows) + 1),
                    "card": card,
                    "card_reference": CARD_URL_TEMPLATE.format(
                        revision=card_revision, card=card
                    ),
                    "finding_index": str(finding_index),
                    "category": finding["category"],
                    "field": finding["field"],
                    "issue": finding["issue"],
                    "screen_claimed_ground_truth": finding["ground_truth"],
                    "verifier_label": "",
                    "evidence_url": "",
                    "notes": "",
                }
            )
        source_records.append(
            {
                "card": card,
                "raw_card_verdict": verdict.get("verdict"),
                "finding_indices": finding_indices,
                "citations": citations,
                "citations_sha256": _canonical_sha256(citations),
            }
        )

    if len(rows) != len(seen_findings):
        raise ValueError("derived worklist does not have unique finding identities")
    return rows, source_records


def read_worklist(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def validate_worklist_against_source(
    worklist_path,
    screen_results_path,
    sample_path,
    card_revision=DEFAULT_CARD_REVISION,
    require_empty_responses=True,
    corpus_cards=DEFAULT_CORPUS_CARDS,
):
    """Require exact columns, rows, order, and raw text identity."""
    screen_results = _load_json(screen_results_path)
    sample = _load_json(sample_path)
    expected_rows, source_records = derive_worklist(
        screen_results,
        sample,
        card_revision=card_revision,
        corpus_cards=corpus_cards,
    )
    fields, rows = read_worklist(worklist_path)
    if fields != WORKLIST_FIELDS:
        raise ValueError(
            f"screen worklist columns differ: got={fields}, expected={WORKLIST_FIELDS}"
        )
    if len({row.get("row_id") for row in rows}) != len(rows):
        raise ValueError("screen worklist row_id values are not unique")
    identities = [(row.get("card"), row.get("finding_index")) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("screen worklist finding identities are not unique")

    actual_by_identity = {
        (row.get("card"), row.get("finding_index")): row for row in rows
    }
    expected_by_identity = {
        (row["card"], row["finding_index"]): row for row in expected_rows
    }
    missing = sorted(set(expected_by_identity) - set(actual_by_identity))
    extra = sorted(set(actual_by_identity) - set(expected_by_identity))
    if missing or extra:
        raise ValueError(
            f"screen worklist finding mismatch: missing={missing}, extra={extra}"
        )
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"screen worklist row count differs: got={len(rows)}, "
            f"source-derived={len(expected_rows)}"
        )

    for position, (actual, expected) in enumerate(zip(rows, expected_rows), 1):
        for field in PROTECTED_FIELDS:
            if actual.get(field, "") != expected[field]:
                raise ValueError(
                    f"screen worklist row {position} field {field!r} differs from "
                    "the frozen source-derived value"
                )
        if require_empty_responses and any(
            actual.get(field, "") != "" for field in RESPONSE_FIELDS
        ):
            raise ValueError(f"screen worklist row {position} has a prefilled response")
    return rows, source_records


def write_worklist(path, rows, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing worklist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path,
    *,
    worklist_path,
    screen_results_path,
    sample_path,
    card_revision,
    rows,
    source_records,
    overwrite=False,
):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    card_verdict_counts = {}
    for record in source_records:
        verdict = record["raw_card_verdict"]
        card_verdict_counts[verdict] = card_verdict_counts.get(verdict, 0) + 1
    manifest = {
        "version": "screen-verification-v1-final-revised-2026-07-18",
        "selection_rule": "finding.severity == needs-fix, independent of card verdict",
        "order_rule": (
            "minimum category triage priority across all raw findings per candidate "
            "card; frozen S150 sample order as tie-breaker; raw finding array order "
            "within card"
        ),
        "card_revision": card_revision,
        "protected_columns": list(PROTECTED_FIELDS),
        "response_columns": list(RESPONSE_FIELDS),
        "n_findings": len(rows),
        "n_cards": len(source_records),
        "candidate_card_verdict_counts": card_verdict_counts,
        "screen_results_sha256": sha256(screen_results_path),
        "sample_sha256": sha256(sample_path),
        "worklist_sha256": sha256(worklist_path),
        "source_records": source_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build the V1 worklist from frozen finding-level screen results"
    )
    parser.add_argument("--screen-results", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--card-revision", default=DEFAULT_CARD_REVISION)
    parser.add_argument("--corpus-cards", default="eval/corpus/cards")
    parser.add_argument("--expected-screen-sha256", default=None)
    parser.add_argument("--expected-sample-sha256", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    screen_results_path = resolve(args.screen_results)
    sample_path = resolve(args.sample)
    out_path = resolve(args.out)
    manifest_path = resolve(args.manifest_out)
    if args.expected_screen_sha256 and sha256(screen_results_path) != args.expected_screen_sha256:
        raise ValueError("screen_results.json SHA-256 differs from the frozen expected hash")
    if args.expected_sample_sha256 and sha256(sample_path) != args.expected_sample_sha256:
        raise ValueError("sample.json SHA-256 differs from the frozen expected hash")

    rows, source_records = derive_worklist(
        _load_json(screen_results_path),
        _load_json(sample_path),
        card_revision=args.card_revision,
        corpus_cards=args.corpus_cards,
    )
    write_worklist(out_path, rows, overwrite=args.force)
    validate_worklist_against_source(
        out_path,
        screen_results_path,
        sample_path,
        card_revision=args.card_revision,
        require_empty_responses=True,
        corpus_cards=args.corpus_cards,
    )
    write_manifest(
        manifest_path,
        worklist_path=out_path,
        screen_results_path=screen_results_path,
        sample_path=sample_path,
        card_revision=args.card_revision,
        rows=rows,
        source_records=source_records,
        overwrite=args.force,
    )
    print(
        f"worklist -> {out_path} ({len(rows)} findings across "
        f"{len(source_records)} cards)"
    )
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
