"""Lock, validate, and score the V1 web-screen verification return.

The V1 instrument contains every finding-level ``needs-fix`` candidate from
the frozen S150 web screen.  ``prepare`` must run before any verifier labels
are available.  It freezes the participant-visible rows, selects a seeded
author-overlap check, and writes the separate contamination worklist.

``validate`` checks a completed verifier CSV but emits no labels, counts, or
metrics.  ``score`` additionally requires the completed blind author-overlap
return before it reports the locked positive-candidate estimands.  The
card-level result is the stratum-weighted share of all 150 S cards with at
least one screen-detected, verifier-confirmed material candidate.  It is not
defect prevalence, screen recall, or screen accuracy.  The screen embargo
clears only when the separate three-card contamination return is also present
and valid.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from build_screen_verification_worklist import (
    PROTECTED_FIELDS,
    derive_worklist,
    sha256,
)
from check_frozen import check


REPO = Path(__file__).resolve().parents[1]
LABELS = (
    "confirmed-material",
    "confirmed-trivial",
    "not-a-defect",
    "unsure",
)
PARTICIPANT_RESPONSE_FIELDS = ("verifier_label", "evidence_url", "notes")
PARTICIPANT_WORKLIST_FIELDS = PROTECTED_FIELDS + PARTICIPANT_RESPONSE_FIELDS
AUTHOR_RESPONSE_FIELDS = (
    "author_label",
    "author_evidence_url",
    "author_notes",
)
CONTAMINATION_CARDS = ("mc-bench", "sunrgbd", "chartqa")
CONTAMINATION_LABELS = (
    "confirmed-contamination",
    "not-contamination",
    "unsure",
)
CONTAMINATION_PROTECTED_FIELDS = (
    "card",
    "card_reference",
    "screen_hf_repo_assessment",
    "screen_repo_findings_json",
    "screen_summary",
    "screen_citations_json",
)
CONTAMINATION_RESPONSE_FIELDS = (
    "author_assessment",
    "author_evidence_url",
    "author_notes",
)
CONTAMINATION_WORKLIST_FIELDS = (
    CONTAMINATION_PROTECTED_FIELDS + CONTAMINATION_RESPONSE_FIELDS
)
AUTHOR_OVERLAP_SEED = 20260718
AUTHOR_CARDS_PER_STRATUM = 5
EXPECTED_FINDINGS = 154
EXPECTED_FINDING_CARDS = 77
EXPECTED_SAMPLE_CARDS = 150
EXPECTED_CORPUS_CARDS = 530
EXPECTED_SAMPLE_PER_STRATUM = 75
STRATA = ("flagged", "unflagged")
EXPECTED_STRATUM_POPULATION = {"flagged": 152, "unflagged": 378}
EXPECTED_FINDING_CARDS_BY_STRATUM = {"flagged": 35, "unflagged": 42}
LOCK_VERSION = "screen-verification-scoring-lock-v2-2026-07-18"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        rows = list(reader)
        for line_no, row in enumerate(rows, 2):
            if None in row:
                raise ValueError(f"{path}:{line_no}: CSV row has extra cells")
            missing_cells = [field for field in fields if row.get(field) is None]
            if missing_cells:
                raise ValueError(
                    f"{path}:{line_no}: CSV row has missing cells for "
                    f"{missing_cells}"
                )
        return fields, rows


def _valid_http_url(value):
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_response(
    row,
    *,
    label_field,
    evidence_field,
    notes_field,
    valid_labels,
    context,
):
    """Validate one decisive-or-unsure response and return its exact cells."""
    label = row.get(label_field)
    evidence_url = row.get(evidence_field)
    notes = row.get(notes_field)
    if not label:
        raise ValueError(f"{context} {label_field} is blank")
    if label not in valid_labels:
        raise ValueError(f"{context} has invalid {label_field} {label!r}")
    if evidence_url and not _valid_http_url(evidence_url):
        raise ValueError(f"{context} has invalid {evidence_field} {evidence_url!r}")
    if label == "unsure":
        if not notes or not notes.strip():
            raise ValueError(f"{context} unsure requires nonempty {notes_field}")
    elif not evidence_url:
        raise ValueError(
            f"{context} decisive label {label!r} requires {evidence_field}"
        )
    return label, evidence_url or "", notes or ""


def validate_empty_worklist_against_source(
    worklist_path, screen_results_path, sample_path
):
    """Require the new participant schema and exact source-derived protected rows."""
    screen_results = _load_json(screen_results_path)
    sample = _load_json(sample_path)
    expected_rows, source_records = derive_worklist(screen_results, sample)
    fields, rows = _read_csv(worklist_path)
    if fields != PARTICIPANT_WORKLIST_FIELDS:
        raise ValueError(
            f"screen worklist columns differ: got={fields}, "
            f"expected={PARTICIPANT_WORKLIST_FIELDS}"
        )
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"screen worklist row count differs: got={len(rows)}, "
            f"source-derived={len(expected_rows)}"
        )
    ids = [row.get("row_id") for row in rows]
    if any(not item_id for item_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("screen worklist row_id values are blank or duplicated")
    identities = [(row.get("card"), row.get("finding_index")) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("screen worklist finding identities are duplicated")
    for position, (actual, expected) in enumerate(zip(rows, expected_rows), 1):
        for field in PROTECTED_FIELDS:
            if actual.get(field) != expected.get(field):
                raise ValueError(
                    f"screen worklist row {position} protected field {field!r} "
                    "differs from the frozen source-derived value"
                )
        if any(actual.get(field) != "" for field in PARTICIPANT_RESPONSE_FIELDS):
            raise ValueError(f"screen worklist row {position} has a prefilled response")
    return rows, source_records


def _write_csv(path, fields, rows, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_sample_design(sample, *, expected_n=EXPECTED_SAMPLE_CARDS):
    """Return the frozen S card frame with exact stratum analysis weights."""
    cards = sample.get("cards")
    if not isinstance(cards, list) or len(cards) != expected_n:
        raise ValueError(
            f"sample must contain exactly {expected_n} cards, got "
            f"{len(cards) if isinstance(cards, list) else 'invalid'}"
        )
    names = [row.get("name") for row in cards]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("sample contains a blank/non-string card name")
    if len(names) != len(set(names)):
        raise ValueError("sample card names are not unique")

    strata_meta = sample.get("strata")
    if not isinstance(strata_meta, dict) or set(strata_meta) != set(STRATA):
        raise ValueError(f"sample strata must be exactly {list(STRATA)}")
    observed = Counter(row.get("stratum") for row in cards)
    frame = []
    total_population = 0
    locked_strata = {}
    for stratum in STRATA:
        meta = strata_meta[stratum]
        try:
            population_n = int(meta["N"])
            sample_n = int(meta["n"])
            recorded_weight = float(meta["weight"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"sample stratum {stratum!r} has invalid design metadata") from None
        if sample_n != observed[stratum]:
            raise ValueError(
                f"sample stratum {stratum!r} declares n={sample_n}, "
                f"but contains {observed[stratum]} cards"
            )
        if expected_n == EXPECTED_SAMPLE_CARDS and sample_n != EXPECTED_SAMPLE_PER_STRATUM:
            raise ValueError(
                f"S150 stratum {stratum!r} must contain "
                f"{EXPECTED_SAMPLE_PER_STRATUM} cards"
            )
        if (
            expected_n == EXPECTED_SAMPLE_CARDS
            and population_n != EXPECTED_STRATUM_POPULATION[stratum]
        ):
            raise ValueError(
                f"S150 stratum {stratum!r} must represent "
                f"{EXPECTED_STRATUM_POPULATION[stratum]} corpus cards"
            )
        if population_n <= 0 or sample_n <= 0:
            raise ValueError(f"sample stratum {stratum!r} has non-positive N/n")
        exact_weight = population_n / sample_n
        if not math.isclose(recorded_weight, exact_weight, rel_tol=1e-6, abs_tol=5e-7):
            raise ValueError(
                f"sample stratum {stratum!r} weight {recorded_weight} "
                f"does not match N/n={exact_weight}"
            )
        locked_strata[stratum] = {
            "population_cards": population_n,
            "sample_cards": sample_n,
            "analysis_weight": exact_weight,
        }
        total_population += population_n

    if expected_n == EXPECTED_SAMPLE_CARDS and total_population != EXPECTED_CORPUS_CARDS:
        raise ValueError(
            f"S150 strata must represent {EXPECTED_CORPUS_CARDS} corpus cards, "
            f"got {total_population}"
        )
    for row in cards:
        name = row["name"]
        stratum = row.get("stratum")
        if stratum not in STRATA:
            raise ValueError(f"{name}: invalid sample stratum {stratum!r}")
        if row.get("weight") is not None and not math.isclose(
            float(row["weight"]), locked_strata[stratum]["analysis_weight"],
            rel_tol=1e-6, abs_tol=5e-7,
        ):
            raise ValueError(f"{name}: per-card sample weight disagrees with N/n")
        frame.append({"card": name, "stratum": stratum})
    return {
        "n_cards": len(frame),
        "population_cards": total_population,
        "strata": locked_strata,
        "cards": frame,
    }


def select_author_overlap_cards(
    worklist_cards,
    sample_design,
    *,
    seed=AUTHOR_OVERLAP_SEED,
    cards_per_stratum=AUTHOR_CARDS_PER_STRATUM,
):
    """Seeded, stratum-balanced card selection excluding contamination checks."""
    if cards_per_stratum <= 0:
        raise ValueError("cards_per_stratum must be positive")
    stratum_by_card = {row["card"]: row["stratum"] for row in sample_design["cards"]}
    pools = defaultdict(list)
    for card in sorted(set(worklist_cards) - set(CONTAMINATION_CARDS)):
        if card not in stratum_by_card:
            raise ValueError(f"worklist card {card!r} is not in the S sample")
        pools[stratum_by_card[card]].append(card)
    rng = random.Random(seed)
    selected = []
    pool_sizes = {}
    for stratum in STRATA:
        pool = sorted(pools[stratum])
        pool_sizes[stratum] = len(pool)
        if len(pool) < cards_per_stratum:
            raise ValueError(
                f"author-overlap pool for {stratum} has only {len(pool)} cards"
            )
        selected.extend(rng.sample(pool, cards_per_stratum))
    return sorted(selected), pool_sizes


def _reference_template(rows):
    prefixes = set()
    for row in rows:
        card = row["card"]
        reference = row.get("card_reference", "")
        suffix = f"/cards/{card}.json"
        if not reference.endswith(suffix):
            raise ValueError(f"{card}: card_reference is not pinned to its card JSON")
        prefixes.add(reference[: -len(suffix)])
    if len(prefixes) != 1:
        raise ValueError("worklist card references do not share one pinned revision")
    return prefixes.pop() + "/cards/{card}.json"


def build_contamination_rows(screen_results, card_reference_template):
    """Build the separate three-card author contamination check."""
    per_card = screen_results.get("per_card")
    if not isinstance(per_card, dict):
        raise ValueError("screen results must contain a per_card object")
    rows = []
    for card in CONTAMINATION_CARDS:
        verdict = per_card.get(card)
        if not isinstance(verdict, dict):
            raise ValueError(f"screen results are missing contamination card {card}")
        assessment = verdict.get("hf_repo_assessment")
        if not isinstance(assessment, str) or "contamination" not in assessment.lower():
            raise ValueError(
                f"{card}: expected a contamination hf_repo_assessment, got {assessment!r}"
            )
        citations = verdict.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError(f"{card}: contamination check has no screen citations")
        repo_findings = [
            finding for finding in verdict.get("findings", [])
            if "repo" in str(finding.get("field", "")).lower()
            or "huggingface" in str(finding.get("issue", "")).lower()
        ]
        rows.append({
            "card": card,
            "card_reference": card_reference_template.format(card=card),
            "screen_hf_repo_assessment": assessment,
            "screen_repo_findings_json": json.dumps(repo_findings, ensure_ascii=False),
            "screen_summary": verdict.get("summary", ""),
            "screen_citations_json": json.dumps(citations, ensure_ascii=False),
            "author_assessment": "",
            "author_evidence_url": "",
            "author_notes": "",
        })
    return rows


def prepare_scoring_lock(
    *,
    worklist_path,
    screen_results_path,
    sample_path,
    expected_findings=EXPECTED_FINDINGS,
    expected_finding_cards=EXPECTED_FINDING_CARDS,
    expected_sample_cards=EXPECTED_SAMPLE_CARDS,
):
    """Validate the empty packet and return lock plus author worklist rows."""
    rows, _ = validate_empty_worklist_against_source(
        worklist_path,
        screen_results_path,
        sample_path,
    )
    if len(rows) != expected_findings:
        raise ValueError(
            f"V1 must contain {expected_findings} findings, got {len(rows)}"
        )
    worklist_cards = sorted({row["card"] for row in rows})
    if len(worklist_cards) != expected_finding_cards:
        raise ValueError(
            f"V1 must contain {expected_finding_cards} finding cards, "
            f"got {len(worklist_cards)}"
        )
    sample = _load_json(sample_path)
    sample_design = validate_sample_design(sample, expected_n=expected_sample_cards)
    sample_names = {row["card"] for row in sample_design["cards"]}
    if not set(worklist_cards) <= sample_names:
        raise ValueError(
            f"V1 contains cards outside S: {sorted(set(worklist_cards)-sample_names)}"
        )

    protected_rows = [
        {field: row.get(field, "") for field in PROTECTED_FIELDS} for row in rows
    ]
    selected_cards, pool_sizes = select_author_overlap_cards(
        worklist_cards,
        sample_design,
        seed=AUTHOR_OVERLAP_SEED,
        cards_per_stratum=AUTHOR_CARDS_PER_STRATUM,
    )
    selected_set = set(selected_cards)
    selected_rows = [row for row in protected_rows if row["card"] in selected_set]
    selected_ids = [row["row_id"] for row in selected_rows]
    screen_results = _load_json(screen_results_path)
    contamination_rows = build_contamination_rows(
        screen_results, _reference_template(rows)
    )
    contamination_protected_rows = [
        {field: row.get(field, "") for field in CONTAMINATION_PROTECTED_FIELDS}
        for row in contamination_rows
    ]

    lock = {
        "version": LOCK_VERSION,
        "labels": list(LABELS),
        "estimand_lock": {
            "decided": "confirmed-material + confirmed-trivial + not-a-defect",
            "candidate_defect_confirmation": (
                "(confirmed-material + confirmed-trivial) / decided"
            ),
            "material_defect_confirmation": "confirmed-material / decided",
            "non_material_severe_call_share": (
                "(confirmed-trivial + not-a-defect) / decided"
            ),
            "non_defect_share_among_decided_screen_positives": (
                "not-a-defect / decided"
            ),
            "unsure_sensitivity": (
                "lower and upper rates over all 154 candidates, treating every "
                "unsure as respectively outside or inside each target class"
            ),
            "card_result": (
                "stratum-weighted share of all S=150 cards with at least one "
                "screen-detected, verifier-confirmed material candidate"
            ),
            "card_design_interval": (
                "approximate normal 95% interval from stratified SRSWOR variance "
                "with finite-population corrections"
            ),
            "guard": (
                "not defect prevalence, screen recall, screen accuracy, a standard "
                "false-positive rate, or human reliability"
            ),
        },
        "packet": {
            "columns": list(PARTICIPANT_WORKLIST_FIELDS),
            "protected_columns": list(PROTECTED_FIELDS),
            "response_columns": list(PARTICIPANT_RESPONSE_FIELDS),
            "sha256": sha256(worklist_path),
            "protected_rows_sha256": _canonical_sha256(protected_rows),
            "n_findings": len(rows),
            "n_cards": len(worklist_cards),
            "rows": protected_rows,
        },
        "source_artifacts": {
            "screen_results_sha256": sha256(screen_results_path),
            "sample_sha256": sha256(sample_path),
        },
        "sample_design": sample_design,
        "author_overlap": {
            "seed": AUTHOR_OVERLAP_SEED,
            "selection_rule": (
                "Python Random(seed) sample of 5 sorted eligible "
                "finding cards per S flag stratum; contamination cards excluded"
            ),
            "cards_per_stratum": AUTHOR_CARDS_PER_STRATUM,
            "excluded_cards": list(CONTAMINATION_CARDS),
            "eligible_pool_cards_by_stratum": pool_sizes,
            "cards": selected_cards,
            "row_ids": selected_ids,
            "n_rows": len(selected_rows),
            "columns": list(PROTECTED_FIELDS + AUTHOR_RESPONSE_FIELDS),
            "protected_columns": list(PROTECTED_FIELDS),
            "response_columns": list(AUTHOR_RESPONSE_FIELDS),
            "protected_rows_sha256": _canonical_sha256(selected_rows),
        },
        "contamination_check": {
            "cards": list(CONTAMINATION_CARDS),
            "separate_from_author_overlap": True,
            "not_part_of_v1_scores": True,
            "labels": list(CONTAMINATION_LABELS),
            "columns": list(CONTAMINATION_WORKLIST_FIELDS),
            "protected_columns": list(CONTAMINATION_PROTECTED_FIELDS),
            "response_columns": list(CONTAMINATION_RESPONSE_FIELDS),
            "rows": contamination_protected_rows,
            "protected_rows_sha256": _canonical_sha256(
                contamination_protected_rows
            ),
        },
    }
    return lock, selected_rows, contamination_rows


def validate_lock(lock):
    """Reject a malformed or changed scoring lock before reading labels."""
    if lock.get("version") != LOCK_VERSION:
        raise ValueError(f"unexpected V1 scoring-lock version {lock.get('version')!r}")
    if tuple(lock.get("labels") or ()) != LABELS:
        raise ValueError("scoring-lock label space differs from the frozen labels")
    packet = lock.get("packet")
    if not isinstance(packet, dict):
        raise ValueError("scoring lock has no packet block")
    if tuple(packet.get("columns") or ()) != PARTICIPANT_WORKLIST_FIELDS:
        raise ValueError("scoring-lock packet columns differ from the frozen schema")
    if tuple(packet.get("protected_columns") or ()) != PROTECTED_FIELDS:
        raise ValueError("scoring-lock protected columns differ from the frozen schema")
    if tuple(packet.get("response_columns") or ()) != PARTICIPANT_RESPONSE_FIELDS:
        raise ValueError("scoring-lock response columns differ from the frozen schema")
    rows = packet.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scoring lock has no protected packet rows")
    if packet.get("n_findings") != EXPECTED_FINDINGS:
        raise ValueError(f"scoring lock must contain exactly {EXPECTED_FINDINGS} findings")
    if packet.get("n_cards") != EXPECTED_FINDING_CARDS:
        raise ValueError(
            f"scoring lock must contain exactly {EXPECTED_FINDING_CARDS} finding cards"
        )
    if packet.get("n_findings") != len(rows):
        raise ValueError("scoring-lock finding count disagrees with its rows")
    if packet.get("n_cards") != len({row.get("card") for row in rows}):
        raise ValueError("scoring-lock card count disagrees with its rows")
    if packet.get("protected_rows_sha256") != _canonical_sha256(rows):
        raise ValueError("scoring-lock protected-row hash is invalid")
    ids = [row.get("row_id") for row in rows]
    if any(not item_id for item_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("scoring-lock row_id values are blank or duplicated")
    identities = [(row.get("card"), row.get("finding_index")) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("scoring-lock finding identities are duplicated")

    sample_design = lock.get("sample_design")
    if not isinstance(sample_design, dict):
        raise ValueError("scoring lock has no sample design")
    sample_cards = sample_design.get("cards")
    if not isinstance(sample_cards, list) or len(sample_cards) != EXPECTED_SAMPLE_CARDS:
        raise ValueError("scoring lock must carry all 150 sampled cards")
    sample_names = [row.get("card") for row in sample_cards]
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("scoring-lock sample card names are duplicated")
    sample_strata = Counter(row.get("stratum") for row in sample_cards)
    if sample_strata != Counter({stratum: 75 for stratum in STRATA}):
        raise ValueError("scoring-lock sample must contain 75 cards per flag stratum")
    if sample_design.get("population_cards") != EXPECTED_CORPUS_CARDS:
        raise ValueError("scoring-lock sample must represent the 530-card corpus")
    strata = sample_design.get("strata")
    if not isinstance(strata, dict) or set(strata) != set(STRATA):
        raise ValueError("scoring-lock sample strata are invalid")
    for stratum in STRATA:
        meta = strata[stratum]
        if meta.get("sample_cards") != EXPECTED_SAMPLE_PER_STRATUM:
            raise ValueError(f"scoring-lock {stratum} sample count is not 75")
        if meta.get("population_cards") != EXPECTED_STRATUM_POPULATION[stratum]:
            raise ValueError(
                f"scoring-lock {stratum} population count is invalid"
            )
        expected_weight = meta.get("population_cards") / meta.get("sample_cards")
        if not math.isclose(meta.get("analysis_weight"), expected_weight):
            raise ValueError(f"scoring-lock {stratum} analysis weight is invalid")
    if not {row["card"] for row in rows} <= set(sample_names):
        raise ValueError("scoring-lock packet contains a card outside S")
    stratum_by_card = {row["card"]: row["stratum"] for row in sample_cards}
    finding_card_strata = Counter(
        stratum_by_card[card] for card in {row["card"] for row in rows}
    )
    if finding_card_strata != Counter(EXPECTED_FINDING_CARDS_BY_STRATUM):
        raise ValueError(
            "scoring-lock finding cards must be 35 flagged and 42 unflagged"
        )

    overlap = lock.get("author_overlap")
    if not isinstance(overlap, dict):
        raise ValueError("scoring lock has no author-overlap block")
    if overlap.get("seed") != AUTHOR_OVERLAP_SEED:
        raise ValueError("author-overlap seed differs from the pre-label lock")
    if overlap.get("cards_per_stratum") != AUTHOR_CARDS_PER_STRATUM:
        raise ValueError("author-overlap allocation differs from the pre-label lock")
    if tuple(overlap.get("columns") or ()) != (
        PROTECTED_FIELDS + AUTHOR_RESPONSE_FIELDS
    ):
        raise ValueError("author-overlap columns differ from the frozen schema")
    if tuple(overlap.get("protected_columns") or ()) != PROTECTED_FIELDS:
        raise ValueError("author-overlap protected columns differ from the frozen schema")
    if tuple(overlap.get("response_columns") or ()) != AUTHOR_RESPONSE_FIELDS:
        raise ValueError("author-overlap response columns differ from the frozen schema")
    row_by_id = {row["row_id"]: row for row in rows}
    overlap_ids = overlap.get("row_ids")
    if not isinstance(overlap_ids, list) or len(overlap_ids) != len(set(overlap_ids)):
        raise ValueError("author-overlap row IDs are invalid")
    if not set(overlap_ids) <= set(row_by_id):
        raise ValueError("author-overlap contains a row outside the packet")
    overlap_rows = [row_by_id[item_id] for item_id in overlap_ids]
    if overlap.get("protected_rows_sha256") != _canonical_sha256(overlap_rows):
        raise ValueError("author-overlap protected-row hash is invalid")
    if set(overlap.get("cards") or ()) & set(CONTAMINATION_CARDS):
        raise ValueError("author-overlap must stay separate from contamination checks")
    expected_overlap_cards, expected_pool_sizes = select_author_overlap_cards(
        {row["card"] for row in rows},
        sample_design,
        seed=AUTHOR_OVERLAP_SEED,
        cards_per_stratum=AUTHOR_CARDS_PER_STRATUM,
    )
    if overlap.get("cards") != expected_overlap_cards:
        raise ValueError("author-overlap cards differ from the fixed-seed preselection")
    if overlap.get("eligible_pool_cards_by_stratum") != expected_pool_sizes:
        raise ValueError("author-overlap eligible-pool counts are invalid")
    expected_overlap_ids = [
        row["row_id"] for row in rows if row["card"] in set(expected_overlap_cards)
    ]
    if overlap_ids != expected_overlap_ids or overlap.get("n_rows") != len(
        expected_overlap_ids
    ):
        raise ValueError("author-overlap rows differ from the selected cards")
    contamination = lock.get("contamination_check")
    if tuple((contamination or {}).get("cards") or ()) != CONTAMINATION_CARDS:
        raise ValueError("contamination worklist card lock is invalid")
    if tuple(contamination.get("labels") or ()) != CONTAMINATION_LABELS:
        raise ValueError("contamination label space differs from its lock")
    if tuple(contamination.get("columns") or ()) != CONTAMINATION_WORKLIST_FIELDS:
        raise ValueError("contamination worklist columns differ from their lock")
    if tuple(contamination.get("protected_columns") or ()) != (
        CONTAMINATION_PROTECTED_FIELDS
    ):
        raise ValueError("contamination protected columns differ from their lock")
    if tuple(contamination.get("response_columns") or ()) != (
        CONTAMINATION_RESPONSE_FIELDS
    ):
        raise ValueError("contamination response columns differ from their lock")
    contamination_rows = contamination.get("rows")
    if not isinstance(contamination_rows, list) or [
        row.get("card") for row in contamination_rows
    ] != list(CONTAMINATION_CARDS):
        raise ValueError("contamination protected rows do not cover the locked cards")
    if contamination.get("protected_rows_sha256") != _canonical_sha256(
        contamination_rows
    ):
        raise ValueError("contamination protected-row hash is invalid")
    return rows


def validate_verifier_return(path, lock):
    """Require an exact complete return and return normalized response rows."""
    expected_rows = validate_lock(lock)
    fields, rows = _read_csv(path)
    if fields != tuple(lock["packet"]["columns"]):
        raise ValueError(
            f"{path}: columns differ from the frozen packet; got={fields}"
        )
    ids = [row.get("row_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate row_id in verifier return")
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"{path}: row count {len(rows)} differs from frozen "
            f"{len(expected_rows)}"
        )
    out = []
    for position, (actual, expected) in enumerate(zip(rows, expected_rows), 1):
        for field in PROTECTED_FIELDS:
            if actual.get(field) != expected.get(field):
                raise ValueError(
                    f"{path}: row {position} protected field {field!r} "
                    "differs from the frozen packet"
                )
        label, evidence_url, notes = _validate_response(
            actual,
            label_field="verifier_label",
            evidence_field="evidence_url",
            notes_field="notes",
            valid_labels=LABELS,
            context=f"{path}: row {position}",
        )
        out.append({
            **expected,
            "verifier_label": label,
            "evidence_url": evidence_url,
            "notes": notes,
        })
    return out


def _expected_author_rows(lock):
    rows = validate_lock(lock)
    row_by_id = {row["row_id"]: row for row in rows}
    return [row_by_id[item_id] for item_id in lock["author_overlap"]["row_ids"]]


def validate_author_return(path, lock):
    """Validate the preselected blind author worklist against its frozen rows."""
    expected_rows = _expected_author_rows(lock)
    expected_fields = tuple(PROTECTED_FIELDS) + AUTHOR_RESPONSE_FIELDS
    fields, rows = _read_csv(path)
    if fields != expected_fields:
        raise ValueError(f"{path}: author-overlap columns differ from the frozen schema")
    ids = [row.get("row_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate row_id in author-overlap return")
    if len(rows) != len(expected_rows):
        raise ValueError(f"{path}: author-overlap row count differs from the lock")
    out = []
    for position, (actual, expected) in enumerate(zip(rows, expected_rows), 1):
        for field in PROTECTED_FIELDS:
            if actual.get(field) != expected.get(field):
                raise ValueError(
                    f"{path}: author row {position} protected field {field!r} differs"
                )
        label, evidence_url, notes = _validate_response(
            actual,
            label_field="author_label",
            evidence_field="author_evidence_url",
            notes_field="author_notes",
            valid_labels=LABELS,
            context=f"{path}: author row {position}",
        )
        out.append({
            **expected,
            "author_label": label,
            "author_evidence_url": evidence_url,
            "author_notes": notes,
        })
    return out


def validate_contamination_return(path, lock):
    """Validate the separate documented author check for all three candidates."""
    validate_lock(lock)
    expected_rows = lock["contamination_check"]["rows"]
    fields, rows = _read_csv(path)
    if fields != CONTAMINATION_WORKLIST_FIELDS:
        raise ValueError(
            f"{path}: contamination columns differ from the frozen schema"
        )
    cards = [row.get("card") for row in rows]
    if len(cards) != len(set(cards)):
        raise ValueError(f"{path}: duplicate card in contamination return")
    if len(rows) != len(expected_rows):
        raise ValueError(f"{path}: contamination row count differs from the lock")
    out = []
    for position, (actual, expected) in enumerate(zip(rows, expected_rows), 1):
        for field in CONTAMINATION_PROTECTED_FIELDS:
            if actual.get(field) != expected.get(field):
                raise ValueError(
                    f"{path}: contamination row {position} protected field "
                    f"{field!r} differs"
                )
        assessment, evidence_url, notes = _validate_response(
            actual,
            label_field="author_assessment",
            evidence_field="author_evidence_url",
            notes_field="author_notes",
            valid_labels=CONTAMINATION_LABELS,
            context=f"{path}: contamination row {position}",
        )
        out.append({
            **expected,
            "author_assessment": assessment,
            "author_evidence_url": evidence_url,
            "author_notes": notes,
        })
    return out


def finding_score_block(rows):
    counts = Counter(row["verifier_label"] for row in rows)
    raw_counts = {label: counts[label] for label in LABELS}
    material = counts["confirmed-material"]
    trivial = counts["confirmed-trivial"]
    not_defect = counts["not-a-defect"]
    unsure = counts["unsure"]
    decided = material + trivial + not_defect

    def ratio(numerator):
        return numerator / decided if decided else None

    def all_candidate_range(lower_numerator):
        n = len(rows)
        return {
            "lower_numerator": lower_numerator,
            "upper_numerator": lower_numerator + unsure,
            "lower_rate": lower_numerator / n if n else None,
            "upper_rate": (lower_numerator + unsure) / n if n else None,
        }

    return {
        "n_findings": len(rows),
        "raw_label_counts": raw_counts,
        "n_unsure": unsure,
        "n_decided": decided,
        "decided_denominator_excludes_unsure": True,
        "candidate_defect_confirmation": ratio(material + trivial),
        "material_defect_confirmation": ratio(material),
        "non_material_severe_call_share": ratio(trivial + not_defect),
        "non_defect_share_among_decided_screen_positives": ratio(not_defect),
        "unsure_sensitivity_all_candidates": {
            "denominator": len(rows),
            "scope": (
                "each range separately assigns every unsure finding outside or "
                "inside the named target class"
            ),
            "candidate_defect_confirmation": all_candidate_range(
                material + trivial
            ),
            "material_defect_confirmation": all_candidate_range(material),
            "non_material_severe_call_share": all_candidate_range(
                trivial + not_defect
            ),
            "non_defect_share_among_screen_positives": all_candidate_range(
                not_defect
            ),
        },
        "inference_guard": (
            "positive-candidate confirmation only; not screen prevalence, recall, "
            "accuracy, a standard false-positive rate, or reliability"
        ),
    }


def _stratified_card_rate(detected_cards, sample):
    by_stratum = {}
    weighted_numerator = 0.0
    variance = 0.0
    population_total = sample["population_cards"]
    for stratum in STRATA:
        cards = [
            row["card"] for row in sample["cards"] if row["stratum"] == stratum
        ]
        indicators = [int(card in detected_cards) for card in cards]
        detected = sum(indicators)
        meta = sample["strata"][stratum]
        population_n = meta["population_cards"]
        sample_n = len(cards)
        weight = meta["analysis_weight"]
        weighted_detected = detected * weight
        weighted_total = population_n
        weighted_numerator += weighted_detected
        sample_share = detected / sample_n
        sample_variance = (
            sum((value - sample_share) ** 2 for value in indicators)
            / (sample_n - 1)
        )
        sampling_fraction = sample_n / population_n
        variance_component = (
            (population_n / population_total) ** 2
            * (1 - sampling_fraction)
            * sample_variance
            / sample_n
        )
        variance += variance_component
        by_stratum[stratum] = {
            "n_sampled_cards": sample_n,
            "n_screen_detected_confirmed_material_cards": detected,
            "sample_share": sample_share,
            "population_cards": population_n,
            "analysis_weight": weight,
            "weighted_detected_cards": weighted_detected,
            "weighted_total_cards": weighted_total,
            "sample_variance_binary": sample_variance,
            "srswor_variance_component": variance_component,
        }
    rate = weighted_numerator / population_total
    standard_error = math.sqrt(max(variance, 0.0))
    critical = 1.959963984540054
    interval = {
        "method": (
            "normal approximation using stratified SRSWOR variance and "
            "finite-population corrections"
        ),
        "level": 0.95,
        "standard_error": standard_error,
        "lower": max(0.0, rate - critical * standard_error),
        "upper": min(1.0, rate + critical * standard_error),
        "scope": "sampling uncertainty only; excludes verifier measurement error",
    }
    return {
        "n_detected_cards_raw": len(detected_cards),
        "weighted_detected_cards": weighted_numerator,
        "weighted_total_cards": population_total,
        "rate": rate,
        "approximate_design_interval95": interval,
        "by_stratum": by_stratum,
    }


def card_detected_material_block(rows, lock):
    """Score the screen-detected, verifier-confirmed material card indicator."""
    sample = lock["sample_design"]
    material_cards = {
        row["card"] for row in rows
        if row["verifier_label"] == "confirmed-material"
    }
    unsure_cards = {
        row["card"] for row in rows if row["verifier_label"] == "unsure"
    }
    confirmed = _stratified_card_rate(material_cards, sample)
    unsure_upper = _stratified_card_rate(material_cards | unsure_cards, sample)
    return {
        "name": (
            "stratum-weighted share of cards with at least one screen-detected, "
            "verifier-confirmed material candidate"
        ),
        "scope": "all 150 sampled cards with original S-to-corpus stratum weights",
        "inference_guard": [
            "defect prevalence",
            "screen recall",
            "screen accuracy",
            "standard false-positive rate",
            "human reliability",
        ],
        "n_sample_cards": sample["n_cards"],
        "n_screen_detected_verifier_confirmed_material_cards_raw": confirmed[
            "n_detected_cards_raw"
        ],
        "weighted_detected_cards": confirmed["weighted_detected_cards"],
        "weighted_total_cards": confirmed["weighted_total_cards"],
        "rate": confirmed["rate"],
        "approximate_design_interval95": confirmed[
            "approximate_design_interval95"
        ],
        "by_stratum": confirmed["by_stratum"],
        "unsure_sensitivity": {
            "lower_assignment": (
                "all unsure findings are treated as not confirmed-material"
            ),
            "upper_assignment": (
                "every card with at least one unsure finding is treated as having "
                "a material candidate"
            ),
            "lower_n_cards_raw": confirmed["n_detected_cards_raw"],
            "upper_n_cards_raw": unsure_upper["n_detected_cards_raw"],
            "lower_rate": confirmed["rate"],
            "upper_rate": unsure_upper["rate"],
        },
    }


def author_agreement_block(verifier_rows, author_rows, lock):
    verifier_by_id = {row["row_id"]: row for row in verifier_rows}
    matrix = {
        author: {verifier: 0 for verifier in LABELS} for author in LABELS
    }
    exact = 0
    for row in author_rows:
        verifier_label = verifier_by_id[row["row_id"]]["verifier_label"]
        author_label = row["author_label"]
        matrix[author_label][verifier_label] += 1
        exact += author_label == verifier_label
    n = len(author_rows)
    return {
        "status": "complete",
        "comparison": (
            "descriptive author-V1 consistency on the fixed answer-blind overlap"
        ),
        "inference_guard": (
            "not human reliability, consensus, or category-specific reliability; "
            "both reviewers saw the screen diagnosis and claimed ground truth"
        ),
        "n_selected_cards": len(lock["author_overlap"]["cards"]),
        "n_overlap_findings": n,
        "n_exact_agreement": exact,
        "exact_agreement": exact / n if n else None,
        "confusion_orientation": {
            "rows": "author_label",
            "columns": "verifier_label",
        },
        "confusion": matrix,
    }


def _require_validated_rows(
    actual_rows,
    expected_rows,
    *,
    protected_fields,
    label_field,
    evidence_field,
    notes_field,
    valid_labels,
    context,
):
    if len(actual_rows) != len(expected_rows):
        raise ValueError(f"{context} row count differs from its lock")
    for position, (actual, expected) in enumerate(
        zip(actual_rows, expected_rows), 1
    ):
        if any(actual.get(field) != expected.get(field) for field in protected_fields):
            raise ValueError(f"{context} row {position} differs from its lock")
        _validate_response(
            actual,
            label_field=label_field,
            evidence_field=evidence_field,
            notes_field=notes_field,
            valid_labels=valid_labels,
            context=f"{context} row {position}",
        )


def validation_receipt(lock, verifier_rows):
    """A label-blind receipt: validate the return without exposing results."""
    expected_rows = validate_lock(lock)
    _require_validated_rows(
        verifier_rows,
        expected_rows,
        protected_fields=PROTECTED_FIELDS,
        label_field="verifier_label",
        evidence_field="evidence_url",
        notes_field="notes",
        valid_labels=LABELS,
        context="verifier",
    )
    return {
        "version": "screen-verification-validation-receipt-v2",
        "v1_return_validation_complete": True,
        "protected_identity_exact": True,
        "responses_complete_and_in_locked_space": True,
        "v1_metrics_withheld_pending_blind_author_overlap": True,
        "screen_embargo_clear": False,
        "pending_components": [
            "completed exact author-overlap return",
            "completed documented contamination return",
        ],
    }


def contamination_completion_block(contamination_rows):
    if contamination_rows is None:
        return {
            "status": "pending",
            "required_cards": list(CONTAMINATION_CARDS),
            "included_in_v1_scores": False,
        }
    return {
        "status": "complete",
        "included_in_v1_scores": False,
        "checks": [
            {
                "card": row["card"],
                "author_assessment": row["author_assessment"],
                "author_evidence_url": row["author_evidence_url"],
                "author_notes": row["author_notes"],
            }
            for row in contamination_rows
        ],
    }


def score_screen_verification(
    lock, verifier_rows, author_rows, contamination_rows=None
):
    expected_verifier_rows = validate_lock(lock)
    _require_validated_rows(
        verifier_rows,
        expected_verifier_rows,
        protected_fields=PROTECTED_FIELDS,
        label_field="verifier_label",
        evidence_field="evidence_url",
        notes_field="notes",
        valid_labels=LABELS,
        context="verifier",
    )
    if author_rows is None:
        raise ValueError(
            "V1 metrics are withheld until the completed blind author-overlap "
            "return is supplied"
        )
    expected_author_rows = _expected_author_rows(lock)
    _require_validated_rows(
        author_rows,
        expected_author_rows,
        protected_fields=PROTECTED_FIELDS,
        label_field="author_label",
        evidence_field="author_evidence_url",
        notes_field="author_notes",
        valid_labels=LABELS,
        context="author overlap",
    )
    if contamination_rows is not None:
        expected_contamination_rows = lock["contamination_check"]["rows"]
        _require_validated_rows(
            contamination_rows,
            expected_contamination_rows,
            protected_fields=CONTAMINATION_PROTECTED_FIELDS,
            label_field="author_assessment",
            evidence_field="author_evidence_url",
            notes_field="author_notes",
            valid_labels=CONTAMINATION_LABELS,
            context="contamination",
        )
    author = author_agreement_block(verifier_rows, author_rows, lock)
    contamination = contamination_completion_block(contamination_rows)
    embargo_clear = contamination_rows is not None
    return {
        "version": "screen-verification-scores-v2",
        "v1_return_validation_complete": True,
        "v1_scoring_complete": True,
        "screen_embargo_clear": embargo_clear,
        "pending_components": (
            [] if embargo_clear else ["completed documented contamination return"]
        ),
        "finding_level": finding_score_block(verifier_rows),
        "card_level": card_detected_material_block(verifier_rows, lock),
        "author_v1_overlap": author,
        "contamination_check": contamination,
    }


def _prepare_command(args):
    worklist = resolve(args.worklist)
    screen_results = resolve(args.screen_results)
    sample = resolve(args.sample)
    lock, author_rows, contamination_rows = prepare_scoring_lock(
        worklist_path=worklist,
        screen_results_path=screen_results,
        sample_path=sample,
    )
    author_out = resolve(args.author_overlap_out)
    contamination_out = resolve(args.contamination_out)
    lock_out = resolve(args.lock_out)
    _write_csv(
        author_out,
        tuple(PROTECTED_FIELDS) + AUTHOR_RESPONSE_FIELDS,
        [
            dict(
                row,
                author_label="",
                author_evidence_url="",
                author_notes="",
            )
            for row in author_rows
        ],
        overwrite=args.force,
    )
    _write_csv(
        contamination_out,
        CONTAMINATION_WORKLIST_FIELDS,
        contamination_rows,
        overwrite=args.force,
    )
    lock["author_overlap"]["empty_worklist_sha256"] = sha256(author_out)
    lock["contamination_check"]["empty_worklist_sha256"] = sha256(contamination_out)
    _write_json(lock_out, lock, overwrite=args.force)
    print(
        f"scoring lock -> {lock_out} ({lock['packet']['n_findings']} findings, "
        f"{lock['packet']['n_cards']} finding cards, S={lock['sample_design']['n_cards']})"
    )
    print(
        f"blind author overlap -> {author_out} "
        f"({len(lock['author_overlap']['cards'])} cards, "
        f"{lock['author_overlap']['n_rows']} findings, "
        f"seed={lock['author_overlap']['seed']})"
    )
    print(f"separate contamination worklist -> {contamination_out}")


def _validate_command(args):
    lock = _load_json(resolve(args.lock))
    verifier_rows = validate_verifier_return(resolve(args.return_csv), lock)
    receipt = validation_receipt(lock, verifier_rows)
    out = resolve(args.out)
    _write_json(out, receipt, overwrite=args.force)
    print(f"validation receipt -> {out}")
    print("return is structurally complete; all V1 results remain withheld")


def _score_command(args):
    lock = _load_json(resolve(args.lock))
    verifier_rows = validate_verifier_return(resolve(args.return_csv), lock)
    author_rows = validate_author_return(resolve(args.author_return), lock)
    contamination_rows = (
        validate_contamination_return(resolve(args.contamination_return), lock)
        if args.contamination_return else None
    )
    result = score_screen_verification(
        lock, verifier_rows, author_rows, contamination_rows
    )
    out = resolve(args.out)
    _write_json(out, result, overwrite=args.force)
    counts = result["finding_level"]["raw_label_counts"]
    print(f"scores -> {out}")
    print(
        f"findings={len(verifier_rows)}, counts={json.dumps(counts, sort_keys=True)}, "
        f"decided={result['finding_level']['n_decided']}"
    )
    print(
        "weighted cards with a screen-detected verifier-confirmed material candidate="
        f"{result['card_level']['rate']:.6f}"
    )
    print(f"screen_embargo_clear={str(result['screen_embargo_clear']).lower()}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and score the locked V1 screen-verification instrument"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="freeze the empty packet and preselect author checks"
    )
    prepare.add_argument("--worklist", required=True)
    prepare.add_argument("--screen-results", required=True)
    prepare.add_argument("--sample", required=True)
    prepare.add_argument("--lock-out", required=True)
    prepare.add_argument("--author-overlap-out", required=True)
    prepare.add_argument("--contamination-out", required=True)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=_prepare_command)

    validate = subparsers.add_parser(
        "validate", help="validate one return without exposing labels or metrics"
    )
    validate.add_argument("--return", dest="return_csv", required=True)
    validate.add_argument("--lock", required=True)
    validate.add_argument("--out", required=True)
    validate.add_argument("--force", action="store_true")
    validate.set_defaults(func=_validate_command)

    score = subparsers.add_parser(
        "score", help="score after the blind author-overlap return is complete"
    )
    score.add_argument("--return", dest="return_csv", required=True)
    score.add_argument("--lock", required=True)
    score.add_argument("--author-return", required=True)
    score.add_argument("--contamination-return", default=None)
    score.add_argument("--out", required=True)
    score.add_argument("--force", action="store_true")
    score.set_defaults(func=_score_command)

    args = parser.parse_args()
    check()
    try:
        args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"screen verification failed: {exc}")


if __name__ == "__main__":
    main()
