"""Build the final revised screen-verifier V1 ZIP and a local review copy."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from build_screen_verification_worklist import (
    DEFAULT_CARD_REVISION,
    sha256 as source_sha256,
    validate_worklist_against_source,
)


REPO = Path(__file__).resolve().parents[1]
SCREEN_ZIP_TIME = (2026, 7, 18, 0, 0, 0)
VOID_SCREEN_V1_FINDINGS = 144
VOID_SCREEN_V1_SHA256 = "d8928ea640e2a4e24dfd907a164f34d77f1ee60c8ea8c871fa77d7b47d3a7162"
SUPERSEDED_REPAIRED_V1_FINDINGS = 154
SUPERSEDED_REPAIRED_V1_SHA256 = (
    "fcd9231a10aa2a4784b661d8964b21b67313c7748962044027e64ece1774d069"
)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_tree(src, dest, zip_time):
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = path.relative_to(src).as_posix()
            if any(part.startswith(".") for part in Path(rel).parts):
                raise ValueError(f"hidden file reached ZIP staging: {rel}")
            info = zipfile.ZipInfo(rel, zip_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, path.read_bytes())


def validate_screen_zip(path, instructions, worklist):
    expected = {
        "INSTRUCTIONS.txt": instructions.read_bytes(),
        "findings_worklist.csv": worklist.read_bytes(),
    }
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != list(expected):
            raise ValueError(f"screen ZIP contents differ: got={names}, expected={list(expected)}")
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"screen ZIP has a corrupt member: {corrupt}")
        for name, source_bytes in expected.items():
            if archive.read(name) != source_bytes:
                raise ValueError(f"screen ZIP member differs from staging input: {name}")
    return names


def validate_screen(worklist, screen_results, sample, card_revision):
    rows, source_records = validate_worklist_against_source(
        worklist,
        screen_results,
        sample,
        card_revision=card_revision,
        require_empty_responses=True,
    )
    return rows, source_records


def validate_screen_instructions(path, n_findings, n_cards):
    text = path.read_text(encoding="utf-8")
    scope_line = (
        f"You will check {n_findings} possible problems across {n_cards} benchmark "
        "cards. Expect about 10 to 13 hours. Hard cases may take longer. Split the "
        "work across multiple sessions."
    )
    competing_scope_lines = [
        line.strip() for line in text.splitlines()
        if "possible problems across" in line or ("Expect about" in line and "hours" in line)
    ]
    if competing_scope_lines != [scope_line]:
        raise ValueError(
            "screen instructions must contain exactly one current scope/workload "
            f"line; got={competing_scope_lines}"
        )
    required = (
        "confirmed-material",
        "confirmed-trivial",
        "not-a-defect",
        "unsure",
        "including its displayed category",
        "Do not confirm it merely because you find a different problem",
        "choose not-a-defect for this row and briefly explain it in notes",
        "The card may still have a different problem.",
        "one authoritative http(s) URL in evidence_url",
        "For unsure, briefly explain why in notes. evidence_url is optional.",
        "several claims about the same defect",
        "Do not use this rule to substitute a different defect or category.",
    )
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        raise ValueError(f"screen instructions missing current text: {missing}")


def _card_verdict_counts(source_records):
    counts = {}
    for record in source_records:
        verdict = record.get("raw_card_verdict")
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(
        description="Build the final revised screen-verifier V1 release"
    )
    ap.add_argument("--screen-worklist", required=True)
    ap.add_argument("--screen-results", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--screen-instructions", required=True)
    ap.add_argument("--card-revision", default=DEFAULT_CARD_REVISION)
    ap.add_argument("--expected-screen-sha256", required=True)
    ap.add_argument("--expected-sample-sha256", required=True)
    ap.add_argument(
        "--screen-only",
        action="store_true",
        help="required safety flag; this builder has no judge-package mode",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--review-copy", default=None)
    args = ap.parse_args()
    if not args.screen_only:
        ap.error("only --screen-only mode is supported")

    screen_worklist = resolve(args.screen_worklist)
    screen_results = resolve(args.screen_results)
    sample = resolve(args.sample)
    screen_instructions = resolve(args.screen_instructions)
    out = resolve(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite existing release directory: {out}")
    if source_sha256(screen_results) != args.expected_screen_sha256:
        raise ValueError("screen_results.json SHA-256 differs from the frozen expected hash")
    if source_sha256(sample) != args.expected_sample_sha256:
        raise ValueError("sample.json SHA-256 differs from the frozen expected hash")

    screen_rows, screen_source_records = validate_screen(
        screen_worklist, screen_results, sample, args.card_revision
    )
    screen_cards = sorted({row["card"] for row in screen_rows})
    validate_screen_instructions(
        screen_instructions, len(screen_rows), len(screen_cards)
    )

    staging = out / "staging"
    staging.mkdir(parents=True)
    screen_dest = staging / "screen_verifier_V1"
    screen_dest.mkdir()
    shutil.copyfile(screen_instructions, screen_dest / "INSTRUCTIONS.txt")
    shutil.copyfile(screen_worklist, screen_dest / "findings_worklist.csv")
    screen_zip = out / "screen_verifier_V1.zip"
    zip_tree(screen_dest, screen_zip, SCREEN_ZIP_TIME)
    zip_names = validate_screen_zip(screen_zip, screen_instructions, screen_worklist)

    checksums = [(sha256(screen_zip), screen_zip.name)]
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in checksums),
        encoding="utf-8",
    )

    audit = [
        "ANNOTATION PACKAGE AUDIT",
        "",
        (
            "PASS: screen worklist exactly matches every finding-level needs-fix "
            "finding in frozen screen_results.json."
        ),
        (
            f"PASS: screen worklist contains {len(screen_rows)} unique rows across "
            f"{len(screen_cards)} cards with pinned card links."
        ),
        (
            "PASS: response columns verifier_label, evidence_url, and notes are "
            "present in that order and exactly empty."
        ),
        (
            "PASS: instructions require one authoritative http(s) evidence URL for "
            "every decisive label and notes for unsure."
        ),
        f"PASS: frozen screen SHA-256 is {source_sha256(screen_results)}.",
        f"PASS: frozen sample SHA-256 is {source_sha256(sample)}.",
        f"PASS: participant worklist SHA-256 is {sha256(screen_worklist)}.",
        f"PASS: participant instructions SHA-256 is {sha256(screen_instructions)}.",
        f"PASS: public card revision is pinned to {args.card_revision}.",
        (
            "PASS: candidate card-level verdict counts are "
            f"{json.dumps(_card_verdict_counts(screen_source_records), sort_keys=True)}."
        ),
        (
            "PASS: screen ZIP contains exactly INSTRUCTIONS.txt and "
            "findings_worklist.csv, byte-identical to staging, with no hidden files."
        ),
        (
            f"PASS: held {VOID_SCREEN_V1_FINDINGS}-row V1 with SHA-256 "
            f"{VOID_SCREEN_V1_SHA256} is void for sending and retained separately."
        ),
        (
            f"PASS: earlier repaired {SUPERSEDED_REPAIRED_V1_FINDINGS}-row V1 "
            f"with SHA-256 {SUPERSEDED_REPAIRED_V1_SHA256} is superseded by the "
            "final evidence-capture revision and retained separately."
        ),
        "PASS: screen-only builder has no judge-package path and did not read or rebuild R1-R3.",
    ]
    audit += ["", "ZIP CONTENTS", f"{screen_zip.name}: {len(zip_names)} files"]
    audit.extend(f"  {name}" for name in zip_names)
    audit += ["", "SHA-256", *(f"{d}  {n}" for d, n in checksums)]
    (out / "PACKAGE_AUDIT.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    (out / "HOLD_NOT_SENT.txt").write_text(
        "UNSENT: final revised screen-verifier V1. Await Aris's final review and "
        f"explicit send decision.\nThe held V7 {VOID_SCREEN_V1_FINDINGS}-row V1 "
        f"({VOID_SCREEN_V1_SHA256}) remains void for sending and is retained "
        "separately as audit history.\nThe earlier repaired "
        f"{SUPERSEDED_REPAIRED_V1_FINDINGS}-row V1 "
        f"({SUPERSEDED_REPAIRED_V1_SHA256}) is also superseded and retained "
        "separately as audit history.\n",
        encoding="utf-8",
    )

    if args.review_copy:
        review = resolve(args.review_copy)
        if review.exists():
            sys.exit(f"release built, but refusing to overwrite review copy: {review}")
        shutil.copytree(out, review)
        print(f"review copy -> {review}")
    print(f"release -> {out}")
    print(
        f"screen rows={len(screen_rows)}, cards={len(screen_cards)}; "
        "judge packages untouched"
    )
    for digest, name in checksums:
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
