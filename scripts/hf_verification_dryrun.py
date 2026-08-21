#!/usr/bin/env python
"""Offline dry-run for the resolve-time HF-match verifier: bucket each benchmark name into the
3-tier gate DETERMINISTICALLY (no LLM), so the LLM-cost cohort (Tier-2) can be measured before a
broad run.

For each input name it runs resolve_hf_repo (mechanical), and for resolved repos fetches HF
metadata and computes the same deterministic signals run_hf uses (_repo_corroboration name-only,
_token_is_namelike on the original-case name, _subject_overlap). It reports per-tier counts, the
identity subject-text length distribution (proxy for thin-identity fraction), and the list of
Tier-2-bound names (the true LLM cohort). NO LLM is called.

Inputs (one of):
  --names-file PATH   newline-delimited benchmark names (optionally "name<TAB>existing_hf_repo")
  --eee-dir PATH      an EEE scan directory (uses scan_eee_directory)

Usage:
  PYTHONPATH=src python scripts/hf_verification_dryrun.py --names-file names.txt
  PYTHONPATH=src python scripts/hf_verification_dryrun.py --eee-dir /path/to/eee --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from auto_benchmarkcard.tools.eee.eee_tool import resolve_hf_repo
from auto_benchmarkcard.tools.eee.paper_resolver import _repo_corroboration, _token_is_namelike
from auto_benchmarkcard.tools.composer.composer_tool import (
    _subject_overlap, _get_hf_readme, _eee_identity_subject)
from auto_benchmarkcard.tools.hf.hf_tool import hf_dataset_metadata
from auto_benchmarkcard.workers import HF_TIER1_ACCEPT_OVERLAP


def _classify(name, existing_hf_repo, eee_metadata):
    """Return (bucket, detail) deterministically. bucket in
    {no_repo, tier0_reject, tier1_accept, tier2_bound, fetch_error}."""
    repo_id = resolve_hf_repo(name, existing_hf_repo)
    if not repo_id:
        return "no_repo", {"repo_id": None}
    try:
        hf_data = hf_dataset_metadata.func(repo_id=repo_id)
    except Exception as e:
        return "fetch_error", {"repo_id": repo_id, "error": str(e)}

    # Mirror run_hf's repo_source: an EEE-provided repo is exempt from the Tier-0 aggregate reject
    # (its name-vs-basename mismatch is an expected abbreviation, not a wrong-repo signal).
    is_eee = bool(existing_hf_repo) and repo_id == existing_hf_repo
    corrob = _repo_corroboration(repo_id, name)
    namelike = _token_is_namelike(name)
    overlap = _subject_overlap(_get_hf_readme(hf_data), _eee_identity_subject(eee_metadata))
    detail = {"repo_id": repo_id, "repo_source": "eee" if is_eee else "search",
              "name_corroboration": corrob,
              "name_class": "coined" if namelike else "generic", "subject_overlap": overlap}
    if corrob == "aggregate_repo" and not is_eee:
        return "tier0_reject", detail
    if (corrob is None and namelike
            and overlap is not None and overlap >= HF_TIER1_ACCEPT_OVERLAP):
        return "tier1_accept", detail
    return "tier2_bound", detail


def _names_from_file(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            name = parts[0].strip()
            existing = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            out.append((name, existing, {"benchmark_name": name}))
    return out


def _names_from_eee(eee_dir):
    # scan_eee_folder yields the {name: bench} mapping; eee_to_pipeline_inputs(bench) gives each its
    # REAL eee_metadata (metric evaluation_descriptions), and bench.hf_repo is the raw EEE-provided
    # repo (pre-resolution) the classifier needs to detect the eee path.
    from auto_benchmarkcard.tools.eee.eee_tool import scan_eee_folder, eee_to_pipeline_inputs
    scan = scan_eee_folder(eee_dir)
    out = []
    for name, bench in scan.benchmarks.items():
        inputs = eee_to_pipeline_inputs(bench)
        eee_metadata = inputs["eee_metadata"]
        out.append((bench.name, bench.hf_repo, eee_metadata))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--names-file")
    g.add_argument("--eee-dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    items = _names_from_file(args.names_file) if args.names_file else _names_from_eee(args.eee_dir)
    if args.limit:
        items = items[:args.limit]

    buckets = Counter()
    subject_lens = []
    tier2_names = []
    records = []
    for name, existing, eee_metadata in items:
        bucket, detail = _classify(name, existing, eee_metadata)
        buckets[bucket] += 1
        subject_lens.append(len(_eee_identity_subject(eee_metadata).split()))
        if bucket == "tier2_bound":
            tier2_names.append(name)
        records.append({"name": name, "bucket": bucket, **detail})
        print(f"{bucket:14s} {name}")

    subject_lens.sort()
    n = len(subject_lens) or 1
    summary = {
        "n_names": len(items),
        "buckets": dict(buckets),
        "llm_cohort_tier2": buckets["tier2_bound"],
        "subject_words": {
            "min": subject_lens[0] if subject_lens else 0,
            "median": subject_lens[n // 2] if subject_lens else 0,
            "max": subject_lens[-1] if subject_lens else 0,
            "thin_lt8": sum(1 for x in subject_lens if x < 8),
        },
        "tier2_names": tier2_names,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"summary": summary, "records": records}, fh, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
