# Frozen v3 evaluation

This directory contains the public, reviewable evaluation snapshot used for
the paper. It contains the frozen sample, automatic judge and screen outputs,
anonymized labels, final scores, and artifact manifests.

The concise result record is [`results_summary.json`](results_summary.json).
Every distributed file is covered by [`SHA256SUMS.txt`](SHA256SUMS.txt).

## Artifact map

```text
eval/
├── corpus/
│   ├── manifest.json
│   ├── corpus_stats.json
│   ├── attempted_universe.txt
│   ├── hf_published.json
│   ├── frozen_code_hashes_md5.json
│   └── schema_fill_summary.json
├── s150/
│   ├── PAPER_EXTENSION_ANALYSIS.md
│   ├── paper_extension_analysis.json
│   ├── paper_extension_field_matrix.csv
│   ├── sample.json
│   ├── manifest.json
│   ├── staging_report.json
│   ├── judge/
│   │   ├── results.json
│   │   ├── analysis_frame.json
│   │   ├── summary.json
│   │   ├── input_manifest.json
│   │   ├── verdict_manifest.json
│   │   └── run_summary.json
│   ├── human_validation/
│   │   ├── key.json
│   │   ├── ratings_r1.csv
│   │   ├── ratings_r2.csv
│   │   ├── ratings_r3.csv
│   │   ├── adjudication.csv
│   │   └── scores.json
│   ├── screen/
│   │   ├── screen_results.json
│   │   ├── scoring_lock.json
│   │   ├── verifier_ratings.csv
│   │   ├── author_overlap.csv
│   │   ├── contamination.csv
│   │   ├── verification_scores.json
│   │   └── run_summary.json
│   └── source_complexity/
│       ├── README.md
│       ├── exposures_outcome_free.csv
│       ├── exposure_reconstruction_audit.json
│       ├── analysis_joined_card_table.csv
│       ├── analysis_results.json
│       └── ANALYSIS_REPORT.md
├── provenance.json
├── results_summary.json
└── SHA256SUMS.txt
```

## Result scopes

The frozen corpus contains 530 published cards from 531 attempted entries. The
evaluation sample contains 150 cards, split evenly between the 152-card
flagged stratum and the 378-card unflagged stratum.

The source-bounded judge covers 23 content fields per card, for 3,450 field
rows. There are 2,035 filled rows and 1,415 `Not specified` rows. S-weighted
filled-field results are:

- supported, including EEE-dependent evidence: 86.08%
- partial: 12.43%
- unsupported: 1.48%

The same automated judge separately assessed the candidate risks attached to
the sampled cards. It classified 547 of 761 entries as relevant and grounded
in the supplied evidence and 214 as not. The complete judge summary retains
the S-weighted grounded-rate estimate of 74.84% (95% CI 69.38% to 79.95%) for
audit and replay. The paper reports only the unweighted sample counts because
the risk judgements were not human-validated. Neither the counts nor the
weighted rate is a headline source-support result, and every candidate risk
requires human review.

The paper-extension analysis also places every one of the 3,450 field-card
slots on one common denominator:

- filled and fully supported: 51.89%
- filled and partially supported: 7.49%
- filled and unsupported: 0.89%
- `Not specified`, information available in the supplied evidence: 7.29%
- `Not specified`, no information found in the supplied evidence: 32.42%

The full 530-card output contains 1,337 `Not specified` values among the 1,590
slots for the three prose fields in Ethical and Legal Considerations. This is a
post-hoc, schema-defined coverage description. It does not show public
non-disclosure, non-applicability, or regulatory noncompliance. In the held-out
analysis, "no information" means that the source judge found no fillable
information in the evidence supplied to it.

Three raters evaluated 75 items across 49 cards. One true three-way split was
blind-adjudicated. In the probability arm, filled-field judge-human agreement
is 89.06% with Cohen's kappa 0.7087 across 17 rows. All 15 probability-arm
`Not specified` decisions agree. The small number of sampled card clusters
means these agreement estimates are low precision.

The public-source screen produced 154 candidates across 77 cards. The verifier
labelled 111 material, 20 trivial, 23 not a defect, and 0 unsure. Using the
original sample weights, 47.80% of cards have at least one screen-detected,
verifier-confirmed material finding, with an approximate design interval of
40.17% to 55.44%. This is not true defect prevalence, screen recall, or screen
accuracy.

Among the 111 confirmed material findings, 43 named at least one exact canonical
path from the 23-field source-judge frame. They yielded 52 finding-path checks
across 35 cards. The source judge had classified 20 as supported by documentary
sources, 9 as supported with evaluation-record evidence, 17 as partial, and 6
as unsupported. These selected, potentially dependent checks do not estimate a
population error rate. A finding without an exact path match is not necessarily
conceptually outside the 23-field universe.

Within the overlapping 23-field filled universe, 33 fields were flagged and
30 were labelled unsupported. One field overlaps. Weighted flag precision is
3.03%, recall is 1.86%, and the error miss share is 98.14%.

See [`s150/PAPER_EXTENSION_ANALYSIS.md`](s150/PAPER_EXTENSION_ANALYSIS.md) for
the complete reproduction instructions and interpretation guards. Aggregate
intervals use 5,000 stratified whole-card bootstrap replicates. The 23-field
matrix records a Wilson-effective-sample-size fallback for rare field-level
events. No finite-population correction is applied.

## Public projection policy

Raw participant returns, identity mappings, messages, consent records, local
paths, and request identifiers are private. Copied paper and web-page source
snapshots are also excluded because their redistribution rights vary.

The public projections follow these rules:

- R1, R2, and R3 labels are unchanged and use the common seven-column CSV
  format.
- R1 was normalized from the participant-confirmed return before projection.
  The rule uses `final_label` when present and otherwise `auto_label`.
- Optional participant notes are blanked uniformly for R1 through R3 and V1.
  They are not used by either scorer.
- Author-overlap notes are also blanked because they are not scored.
- The three contamination notes are retained because they are included in the
  final score record.
- One public academic email address quoted inside a screen finding is replaced
  with `[redacted email]` in the public screen artifacts.
- Full private input hashes are recorded in `provenance.json`.

This policy does not change participant labels, evidence URLs, counts, or
reported metrics.

## Corpus fingerprint note

The frozen `sample.json` records corpus fingerprint
`c4f518e471ca09c14acf90fdcdf7cc56`. The final published staging tree has
fingerprint `81a7a3001f2ed7ec4ca1f914a03d8542`.

The sample file is not rewritten. Release verification confirms:

- all 150 sampled card MD5 values match the final published card files;
- the final corpus still contains 152 flagged and 378 unflagged cards;
- seed `20260704` reproduces exactly the same 75 plus 75 selection.

Both fingerprints and per-card SHA-256 values are retained in
`corpus/manifest.json`.

## Reproduce the final scores

Use Python 3.12.10 for byte-identical float serialization.

```bash
python3.12 scripts/check_frozen.py

python3.12 scripts/score_calibration.py \
  --ratings \
    eval/s150/human_validation/ratings_r1.csv \
    eval/s150/human_validation/ratings_r2.csv \
    eval/s150/human_validation/ratings_r3.csv \
  --key eval/s150/human_validation/key.json \
  --adjudicated eval/s150/human_validation/adjudication.csv \
  --out /tmp/calibration_scores.json

python3.12 scripts/score_screen_verification.py score \
  --return eval/s150/screen/verifier_ratings.csv \
  --lock eval/s150/screen/scoring_lock.json \
  --author-return eval/s150/screen/author_overlap.csv \
  --contamination-return eval/s150/screen/contamination.csv \
  --out /tmp/screen_scores.json
```

Expected SHA-256 values:

```text
98e0938fd3d501d81a9b6d648c19dca813324e6ff0be5bd786ea47e3e456a150  /tmp/calibration_scores.json
b35b38e4e023a2211ca17d2b0638d9703e32c57c224ce521cd0242d22312aca4  /tmp/screen_scores.json
```

Reproduce the paper-extension analysis from the pinned published corpus using
Python 3.11.7 and NumPy 2.2.6:

```bash
git clone https://huggingface.co/datasets/evaleval/auto-benchmarkcards \
  /tmp/auto-benchmarkcards-corpus
git -C /tmp/auto-benchmarkcards-corpus checkout \
  0a86cea5b55d6070bd7f1f020f01281e1631adba

python scripts/analyze_paper_extensions.py \
  --corpus-cards /tmp/auto-benchmarkcards-corpus/cards
```

Replay the exploratory documentary-source-complexity analysis from the same
corpus and the public evaluation projections:

```bash
python scripts/analyze_source_complexity.py \
  --exposures eval/s150/source_complexity/exposures_outcome_free.csv \
  --sample eval/s150/sample.json \
  --judge eval/s150/judge/analysis_frame.json \
  --verifier eval/s150/screen/verifier_ratings.csv \
  --corpus-cards /tmp/auto-benchmarkcards-corpus/cards \
  --out-dir /tmp/source-complexity-replay
```

See [`s150/source_complexity/README.md`](s150/source_complexity/README.md) for
the frozen exposure-construction boundary and numerical-runtime note.

Verify the distributed snapshot:

```bash
cd eval
shasum -a 256 -c SHA256SUMS.txt
```

## Rebuild the public projection

Maintainers with access to the private frozen workspace can rebuild this
directory through the explicit allowlist:

```bash
python scripts/build_public_evaluation.py \
  --source-root /path/to/private/integration-worktree \
  --out eval
```

Never replace this with a recursive copy of a private worktree.
