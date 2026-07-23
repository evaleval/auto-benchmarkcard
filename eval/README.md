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
│   └── frozen_code_hashes_md5.json
├── s150/
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
│   └── screen/
│       ├── screen_results.json
│       ├── scoring_lock.json
│       ├── verifier_ratings.csv
│       ├── author_overlap.csv
│       ├── contamination.csv
│       ├── verification_scores.json
│       └── run_summary.json
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

Within the overlapping 23-field filled universe, 33 fields were flagged and
30 were labelled unsupported. One field overlaps. Weighted flag precision is
3.03%, recall is 1.86%, and the error miss share is 98.14%.

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
