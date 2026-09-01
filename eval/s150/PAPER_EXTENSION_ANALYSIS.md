# Paper extension analysis

This analysis adds field-level results to the frozen corpus-v3 evaluation
without making new model calls, collecting new annotations, or tuning the
pipeline.

## Reproduction

The analysis requires the 530 JSON cards from the published corpus revision
recorded in `eval/corpus/manifest.json`. Download that revision and supply its
`cards` directory:

```bash
python scripts/analyze_paper_extensions.py \
  --corpus-cards /path/to/published-corpus/cards
```

The producer validates:

- the repository's frozen artifacts;
- the 150-card source-judge frame;
- all three human-rating tables and the adjudicated human reference;
- the verifier return against the pre-label screen scoring lock;
- all 530 card filenames, byte counts, and SHA-256 values against the published
  corpus manifest.

The frozen run used Python 3.11.7 and NumPy 2.2.6. It produced:

- `paper_extension_analysis.json`, SHA-256
  `62301484bfa51236f7007f3296cb7494d9c4b31f30dd10cf2354551ca49eff2e`;
- `paper_extension_field_matrix.csv`, SHA-256
  `d044fb9e2c90740b0db937a67ed30bf2b8cf70590126ce791e0af02bb73a039b`.

Two consecutive runs in that environment produced byte-identical files.

## Outputs

`paper_extension_analysis.json` contains:

- the mutually exclusive five-state decomposition over all 3,450 evaluated
  field-card slots;
- the three-field Ethical and Legal Considerations comparison;
- the field distribution of human-confirmed judge-unsupported decisions;
- the deterministic exact-path overlap between verifier-confirmed material
  findings and source-judge verdicts;
- the complete 23-field matrix.

`paper_extension_field_matrix.csv` is a flat, supplement-ready projection of
the 23-field matrix. Each state records its point estimate, interval, interval
method, raw numerator and denominator, and by-stratum counts. Aggregate
intervals use the stratified whole-card bootstrap. Rare field-level events use
the frozen Wilson-effective-sample-size fallback. No finite-population
correction is applied.

## Interpretation guards

- Source-judge outcomes are relative to the evidence supplied to the judge.
- A generated value of `Not specified` does not prove that information is
  absent from every public source or that a field applies to every benchmark.
- Ethical and legal coverage does not establish regulatory noncompliance.
- The overlap contains exact canonical paths named by screen-selected,
  verifier-confirmed findings. It is not a probability sample or population
  error rate.
- A finding with no exact path match is not necessarily conceptually outside
  the 23-field universe.
- The source judge and public-source screen used the same model family.
