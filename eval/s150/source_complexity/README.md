# Exploratory source-complexity analysis

This directory supports the paper's limited statement that documentary source
count had a mixed, non-monotonic relationship with screen-detected,
verifier-confirmed material findings.

## Public replay

The outcome-free exposure table, sampled-card record, source-judge frame,
verifier labels, and final 530-card corpus are sufficient to replay the
statistical join:

```bash
python scripts/analyze_source_complexity.py \
  --exposures eval/s150/source_complexity/exposures_outcome_free.csv \
  --sample eval/s150/sample.json \
  --judge eval/s150/judge/analysis_frame.json \
  --verifier eval/s150/screen/verifier_ratings.csv \
  --corpus-cards /path/to/frozen-corpus/cards \
  --out-dir /path/to/replay-output
```

The frozen corpus is Hugging Face revision
`0a86cea5b55d6070bd7f1f020f01281e1631adba`; it is also included in the
paper's code-and-data archive. With Python 3.11.7, NumPy 2.2.6, and SciPy
1.17.1, the regenerated report and joined table are byte-identical to the
files here. Floating-point tail digits in `analysis_results.json` can vary
with the numerical runtime; the rounded reported estimates do not.

## Frozen construction boundary

`build_source_complexity_exposures.py` records how the exposure table was
constructed before outcomes were joined. Rebuilding that table requires the
raw source-run tree and recovered-abstract manifest. Those inputs are not
redistributed because they contain copied papers, repository pages, and web
material with varying terms. The audit file records their hashes and the
hashes of the retained source artifacts without publishing the source text.

The exposure counts available documentary source types, not unique documents.
Budgeted character counts are retrieval proxies, not exact prompt lengths.
The analysis is descriptive, not causal, and its outcome is not overall defect
prevalence.
