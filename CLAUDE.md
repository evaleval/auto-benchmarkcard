# Repository guide

This file gives code assistants a compact map of the public repository.

## Work in these paths

- Package code: `src/auto_benchmarkcard/`
- Maintained tests: `tests/`
- Operational and evaluation programs: `scripts/`
- Public frozen artifacts: `eval/`
- Documentation and figures: `docs/`

## Before changing code

1. Read `README.md` and `DEVELOPMENT.md`.
2. Run the focused tests for the affected module.
3. Run `python -m pytest` before handing off a code change.
4. Run `python scripts/check_frozen.py` before changing evaluation code.

## Evaluation invariants

- Frozen corpus: 530 cards at the revision recorded in
  `eval/corpus/manifest.json`.
- Frozen sample: 150 cards, 75 from each flag stratum.
- Canonical aggregate record: `eval/results_summary.json`.
- Human labels are in `eval/s150/human_validation/`.
- Screen labels are in `eval/s150/screen/`.
- Raw participant returns, identity mappings, copied source text, and request
  identifiers are private and must not be added.
- Do not replace a frozen artifact with an older similarly named file.

## Repository hygiene

Keep paths relative, keep credentials in `.env`, and do not add local agent
state or generated output. Use normal Git commits without assistant
attribution, signoff trailers, or co-author trailers unless a maintainer asks
for them.
