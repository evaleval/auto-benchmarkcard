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
2. Select and run focused manual checks appropriate to the affected code.
3. Keep development on `main` distinct from the frozen `v0.2.0` research
   release. Use that tag or the release archive for historical evaluation
   commands, including `python scripts/check_frozen.py`.

Do not add GitHub Actions workflows, automatic test or secret-scan jobs,
Git hooks, required CI, or branch-protection gates. Keep maintenance manual
and straightforward; do not request expanded workflow permissions.

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
- Preserve the frozen checksum baseline. Current runtime helpers differ from
  the evaluated instrument; a failed historical instrument check on `main`
  must not be fixed by rewriting that baseline.

## Repository hygiene

Keep paths relative, keep credentials in `.env`, and do not add local agent
state or generated output. Use normal Git commits without assistant
attribution, signoff trailers, or co-author trailers unless a maintainer asks
for them.
