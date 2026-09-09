# Development

## Setup

Use Python 3.11 or newer for current development on `main`. Use the `v0.2.0`
checkout or release archive for the frozen paper evaluation; Python 3.12.10
is required when byte-identical reproduction of its score JSON files matters.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Do not commit `.env`, generated cards, source caches, participant raw returns,
or local agent state.

## Manual checks

Choose checks relevant to the change. For example, a composer schema change
can be checked with:

```bash
python -m pytest tests/test_card_schema.py
```

For packaging changes, `python -m build` builds the wheel and source archive.
Tests are discovered under `tests/`; historical checks under `scripts/` are
invoked separately when needed. Maintenance is manual: do not add GitHub
Actions workflows, automatic test or scanning jobs, Git hooks, required CI,
or branch-protection gates.

## Main entry points

- `src/auto_benchmarkcard/cli.py`: command-line interface
- `src/auto_benchmarkcard/workflow.py`: LangGraph workflow
- `src/auto_benchmarkcard/workers.py`: workflow workers
- `src/auto_benchmarkcard/tools/composer/`: card composition and validation
- `src/auto_benchmarkcard/tools/eee/`: Every Eval Ever input and source binding
- `scripts/batch_generate.py`: corpus batch runner
- `scripts/score_calibration.py`: three-rater judge validation scorer
- `scripts/score_screen_verification.py`: public-source screen scorer

## Frozen evaluation

The files in `eval/` and the instrument baseline in
`eval/corpus/frozen_code_hashes_md5.json` describe the `v0.2.0` paper release.
Run its evaluation commands, including `python scripts/check_frozen.py`, from
that tag or the release archive. Current runtime helpers have evolved, so
the historical instrument check on `main` refuses to treat them as the
evaluated instrument. Do not update the baseline to make that check pass.
An evaluation of changed instruments needs its own version and sample.

The current public artifact map and exact reproduction commands are in
`eval/README.md`. Private source snapshots and raw participant returns are
inputs to `scripts/build_public_evaluation.py`, not repository content.
