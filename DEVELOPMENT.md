# Development

## Setup

Use Python 3.11 or newer for the package. Python 3.12.10 is required when
byte-identical reproduction of the frozen evaluation score JSON files matters.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Do not commit `.env`, generated cards, source caches, participant raw returns,
or local agent state.

## Checks

```bash
python -m pytest
python scripts/check_frozen.py
python -m build
```

`pytest` is intentionally limited to `tests/`. Historical one-off checks under
`scripts/` are not part of the maintained suite.

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

Files listed in `eval/corpus/frozen_code_hashes_md5.json` are frozen
instruments. Do not update their baseline to make a failing check pass. A
planned instrument change requires a new evaluation version and a new sample.

The current public artifact map and exact reproduction commands are in
`eval/README.md`. Private source snapshots and raw participant returns are
inputs to `scripts/build_public_evaluation.py`, not repository content.
