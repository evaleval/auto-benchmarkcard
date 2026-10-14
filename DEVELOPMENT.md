# Development

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required. Credentials go into `.env`, which is
ignored by Git; never commit generated cards, source caches or run output.

## Tests

The test suite is offline. Run it with the Hugging Face cache disabled so
that no test reaches the network:

```bash
HF_HUB_OFFLINE=1 python -m pytest
```

`tests/conftest.py` puts this checkout's `src/` first on the import path, so
the suite always exercises the code in the working tree.

## Where things are

- `src/auto_benchmarkcard/cli.py`: the `benchmarkcard` command
- `src/auto_benchmarkcard/workflow.py`: the LangGraph graph and the
  orchestrator
- `src/auto_benchmarkcard/workers.py`: one worker per stage
- `src/auto_benchmarkcard/eee_workflow.py`: the record-driven run, the
  identity gate and the finalisation step
- `src/auto_benchmarkcard/config.py`: configuration from the environment
- `src/auto_benchmarkcard/tools/composer/`: evidence extraction, referent
  frame, card generation and validation
- `src/auto_benchmarkcard/tools/eee/`: record parsing, dataset binding and
  paper resolution
- `src/auto_benchmarkcard/tools/rag/` and `tools/factreasoner/`: claim
  atomisation, retrieval and factuality scoring
- `scripts/`: batch runners and the field-status report

## Conventions

- Comments and docstrings describe behaviour, not history. A comment that
  explains a rule should state the rule.
- Line length is 100 columns (`pyproject.toml` configures black and ruff).
- A change to the card schema lives in `tools/composer/field_spec.py` and
  the section models in `composer_tool.py`; `tests/test_card_schema.py` and
  `tests/test_card_shape.py` cover it.
- Packaging changes are checked with `python -m build`.
