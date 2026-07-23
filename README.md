# Auto-BenchmarkCard

Auto-BenchmarkCard generates structured documentation for AI benchmarks from
evaluation records, papers, dataset repositories, and linked web sources. It
records source provenance, composes a BenchmarkCard, and checks card claims
against the collected evidence.

![Auto-BenchmarkCard pipeline](docs/figures/pipeline.png)

The editable figure is available as
[`docs/figures/pipeline.drawio`](docs/figures/pipeline.drawio).

## Repository contents

- `src/auto_benchmarkcard/`: package and command-line application
- `scripts/`: batch generation, corpus assembly, and evaluation programs
- `tests/`: offline and artifact-backed regression tests
- `eval/`: frozen v3 evaluation sample, results, public human-label
  projections, manifests, and checksums
- `docs/`: architecture notes and figure sources
- `spaces/benchmarkcard-webhook/`: optional Hugging Face Space integration

The public evaluation snapshot is based on 531 attempted entries and 530
published cards. The exact card corpus is pinned to Hugging Face revision
[`0a86cea5b55d6070bd7f1f020f01281e1631adba`](https://huggingface.co/datasets/evaleval/auto-benchmarkcards/tree/0a86cea5b55d6070bd7f1f020f01281e1631adba).
See [`eval/README.md`](eval/README.md) for the result scopes, artifact map,
sanitization policy, and reproduction commands.

## Install

Auto-BenchmarkCard requires Python 3.11 or newer.

```bash
git clone https://github.com/evaleval/auto-benchmarkcard.git
cd auto-benchmarkcard
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Copy `.env.example` to `.env` and add only the credentials needed by your
chosen backends. `.env` is ignored by Git.

For the validation phase, build
[Merlin](https://github.com/arishofmann/merlin) at `external/merlin`:

```bash
git clone https://github.com/arishofmann/merlin.git external/merlin
make -C external/merlin
```

## Run

Generate from an Every Eval Ever export:

```bash
benchmarkcard generate ./external/eee_samples \
  -b "MMLU,TruthfulQA" \
  -o ./output
```

Generate from Unitxt:

```bash
benchmarkcard generate-unitxt glue -o ./output
```

Inspect the available commands:

```bash
benchmarkcard --help
benchmarkcard validate
```

Each run writes a timestamped directory containing the final card, source
artifacts, provenance, and factuality results. Generated outputs are ignored
by Git.

## Evaluation snapshot

The primary public result record is
[`eval/results_summary.json`](eval/results_summary.json). The machine-readable
source artifacts remain available beside it. Important scope constraints:

- The source-bounded judge covers 23 content fields per sampled card.
- Human judge agreement is estimated from the probability arm. The filled
  result contains 17 rows, so its uncertainty is low precision.
- The public-source screen reports the weighted share of cards with at least
  one screen-detected, verifier-confirmed material finding. It is not an
  estimate of true defect prevalence because screen recall is unknown.
- Validation-flag precision and recall use only the overlapping filled-field
  judge universe.

Raw participant returns and copied third-party source text are not included.
Anonymized labels and evidence URLs are retained. Optional participant notes
are omitted uniformly and are not used by the scorers.

## Development

Run the maintained test suite:

```bash
python -m pytest
```

Additional setup, package checks, and frozen-artifact rules are documented in
[`DEVELOPMENT.md`](DEVELOPMENT.md). Tool-specific repository guidance is in
[`CLAUDE.md`](CLAUDE.md).

## References

- A. Sokol et al. "BenchmarkCards: Standardized Documentation for Large
  Language Model Benchmarks." arXiv:2410.12974, 2025.
- R. Marinescu et al. "FactReasoner: A Probabilistic Approach to Long-Form
  Factuality Assessment for Large Language Models." arXiv:2502.18573, 2025.
- F. Bagehorn et al. "AI Risk Atlas: Taxonomy and Tooling for Navigating AI
  Risks and Resources." arXiv:2503.05780, 2025.

## License

The repository code is available under the MIT License. Third-party content
and dependencies retain their original licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
