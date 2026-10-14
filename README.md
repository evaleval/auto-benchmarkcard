# Auto-BenchmarkCard

Auto-BenchmarkCard writes structured documentation for AI benchmarks. Given a
benchmark name and its evaluation records, the pipeline finds the benchmark's
paper, dataset repository and web pages, extracts verbatim evidence from them,
composes a card with a fixed schema, and checks the card's claims against the
collected sources. Every filled field carries provenance back to the quote it
was built from, and a field the pipeline cannot support is recorded as
"Not specified" rather than guessed.

The pipeline reads evaluation records in the format of
[Every Eval Ever](https://github.com/evaleval/every_eval_ever) and can also
start from a [Unitxt](https://github.com/IBM/unitxt) catalog entry. Cards
produced with it are published in the
[evaleval/auto-benchmarkcards](https://huggingface.co/datasets/evaleval/auto-benchmarkcards)
dataset.

## What a card contains

A card is one JSON document with five sections and a few top-level keys.

| Section | Fields |
|---|---|
| `benchmark_details` | name, overview, data type, domains, languages, similar benchmarks, resources, authors, organisation, benchmark type, and the suites the benchmark appears in |
| `purpose_and_intended_users` | goal, audience, tasks, limitations, out-of-scope uses |
| `data` | source, size, size breakdown, format, annotation, collection date, contamination controls |
| `methodology` | methods, metrics, calculation, interpretation, baseline results, validation, human baseline, judge model details, validity justification |
| `ethical_and_legal_considerations` | privacy and anonymity, data licensing, consent procedures, compliance with regulations |

`possible_risks` lists candidate risks from the IBM AI Risk Atlas with a
justification each. `flagged_fields` holds the fields whose support in the
sources was weak, and `missing_fields` the fields left as "Not specified".
`card_info` records the model, schema version and run metadata. A provenance
sidecar maps every field to the evidence items behind it, and the evidence
items carry the exact quote, its character offsets and the source document.

## How the pipeline works

The pipeline is a [LangGraph](https://github.com/langchain-ai/langgraph)
graph. An orchestrator node decides which worker runs next from the state
produced so far, and each worker writes its raw output to the run directory
before the next one starts.

**Identification.** Evaluation records are grouped by dataset name. Subject
variants of one benchmark, such as the subjects of MMLU, are collapsed into a
single benchmark, and a folder that holds several benchmarks is treated as a
composite suite whose members are documented first. Entries that reach the
pipeline only through an aggregator, without a repository or paper of their
own, pass an identity gate before any source is fetched.

**Source binding.** The benchmark is bound to a Hugging Face dataset, taken
from the record when it names one and otherwise searched on the Hub with a
strict name match. The binding is verified in three tiers, deterministic
reject, deterministic accept, and one model call for the ambiguous cases.
The paper is found through the record, the dataset card, or an open search
over OpenAlex, Semantic Scholar and arXiv, and every candidate passes a
verification step that checks that the paper introduces this benchmark and
not a sibling or a derivative. The paper is converted to text with Docling,
web pages with trafilatura, the GitHub README through the GitHub API, and
structural facts such as row counts and licence come from the dataset's
Croissant metadata.

**Composition.** The composer works in two stages. Stage A reads each source
in isolation and extracts candidate evidence as verbatim quotes assigned to
card fields. A quote survives only if it is an exact substring of its source
after whitespace normalisation. A referent frame, built from the document
structure and the benchmark's metadata, records which names in the paper
denote the target benchmark and which denote siblings, subsets, versions or
splits, so that evidence about a neighbouring benchmark can be demoted before
it reaches the card. An entity-attribution pass re-checks the surviving
quotes against that frame. Stage B then produces the card sections in three
grouped structured-output calls from the evidence digest alone, validates
each group against the schema and the evidence links, and repairs
violations for up to three rounds. Values that can be derived
deterministically from metadata, such as languages, size and licence,
override the model's output.

**Risks.** A retriever proposes candidate risks from the IBM AI Risk Atlas,
and one model call selects at most five with a justification each.

**Factuality check.** The card's prose fields are split into atomic claims.
Each claim is scored by [FactReasoner](https://github.com/IBM/FactReasoner)
against passages retrieved from the bound sources, with the probabilistic
inference done by the Merlin engine. Fields whose claims fall below the
support threshold are listed in `flagged_fields`, with an exception for
identity fields whose cited evidence passed the attribution check.

**Finalisation.** For runs that started from evaluation records, the final
step adds the recorded scores as baseline results, maps the benchmark type,
records the suites the benchmark appears in, and recomputes the list of
missing fields.

A more detailed walk through the stages, with the module that implements
each one, is in [`docs/architecture.md`](docs/architecture.md).

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/evaleval/auto-benchmarkcard.git
cd auto-benchmarkcard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The install pulls FactReasoner from a pinned commit of its repository. The
factuality check additionally needs the Merlin binary. Build it from the
[Merlin repository](https://github.com/arishofmann/merlin) and point
`MERLIN_BIN` at the executable, or place it at `external/merlin/bin/merlin`
inside the checkout. Without Merlin, `generate` still writes a card but skips
the factuality check, while `generate-unitxt` reports the run as failed.

## Configuration

Copy `.env.example` to `.env` and fill in what you use. The file documents
every variable; the ones that matter most are listed here.

- `LLM_ENGINE_TYPE` selects the model backend for all calls. Supported values
  are `hf` (Hugging Face inference, the default), `ollama`, `vllm`, `wml`,
  `rits` and `openrouter`. The composer model is read from
  `<ENGINE>_COMPOSER_MODEL`, for example `HF_COMPOSER_MODEL`.
- `COMPOSER_ENGINE_TYPE` reroutes only the composer calls to another backend,
  which is useful when the factuality model and the composer model are served
  in different places.
- `FACTREASONER_MODEL` and `FACTREASONER_ENGINE_TYPE` configure the factuality
  model. It must be served by an endpoint that returns token log
  probabilities; any OpenAI-compatible endpoint can be given with
  `FACTREASONER_API_BASE` and `FACTREASONER_API_KEY`.
- `OPENALEX_API_KEY`, `S2_API_KEY` and `GITHUB_TOKEN` are optional and raise
  the rate limits of the source search.

`benchmarkcard validate` checks that the variables required by the selected
engine are set, that the core imports work and that Merlin is executable.
With `--live` it also makes one model call.

## Running the pipeline

The repository ships a few evaluation records under `external/eee_samples`
so a first run needs no other input:

```bash
benchmarkcard generate ./external/eee_samples -b "TruthfulQA,GPQA" -o runs
```

`-b` restricts the run to the named benchmarks; without it every benchmark
found in the records is documented. A run from a Unitxt catalog entry looks
like this:

```bash
benchmarkcard generate-unitxt glue -o runs
```

Each run creates one session directory per benchmark. With `-o runs` the
layout is

```
runs/output/<slug>_<YYYY-MM-DD_HH-MM>/
  benchmarkcard/benchmark_card_<slug>.json   the card
  tool_output/
    composer/provenance_<slug>.json          field to evidence map
    composer/frame_<slug>.json               referent frame
    paper_resolver/, hf/, docling/, html/, github/, rag/, factreasoner/, ...
```

The `output/` level is always inserted under the directory given with `-o`.
Sessions can be listed and inspected:

```bash
benchmarkcard list -o runs/output
benchmarkcard show runs/output/<session> --detailed
```

For larger jobs, `scripts/batch_generate.py` runs a backlog of benchmark
names, `scripts/run_sharded.sh` runs such a backlog in parallel shards with
a per-card timeout, and `scripts/generate_manual.py` documents benchmarks
that have neither evaluation records nor a Unitxt entry from a small JSON
input file. `scripts/field_status.py` summarises, for a finished batch, how
many fields were filled from quotes, from metadata, or left empty.

## Using the composer as a library

The composer's building blocks are independent of the benchmark schema.
`field_spec.CardSchema` describes a schema's sections, fields and allowed
values, `evidence.build_extraction_prompt` and `evidence.verify_items` run
Stage A against it, `docstructure.build_index` indexes a source document,
and `validator.run_group_with_repair` runs Stage B with the repair loop.
The pipeline's own entry point, `compose_benchmark_card`, is bound to the
benchmark schema. The [model-cards](https://github.com/evaleval/model-cards)
project uses the same building blocks with a model-card schema.

## Limitations

- Sources are fetched live, so two runs of the same benchmark can differ, and
  a card is only as good as the sources that were reachable when it ran.
- Only the dataset binding and the paper binding are verified for subject
  agreement. Web pages and GitHub READMEs enter the composer unverified.
- The factuality check measures support in the collected sources, not truth.
  Flags are cues for a reviewer, and an unflagged field is not a verified one.
- The risk list is a set of candidates for human review.
- The pipeline follows URLs found in its inputs. Run it in an isolated
  environment when the inputs are untrusted; see [`SECURITY.md`](SECURITY.md).

## Repository layout

- `src/auto_benchmarkcard/`: the package, with the CLI in `cli.py`, the graph
  in `workflow.py` and `workers.py`, and the record-driven entry point in
  `eee_workflow.py`
- `src/auto_benchmarkcard/tools/`: one package per stage, with the composer
  under `tools/composer/`
- `scripts/`: batch runners and the field-status report
- `tests/`: offline tests, run with `python -m pytest`
- `external/eee_samples/`: sample evaluation records
- `spaces/benchmarkcard-webhook/`: a Hugging Face Space that regenerates
  cards when new evaluation records arrive
- `docs/architecture.md`: the stages in order, with their modules

## Citation

If you use this software, please cite the paper that introduced it:

```bibtex
@article{hofmann2025autobenchmarkcard,
  title   = {Auto-BenchmarkCard: Automated Synthesis of Benchmark Documentation},
  author  = {Hofmann, Aris and Vejsbjerg, Inge and Salwala, Dhaval and Daly, Elizabeth M.},
  journal = {arXiv preprint arXiv:2512.09577},
  year    = {2025}
}
```

## License

The code is released under the [MIT License](LICENSE). Papers, dataset
cards, web pages and benchmark content fetched at run time remain under their
own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
