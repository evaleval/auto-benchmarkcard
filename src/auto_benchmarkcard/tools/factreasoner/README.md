# FactReasoner validation

This package checks each claim on a composed benchmark card against source text
and flags fields that the sources do not support. It wraps
[FactReasoner](https://github.com/IBM/FactReasoner), which scores claims with
NLI and combines the scores in the Merlin probabilistic solver. In the
pipeline, `workers.run_factreasoner` calls it after the RAG stage.

## Steps

1. Re-retrieve each claim's contexts with BM25 from the card's own paper, web
   page and HF README (`tools/rag/source_retrieval.py`).
2. Rewrite each claim as a hypothesis that names the benchmark and any subset,
   version, metric or related benchmark the composer recorded for the field.
3. Score every claim against its contexts (`evaluate_factuality`).
4. Score claims that stay exactly neutral (0.5) again against overlapping
   12,000-character windows of the full source (`evaluate_factuality_two_tier`).
5. Flag fields whose average score is below 0.8 or whose claims are all
   neutral (`flag_benchmark_card_fields`). A flag is lifted when the field's
   evidence appears in the retrieved text or its value is confirmed by the EEE
   or Hugging Face metadata.

Flags go into the card's `flagged_fields` map, keyed by field path, with reasons
such as `[Possible Hallucination], ...`, `[Probable Hallucination], ...` (still
neutral after step 4) or `[Factuality Score: 0.45], ...`. On a card whose source
was blocked or very short, hallucination flags are relabelled as unverified.

## Setup

- Merlin: build the solver as described in the main README. The pipeline finds
  it through `Config.MERLIN_BIN`, which reads `MERLIN_BIN` and defaults to
  `<repo>/external/merlin/bin/merlin`.
- Model endpoint: NLI needs a chat endpoint that returns token logprobs.
  `FACTREASONER_ENGINE_TYPE` picks the route and falls back to `LLM_ENGINE_TYPE`.
  - `rits`: `RITS_API_KEY` and `RITS_API_URL`.
  - `hf`: `HF_TOKEN`, optionally `HF_API_URL` and `HF_ORG`.
  - any other value: an OpenAI-compatible endpoint at `FACTREASONER_API_BASE`
    with `FACTREASONER_API_KEY` (else `LLM_API_BASE` and `LLM_API_KEY`). On
    OpenRouter, set `FACTREASONER_PROVIDER` to a comma-separated list of
    providers that return logprobs; routing is pinned to them with no fallback.
- `FACTREASONER_MODEL` sets the model (default `llama-3.3-70b-instruct`).

## Output

The pipeline writes to `<session>/tool_output/factreasoner/`:
`factuality_results_<benchmark>.json` (marginals, per-field analysis, atom
summary, escalation details), `source_scoped_rag_<benchmark>.jsonl` (the atoms
and contexts that were scored) and, when the card has flags,
`unflag_policy_<benchmark>.json`.

## Standalone use

```bash
python -m auto_benchmarkcard.tools.factreasoner.factreasoner_tool \
    <session>/tool_output/rag/formatted_rag_results_<benchmark>.jsonl \
    --benchmark-card card.json --merlin-path "$MERLIN_BIN"
```

This runs one scoring pass without escalation or evidence-based unflagging.
