# Pipeline architecture

This page lists the stages in the order they run and names the module that
implements each one. The entry points are `cli.py` for the commands,
`eee_workflow.run_eee_pipeline` for runs that start from evaluation records,
and `workflow.build_workflow` for the LangGraph graph that both commands use.

## Before the graph (evaluation records only)

1. **Scan and group.** `tools.eee.eee_tool.scan_eee_folder` reads the
   evaluation records, groups them by dataset name, collapses subject
   variants of one benchmark into a single entry, and marks folders that hold
   several benchmarks as composite suites.
2. **Filter.** `eee_workflow._apply_benchmark_filter` applies the `-b`
   option. A composite name expands to its members.
3. **Dataset binding.** `tools.eee.eee_tool.resolve_hf_repo` uses the
   repository named in the record, or searches the Hugging Face Hub and
   accepts only a strict name match.
4. **Identity gate.** `eee_workflow.gate_benchmark_identity` runs for entries
   known only through an aggregator. A structural prescreen decides whether
   one model call is needed; the entry is skipped only on a confident
   negative verdict.
5. **Session setup.** `eee_workflow.process_single_benchmark` creates the run
   directory, writes the record, looks up a Unitxt paper reference when the
   record has none, and pre-populates the graph state.

## The graph

The orchestrator in `workflow.py` routes to the next worker from the current
state. Workers live in `workers.py`; each writes its output under
`tool_output/` before returning to the orchestrator.

6. **Unitxt lookup and ID extraction** (`run_unitxt`, `run_extractor`) run
   only for `generate-unitxt`, which starts from a catalog entry rather than
   from evaluation records.
7. **Dataset metadata and verification** (`run_hf`). Fetches the dataset
   card and metadata. The binding is verified in three tiers: deterministic
   reject, deterministic accept, and one model call for the ambiguous band.
   A rejected binding is dropped rather than used.
8. **Paper URL from the dataset card** (`run_hf_extractor`).
9. **Paper resolution** (`run_paper_resolver`, `tools.eee.paper_resolver`).
   A pre-set URL is verified first; otherwise the dataset card's links are
   tried, then an open search over the Entity Registry, OpenAlex and
   Semantic Scholar, with a title prefilter and one batched model
   verification. The verifier rejects papers about a sibling benchmark or a
   derivative of the target.
10. **Paper text** (`run_docling`, `tools.docling`). The URL is normalised
    and probed; PDFs are converted with Docling, and an HTML-only page is
    handed to the web stage instead.
11. **Web pages** (`run_html_extractor`, `tools.html`). Up to three ranked
    non-PDF URLs are extracted with trafilatura.
12. **GitHub README** (`run_github_readme`, `tools.github`). The repository
    is derived from the paper, the website, the dataset card and the
    record's URLs.
13. **Composer** (`run_composer`, `tools.composer`). See below.
14. **Risk identification** (`run_risk_identification`,
    `tools.ai_atlas_nexus`). A retriever proposes candidates from the IBM AI
    Risk Atlas; one model call selects at most five with justifications.
15. **Claim atomisation and retrieval** (`run_rag`, `tools.rag`). The card's
    prose fields are split into atomic claims, and supporting passages are
    retrieved from the bound sources with hybrid BM25 and embedding search.
16. **Factuality scoring** (`run_factreasoner`, `tools.factreasoner`). Each
    claim is scored by FactReasoner with Merlin against source-scoped
    contexts; claims with no decisive context are rescored over wider
    windows of the full source. Fields below the threshold go to
    `flagged_fields`, subject to the unflag rules in `validation_policy.py`.
    This is the last node of the graph.

## Inside the composer

`tools.composer.composer_tool.compose_benchmark_card` runs these steps.

- **Content assembly.** The paper text is used whole when it fits the budget;
  otherwise the introduction plus retrieved passages per section are used.
  The abstract from the resolver is the fallback when conversion failed.
- **Document index and identity anchor** (`docstructure.build_index`). The
  source is indexed by section and table so that quotes can be located and
  their region recorded.
- **Referent frame** (`frame.build_frame`). Records the target benchmark's
  name and its siblings, subsets, versions, splits and metrics, from the
  metadata and the document index, with one optional model pass.
- **Stage A evidence** (`evidence.build_extraction_prompt`,
  `evidence.verify_items`). Each source is read in isolation; the model
  returns verbatim quotes assigned to fields, and every quote must be an
  exact substring of the source after whitespace normalisation.
- **Deterministic facts** (`extract_deterministic_facts`). Languages, data
  type, format, size, licence and metrics are derived from the record, the
  dataset metadata and the Croissant description.
- **Gap fill** (`_fill_paper_gaps`). Fields still without evidence are
  queried against the paper once more.
- **Assignment gates and entity attribution** (`aboutness.apply_assignment_gates`,
  `eav.run_eav`). Quotes that describe a sibling or a subset rather than the
  target are demoted.
- **Stage B generation** (`validator.run_group_with_repair`). Three grouped
  structured-output calls produce the sections from the evidence digest.
  Each group is validated against the schema, the enums and the evidence
  links, and repaired for at most three rounds. Residual violations are
  recorded in `flagged_fields`.
- **Aboutness guard and post-processing.** Judge-model and data-source fields
  that are not about the target are cleared; deterministic values override
  the model's output; size and version conflicts are surfaced; display fields
  such as the logo are attached; a confidence summary is computed.

## After the graph (evaluation records only)

17. **Finalisation** (`eee_workflow.finalize_eee_card`). Adds the recorded
    scores as baseline results, maps the benchmark type, records the suites
    the benchmark appears in, sets the judge fields from the records,
    recomputes `missing_fields`, and writes the card. The best available
    card is used, so a run whose factuality stage failed still produces a
    card without flags.

## Output

Every run writes `benchmarkcard/benchmark_card_<slug>.json` and a
`tool_output/` tree with one folder per stage, including the provenance map,
the referent frame, the retrieval results and the factuality scores. See the
README for the directory layout.
