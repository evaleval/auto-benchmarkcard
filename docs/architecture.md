# Pipeline architecture

The pipeline begins with a benchmark name and an evaluation record. It
identifies one card target, collects candidate evidence, composes the card,
and validates the resulting claims.

## Identification

The Every Eval Ever adapter groups records by benchmark identity. Identity
checks prevent one benchmark, suite, or derivative dataset from being
silently attached to another.

## Extraction

The source finder resolves papers, Hugging Face repositories, and linked web
pages. The source verifier checks subject agreement before a source can
contribute to the card. Extracted content retains source provenance.

## Composition

The information extractor proposes field values. The checker enforces the
selected card schema and source constraints. Risk identification adds relevant
AI risk categories. The generator produces the final structured card.

The composer can also be used as a library. `field_spec.CardSchema` supplies
the schema to evidence extraction, Stage B formatting, and the validator's
repair loop; the benchmark schema is the default. `docstructure` indexes source
text, and `evidence` verifies extracted quotations and builds the evidence
digest. Modelcards supplies its own schema to this same machinery.

## Validation

Card fields are split into claims. Retrieval selects supporting evidence for
each claim, and FactReasoner evaluates the claim-evidence pair. Unsupported or
insufficiently grounded values can be flagged for review.

The implementation entry points are `workflow.py`, `workers.py`, and the
modules under `src/auto_benchmarkcard/tools/`. The current development runtime
on `main` incorporates the reusable composer from `composer-library` commit
`94c7713d1df00730d8c43a35e618de35cf0589bd`. The paper's measured pipeline and
evaluation configuration remain frozen at `v0.2.0`.

## Figure source

- Rendered figure: [`figures/pipeline.png`](figures/pipeline.png)
- Editable draw.io source: [`figures/pipeline.drawio`](figures/pipeline.drawio)
