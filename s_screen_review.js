export const meta = {
  name: 's-screen-review',
  description: 'Web-grounded adversarial defect screen over the S evaluation sample: seeded card list from the sample file, per-card verdicts persisted by each reviewer, reviewing model pinned and recorded, findings cite the URLs used. Rubric unchanged from broad_sample_review.js.',
  phases: [{ title: 'Screen', detail: 'one web-grounded adversarial reviewer per sampled card' }],
}

// args = config object from scripts/gen_screen_config.py (pass the JSON verbatim):
// { cards: [{slug, staged_dir}], outdir, model_alias, model_id, batch }
const CFG = typeof args === 'string' ? JSON.parse(args) : args
if (!CFG || !Array.isArray(CFG.cards) || !CFG.outdir || !CFG.model_alias || !CFG.model_id) {
  throw new Error('s_screen_review: args must be the gen_screen_config.py object')
}

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    card: { type: 'string' },
    real_benchmark: { type: 'string' },
    identity_correct: { type: 'boolean' },
    paper_assessment: { type: 'string', enum: ['correct','wrong','plausible-unverified','correctly-none','missing-should-exist'] },
    hf_repo_assessment: { type: 'string', enum: ['correct-kept','correctly-rejected','wrong-kept-CONTAMINATION','wrongly-rejected-lost-data','no-repo-ok','no-repo-but-one-exists'] },
    content_accurate: { type: 'boolean' },
    fabricated_fact: { type: 'boolean' },
    verdict: { type: 'string', enum: ['clean','minor','needs-fix'] },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: {
        severity: { type: 'string', enum: ['needs-fix','minor','note'] },
        category: { type: 'string', enum: ['wrong-identity','wrong-paper','fabricated-fact','wrong-section-splice','thin','other'] },
        field: { type: 'string' }, issue: { type: 'string' }, ground_truth: { type: 'string' } },
      required: ['severity','category','field','issue','ground_truth'] } },
    citations: { type: 'array', items: { type: 'string' } },
    reviewing_model: { type: 'string' },
    summary: { type: 'string' },
  },
  required: ['card','real_benchmark','identity_correct','paper_assessment','hf_repo_assessment','content_accurate','fabricated_fact','verdict','findings','citations','reviewing_model','summary'],
}

const reviewOne = ({ slug, staged_dir }) => () =>
  agent(`You are an independent, adversarial reviewer of ONE generated AI-benchmark card. Establish real-world ground truth from the WEB, then check the card against it. Be skeptical and specific. WEB-VERIFY before calling anything "fabricated" -- some scores/models that look anachronistic are real and current (the live web is the arbiter, not your priors).

CARD: "${slug}"

STEP 1 - locate + read (read-only): the staged card dir is ${staged_dir}. Read the card JSON at ${staged_dir}/benchmarkcard/benchmark_card_${slug}.json (name/overview/data_type/size/languages/license, metrics, purpose, paper_url, methodology.baseline_results, authors). Read ${staged_dir}/tool_output/paper_resolver/paper-verification.json and ${staged_dir}/tool_output/hf_verifier/hf-verification.json if present (the binding verdicts), and ${staged_dir}/tool_output/eee/*.json for the EEE identity.

STEP 2 - establish GROUND TRUTH from the web (WebSearch/WebFetch; load via ToolSearch if needed): what the benchmark IS, who introduced it, the introducing paper (arxiv id), the canonical HF repo if any, and a few hard facts (size, metrics, languages, license). Do NOT trust the card for ground truth.

STEP 3 - adversarially check + categorize: identity_correct (right benchmark, not a same-named other); paper_assessment ('wrong'=wrong paper [category wrong-paper]; 'correctly-none'; 'missing-should-exist'=recall/thin gap); hf_repo_assessment (a kept different entity = wrong-kept-CONTAMINATION; a rejected repo with no spliced content = correct honest-thin); content_accurate + fabricated_fact (set true for any invented/untraceable concrete fact -- invented leaderboard, score for a non-existent model, size contradicting the breakdown, wrong authors -> category fabricated-fact; a right-benchmark one-wrong-detail = wrong-section-splice; a sparse card = thin).

verdict: clean / minor (presentation/thin/recall, no wrong facts/identity/paper) / needs-fix (wrong fact, wrong identity, wrong paper, fabrication, contamination). Cite the web facts used.

Recording requirements (mandatory):
- set citations = the URLs you actually consulted and that your verdict rests on (never empty unless you used no web sources at all, which itself is a defect in your review);
- set reviewing_model = "${CFG.model_id}" exactly;
- after composing the verdict object, WRITE it as pretty-printed JSON to ${CFG.outdir}/verdicts/${slug}.json using the Write tool, then return the same object.`,
    { label: `screen:${slug}`, phase: 'Screen', schema: SCHEMA, model: CFG.model_alias }
  ).then(r => r || { card: slug, verdict: 'ERROR', identity_correct: null, fabricated_fact: null, findings: [], citations: [], reviewing_model: CFG.model_id, summary: 'agent returned null' })

const BATCH = CFG.batch || 25
const results = []
for (let i = 0; i < CFG.cards.length; i += BATCH) {
  const chunk = CFG.cards.slice(i, i + BATCH)
  log(`screen batch ${1 + i / BATCH}: ${chunk.length} cards`)
  results.push(...await parallel(chunk.map(reviewOne)))
}
return results
