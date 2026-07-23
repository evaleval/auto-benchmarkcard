"""Single source of truth for which card fields the validation stage may flag.

Identity fields describe what the benchmark IS (name, domains, audience).
They're pulled from multiple sources, so NLI-based contradiction checks produce
false positives -- their all-neutral flags are suppressed.

The v2 audience split (audience_evaluators / audience_consumers) is added here as
two identity fields. The legacy single `audience` key is KEPT so the frozen v31
gold cards (which still carry it, several with an all-neutral flag) stay
byte-identical under reapply_flags --self-test; on v2 cards `audience` is simply
absent, so keeping it is a harmless no-op there.

Analytical fields are reasoned by the LLM from context (limitations,
out-of-scope uses). They can't be fact-checked against source docs because they
don't appear verbatim anywhere -- skip NLI flagging entirely.

methodology.interpretation was moved OUT of the analytical set (gold-set audit
2026-06-09: both all-neutral interpretation cases were judge-unsupported
fabrications, zero supported false positives -- suppressing it only hid real
hallucinations).

Shared by workers.py (production flagging), scripts/revalidate_gold_set.py and
scripts/eval_gold_set.py so the flagging policy and the recall_flaggable
denominator can never drift apart.
"""

IDENTITY_FIELDS = frozenset({
    "benchmark_details.overview", "benchmark_details.domains",
    "benchmark_details.data_type", "benchmark_details.similar_benchmarks",
    "purpose_and_intended_users.goal", "purpose_and_intended_users.audience",
    "purpose_and_intended_users.audience_evaluators",
    "purpose_and_intended_users.audience_consumers",
    "purpose_and_intended_users.tasks",
})

ANALYTICAL_FIELDS = frozenset({
    "purpose_and_intended_users.limitations",
    "purpose_and_intended_users.out_of_scope_uses",
})

UNFLAGGABLE_FIELDS = IDENTITY_FIELDS | ANALYTICAL_FIELDS
