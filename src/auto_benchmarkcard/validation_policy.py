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

# v4 per-field unflag policy (production only). The constants above keep their
# original meaning for the frozen instruments (scripts/revalidate_gold_set.py,
# scripts/reapply_flags.py), which still apply the blanket rule: analytical
# always unflagged, identity unflagged on all_neutral. The production block in
# workers.run_factreasoner uses unflag_decision instead: an identity field's
# all-neutral flag is lifted only when its provenance cites evidence that
# survived the EAV audit. Blanket identity unflagging hid real wrong-referent
# errors (the NLI hypothesis never named the referent), so neutrality alone is
# no longer proof of a harmless descriptive mismatch.
IDENTITY_UNFLAG_REQUIRES_EVIDENCE = True

UNFLAG_ANALYTICAL = "analytical_never_flagged"
UNFLAG_IDENTITY_EVIDENCE_INTACT = "identity_neutral_evidence_intact"
KEEP_IDENTITY_EVIDENCE_DEMOTED = "identity_evidence_demoted"
KEEP_IDENTITY_NO_CITED_EVIDENCE = "identity_no_cited_evidence"
KEEP_IDENTITY_CONTRADICTION = "identity_contradiction"
KEEP_NOT_UNFLAGGABLE = "not_unflaggable"


def field_evidence_status(field_name, provenance_entry, eav_demoted):
    """Return (cited_evidence, evidence_demoted) for one card field.

    cited_evidence: the field's provenance entry cites at least one evidence id,
    i.e. the value rests on a quote-verified evidence item rather than on a
    deterministic echo or a bare assertion.

    evidence_demoted: the EAV audit removed evidence for this field. Primary
    signal is a composition_metadata.eav.demoted entry tagged with this field;
    a cited id appearing in the demoted set is caught too (defensive: Stage B
    should never cite a demoted id)."""
    demoted = [d for d in (eav_demoted or []) if isinstance(d, dict)]
    demoted_ids = {str(d.get("id")) for d in demoted if d.get("id")}
    cited_ids = []
    if isinstance(provenance_entry, dict):
        raw = provenance_entry.get("evidence_ids")
        if isinstance(raw, list):
            cited_ids = [str(e) for e in raw if e]
    cited = bool(cited_ids)
    evidence_demoted = any(d.get("field") == field_name for d in demoted) or bool(
        demoted_ids.intersection(cited_ids)
    )
    return cited, evidence_demoted


def unflag_decision(field_name, *, all_neutral, cited_evidence, evidence_demoted):
    """Per-field unflag ruling for the production flag block: (unflag, rule).

    Analytical fields are reasoned content and stay never-flagged. Identity
    fields lose an all-neutral flag only when their cited evidence survived the
    EAV audit; a demoted citation, a missing one, or a contradiction-based flag
    (not all_neutral) keeps the flag. Every other field keeps its flag."""
    if field_name in ANALYTICAL_FIELDS:
        return True, UNFLAG_ANALYTICAL
    if field_name not in IDENTITY_FIELDS:
        return False, KEEP_NOT_UNFLAGGABLE
    if not all_neutral:
        return False, KEEP_IDENTITY_CONTRADICTION
    if evidence_demoted:
        return False, KEEP_IDENTITY_EVIDENCE_DEMOTED
    if not cited_evidence:
        return False, KEEP_IDENTITY_NO_CITED_EVIDENCE
    return True, UNFLAG_IDENTITY_EVIDENCE_INTACT
