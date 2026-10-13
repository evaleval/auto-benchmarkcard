"""Per-field unflag policy.

The production block in workers.run_factreasoner replaced the blanket rules
(analytical always unflagged, identity unflagged on all_neutral) with
unflag_decision: identity fields lose an all-neutral flag only when their cited
evidence survived the EAV audit.
"""

import inspect
import pathlib


from auto_benchmarkcard import validation_policy as VP


def test_analytical_always_unflags():
    for field in VP.ANALYTICAL_FIELDS:
        for neutral in (True, False):
            unflag, rule = VP.unflag_decision(
                field, all_neutral=neutral, cited_evidence=False, evidence_demoted=True)
            assert unflag is True
            assert rule == VP.UNFLAG_ANALYTICAL


def test_identity_neutral_intact_evidence_unflags():
    unflag, rule = VP.unflag_decision(
        "benchmark_details.overview",
        all_neutral=True, cited_evidence=True, evidence_demoted=False)
    assert unflag is True
    assert rule == VP.UNFLAG_IDENTITY_EVIDENCE_INTACT


def test_identity_neutral_demoted_evidence_keeps_flag():
    unflag, rule = VP.unflag_decision(
        "benchmark_details.overview",
        all_neutral=True, cited_evidence=True, evidence_demoted=True)
    assert unflag is False
    assert rule == VP.KEEP_IDENTITY_EVIDENCE_DEMOTED


def test_identity_neutral_without_cited_evidence_keeps_flag():
    unflag, rule = VP.unflag_decision(
        "purpose_and_intended_users.goal",
        all_neutral=True, cited_evidence=False, evidence_demoted=False)
    assert unflag is False
    assert rule == VP.KEEP_IDENTITY_NO_CITED_EVIDENCE


def test_identity_contradiction_keeps_flag():
    # a low-factuality (contradiction) flag is not neutrality: it stands even
    # with intact evidence
    unflag, rule = VP.unflag_decision(
        "benchmark_details.domains",
        all_neutral=False, cited_evidence=True, evidence_demoted=False)
    assert unflag is False
    assert rule == VP.KEEP_IDENTITY_CONTRADICTION


def test_other_fields_never_unflagged_here():
    unflag, rule = VP.unflag_decision(
        "data.size", all_neutral=True, cited_evidence=True, evidence_demoted=False)
    assert unflag is False
    assert rule == VP.KEEP_NOT_UNFLAGGABLE


def test_field_evidence_status_cited_and_intact():
    entry = {"source": "paper", "evidence": "q", "evidence_ids": ["E01", "E02"]}
    cited, demoted = VP.field_evidence_status("data.size", entry, [])
    assert (cited, demoted) == (True, False)


def test_field_evidence_status_field_tagged_demotion():
    # the real EAV signal: telemetry entry tagged with this field (the demoted
    # item never reaches provenance, Stage B cannot cite it)
    entry = {"source": "paper", "evidence": "q", "evidence_ids": ["E02"]}
    eav = [{"id": "E07", "field": "data.size", "referent": "subset:hard",
            "reason": "field_mismatch"}]
    cited, demoted = VP.field_evidence_status("data.size", entry, eav)
    assert (cited, demoted) == (True, True)
    # a different field's demotion does not taint this one
    cited, demoted = VP.field_evidence_status("data.format", entry, eav)
    assert (cited, demoted) == (True, False)


def test_field_evidence_status_cited_id_demoted():
    # defensive: a cited id in the demoted set counts as demoted
    entry = {"evidence_ids": ["E07"]}
    eav = [{"id": "E07", "field": "methodology.metrics", "reason": "referent_unresolved"}]
    cited, demoted = VP.field_evidence_status("data.size", entry, eav)
    assert (cited, demoted) == (True, True)


def test_field_evidence_status_missing_entry():
    assert VP.field_evidence_status("data.size", None, []) == (False, False)
    assert VP.field_evidence_status("data.size", {"evidence_ids": []}, []) == (False, False)


def test_run_factreasoner_uses_per_field_policy():
    import auto_benchmarkcard.workers as W

    src = inspect.getsource(W.run_factreasoner)
    assert "unflag_decision" in src
    assert "field_evidence_status" in src
    # the blanket rule is gone from production
    assert "Field-type-aware scoring" not in src


def test_frozen_constants_unchanged():
    assert VP.IDENTITY_FIELDS == frozenset({
        "benchmark_details.overview", "benchmark_details.domains",
        "benchmark_details.data_type", "benchmark_details.similar_benchmarks",
        "purpose_and_intended_users.goal", "purpose_and_intended_users.audience",
        "purpose_and_intended_users.audience_evaluators",
        "purpose_and_intended_users.audience_consumers",
        "purpose_and_intended_users.tasks",
    })
    assert VP.ANALYTICAL_FIELDS == frozenset({
        "purpose_and_intended_users.limitations",
        "purpose_and_intended_users.out_of_scope_uses",
    })
    assert VP.UNFLAGGABLE_FIELDS == VP.IDENTITY_FIELDS | VP.ANALYTICAL_FIELDS
