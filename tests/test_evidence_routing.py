"""Tests for the Stage-A evidence router (evidence.route_evidence) and its classifiers.

The router is structural and model-independent, so it is exercised directly with synthetic
6-key evidence records. Cases cover the three classifiers (recall + precision, grounded in the
real biolp/bigcode artifact quotes), the construction->annotation and scope->out_of_scope
re-filing, the metric-only data_type drop, dense-id / digest_ids invariants, and a regression
guard that unmarked evidence is passed through unchanged.
"""

import pytest

from auto_benchmarkcard.tools.composer import aboutness as A
from auto_benchmarkcard.tools.composer import evidence as E

# Real evidence strings from the d6 artifacts (grounding the fixtures).
Q_BIOLP_CONSTRUCTION = (
    "we introduced in these protocols numerous mistakes that would still allow them to function "
    "correctly. After that we introduced in each protocol a single mistake that would cause it to "
    "fail. We then presented these modified protocols to an LLM, prompting it to identify the mistake"
)
Q_BIOLP_METRIC = (
    "We then presented these modified protocols to an LLM, prompting it to identify the mistake that "
    "would cause it to fail, and measured the accuracy of a model in identifying such mistakes across "
    "many test cases."
)
Q_SCOPE_BIGCODE = "It serves as a fundamental benchmark for LLMs rather than LLM agents"
Q_SCOPE_SWE = "SWE-bench is not suitable for evaluating agentic workflows"
Q_LEGIT_TASK = "Identify the correct answer to each multiple-choice physics question"
Q_TASK_RATHER_THAN = "classify the sentiment rather than the topic of each review"
Q_LEGIT_DATATYPE = "The dataset consists of text passages and code snippets"
Q_DATATYPE_TEXT_METRIC = "Text classification accuracy is the primary metric"


def _rec(field, quote, cs=0, doc="docling", kind="stated", note=""):
    return {"field": field, "quote": quote, "char_start": cs, "doc": doc, "kind": kind, "note": note}


# --------------------------------------------------------------- classifiers ----

@pytest.mark.parametrize("quote,expected", [
    (Q_BIOLP_CONSTRUCTION, True),                                    # introduced ... mistake
    ("annotators used a rubric to label the data", True),           # _CONSTRUCTION_VETO_RE reuse
    ("we constructed the benchmark from public protocols", True),
    (Q_LEGIT_TASK, False),
    (Q_SCOPE_BIGCODE, False),
    ("", False),
])
def test_is_construction_quote(quote, expected):
    assert A.is_construction_quote(quote) is expected


@pytest.mark.parametrize("quote,expected", [
    (Q_SCOPE_BIGCODE, True),
    (Q_SCOPE_SWE, True),
    ("not designed for code-only evaluation", True),
    (Q_LEGIT_TASK, False),
    (Q_TASK_RATHER_THAN, False),     # bare "rather than" must NOT fire (precision)
    ("", False),
])
def test_is_scope_framing(quote, expected):
    assert A.is_scope_framing(quote) is expected


@pytest.mark.parametrize("quote,expected", [
    (Q_BIOLP_METRIC, True),                          # metric, no modality term -> drop
    ("Pass@1 with greedy decoding in the zero-shot setting", True),
    (Q_LEGIT_DATATYPE, False),                       # has "text"/"code"
    (Q_DATATYPE_TEXT_METRIC, False),                 # has "text" even though it says accuracy
    ("image and audio recordings", False),           # pure modality
    ("", False),
])
def test_is_metric_only_data_type(quote, expected):
    assert A.is_metric_only_data_type(quote) is expected


# ----------------------------------------------------------- route_evidence ----

def test_construction_tasks_quote_routes_to_annotation():
    routed = E.route_evidence([_rec(E._TASKS_FIELD, Q_BIOLP_CONSTRUCTION, cs=20)])
    fields = [r["field"] for r in routed]
    assert E._TASKS_FIELD in fields                  # original kept
    assert E._ANNOTATION_FIELD in fields             # derived annotation added
    annotation = [r for r in routed if r["field"] == E._ANNOTATION_FIELD][0]
    assert annotation["quote"] == Q_BIOLP_CONSTRUCTION
    assert annotation["char_start"] == 20 and annotation["doc"] == "docling"


def test_scope_tasks_quote_routes_to_out_of_scope():
    routed = E.route_evidence([_rec(E._TASKS_FIELD, Q_SCOPE_BIGCODE)])
    fields = [r["field"] for r in routed]
    assert E._TASKS_FIELD in fields
    assert E._OUT_OF_SCOPE_FIELD in fields


def test_metric_data_type_dropped_modality_kept():
    routed = E.route_evidence([
        _rec(E._DATA_TYPE_FIELD, Q_BIOLP_METRIC, cs=10),
        _rec(E._DATA_TYPE_FIELD, Q_LEGIT_DATATYPE, cs=0, doc="hf_readme"),
    ])
    dt = [r["quote"] for r in routed if r["field"] == E._DATA_TYPE_FIELD]
    assert Q_BIOLP_METRIC not in dt                  # dropped
    assert Q_LEGIT_DATATYPE in dt                    # legit modality survives


def test_plain_task_not_routed():
    routed = E.route_evidence([_rec(E._TASKS_FIELD, Q_LEGIT_TASK)])
    assert [r["field"] for r in routed] == [E._TASKS_FIELD]   # no derivation


def test_unmarked_records_passed_through_unchanged():
    raw = [
        _rec("benchmark_details.name", "BioLP-bench"),
        _rec("data.size", "1000 examples"),
        _rec(E._TASKS_FIELD, Q_LEGIT_TASK),
    ]
    assert E.route_evidence(list(raw)) == raw         # no drop, no add, order preserved


def test_dedup_skips_existing_annotation_span():
    # an annotation record already covering the same (span, doc) blocks the derived duplicate
    routed = E.route_evidence([
        _rec(E._TASKS_FIELD, Q_BIOLP_CONSTRUCTION, cs=20),
        _rec(E._ANNOTATION_FIELD, Q_BIOLP_CONSTRUCTION, cs=20),
    ])
    annotation = [r for r in routed if r["field"] == E._ANNOTATION_FIELD]
    assert len(annotation) == 1                       # not duplicated


# ----------------------------------------------- finalize_evidence integration ----

def test_finalize_evidence_dense_ids_and_digest_invariant():
    paper = [
        _rec("benchmark_details.name", "BioLP-bench", 5),
        _rec(E._DATA_TYPE_FIELD, Q_BIOLP_METRIC, 10),
        _rec(E._TASKS_FIELD, Q_BIOLP_CONSTRUCTION, 20),
        _rec(E._TASKS_FIELD, Q_SCOPE_BIGCODE, 40),
        _rec(E._TASKS_FIELD, Q_LEGIT_TASK, 60),
    ]
    hf = [_rec(E._DATA_TYPE_FIELD, Q_LEGIT_DATATYPE, 0, doc="hf_readme")]
    items, digest = E.finalize_evidence(paper, [], [], hf)

    ids = [it["evidence_id"] for it in items]
    assert ids == [f"E{i:02d}" for i in range(1, len(items) + 1)]   # dense, no gaps
    digest_ids = {it["evidence_id"] for v in digest.values() for it in v}
    assert digest_ids == set(ids)                                   # composer_tool.py:~1640 invariant

    assert len(digest[E._TASKS_FIELD]) == 3                         # all originals retained
    assert E._ANNOTATION_FIELD in digest and E._OUT_OF_SCOPE_FIELD in digest
    dt = [it["quote"] for it in digest[E._DATA_TYPE_FIELD]]
    assert dt == [Q_LEGIT_DATATYPE]                                 # metric dropped, modality kept
    assert all(Q_BIOLP_METRIC != it["quote"] for it in items)      # dropped from items too


def test_derived_target_fields_are_whitelisted():
    for path in (E._ANNOTATION_FIELD, E._OUT_OF_SCOPE_FIELD, E._TASKS_FIELD):
        assert path in E.CURATOR_FIELDS                             # passes EVIDENCE_FIELD_WHITELIST
