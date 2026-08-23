"""Tests for honest paper-context labels and the un-primed gap-fill pass (referent fix,
phase 1), plus the ranked-widening / second-sweep recall mechanics.

Network-free: retrieval is faked with plain objects and the extraction call is captured
via a monkeypatched _run_extraction (or a fake LLM handler for the full-compose seam).
"""

import json
from types import SimpleNamespace

from auto_benchmarkcard.tools.composer import composer_tool as C
from auto_benchmarkcard.tools.composer import evidence
from auto_benchmarkcard.tools.composer.composer_tool import (
    BenchmarkDetails, DataInfo, EthicalAndLegalConsiderations,
    Methodology, PurposeAndIntendedUsers)

SECTION_MODELS = {
    "benchmark_details": BenchmarkDetails,
    "purpose_and_intended_users": PurposeAndIntendedUsers,
    "data": DataInfo,
    "methodology": Methodology,
    "ethical_and_legal_considerations": EthicalAndLegalConsiderations,
}


# ------------------------------------------------------- excerpt labels ----

def test_excerpt_label_plain_no_fabricated_section():
    label = C._excerpt_label(3, "The dataset contains 500 tasks drawn from forums.")
    assert label == "[Excerpt 3 from paper]"
    assert "Section" not in label


def test_excerpt_label_names_first_markdown_heading():
    text = "tail of prior text.\n## Related Work\nPrior benchmarks include X.\n### Details\nmore"
    assert C._excerpt_label(1, text) == "[Excerpt 1 from paper, near heading: Related Work]"


def test_excerpt_label_does_not_invent_headings():
    # inline "Section 4" prose and a no-space hash line are not markdown headings
    text = "As shown in Section 4, accuracy drops.\n#hashtag stays data"
    assert C._excerpt_label(2, text) == "[Excerpt 2 from paper]"


def test_format_paper_chunks_uses_excerpt_labels_within_budget():
    intro = "INTRO " * 10
    chunks = ["first chunk body", "## Evaluation\nsecond chunk body", "third chunk body"]
    content, used = C._format_paper_chunks(intro, chunks, budget=10_000)
    assert content.startswith(f"[Abstract and Introduction]\n{intro}")
    assert "[Excerpt 1 from paper]\nfirst chunk body" in content
    assert "[Excerpt 2 from paper, near heading: Evaluation]" in content
    assert "[Excerpt 3 from paper]\nthird chunk body" in content
    assert "[Section" not in content
    assert used == len(intro) + sum(len(c) for c in chunks)


def test_format_paper_chunks_budget_clips_and_stops():
    intro = "x" * 100
    chunks = ["a" * 300, "b" * 300]
    content, used = C._format_paper_chunks(intro, chunks, budget=250)
    # first chunk clipped to the 150 remaining chars (>100 so it is kept), second dropped
    assert "[Excerpt 1 from paper]\n" + "a" * 150 in content
    assert "b" not in content.split("\n\n")[-1]
    assert used == 100


# ------------------------------------------------------------- gap-fill ----

class _Retriever:
    """Returns docs_for.get(query, default); raises for queries in raise_for.
    Records every query in .queries so tests can assert what each sweep retrieved."""

    def __init__(self, default, docs_for=None, raise_for=()):
        self.default = default
        self.docs_for = docs_for or {}
        self.raise_for = set(raise_for)
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        if query in self.raise_for:
            raise RuntimeError("retrieval down")
        return [SimpleNamespace(page_content=t)
                for t in self.docs_for.get(query, self.default)]


def _capture_extraction(monkeypatch):
    calls = []

    def fake(prompt, verify_source, doc, label, shown_text="", doc_index=None, frame=None):
        calls.append({"prompt": prompt, "verify_source": verify_source,
                      "doc": doc, "label": label, "doc_index": doc_index, "frame": frame})
        return [], evidence.empty_call_telemetry(doc)

    monkeypatch.setattr(C, "_run_extraction", fake)
    return calls


def test_gap_fill_prompt_has_no_field_binding_and_lists_all_targets(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    targets = evidence.gap_fields()
    broken_query = next(iter(targets.values()))
    retriever = _Retriever(default=["The benchmark was annotated by three experts."],
                           raise_for=[broken_query])
    C._fill_paper_gaps([], retriever, "TestBench", "full paper text here")
    assert len(calls) == 1
    prompt = calls[0]["prompt"]
    assert "[For " not in prompt
    assert "[Excerpt 1 from paper]" in prompt
    # every gap field is offered once, including the one whose retrieval failed
    for path in targets:
        assert path in prompt
    # the un-binding instruction replaces per-chunk priming
    assert "tag each quote with the field it actually supports" in prompt


def test_gap_fill_dedupes_chunks_and_surfaces_headings(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    # every query retrieves the same chunk: it must appear once, labeled by its heading
    retriever = _Retriever(default=["## Data Collection\nWe scraped 10k posts."])
    C._fill_paper_gaps([], retriever, "TestBench", "full paper text here")
    prompt = calls[0]["prompt"]
    assert "[Excerpt 1 from paper, near heading: Data Collection]" in prompt
    assert "[Excerpt 2" not in prompt


def test_gap_fill_without_retriever_uses_paper_text(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    paper = "ShortBench evaluates parsing. It has 200 items."
    C._fill_paper_gaps([], None, "ShortBench", paper)
    assert len(calls) == 1
    prompt = calls[0]["prompt"]
    assert paper in prompt
    assert "research paper" in prompt
    assert calls[0]["verify_source"] == paper


def test_gap_fill_without_retriever_and_empty_source_is_a_noop(monkeypatch):
    # 3-tuple unpack: _fill_paper_gaps now also returns gap-recall telemetry
    calls = _capture_extraction(monkeypatch)
    items, telem, gap = C._fill_paper_gaps([], None, "TestBench", "")
    assert items == [] and calls == []
    assert telem == evidence.empty_call_telemetry(evidence.DOC_PAPER)
    assert gap == {"gap_sweeps": 0, "gap_fields_open_before": [],
                   "gap_fields_open_after": []}


def test_gap_fill_all_fields_covered_is_a_noop(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    covered = [{"field": p} for p in evidence.gap_fields()]
    items, _, gap = C._fill_paper_gaps(covered, None, "TestBench", "paper text")
    assert items == [] and calls == []
    assert gap["gap_sweeps"] == 0


# ------------------------------------- ranked widening + second sweep ----

def _scripted_extraction(monkeypatch, per_call_fields):
    """Fake _run_extraction that returns verified records for the given field lists,
    one list per call in order; extra calls return nothing."""
    calls = []

    def fake(prompt, verify_source, doc, label, shown_text="", doc_index=None, frame=None):
        fields = per_call_fields[len(calls)] if len(calls) < len(per_call_fields) else []
        calls.append({"prompt": prompt, "shown_text": shown_text})
        telem = evidence.empty_call_telemetry(doc)
        telem["emitted"] = telem["verified"] = len(fields)
        return [{"field": f} for f in fields], telem

    monkeypatch.setattr(C, "_run_extraction", fake)
    return calls


_TWO_GAPS = {"data.annotation": "annotation query",
             "methodology.validation": "validation query"}


def test_gap_fill_widens_retrieval_to_top4(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    monkeypatch.setattr(evidence, "gap_fields",
                        lambda: {"data.annotation": "annotation query"})
    chunks = [f"chunk body number {i} " + "x" * 50 for i in range(6)]
    retriever = _Retriever(default=chunks)
    C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    prompt = calls[0]["prompt"]
    # top-4 per query (was top-2); equal scores keep retrieval order
    for i in range(4):
        assert f"chunk body number {i}" in prompt
    assert "chunk body number 4" not in prompt
    assert "[Excerpt 4 from paper]" in prompt


def test_gap_fill_ranks_hint_relevant_excerpts_first(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    monkeypatch.setattr(evidence, "gap_fields",
                        lambda: {"data.annotation": "annotation query"})
    filler = "irrelevant filler prose about nothing in particular"
    relevant = "Labeling was done by experts with strict quality control over annotated items."
    retriever = _Retriever(default=[filler, relevant])
    C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    prompt = calls[0]["prompt"]
    assert f"[Excerpt 1 from paper]\n{relevant}" in prompt
    assert prompt.index(relevant) < prompt.index(filler)


def test_rank_gap_excerpts_deterministic_and_budgeted():
    hints = {"annotation", "quality", "control"}
    a = "annotation quality control " * 10
    b = "annotation only " * 10
    c = "nothing relevant here " * 10
    ranked = C._rank_gap_excerpts([c, b, a], hints, budget=10_000)
    assert ranked == [a, b, c]
    assert C._rank_gap_excerpts([c, b, a], hints, budget=10_000) == ranked
    # ties keep retrieval order
    assert C._rank_gap_excerpts(["annotation alpha", "annotation beta"], hints,
                                budget=1000) == ["annotation alpha", "annotation beta"]
    # an over-budget excerpt is skipped; a smaller lower-ranked one still fits
    big = "annotation quality control " + "z" * 500
    small = "annotation " + "y" * 20
    assert C._rank_gap_excerpts([big, small], hints, budget=len(small) + 10) == [small]
    assert C._rank_gap_excerpts([small, big], hints, budget=len(big) + 10) == [big]


def test_gap_hint_tokens_use_curator_hints():
    tokens = C._gap_hint_tokens(["data.annotation"])
    # from the curator hint "How labeling was done, who annotated, quality control"
    assert {"labeling", "annotated", "quality", "control"} <= tokens
    assert "how" not in tokens and "was" not in tokens


def test_second_sweep_runs_on_remaining_fields_when_productive(monkeypatch):
    monkeypatch.setattr(evidence, "gap_fields", lambda: dict(_TWO_GAPS))
    calls = _scripted_extraction(monkeypatch, [["data.annotation"]])
    retriever = _Retriever(default=["Some chunk about annotation and validation."])
    items, telem, gap = C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    assert len(calls) == 2
    # the second sweep offers only the still-open field
    assert "- methodology.validation" in calls[1]["prompt"]
    assert "- data.annotation" not in calls[1]["prompt"]
    assert [it["field"] for it in items] == ["data.annotation"]
    assert gap == {"gap_sweeps": 2, "gap_fields_open_before": [2, 1],
                   "gap_fields_open_after": [1, 1]}
    assert telem["emitted"] == 1 and telem["verified"] == 1


def test_no_second_sweep_when_first_sweep_unproductive(monkeypatch):
    monkeypatch.setattr(evidence, "gap_fields", lambda: dict(_TWO_GAPS))
    calls = _scripted_extraction(monkeypatch, [[]])
    retriever = _Retriever(default=["Some chunk."])
    items, _, gap = C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    assert len(calls) == 1
    assert items == []
    assert gap == {"gap_sweeps": 1, "gap_fields_open_before": [2],
                   "gap_fields_open_after": [2]}


def test_no_second_sweep_when_first_sweep_fills_everything(monkeypatch):
    monkeypatch.setattr(evidence, "gap_fields", lambda: dict(_TWO_GAPS))
    calls = _scripted_extraction(
        monkeypatch, [["data.annotation", "methodology.validation"]])
    retriever = _Retriever(default=["chunk"])
    items, _, gap = C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    assert len(calls) == 1
    assert gap == {"gap_sweeps": 1, "gap_fields_open_before": [2],
                   "gap_fields_open_after": [0]}


def test_hard_cap_two_calls_even_when_second_sweep_productive(monkeypatch):
    monkeypatch.setattr(evidence, "gap_fields", lambda: {
        "data.annotation": "q1", "methodology.validation": "q2", "data.source": "q3"})
    calls = _scripted_extraction(monkeypatch, [["data.annotation"], ["data.source"]])
    retriever = _Retriever(default=["chunk text"])
    items, telem, gap = C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    # one field is still open after a productive second sweep: no third call
    assert len(calls) == 2
    assert gap == {"gap_sweeps": 2, "gap_fields_open_before": [3, 2],
                   "gap_fields_open_after": [2, 1]}
    assert sorted(it["field"] for it in items) == ["data.annotation", "data.source"]
    assert telem["emitted"] == 2 and telem["verified"] == 2


def test_second_sweep_retrieves_fresh_chunks_for_remaining_queries_only(monkeypatch):
    monkeypatch.setattr(evidence, "gap_fields", lambda: dict(_TWO_GAPS))
    calls = _scripted_extraction(monkeypatch, [["data.annotation"]])
    retriever = _Retriever(default=["first chunk body"],
                           docs_for={"validation query": ["second chunk about validation"]})
    C._fill_paper_gaps([], retriever, "TestBench", "full paper text")
    assert retriever.queries == ["annotation query", "validation query", "validation query"]
    assert "second chunk about validation" in calls[1]["prompt"]
    assert "first chunk body" not in calls[1]["prompt"]


def test_second_sweep_short_paper_uses_full_text_again(monkeypatch):
    monkeypatch.setattr(evidence, "gap_fields", lambda: dict(_TWO_GAPS))
    calls = _scripted_extraction(monkeypatch, [["data.annotation"]])
    paper = "ShortBench evaluates parsing. Annotated by experts."
    items, _, gap = C._fill_paper_gaps([], None, "ShortBench", paper)
    assert len(calls) == 2
    assert paper in calls[1]["prompt"]
    assert calls[1]["shown_text"] == paper
    assert "- methodology.validation" in calls[1]["prompt"]
    assert "- data.annotation" not in calls[1]["prompt"]
    assert gap["gap_sweeps"] == 2


def test_fold_gap_telemetry_sums_counts():
    a = evidence.empty_call_telemetry(evidence.DOC_PAPER)
    a["emitted"], a["verified"], a["rejected"] = 3, 2, 1
    a["reject_reasons"]["no_match"] = 1
    b = evidence.empty_call_telemetry(evidence.DOC_PAPER)
    b["emitted"], b["verified"], b["rejected"] = 2, 1, 1
    b["reject_reasons"]["empty"] = 1
    b["referent_resolution"] = {"default_target": 1}
    C._fold_gap_telemetry(a, b)
    assert a["emitted"] == 5 and a["verified"] == 3 and a["rejected"] == 2
    assert a["reject_reasons"]["no_match"] == 1 and a["reject_reasons"]["empty"] == 1
    assert a["referent_resolution"] == {"default_target": 1}
    assert a["doc"] == evidence.DOC_PAPER


# ------------------------------------------- compose seam (short paper) ----

def _all_ns_response():
    out = {}
    for sec, cls in SECTION_MODELS.items():
        out[sec] = {f: "Not specified" for f in cls.model_fields if f != "provenance"}
        out[sec]["provenance"] = {}
    return json.dumps(out)


class _FakeHandler:
    model_name = "fake-model"

    def generate(self, prompt, response_format=None):
        return _all_ns_response()

    def generate_with_meta(self, prompt, response_format=None, max_completion_tokens=None):
        return _all_ns_response(), None


def test_short_paper_still_gets_gap_fill_pass(monkeypatch):
    # the paper fits in the budget so no retriever is built; gap-fill must run anyway
    # (fake returns the 3-tuple shape and its gap-recall dict must reach the metadata)
    rec = {}

    def fake_gaps(paper_items, paper_retriever, benchmark_name, verify_source,
                  identity_anchor="", doc_index=None, frame=None):
        rec["retriever"] = paper_retriever
        rec["verify_source"] = verify_source
        gap = {"gap_sweeps": 1, "gap_fields_open_before": [3], "gap_fields_open_after": [3]}
        return [], evidence.empty_call_telemetry(evidence.DOC_PAPER), gap

    monkeypatch.setattr(C, "_fill_paper_gaps", fake_gaps)
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler())

    docling = {"success": True,
               "filtered_text": "ShortBench is a benchmark for testing parsers. " * 20,
               "metadata": {"title": "ShortBench"}}
    fn = getattr(C.compose_benchmark_card, "func", C.compose_benchmark_card)
    res = fn(query="ShortBench", docling_output=docling)
    assert res["benchmark_card"]
    assert rec["retriever"] is None
    assert "ShortBench is a benchmark" in rec["verify_source"]
    assert res["composition_metadata"]["gap_recall"] == {
        "gap_sweeps": 1, "gap_fields_open_before": [3], "gap_fields_open_after": [3]}


def test_gap_fill_forwards_index_and_frame(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    frame = {"target_id": "target", "nodes": [
        {"id": "target", "type": "target", "name": "TestBench", "aliases": []}]}
    idx = object()
    C._fill_paper_gaps([], None, "TestBench", "full paper text here",
                       doc_index=idx, frame=frame)
    assert calls[0]["doc_index"] is idx
    assert calls[0]["frame"] is frame
    assert "target: TestBench (target)" in calls[0]["prompt"]
