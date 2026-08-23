"""Tests for honest paper-context labels and the un-primed gap-fill pass (referent fix, phase 1).

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
    """Returns docs_for.get(query, default); raises for queries in raise_for."""

    def __init__(self, default, docs_for=None, raise_for=()):
        self.default = default
        self.docs_for = docs_for or {}
        self.raise_for = set(raise_for)

    def invoke(self, query):
        if query in self.raise_for:
            raise RuntimeError("retrieval down")
        return [SimpleNamespace(page_content=t)
                for t in self.docs_for.get(query, self.default)]


def _capture_extraction(monkeypatch):
    calls = []

    def fake(prompt, verify_source, doc, label, shown_text=""):
        calls.append({"prompt": prompt, "verify_source": verify_source,
                      "doc": doc, "label": label})
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
    calls = _capture_extraction(monkeypatch)
    items, telem = C._fill_paper_gaps([], None, "TestBench", "")
    assert items == [] and calls == []
    assert telem == evidence.empty_call_telemetry(evidence.DOC_PAPER)


def test_gap_fill_all_fields_covered_is_a_noop(monkeypatch):
    calls = _capture_extraction(monkeypatch)
    covered = [{"field": p} for p in evidence.gap_fields()]
    items, _ = C._fill_paper_gaps(covered, None, "TestBench", "paper text")
    assert items == [] and calls == []


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
    rec = {}

    def fake_gaps(paper_items, paper_retriever, benchmark_name, verify_source,
                  identity_anchor=""):
        rec["retriever"] = paper_retriever
        rec["verify_source"] = verify_source
        return [], evidence.empty_call_telemetry(evidence.DOC_PAPER)

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
