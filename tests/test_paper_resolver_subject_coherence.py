"""FINISH-5 subject-coherence: a verified paper must not introduce a plainly
differently-named benchmark than the identity (the beyond-aime defect, where the LLM accepted
'MATH-Beyond' for 'Beyond AIME').

Synthetic and real-title fixtures; network/LLM are mocked. The guard is a general mechanism keyed
on the paper's declared name vs the identity, never on a benchmark list.
"""

import json

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr


@pytest.fixture(autouse=True)
def _no_sleep_clean_state(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._author_cache.clear()
    pr._metadata_cache.clear()
    pr._SOURCE_STATUS.clear()


def _cand(title, arxiv_id, sim, authors=("A",)):
    return {"title": title, "abstract": "", "arxiv_id": arxiv_id, "doi": "", "url": "",
            "year": 2024, "citationCount": 0, "authors": list(authors),
            "_title_similarity": sim, "_source": "openalex", "_query_used": "q"}


def _mock_resolver(monkeypatch, candidates, llm_result, *, full_name="WidgetBench"):
    monkeypatch.setattr(pr, "_query_benchmark_metadata",
                        lambda *a, **k: {"full_name": full_name, "domain": "evaluation"})
    monkeypatch.setattr(pr, "_lookup_display_name", lambda *a, **k: full_name)
    monkeypatch.setattr(pr, "_SOURCES", [("openalex", lambda q: [{"_raw": True}], lambda p: dict(p))])
    monkeypatch.setattr(pr, "_prefilter_candidates", lambda *a, **k: [dict(c) for c in candidates])
    monkeypatch.setattr(pr, "_batch_verify_with_llm", lambda *a, **k: llm_result)


# --- helper units: _declared_benchmark_name / _introduces_other_benchmark --------------------

def test_declared_name_parses_leading_coined_name():
    assert pr._declared_benchmark_name("MATH-Beyond: A Benchmark for RL") == "MATH-Beyond"
    assert pr._declared_benchmark_name("CharXiv: Charting Gaps in Chart Understanding") == "CharXiv"
    assert pr._declared_benchmark_name("Length-Controlled AlpacaEval: A Simple Way") \
        == "Length-Controlled AlpacaEval"


def test_declared_name_none_for_method_titles_and_generic_heads():
    assert pr._declared_benchmark_name("Training Verifiers to Solve Math Word Problems") is None
    assert pr._declared_benchmark_name("Attention Is All You Need") is None
    # leading colon head that is a generic Title-cased phrase, not a coined name
    assert pr._declared_benchmark_name("A Large-Scale Study of Models: Findings") is None


def test_introduces_other_rejects_differently_named_benchmark():
    # shares the token 'beyond' but the distinctive bases are disjoint as strings -> reject
    assert pr._introduces_other_benchmark(
        "MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model",
        "beyond-aime", "Beyond AIME") == "MATH-Beyond"


@pytest.mark.parametrize("title,identities", [
    ("CharXiv: Charting Gaps in Realistic Chart Understanding", ("charxiv-d", "CharXiv")),
    ("ActivityNet: A large-scale video benchmark", ("activitynet",)),
    ("SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ("swe-bench",)),
    ("SuperGLUE: A Stickier Benchmark", ("superglue",)),
    ("HellaSwag: Can a Machine Really Finish Your Sentence?", ("hellaswag",)),
    ("Length-Controlled AlpacaEval: A Simple Way to Debias", ("alpaca_eval",)),
    ("BOLD: Dataset and Metrics for Measuring Biases", ("bold",)),  # <5 char base -> undecidable keep
    ("Training Verifiers to Solve Math Word Problems", ("gsm8k",)),  # no declared name
])
def test_introduces_other_keeps_coherent_or_undecidable(title, identities):
    assert pr._introduces_other_benchmark(title, *identities) is None


# --- resolve_paper integration: search path ---------------------------------------------------

def test_resolve_rejects_wrong_paper_subject_coherence(monkeypatch, tmp_path):
    # beyond-aime regression: LLM confidently matches a paper that introduces a DIFFERENT benchmark.
    cand = _cand("MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model", "2510.11653", 70.0)
    _mock_resolver(monkeypatch, [cand],
                   {"match_index": 0, "confidence": 0.95, "reasoning": "close", "error": False},
                   full_name="Beyond AIME")
    result = pr.resolve_paper("beyond-aime", full_name="Beyond AIME", output_dir=tmp_path)
    assert result is None
    log = json.loads((tmp_path / "paper-verification.json").read_text())
    assert log["resolved_url"] is None
    assert log["subject_coherence"]["declared"] == "MATH-Beyond"
    assert log["subject_coherence"]["action"] == "rejected"


def test_resolve_keeps_coherent_paper(monkeypatch):
    # A same-named paper is unaffected by the guard.
    cand = _cand("BOLD: Dataset and Metrics for Measuring Biases", "2101.11718", 99.0)
    _mock_resolver(monkeypatch, [cand],
                   {"match_index": 0, "confidence": 1.0, "reasoning": "ok", "error": False},
                   full_name="BOLD")
    result = pr.resolve_paper("bold", full_name="BOLD")
    assert result is not None and "2101.11718" in result["url"]


# --- B documentation: derivative subset/extension are indistinguishable structurally ----------

def test_repo_corroboration_indistinguishable_subset_vs_extension():
    # Both a canonical extension and a wrong subset return the same reason: no structural signal
    # separates keep-apex-v1 from strip-swe-bench (FINISH-5 B deferred to the composer guard).
    assert pr._repo_corroboration("org/Widget-extended", "widget") == "derivative_repo"
    assert pr._repo_corroboration("org/Widget_Verified", "widget") == "derivative_repo"
