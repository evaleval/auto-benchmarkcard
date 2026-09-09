"""Referent-explicit NLI hypotheses (SATORI pattern).

The wrap puts the referent INSIDE what NLI checks: every atom text becomes a
full claim naming the benchmark, and the field's non-target referent when the
composer's provenance records one. Atom decomposition granularity must not
change: the wrap rewrites only the hypothesis text.
"""

import inspect

from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import (
    FIELD_DESCRIPTION_PHRASES,
    field_referents_from_provenance,
    verbalize_hypothesis,
    verbalize_rag_hypotheses,
)
from auto_benchmarkcard.tools.rag.atomizer import text_to_statements


def test_target_only_format():
    got = verbalize_hypothesis(
        "The dataset contains 10,000 examples", "data.size", "ETHOS")
    assert got == (
        'For the benchmark "ETHOS", the size of the benchmark\'s dataset: '
        "The dataset contains 10,000 examples"
    )


def test_subset_referent_format():
    ref = {"id": "subset:hard", "name": "hard", "type": "subset"}
    got = verbalize_hypothesis(
        "The best model achieved 55% accuracy", "methodology.baseline_results",
        "BigBench", referent=ref)
    assert got == (
        'For the "hard" subset of benchmark "BigBench", '
        "the benchmark's baseline results: The best model achieved 55% accuracy"
    )


def test_sibling_referent_names_the_contrast():
    ref = {"id": "sibling:fibble", "name": "Fibble", "type": "sibling"}
    got = verbalize_hypothesis(
        "It covers 12 languages", "benchmark_details.languages", "ETHOS", referent=ref)
    assert got.startswith('For the related benchmark "Fibble" (not "ETHOS" itself), ')


def test_unknown_field_falls_back_to_field_name():
    got = verbalize_hypothesis("Something", "data.weird", "X")
    assert got == 'For the benchmark "X", the field "data.weird": Something'


def test_missing_field_still_wraps_with_target():
    got = verbalize_hypothesis("Something", None, "X")
    assert got == 'For the benchmark "X", the card\'s content: Something'


def test_phrase_map_covers_the_atomizer_whitelist():
    # Every field path the atomizer can emit has a curated phrase, so the
    # fallback only ever fires for genuinely unknown paths.
    raw = (
        "- The dataset contains 10,000 examples [data.size]\n"
        "- The benchmark name is ETHOS [benchmark_details.name]\n"
    )
    for stmt in text_to_statements(raw):
        assert stmt["field"] in FIELD_DESCRIPTION_PHRASES


def test_field_referents_from_provenance():
    provenance = {
        "data": {
            "size": {"source": "paper", "evidence": "q", "evidence_ids": ["E01"],
                     "referent": "subset:hard"},
        },
        "methodology": {
            # target referent: no context needed
            "metrics": {"source": "eee", "evidence": "q", "evidence_ids": [],
                        "referent": "target"},
            # display label map, not a provenance entry: must be skipped
            "metrics_labels": {"acc": "Accuracy"},
        },
        "benchmark_details": {
            # no referent key: target-only wrap
            "overview": {"source": "hf", "evidence": "q", "evidence_ids": []},
        },
    }
    frame = {"target_id": "target",
             "nodes": [{"id": "subset:hard", "type": "subset", "name": "Hard"}]}
    got = field_referents_from_provenance(provenance, frame)
    assert got == {"data.size": {"id": "subset:hard", "name": "Hard", "type": "subset"}}


def test_field_referent_unknown_node_degrades_to_id_parts():
    provenance = {"data": {"size": {"referent": "split:dev"}}}
    got = field_referents_from_provenance(provenance, {"target_id": "target", "nodes": []})
    assert got == {"data.size": {"id": "split:dev", "name": "dev", "type": "split"}}


def test_field_referents_empty_inputs():
    assert field_referents_from_provenance(None, None) == {}
    assert field_referents_from_provenance({}, {}) == {}


def _rag():
    return {
        "input": "in", "output": "out", "topic": "ETHOS",
        "atoms": [
            {"id": "a1", "text": "The dataset contains 10,000 examples",
             "field": "data.size", "contexts": ["c1", "c2"]},
            {"id": "a2", "text": "It is a hate speech benchmark",
             "field": "benchmark_details.overview", "contexts": ["c1"]},
            {"id": "a3", "text": "Statement without field", "contexts": []},
        ],
        "contexts": [{"id": "c1", "text": "t"}, {"id": "c2", "text": "u"}],
    }


def test_verbalize_rag_keeps_granularity():
    rag = _rag()
    refs = {"data.size": {"id": "subset:hard", "name": "hard", "type": "subset"}}
    out, telem = verbalize_rag_hypotheses(rag, "ETHOS", field_referents=refs)
    # same atoms, same order, same ids/fields/context assignments
    assert [a["id"] for a in out["atoms"]] == ["a1", "a2", "a3"]
    assert [a.get("field") for a in out["atoms"]] == [
        "data.size", "benchmark_details.overview", None]
    assert [a["contexts"] for a in out["atoms"]] == [["c1", "c2"], ["c1"], []]
    assert out["contexts"] == rag["contexts"]
    # only the text is wrapped; the original survives alongside
    for orig, wrapped in zip(rag["atoms"], out["atoms"]):
        assert wrapped["text_unverbalized"] == orig["text"]
        assert wrapped["text"].endswith(orig["text"])
        assert wrapped["text"].startswith("For ")
    # the input dict is not mutated
    assert rag["atoms"][0]["text"] == "The dataset contains 10,000 examples"
    assert "text_unverbalized" not in rag["atoms"][0]


def test_verbalize_rag_telemetry_counts():
    refs = {"data.size": {"id": "subset:hard", "name": "hard", "type": "subset"}}
    _, telem = verbalize_rag_hypotheses(_rag(), "ETHOS", field_referents=refs)
    assert telem["atoms_total"] == 3
    assert telem["verbalized"] == 3
    assert telem["with_referent_context"] == 1
    assert telem["referent_fields"] == ["data.size"]


def test_verbalize_rag_no_referents():
    _, telem = verbalize_rag_hypotheses(_rag(), "ETHOS")
    assert telem["with_referent_context"] == 0
    assert telem["referent_fields"] == []


def test_atomizer_split_pinned():
    # Pins the decomposition: verbalization must not change how the atomizer
    # splits or attributes statements (the wrap happens downstream of RAG).
    raw = (
        "- The dataset contains 10,000 examples [data.size]\n"
        "- The benchmark name is ETHOS [benchmark_details.name]\n"
        "- Model achieved 85% accuracy [methodology.result]\n"
    )
    assert text_to_statements(raw) == [
        {"text": "The dataset contains 10,000 examples", "field": "data.size"},
        {"text": "The benchmark name is ETHOS", "field": "benchmark_details.name"},
        {"text": "Model achieved 85% accuracy", "field": "methodology.interpretation"},
    ]


def test_run_factreasoner_wires_verbalization():
    import auto_benchmarkcard.workers as W

    src = inspect.getsource(W.run_factreasoner)
    assert "verbalize_rag_hypotheses" in src
    assert "formatted_rag_results=verbalized_rag" in src
    # retrieval scoping still runs on the raw atom texts (wrap comes after)
    assert src.index("source_scoped_rag(") < src.index("verbalize_rag_hypotheses(")
