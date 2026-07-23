"""Regression tests for per-benchmark RAG collection isolation.

In a single batch process every RAGRetriever used to share chromadb's default
"langchain" collection (chromadb caches clients by settings), so a later
benchmark retrieved an earlier one's chunks. These tests pin the fix: each
retriever indexes into its own collection and cleans it up afterwards.
"""

import re

import pytest
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from auto_benchmarkcard.output import sanitize_chroma_collection_name
from auto_benchmarkcard.tools.rag.rag_retriever import RAGRetriever


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline embeddings so tests don't download a HF model."""

    def _vector(self, text):
        tokens = text.lower().split()
        vec = [0.0] * 16
        for tok in tokens:
            vec[hash(tok) % 16] += 1.0
        return vec or [0.0] * 16

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def _make_retriever(monkeypatch, collection_name=None):
    """Build a retriever with fake embeddings and no LLM/network dependencies."""
    monkeypatch.setattr(
        RAGRetriever, "_initialize_embeddings", lambda self, model: _FakeEmbeddings()
    )
    return RAGRetriever(
        embedding_model="minilm",
        enable_llm_reranking=False,
        enable_hybrid_search=False,
        enable_query_expansion=False,
        collection_name=collection_name,
    )


def test_sequential_index_builds_do_not_cross_retrieve(monkeypatch):
    """Two retrievers indexed in sequence must not see each other's chunks."""
    docs_a = [Document(page_content="ALPHA_SENTINEL bigcodebench coding tasks dataset", metadata={"source": "a"})]
    docs_b = [Document(page_content="BRAVO_SENTINEL biolp biology protocols dataset", metadata={"source": "b"})]

    retriever_a = _make_retriever(monkeypatch, "bc_bigcodebench_aaaaaaaa")
    retriever_b = _make_retriever(monkeypatch, "bc_biolp_bbbbbbbb")
    try:
        retriever_a.index_documents(docs_a)
        retriever_b.index_documents(docs_b)

        # Pull everything out of B's collection; A's content must never appear.
        b_contents = [
            d.page_content for d in retriever_b.vectorstore.similarity_search("dataset", k=50)
        ]
        assert any("BRAVO_SENTINEL" in c for c in b_contents)
        assert not any("ALPHA_SENTINEL" in c for c in b_contents)

        # Symmetric check on A's collection.
        a_contents = [
            d.page_content for d in retriever_a.vectorstore.similarity_search("dataset", k=50)
        ]
        assert any("ALPHA_SENTINEL" in c for c in a_contents)
        assert not any("BRAVO_SENTINEL" in c for c in a_contents)
    finally:
        retriever_a.cleanup()
        retriever_b.cleanup()


def test_default_collection_names_are_unique(monkeypatch):
    """Without an explicit name, retrievers must not share the default collection."""
    r1 = _make_retriever(monkeypatch)
    r2 = _make_retriever(monkeypatch)
    assert r1.collection_name != r2.collection_name
    assert r1.collection_name != "langchain"


def test_cleanup_removes_collection(monkeypatch):
    """cleanup() drops the collection and is safe to call more than once."""
    docs = [Document(page_content="GAMMA_SENTINEL evaluation harness details", metadata={"source": "g"})]
    retriever = _make_retriever(monkeypatch, "bc_gamma_cccccccc")
    retriever.index_documents(docs)
    assert retriever.vectorstore is not None

    retriever.cleanup()
    assert retriever.vectorstore is None
    assert retriever.retriever is None
    # Idempotent: a second cleanup must not raise.
    retriever.cleanup()


# -------------------------------------------------- chroma name sanitizer ----
# chromadb rejects collection names that are not 3-63 chars, don't start/end with
# an alphanumeric, contain anything outside [a-zA-Z0-9._-], have consecutive dots,
# or are IPv4-shaped. sanitize_benchmark_name() is filesystem-oriented and can
# produce all of these, which would make Chroma.from_documents raise and silently
# skip RAG. sanitize_chroma_collection_name() must yield a valid name for any input.

_CHROMA_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$")
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

ADVERSARIAL_NAMES = [
    "BIG-Bench Hard",            # spaces
    "τ-bench (v2)",         # non-ascii + parens
    "v1..2...3",                 # consecutive dots
    "1.2.3.4",                   # IPv4-shaped
    "MMLU.Pro",                  # single version dot
    "...---...",                 # empty after stripping non-alphanumerics
    "!!!",                       # empty after stripping
    "",                          # empty
    "数据集",        # all non-ascii
    "a" * 200,                   # over-length
    "-leading-and-trailing-",    # leading/trailing non-alphanumerics
]


def _is_valid_chroma_name(name):
    return (
        isinstance(name, str)
        and 3 <= len(name) <= 63
        and bool(_CHROMA_NAME_RE.match(name))
        and ".." not in name
        and not _IPV4_RE.match(name)
    )


@pytest.mark.parametrize("raw", ADVERSARIAL_NAMES)
def test_chroma_sanitizer_yields_valid_names(raw):
    assert _is_valid_chroma_name(sanitize_chroma_collection_name(raw))
    assert _is_valid_chroma_name(sanitize_chroma_collection_name(raw, suffix="deadbeef"))


def test_chroma_sanitizer_suffix_unique_and_survives_truncation():
    """A long name must still keep its uniqueness suffix (else collisions return)."""
    base = "x" * 200
    a = sanitize_chroma_collection_name(base, suffix="aaaaaaaa")
    b = sanitize_chroma_collection_name(base, suffix="bbbbbbbb")
    assert a != b                       # the suffix differentiates collections
    assert a.endswith("aaaaaaaa")       # suffix is never truncated away
    assert _is_valid_chroma_name(a) and len(a) <= 63


def test_sanitized_names_accepted_by_chromadb():
    """The regex above is our spec; this proves chromadb itself accepts the names."""
    emb = _FakeEmbeddings()
    docs = [Document(page_content="content for indexing", metadata={"source": "s"})]
    for i, raw in enumerate(ADVERSARIAL_NAMES):
        name = sanitize_chroma_collection_name(raw, suffix="t%02d" % i)
        vs = Chroma.from_documents(docs, emb, collection_name=name)  # raises if invalid
        try:
            assert vs is not None
        finally:
            vs.delete_collection()


# ----------------------------------------- composer inline paper-RAG (site 2) ----

def _ns_section_response():
    """An all-"Not specified" Stage-B section response so compose runs offline."""
    import json

    from auto_benchmarkcard.tools.composer.composer_tool import (
        BenchmarkDetails, DataInfo, EthicalAndLegalConsiderations, Methodology,
        PurposeAndIntendedUsers)

    models = {
        "benchmark_details": BenchmarkDetails,
        "purpose_and_intended_users": PurposeAndIntendedUsers,
        "data": DataInfo,
        "methodology": Methodology,
        "ethical_and_legal_considerations": EthicalAndLegalConsiderations,
    }
    out = {}
    for sec, cls in models.items():
        out[sec] = {f: "Not specified" for f in cls.model_fields if f != "provenance"}
        out[sec]["provenance"] = {}
    return json.dumps(out)


def test_compose_inline_paper_rag_unique_collection_and_cleanup(monkeypatch):
    """compose_benchmark_card's inline paper-RAG path must index into a unique,
    chromadb-valid collection and delete it afterwards, so a later benchmark in the
    same process can't retrieve this one's paper chunks via the default collection."""
    from auto_benchmarkcard.tools.composer import composer_tool as C

    rec = {"collection_name": None, "deleted": False}

    class _FakeRetriever:
        def invoke(self, _query):
            return [Document(page_content="the benchmark dataset evaluates models", metadata={})]

    class _FakeVS:
        def as_retriever(self, **_kwargs):
            return _FakeRetriever()

        def delete_collection(self):
            rec["deleted"] = True

    class _FakeChroma:
        @staticmethod
        def from_documents(documents, embedding, collection_name=None, **_kwargs):
            rec["collection_name"] = collection_name
            rec["n_docs"] = len(documents)
            return _FakeVS()

    class _FakeHandler:
        model_name = "fake-model"

        def generate(self, prompt, response_format=None):
            return _ns_section_response()

    # Avoid the real chroma client, the HF embedding download, and the network LLM.
    monkeypatch.setattr(C, "Chroma", _FakeChroma)
    monkeypatch.setattr(C, "HuggingFaceEmbeddings", lambda *a, **k: object())
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler())

    # filtered_text must exceed PAPER_EXTRACTION_BUDGET to enter the RAG branch.
    big_paper = "This benchmark measures reasoning over coding tasks. " * 2000
    docling = {"success": True, "filtered_text": big_paper, "metadata": {"title": "Demo Bench"}}

    fn = getattr(C.compose_benchmark_card, "func", C.compose_benchmark_card)
    res = fn(query="Weird/Name v1.0!!", docling_output=docling)

    # A real per-benchmark collection name was passed (not chromadb's default).
    assert rec["collection_name"], "no collection_name passed to inline paper-RAG"
    assert rec["collection_name"] != "langchain"
    assert _is_valid_chroma_name(rec["collection_name"])
    # And the collection was cleaned up; nothing leaks into the next benchmark.
    assert rec["deleted"] is True
    assert res["benchmark_card"]
