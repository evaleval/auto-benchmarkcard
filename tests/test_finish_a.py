"""Finish-Composer Brief A: deterministic fills & metadata (A1-A5, A7-A9).

All network-free: HTTP is monkeypatched. The wave2 (@1e9e9b8) pilot defects are asserted
via small inline fixtures, not by depending on output/ run dirs.
"""

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr
from auto_benchmarkcard.tools.composer import composer_tool as ct
from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import omit_empty_risk_urls


class _Resp:
    def __init__(self, *, text="", json_data=None, status=200):
        self._text, self._json, self.status_code = text, json_data, status

    @property
    def text(self):
        return self._text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --------------------------------------------------------------------------- A1
_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>BigCodeBench: Benchmarking Code Generation with Diverse Function Calls</title>
    <published>2024-06-22T17:59:59Z</published>
    <author><name>Terry Yue Zhuo</name></author>
    <author><name>Minh Chien Vu</name></author>
  </entry>
</feed>"""


def test_a1_arxiv_authors_title_year(monkeypatch):
    pr._author_cache.clear()
    monkeypatch.setattr(pr.requests, "get", lambda *a, **k: _Resp(text=_ARXIV_ATOM))
    meta = pr._fetch_paper_authors("https://arxiv.org/abs/2406.15877")
    assert meta["authors"] == ["Terry Yue Zhuo", "Minh Chien Vu"]
    assert meta["title"].startswith("BigCodeBench")
    assert meta["year"] == 2024


def test_a1_doi_via_openalex(monkeypatch):
    pr._author_cache.clear()
    oa = {"title": "WinoGrande", "publication_year": 2020,
          "authorships": [{"author": {"display_name": "Keisuke Sakaguchi"}}],
          "ids": {}, "locations": [], "primary_location": {}}
    monkeypatch.setattr(pr.requests, "get", lambda *a, **k: _Resp(json_data=oa))
    meta = pr._fetch_paper_authors("https://doi.org/10.1609/aaai.v34i05.6399")
    assert meta["authors"] == ["Keisuke Sakaguchi"]
    assert meta["title"] == "WinoGrande"


def test_a1_failsoft_unknown_not_negative_cached(monkeypatch):
    pr._author_cache.clear()
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)  # no backoff delay in tests
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("network down")

    monkeypatch.setattr(pr.requests, "get", boom)
    url = "https://arxiv.org/abs/1234.56789"
    assert pr._fetch_paper_authors(url) == {"authors": [], "title": "", "year": None}
    # a transient miss must NOT be negative-cached (else it poisons same-URL suite families)
    assert url not in pr._author_cache
    # non-arxiv, non-doi URL -> no fetch at all
    before = calls["n"]
    assert pr._fetch_paper_authors("https://example.com/foo") == \
        {"authors": [], "title": "", "year": None}
    assert calls["n"] == before
    # a repeat of the failed arxiv URL re-attempts rather than returning a cached NS
    pr._fetch_paper_authors(url)
    assert calls["n"] > before


# --------------------------------------------------------------------------- A2
def _clear_logo_caches():
    ct._logo_cache.clear()
    ct._org_overview_cache.clear()


def test_a2_logo_org_overview(monkeypatch):
    # A confirmed org with a real uploaded avatar (cdn-avatars, no '/avatars/' path).
    _clear_logo_caches()
    monkeypatch.setattr(ct.requests, "get", lambda url, **k: (
        _Resp(json_data={"avatarUrl": "https://cdn-avatars.huggingface.co/v1/x.png"})
        if "organizations/bigcode/overview" in url else _Resp(status=404)))
    assert ct._fetch_org_logo("bigcode") == "https://cdn-avatars.huggingface.co/v1/x.png"


def test_a2_logo_rejects_identicon(monkeypatch):
    # A confirmed org whose avatarUrl is a default identicon (/avatars/<hash>.svg) has no real
    # logo: rejected, and with no github fallback the result is None.
    _clear_logo_caches()
    monkeypatch.setattr(ct.requests, "get", lambda url, **k: (
        _Resp(json_data={"avatarUrl": "/avatars/6f9b.svg"})
        if "organizations/acme/overview" in url else _Resp(status=404)))
    monkeypatch.setattr(ct.requests, "head", lambda *a, **k: _Resp(status=404))
    assert ct._fetch_org_logo("acme") is None


def test_a2_logo_github_fallback_for_confirmed_org(monkeypatch):
    # Confirmed org but no usable avatar -> github.com/<org>.png.
    _clear_logo_caches()
    monkeypatch.setattr(ct.requests, "get", lambda url, **k: (
        _Resp(json_data={})  # confirmed org, no avatarUrl
        if "organizations/bigcode/overview" in url else _Resp(status=404)))
    monkeypatch.setattr(ct.requests, "head", lambda *a, **k: _Resp(status=200))
    assert ct._fetch_org_logo("bigcode") == "https://github.com/bigcode.png"


def test_a2_logo_user_handle_is_none(monkeypatch):
    # A user handle (organizations/<user> 404s) gets no logo -- never the users/ photo, never
    # github.com/<user>.png (a wrong person photo), even when github.com/<user>.png exists.
    _clear_logo_caches()
    monkeypatch.setattr(ct.requests, "get", lambda *a, **k: _Resp(status=404))
    monkeypatch.setattr(ct.requests, "head", lambda *a, **k: _Resp(status=200))
    assert ct._fetch_org_logo("baceolus") is None


def test_a2_logo_failsoft_none(monkeypatch):
    _clear_logo_caches()
    monkeypatch.setattr(ct.requests, "get", lambda *a, **k: _Resp(status=404))
    monkeypatch.setattr(ct.requests, "head", lambda *a, **k: _Resp(status=404))
    assert ct._fetch_org_logo("nobody") is None


# --------------------------------------------------------------------------- A3
def test_a3_caps_name_resolves_to_short_identity():
    name = ("BIGCODEBENCH : BENCHMARKING CODE GENERATION WITH DIVERSE "
            "FUNCTION CALLS AND COMPLEX INSTRUCTIONS")
    out = ct._clean_caps_name(name, ("BigCodeBench: Benchmarking Code Generation",))
    assert out == "BigCodeBench"


def test_a3_short_acronym_unchanged():
    assert ct._clean_caps_name("AA-LCR", ()) == "AA-LCR"


def test_a3_mixed_case_unchanged():
    n = "Biological Lab Protocol benchmark (BioLP-bench)"
    assert ct._clean_caps_name(n, ()) == n


def test_a3_caps_without_candidate_drops_subtitle():
    assert ct._clean_caps_name("FOOBENCH : A GREAT BENCHMARK", ()) == "FOOBENCH"


# --------------------------------------------------------------------------- A4
def test_a4_provenance_source_from_evidence_doc():
    prov = {
        "benchmark_details": {
            "name": {"source": "E01", "evidence_ids": ["E01"]},
            "overview": {"source": "docling/stated", "evidence_ids": ["E02"]},
            "data_type": {"source": "deterministic", "evidence_ids": []},
        },
        "methodology": {"metrics": {"source": "eee", "evidence_ids": []}},
    }
    items = [{"evidence_id": "E01", "doc": "abstract"},
             {"evidence_id": "E02", "doc": "github_readme"}]
    ct._normalize_provenance_sources(prov, items)
    assert prov["benchmark_details"]["name"]["source"] == "abstract"
    assert prov["benchmark_details"]["overview"]["source"] == "github"
    assert prov["benchmark_details"]["data_type"]["source"] == "deterministic"
    assert prov["methodology"]["metrics"]["source"] == "eee"


def test_a4_compound_source_recovered_without_items():
    prov = {"benchmark_details": {"name": {"source": "docling/stated", "evidence_ids": []}}}
    ct._normalize_provenance_sources(prov, [])
    assert prov["benchmark_details"]["name"]["source"] == "paper"


def test_a4_display_fields_write_sidecar_provenance(monkeypatch):
    monkeypatch.setattr(ct, "_fetch_org_logo", lambda org: None)
    monkeypatch.setattr(ct, "_hf_org_overview", lambda owner: {})  # confirmed org
    card, prov = {"benchmark_details": {}}, {}
    ct.apply_display_fields(
        card, hf_metadata=None,
        extracted_ids={"hf_repo": "bigcode/bigcodebench", "paper_authors": ["Terry Zhuo"]},
        provenance=prov,
    )
    bd = prov["benchmark_details"]
    assert bd["authors"]["source"] == "paper" and bd["authors"]["status"] == "derived"
    assert bd["org_url"]["source"] == "hf"
    assert card["benchmark_details"]["authors"] == ["Terry Zhuo"]


def test_a4_hf_author_handle_not_treated_as_person():
    # HF card_data may carry a lone org handle in `authors`; it must not become authors.
    card = {"benchmark_details": {}}
    ct.apply_display_fields(
        card, hf_metadata={"tags": [], "card_data": {"authors": ["bigcode"]}},
        extracted_ids={"paper_authors": ["Real Person"]}, provenance={},
    )
    assert card["benchmark_details"]["authors"] == ["Real Person"]


# --------------------------------------------------------------------------- A5
def test_a5_omit_empty_risk_urls():
    card = {"possible_risks": [
        {"category": "Evaluation bias (single LLM judge)", "description": "x", "url": None},
        {"category": "Data bias", "description": "y", "url": "https://ibm.com/data-bias"},
    ]}
    omit_empty_risk_urls(card)
    r0, r1 = card["possible_risks"]
    assert "url" not in r0
    assert r1["url"] == "https://ibm.com/data-bias"


# --------------------------------------------------------------------------- A7
def test_a7_metric_labels_by_token_and_raw_id():
    eee = {"metrics": {
        "bigcodebench_hard_set_pass_1": {
            "metric_name": "BigCodeBench (Hard Set) Pass@1", "metric_unit": "points"},
        "llm_stats.bigcodebench.score": {
            "metric_name": "BigCodeBench score", "metric_unit": "proportion",
            "score_type": "continuous"},
    }}
    labels = ct.build_metric_labels(eee)
    assert labels["pass@1"]["name"] == "BigCodeBench (Hard Set) Pass@1"
    assert labels["other:bigcodebench.score"]["name"] == "BigCodeBench score"
    assert labels["other:bigcodebench.score"]["unit"] == "proportion"
    # also addressable by the raw id
    assert labels["bigcodebench_hard_set_pass_1"]["name"] == "BigCodeBench (Hard Set) Pass@1"


def test_a7_empty_when_no_metrics():
    assert ct.build_metric_labels({}) == {}
    assert ct.build_metric_labels(None) == {}


# --------------------------------------------------------------------------- A8
def test_a8_drops_judge_detail_when_no_llm_judge():
    card = {"methodology": {"judge_uses_llm": False}}
    missing = ["benchmark_details.similar_benchmarks", "methodology.judge_num",
               "methodology.judge_models", "methodology.judge_score_consolidation",
               "data.source"]
    out = ct.drop_inapplicable_judge_missing(card, missing)
    assert out == ["benchmark_details.similar_benchmarks", "data.source"]


def test_a8_keeps_judge_detail_when_llm_judge():
    card = {"methodology": {"judge_uses_llm": True}}
    missing = ["methodology.judge_score_consolidation"]
    assert ct.drop_inapplicable_judge_missing(card, missing) == missing


def test_a8_noop_when_judge_unknown():
    card = {"methodology": {"judge_uses_llm": "Not specified"}}
    missing = ["methodology.judge_num"]
    assert ct.drop_inapplicable_judge_missing(card, missing) == missing


# --------------------------------------------------------------------------- A9
def test_a9_format_from_files():
    # The dominant data-file extension is a storage token, labelled as the hosting format.
    assert ct._format_from_files(["data/x-00000.parquet", "README.md", "data/y.parquet"]) == \
        "parquet (HuggingFace hosting format)"
    assert ct._format_from_files(["a.csv", "b.csv", "c.json"]) == "csv (HuggingFace hosting format)"
    assert ct._format_from_files(["README.md", "LICENSE"]) is None
    assert ct._format_from_files(None) is None


def test_a9_size_breakdown_from_counts_table():
    md = (
        "| Domain | Questions |\n"
        "| --- | --- |\n"
        "| Company Documents | 63 |\n"
        "| Industry Reports | 8 |\n"
        "| Legal | 6 |\n"
    )
    assert ct._size_breakdown_from_readme(md) == {
        "Company Documents": 63, "Industry Reports": 8, "Legal": 6}


def test_a9_size_breakdown_ignores_decimal_score_table():
    md = "| Model | Score |\n| --- | --- |\n| GPT-4 | 0.42 |\n| Claude | 0.55 |\n"
    assert ct._size_breakdown_from_readme(md) is None


def test_a9_size_breakdown_none_without_table():
    assert ct._size_breakdown_from_readme("no table here") is None
    assert ct._size_breakdown_from_readme(None) is None


def test_a9_overrides_are_fill_only():
    # bigcode already has a real size_breakdown dict -> must be preserved (fill-only);
    # collection_date is NS -> filled.
    card = {"data": {"collection_date": "Not specified", "size_breakdown": {"Domain": 7}}}
    facts = {"data.collection_date": "2024", "data.size_breakdown": {"X": 1, "Y": 2}}
    ct.apply_deterministic_overrides(card, facts)
    assert card["data"]["collection_date"] == "2024"
    assert card["data"]["size_breakdown"] == {"Domain": 7}
