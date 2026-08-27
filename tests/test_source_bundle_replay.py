"""Source-bundle replay (WP1.5): worker nodes read their inputs from a prior run's
tool_output instead of fetching live, so two conditions compose from byte-identical
sources.

Failure classes pinned here:
  1. Source drift between conditions: with a bundle active, no node may fetch live
     (the live functions are patched to raise).
  2. Attachment drift: a card whose bundle run bound NO paper must get no paper in the
     replay, never a fresh live resolution.
  3. Silent fallthrough: without SOURCE_BUNDLE_DIR the helpers return None and the nodes
     behave as before.
"""

import json
import os

import pytest

import auto_benchmarkcard.workers as W


class _FakeOM:
    def __init__(self):
        self.writes = {}

    def save_tool_output(self, data, tool, filename):
        self.writes[(tool, filename)] = data
        return f"/fake/{tool}/{filename}"

    def get_tool_output_path(self, tool, create_if_missing=True):
        return f"/fake/{tool}"


def _state(name, **extra):
    st = {"query": name, "extracted_ids": {}, "eee_metadata": {"benchmark_name": name, "metrics": {}},
          "hf_json": None, "hf_repo": "acme/" + name, "output_manager": _FakeOM()}
    st.update(extra)
    return st


def _write(bundle, slug, stamp, tool, filename, payload):
    d = bundle / f"{slug}_{stamp}" / "tool_output" / tool
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps(payload))


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    monkeypatch.setenv("SOURCE_BUNDLE_DIR", str(root))
    return root


@pytest.fixture()
def no_live(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("live fetch must not run under a source bundle")

    class _Tool:
        func = staticmethod(boom)

    monkeypatch.setattr(W, "extract_paper_with_docling", _Tool())
    monkeypatch.setattr(W, "hf_dataset_metadata", _Tool())
    monkeypatch.setattr(W, "extract_html_content", _Tool())
    monkeypatch.setattr(W, "fetch_github_readme", boom)
    monkeypatch.setattr(W, "resolve_paper", boom)
    monkeypatch.setattr(W, "_verify_paper_binding_gate", boom)
    monkeypatch.setattr(W, "_fetch_paper_authors",
                        lambda url: {"authors": ["A. Author"], "title": "T", "year": 2025})


def test_helpers_inactive_without_env(monkeypatch):
    monkeypatch.delenv("SOURCE_BUNDLE_DIR", raising=False)
    assert W._bundle_dir() is None
    assert W._bundle_tool_output(_state("x"), "docling", "x.json") is None


def test_newest_dir_with_file_wins(bundle):
    _write(bundle, "x", "2026-01-01_00-00", "docling", "x.json", {"v": "old"})
    _write(bundle, "x", "2026-02-02_00-00", "docling", "x.json", {"v": "new"})
    _write(bundle, "x", "2026-03-03_00-00", "eee", "x.json", {"v": "later attempt, no docling"})
    assert W._bundle_tool_output(_state("x"), "docling", "x.json") == {"v": "new"}


def test_docling_replays_from_bundle(bundle, no_live):
    doc = {"success": True, "filtered_text": "Paper body " * 20, "metadata": {"title": "T"}}
    _write(bundle, "x", "2026-01-01_00-00", "docling", "x.json", doc)
    st = _state("x", extracted_ids={"paper_url": "https://arxiv.org/abs/1234.5678"})
    out = W.run_docling(st)
    assert out["docling_output"] == doc
    assert out["docling_telemetry"]["reason"] == "ok"
    assert "replayed" in out["completed"][0]
    assert ("docling", "x.json") in st["output_manager"].writes   # re-recorded in the new run


def test_docling_bundle_without_paper_does_not_convert_live(bundle, no_live):
    st = _state("x", extracted_ids={"paper_url": "https://arxiv.org/abs/1234.5678"})
    out = W.run_docling(st)
    assert out["docling_output"] is None
    assert out["docling_telemetry"]["reason"] == "bundle_no_docling"


def test_paper_resolver_replays_binding_and_no_paper(bundle, no_live):
    _write(bundle, "x", "2026-01-01_00-00", "paper_resolver", "paper-verification.json",
           {"resolved_url": "https://arxiv.org/abs/1234.5678", "binding": {"verdict": "confirmed"}})
    out = W.run_paper_resolver(_state("x"))
    assert out["extracted_ids"]["paper_url"] == "https://arxiv.org/abs/1234.5678"
    assert out["extracted_ids"]["paper_authors"] == ["A. Author"]
    assert out["paper_verified"] is True

    _write(bundle, "y", "2026-01-01_00-00", "paper_resolver", "paper-verification.json",
           {"resolved_url": None, "binding": {"verdict": "rejected"}})
    st = _state("y", extracted_ids={"paper_url": "https://arxiv.org/abs/9999.0001"})
    out = W.run_paper_resolver(st)
    assert "paper_url" not in out["extracted_ids"]       # attachment parity: stays unattached
    assert "replayed" in out["completed"][0]


def test_paper_resolver_bundle_without_sidecar_means_no_paper(bundle, no_live):
    out = W.run_paper_resolver(_state("z"))
    assert "paper_url" not in out["extracted_ids"]


def test_hf_replays_metadata_and_rejection(bundle, no_live):
    hf = {"id": "acme/x", "tags": ["license:mit"], "readme_markdown": "# x\n" * 30}
    _write(bundle, "x", "2026-01-01_00-00", "hf", "x.json", hf)
    out = W.run_hf(_state("x"))
    assert out["hf_json"] == hf and "replayed" in out["completed"][0]

    _write(bundle, "y", "2026-01-01_00-00", "hf", "y.json", hf)
    _write(bundle, "y", "2026-01-01_00-00", "hf_verifier", "hf-verification.json",
           {"final_action": "rejected_to_none", "verdict": "rejected"})
    out = W.run_hf(_state("y"))
    assert out["hf_repo"] is None and out["hf_rejected"] is True


def test_hf_bundle_without_metadata_is_honest_thin(bundle, no_live):
    out = W.run_hf(_state("q"))
    assert "hf_json" not in out or out.get("hf_json") is None


def test_html_and_github_replay(bundle, no_live):
    html = {"success": True, "text": "page text " * 20, "url": "https://x.org", "title": "x"}
    gh = {"success": True, "text": "readme " * 20, "url": "https://github.com/a/x"}
    _write(bundle, "x", "2026-01-01_00-00", "html", "x.json", html)
    _write(bundle, "x", "2026-01-01_00-00", "github", "x.json", gh)
    st = _state("x", eee_metadata={"benchmark_name": "x", "metrics": {},
                                   "source_urls": ["https://x.org/page.html"]})
    assert W.run_html_extractor(st)["html_content"] == html
    out = W.run_github_readme(st)
    assert out["github_readme"] == gh and out["github_telemetry"]["reason"] == "ok"


def test_html_and_github_absent_in_bundle_stay_absent(bundle, no_live):
    st = _state("x", eee_metadata={"benchmark_name": "x", "metrics": {},
                                   "source_urls": ["https://x.org/page.html"]})
    assert W.run_html_extractor(st)["html_content"]["success"] is False
    assert W.run_github_readme(st)["github_readme"]["success"] is False


def test_croissant_replay_helper(bundle):
    _write(bundle, "x", "2026-01-01_00-00", "hf", "croissant_x.json", {"data.size": "10 examples"})
    assert W._bundle_croissant_facts(_state("x")) == {"data.size": "10 examples"}
    assert W._bundle_croissant_facts(_state("nope")) is None
