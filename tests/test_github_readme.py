"""Tests for the GitHub-README extraction source: repo derivation (structured + prose),
the fail-loud telemetry taxonomy from run_github_readme, the sidecar written by
run_composer, and evidence routing via extract_facts_from_github_readme.

Network-free: github_tool.requests.get is monkeypatched.
"""

import json
import logging
import os

import requests

import auto_benchmarkcard.workers as W
import auto_benchmarkcard.tools.github.github_tool as GH
from auto_benchmarkcard.tools.composer import composer_tool, evidence
from auto_benchmarkcard.output import OutputManager, sanitize_benchmark_name


class _FakeOM:
    def save_tool_output(self, data, tool, filename):
        return f"/tmp/{tool}/{filename}"


class _Resp:
    def __init__(self, status_code=200, text="# Bench\n\nA dataset of tasks."):
        self.status_code = status_code
        self.text = text


def _state(extracted_ids=None, **extra):
    s = {"query": "demo-bench", "extracted_ids": extracted_ids or {}, "output_manager": _FakeOM()}
    s.update(extra)
    return s


# --- repo derivation -------------------------------------------------------------

def test_derive_structured_url():
    assert GH.derive_github_repo("https://github.com/baceolus/BioLP-bench/tree/main") == \
        "https://github.com/baceolus/BioLP-bench"


def test_derive_skips_reserved_and_none():
    assert GH.derive_github_repo("https://github.com/features/actions") is None
    assert GH.derive_github_repo("https://example.com/x") is None


# --- run_github_readme reason taxonomy -------------------------------------------

def test_ok(monkeypatch):
    monkeypatch.setattr(GH.requests, "get", lambda url, **k: _Resp(200, "# BioLP\n\n" + "x" * 200))
    out = W.run_github_readme(_state({"website_url": "https://github.com/baceolus/BioLP-bench"}))
    assert out["github_readme"]["success"] is True
    assert out["github_telemetry"]["reason"] == "ok"
    assert out["github_telemetry"]["degraded_to_no_content"] is False


def test_no_repo_fail_loud():
    out = W.run_github_readme(_state({"website_url": "https://example.com/none"}))
    assert out["github_readme"]["success"] is False
    tel = out["github_telemetry"]
    assert tel["reason"] == "no_github_repo"
    assert tel["repo_url_resolved"] is False


def test_404_no_readme(monkeypatch):
    monkeypatch.setattr(GH.requests, "get", lambda url, **k: _Resp(404, ""))
    tel = W.run_github_readme(_state({"website_url": "https://github.com/owner/missing"}))["github_telemetry"]
    assert tel["reason"] == "no_readme" and tel["http_status"] == 404
    assert tel["degraded_to_no_content"] is True


def test_403_fetch_blocked(monkeypatch):
    monkeypatch.setattr(GH.requests, "get", lambda url, **k: _Resp(403, ""))
    tel = W.run_github_readme(_state({"website_url": "https://github.com/owner/repo"}))["github_telemetry"]
    assert tel["reason"] == "fetch_blocked" and tel["http_status"] == 403
    assert tel["degraded_to_no_content"] is True


def test_timeout(monkeypatch):
    def _boom(url, **k):
        raise requests.Timeout("slow")
    monkeypatch.setattr(GH.requests, "get", _boom)
    tel = W.run_github_readme(_state({"website_url": "https://github.com/owner/repo"}))["github_telemetry"]
    assert tel["reason"] == "timeout"
    assert tel["degraded_to_no_content"] is True


# --- biolp real shape: repo URL only in the resolved paper abstract --------------

def test_biolp_prose_carrier_fires_fetch(monkeypatch):
    calls = {}

    def _get(url, **k):
        calls["url"] = url
        return _Resp(200, "# BioLP-bench\n\n" + "Biology lab protocols. " * 30)

    monkeypatch.setattr(GH.requests, "get", _get)
    state = _state({
        "paper_abstract": "We introduce BioLP. Code and dataset are published at "
                          "https://github.com/baceolus/BioLP-bench",
    })
    out = W.run_github_readme(state)
    assert out["github_readme"]["success"] is True
    assert out["github_readme"]["url"] == "https://github.com/baceolus/BioLP-bench"
    assert "baceolus/BioLP-bench" in calls["url"]  # GitHub API readme endpoint hit


# --- evidence routing ------------------------------------------------------------

def test_evidence_routing_uses_github_doc(monkeypatch):
    captured = {}

    def _fake_run(prompt, source_text, doc, label):
        captured["doc"], captured["label"] = doc, label
        return [], {}

    monkeypatch.setattr(composer_tool, "_run_extraction", _fake_run)
    gh = {"success": True, "text": "# Bench\n\n" + "Real description text. " * 20}
    composer_tool.extract_facts_from_github_readme(gh, "demo-bench")
    assert captured["doc"] == evidence.DOC_GITHUB
    assert captured["label"] == "GitHub README"


def test_evidence_short_text_empty():
    items, _ = composer_tool.extract_facts_from_github_readme({"text": "hi"}, "demo-bench")
    assert items == []


# --- sidecar + WARNING -----------------------------------------------------------

def test_sidecar_written_and_warns_when_degraded(tmp_path, caplog):
    om = OutputManager("demo-bench", base_path=str(tmp_path))
    state = {
        "query": "demo-bench",
        "output_manager": om,
        "github_telemetry": W._github_telemetry(True, False, "no_readme", 404,
                                                 "https://github.com/owner/missing"),
    }
    with caplog.at_level(logging.WARNING, logger="auto_benchmarkcard.workers"):
        W._write_github_telemetry(state)
    path = os.path.join(om.get_tool_output_path("composer"),
                        f"github_telemetry_{sanitize_benchmark_name('demo-bench')}.json")
    with open(path) as f:
        tel = json.load(f)
    assert tel["reason"] == "no_readme" and tel["http_status"] == 404
    assert tel["degraded_to_no_content"] is True
    assert any("GITHUB_DEGRADED" in r.getMessage() for r in caplog.records)
