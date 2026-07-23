"""Tests for the fail-loud docling-degraded telemetry: the reason taxonomy emitted by
run_docling, the separate sidecar written by run_composer, the WARNING signal, and the
gate-scanner rollup.

Network-free: requests.head/get and the docling tool are monkeypatched.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path

import auto_benchmarkcard.workers as W
from auto_benchmarkcard.output import OutputManager, sanitize_benchmark_name


class _FakeOM:
    def save_tool_output(self, data, tool, filename):
        return f"/tmp/{tool}/{filename}"


class _FakeHead:
    def __init__(self, status_code=200, content_type="application/pdf"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}


def _state(paper_url):
    s = {"query": "demo-bench", "extracted_ids": {}, "output_manager": _FakeOM()}
    if paper_url is not None:
        s["extracted_ids"]["paper_url"] = paper_url
    return s


def _fake_tool(result):
    class _T:
        @staticmethod
        def func(paper_url):
            return result
    return _T


# --- reason taxonomy via run_docling ---------------------------------------------

def test_no_paper_url_not_degraded():
    tel = W.run_docling(_state(None))["docling_telemetry"]
    assert tel["reason"] == "no_paper_url"
    assert tel["paper_url_resolved"] is False
    assert tel["degraded_to_abstract_only"] is False


def test_not_extractable_url_degraded():
    tel = W.run_docling(_state("https://www.semanticscholar.org/paper/x"))["docling_telemetry"]
    assert tel["reason"] == "not_extractable"
    assert tel["degraded_to_abstract_only"] is True


def test_fetch_blocked_403_degraded(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(403, "text/html"))
    tel = W.run_docling(_state("https://example.com/paper.pdf"))["docling_telemetry"]
    assert tel["reason"] == "fetch_blocked" and tel["http_status"] == 403
    assert tel["degraded_to_abstract_only"] is True


def test_html_only_degraded(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "text/html"))
    tel = W.run_docling(_state("https://example.com/paper"))["docling_telemetry"]
    assert tel["reason"] == "html_only"
    assert tel["degraded_to_abstract_only"] is True


def test_empty_body_degraded(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "application/pdf"))
    monkeypatch.setattr(W, "extract_paper_with_docling",
                        _fake_tool({"success": True, "filtered_text": "   ", "metadata": {}}))
    tel = W.run_docling(_state("https://example.com/paper.pdf"))["docling_telemetry"]
    assert tel["reason"] == "empty_body"
    assert tel["full_text"] is False and tel["degraded_to_abstract_only"] is True


def test_extraction_failed_degraded(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "application/pdf"))
    monkeypatch.setattr(W, "extract_paper_with_docling",
                        _fake_tool({"success": False, "warning": "Could not extract paper"}))
    tel = W.run_docling(_state("https://example.com/paper.pdf"))["docling_telemetry"]
    assert tel["reason"] == "extraction_failed"
    assert tel["degraded_to_abstract_only"] is True


def test_ok_not_degraded(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "application/pdf"))
    monkeypatch.setattr(W, "extract_paper_with_docling",
                        _fake_tool({"success": True, "filtered_text": "real body text", "metadata": {}}))
    tel = W.run_docling(_state("https://example.com/paper.pdf"))["docling_telemetry"]
    assert tel["reason"] == "ok"
    assert tel["full_text"] is True and tel["degraded_to_abstract_only"] is False


# --- _write_docling_telemetry: separate sidecar + WARNING ------------------------

def test_sidecar_written_and_warns_when_degraded(tmp_path, caplog):
    om = OutputManager("demo-bench", base_path=str(tmp_path))
    state = {
        "query": "demo-bench",
        "output_manager": om,
        "extracted_ids": {"paper_url": "https://doi.org/10.1101/x"},
        "docling_telemetry": W._docling_telemetry(
            True, False, "fetch_blocked", 403, "https://www.biorxiv.org/...full.pdf"),
    }
    with caplog.at_level(logging.WARNING, logger="auto_benchmarkcard.workers"):
        W._write_docling_telemetry(state)

    path = os.path.join(om.get_tool_output_path("composer"),
                        f"docling_telemetry_{sanitize_benchmark_name('demo-bench')}.json")
    with open(path) as f:
        tel = json.load(f)
    assert tel["degraded_to_abstract_only"] is True
    assert tel["reason"] == "fetch_blocked" and tel["http_status"] == 403
    assert any("DOCLING_DEGRADED" in r.getMessage() for r in caplog.records)


def test_sidecar_separate_from_provenance(tmp_path):
    om = OutputManager("demo-bench3", base_path=str(tmp_path))
    safe = sanitize_benchmark_name("demo-bench3")
    prov = {"data": {"size": {"source": "deterministic", "evidence": "1319"}}}
    om.save_tool_output(prov, "composer", f"provenance_{safe}.json")

    state = {
        "query": "demo-bench3",
        "output_manager": om,
        "extracted_ids": {"paper_url": "https://doi.org/10.1101/x"},
        "docling_telemetry": W._docling_telemetry(True, False, "fetch_blocked", 403, "u"),
    }
    W._write_docling_telemetry(state)

    composer_dir = om.get_tool_output_path("composer")
    with open(os.path.join(composer_dir, f"provenance_{safe}.json")) as f:
        prov_after = json.load(f)
    assert prov_after == prov  # untouched
    assert os.path.exists(os.path.join(composer_dir, f"docling_telemetry_{safe}.json"))


def test_no_telemetry_no_paper_url_not_degraded(tmp_path, caplog):
    om = OutputManager("demo-bench4", base_path=str(tmp_path))
    state = {"query": "demo-bench4", "output_manager": om, "extracted_ids": {}}
    with caplog.at_level(logging.WARNING, logger="auto_benchmarkcard.workers"):
        W._write_docling_telemetry(state)
    path = os.path.join(om.get_tool_output_path("composer"),
                        f"docling_telemetry_{sanitize_benchmark_name('demo-bench4')}.json")
    with open(path) as f:
        tel = json.load(f)
    assert tel["reason"] == "no_paper_url"
    assert tel["degraded_to_abstract_only"] is False
    assert not any("DOCLING_DEGRADED" in r.getMessage() for r in caplog.records)


# --- compute_gate_metrics rollup -------------------------------------------------

def _load_cgm():
    p = Path(__file__).resolve().parents[1] / "scripts" / "compute_gate_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_gate_metrics", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_tel(run_dir, name, tel):
    d = run_dir / name / "tool_output" / "composer"
    d.mkdir(parents=True)
    (d / f"docling_telemetry_{name}.json").write_text(json.dumps(tel))


def test_gate_scanner_counts_degraded(tmp_path):
    cgm = _load_cgm()
    _write_tel(tmp_path, "cardA",
               W._docling_telemetry(True, False, "fetch_blocked", 403, "u"))
    _write_tel(tmp_path, "cardB",
               W._docling_telemetry(True, True, "ok", 200, "u"))
    dd = cgm._docling_degraded(str(tmp_path))
    assert dd["count"] == 1
    assert dd["cards"][0]["card"] == "cardA"
    assert dd["cards"][0]["reason"] == "fetch_blocked"
