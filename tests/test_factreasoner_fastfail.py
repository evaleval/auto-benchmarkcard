"""FactReasoner fast-fail when RITS is unreachable.

Failure class: dead time. An unreachable RITS endpoint used to cost ~7 min of read
timeouts per card before FactReasoner failed soft with the same outcome (no flags).
"""

import auto_benchmarkcard.workers as W
from auto_benchmarkcard.config import Config


def test_factreasoner_skips_immediately_when_rits_down(monkeypatch):
    monkeypatch.setattr(Config, "LLM_ENGINE_TYPE", "rits")
    monkeypatch.setattr(W, "_rits_reachable", lambda timeout=8.0: False)

    def boom(*a, **k):
        raise AssertionError("FactReasoner must not be invoked when RITS is unreachable")

    monkeypatch.setattr(W, "source_scoped_rag", boom)
    state = {"query": "x", "rag_results": {"contexts": []}, "errors": []}
    out = W.run_factreasoner(state)
    assert out["completed"] == ["factreasoner evaluation failed"]
    assert any("RITS endpoint unreachable" in e for e in out["errors"])


def test_factreasoner_reachability_check_only_gates_rits(monkeypatch):
    monkeypatch.setattr(Config, "LLM_ENGINE_TYPE", "hf")
    monkeypatch.setattr(W, "_rits_reachable", lambda timeout=8.0: False)
    calls = []

    def fake_scoped(*a, **k):
        calls.append(1)
        raise RuntimeError("stop here")

    monkeypatch.setattr(W, "source_scoped_rag", fake_scoped)
    state = {"query": "x", "rag_results": {"contexts": []}, "errors": []}
    W.run_factreasoner(state)
    assert calls == [1]


def test_rits_reachable_false_without_url(monkeypatch):
    monkeypatch.delenv("RITS_API_URL", raising=False)
    assert W._rits_reachable() is False
