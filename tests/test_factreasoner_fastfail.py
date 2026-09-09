"""FactReasoner fast-fail when RITS is unreachable.

Failure class: dead time. An unreachable RITS endpoint used to cost ~7 min of read
timeouts per card before FactReasoner failed soft with the same outcome (no flags).
"""

import pytest

import auto_benchmarkcard.workers as W
from auto_benchmarkcard.config import Config


@pytest.mark.parametrize("base_engine,override", [("rits", None), ("hf", "rits")])
def test_factreasoner_skips_immediately_when_rits_down(monkeypatch, base_engine, override):
    monkeypatch.setattr(Config, "LLM_ENGINE_TYPE", base_engine)
    if override is None:
        monkeypatch.delenv("FACTREASONER_ENGINE_TYPE", raising=False)
    else:
        monkeypatch.setenv("FACTREASONER_ENGINE_TYPE", override)
    monkeypatch.setattr(W, "_rits_reachable", lambda timeout=8.0: False)

    def boom(*a, **k):
        raise AssertionError("FactReasoner must not be invoked when RITS is unreachable")

    monkeypatch.setattr(W, "source_scoped_rag", boom)
    state = {"query": "x", "rag_results": {"contexts": []}, "errors": []}
    out = W.run_factreasoner(state)
    assert out["completed"] == ["factreasoner evaluation failed"]
    assert any("RITS endpoint unreachable" in e for e in out["errors"])


@pytest.mark.parametrize("base_engine,override", [("hf", None), ("rits", "openai")])
def test_factreasoner_reachability_check_only_gates_rits(monkeypatch, base_engine, override):
    monkeypatch.setattr(Config, "LLM_ENGINE_TYPE", base_engine)
    if override is None:
        monkeypatch.delenv("FACTREASONER_ENGINE_TYPE", raising=False)
    else:
        monkeypatch.setenv("FACTREASONER_ENGINE_TYPE", override)

    def no_rits_probe(*a, **k):
        raise AssertionError("A non-RITS FactReasoner route must not probe RITS")

    monkeypatch.setattr(W, "_rits_reachable", no_rits_probe)
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
