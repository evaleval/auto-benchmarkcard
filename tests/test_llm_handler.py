"""Tests for LLMHandler.generate_with_meta: stop_reason plumbing, <think> stripping,
per-call max_completion_tokens override, and the config default budget.

Network-free: __init__ is bypassed and a FakeEngine is injected, so no real inference
engine / credentials are needed.
"""

import json

from auto_benchmarkcard.llm_handler import (
    LLMHandler,
    clear_usage_log_path,
    set_usage_log_path,
)


class _Out:
    def __init__(self, prediction, stop_reason=None):
        self.prediction = prediction
        self.stop_reason = stop_reason


class FakeEngine:
    def __init__(self, prediction="{}", stop_reason="stop"):
        self.prediction = prediction
        self.stop_reason = stop_reason
        self.parameters = {"temperature": 0.15, "max_completion_tokens": 16384}
        self.seen_max_tokens = []

    def generate(self, prompts, response_format=None, verbose=False):
        self.seen_max_tokens.append(self.parameters.get("max_completion_tokens"))
        return [_Out(self.prediction, self.stop_reason)]


def _handler(engine):
    h = LLMHandler.__new__(LLMHandler)  # bypass __init__ (no real engine / creds)
    h.engine = engine
    h.verbose = False
    return h


def test_generate_with_meta_returns_text_and_stop_reason():
    h = _handler(FakeEngine(prediction='{"a": 1}', stop_reason="length"))
    text, stop_reason = h.generate_with_meta("p")
    assert text == '{"a": 1}'
    assert stop_reason == "length"


def test_strip_closed_think():
    h = _handler(FakeEngine(prediction='<think>reasoning</think>{"a":1}'))
    text, _ = h.generate_with_meta("p")
    assert text == '{"a":1}'


def test_strip_unclosed_think():
    # truncation can cut off mid-reasoning, leaving a dangling open tag
    h = _handler(FakeEngine(prediction="<think>reasoning that got cut off"))
    text, _ = h.generate_with_meta("p")
    assert text == ""


def test_per_call_max_tokens_override_and_restore():
    eng = FakeEngine()
    h = _handler(eng)
    h.generate_with_meta("p", max_completion_tokens=32768)
    assert eng.seen_max_tokens == [32768]                     # engine saw the bumped budget
    assert eng.parameters["max_completion_tokens"] == 16384   # restored afterwards


def test_no_override_leaves_params_untouched():
    eng = FakeEngine()
    h = _handler(eng)
    h.generate_with_meta("p")
    assert eng.seen_max_tokens == [16384]
    assert eng.parameters["max_completion_tokens"] == 16384


def test_config_handler_sets_max_completion_tokens(monkeypatch):
    import auto_benchmarkcard.config as config

    captured = {}

    class CapHandler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # get_llm_handler imports LLMHandler from the llm_handler module at call time
    monkeypatch.setattr("auto_benchmarkcard.llm_handler.LLMHandler", CapHandler)
    config._llm_cache.clear()
    config.get_llm_handler("some-model")
    assert captured["parameters"].get("max_completion_tokens") == 16384
    config._llm_cache.clear()


class _TokenOut:
    def __init__(self, prediction="{}", stop_reason="stop", input_tokens=100, output_tokens=20):
        self.prediction = prediction
        self.stop_reason = stop_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _EngineReturning:
    """Fake engine that returns a fixed result object from generate()."""

    def __init__(self, out):
        self._out = out
        self.parameters = {"temperature": 0.15, "max_completion_tokens": 16384}

    def generate(self, prompts, response_format=None, verbose=False):
        return [self._out]


def _usage_handler(engine):
    h = LLMHandler.__new__(LLMHandler)  # bypass __init__ (no real engine / creds)
    h.engine = engine
    h.verbose = False
    h.model_name = "test-model"
    h.engine_type = "FAKE"
    return h


def test_usage_log_records_line_then_clear_stops(tmp_path):
    log = tmp_path / "usage.jsonl"
    h = _usage_handler(_EngineReturning(_TokenOut(input_tokens=100, output_tokens=20, stop_reason="stop")))
    set_usage_log_path(str(log))
    try:
        h.generate_with_meta("p")
        lines = log.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["model"] == "test-model"
        assert rec["engine"] == "FAKE"
        assert rec["method"] == "generate_with_meta"
        assert rec["input_tokens"] == 100
        assert rec["output_tokens"] == 20
        assert rec["stop_reason"] == "stop"
        assert isinstance(rec["wall_s"], float)
    finally:
        clear_usage_log_path()
    # after clearing the sink, a second call appends nothing
    h.generate_with_meta("p")
    assert len(log.read_text().splitlines()) == 1


def test_usage_log_writes_nulls_for_missing_token_attrs(tmp_path):
    log = tmp_path / "usage.jsonl"
    # _Out has no input_tokens / output_tokens attributes
    h = _usage_handler(_EngineReturning(_Out('{"a": 1}', stop_reason="stop")))
    set_usage_log_path(str(log))
    try:
        text, stop_reason = h.generate_with_meta("p")
        assert text == '{"a": 1}'
        assert stop_reason == "stop"
        rec = json.loads(log.read_text().splitlines()[0])
        assert rec["input_tokens"] is None
        assert rec["output_tokens"] is None
    finally:
        clear_usage_log_path()


def test_usage_log_fail_soft_on_unwritable_path(tmp_path):
    # pointing the sink at a directory makes open(..., "a") raise; must not propagate
    h = _usage_handler(_EngineReturning(_TokenOut()))
    set_usage_log_path(str(tmp_path))
    try:
        text, stop_reason = h.generate_with_meta("p")
        assert text == "{}"
        assert stop_reason == "stop"
    finally:
        clear_usage_log_path()
