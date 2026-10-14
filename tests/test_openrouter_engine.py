"""OpenRouterEngine: OpenAI-compatible serving substitute for the RITS engine.

The engine must mirror the ai-atlas-nexus surface (list outputs with prediction /
stop_reason / token counts), pin the provider with fallbacks disabled, map the pipeline's
max_completion_tokens onto OpenRouter's max_tokens, and carry serving/provider/cost into
the usage telemetry. get_llm_handler reroutes ONLY the composer model when
COMPOSER_ENGINE_TYPE is set; other models keep the default engine.
"""

import json

import pytest

import auto_benchmarkcard.llm_handler as lh
from auto_benchmarkcard.llm_handler import LLMHandler, OpenRouterEngine


class _FakeUsage:
    prompt_tokens = 120
    completion_tokens = 40
    cost = 0.000123
    model_extra = {}


class _FakeMessage:
    content = "hello <think>x</think>world"


class _FakeChoice:
    message = _FakeMessage()
    finish_reason = "stop"


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()
    provider = "novita"
    model_extra = {"provider": "novita"}


class _FakeCompletions:
    def __init__(self):
        self.calls = []
        self.fail_first_with_schema = False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first_with_schema and "response_format" in kwargs:
            import httpx
            from openai import BadRequestError
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = httpx.Response(400, request=req, text="schema unsupported")
            raise BadRequestError("bad request", response=resp, body=None)
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()

        class _Chat:
            pass

        self.chat = _Chat()
        self.chat.completions = self.completions


@pytest.fixture()
def engine(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "novita")
    monkeypatch.setenv("OPENROUTER_QUANTIZATION", "fp8")
    eng = OpenRouterEngine.__new__(OpenRouterEngine)
    eng.model_name = "deepseek-ai/DeepSeek-V4-Flash"
    eng.model_id = "deepseek/deepseek-v4-flash"
    eng.parameters = {"temperature": 0.15, "max_completion_tokens": 16384}
    eng.provider_order = ["Baidu"]
    eng.quantization = "fp8"
    eng.client = _FakeClient()
    return eng


def test_output_mapping_and_provider_pin(engine):
    outs = engine.generate(["hi"], response_format={"type": "object"})
    out = outs[0]
    assert out.prediction.startswith("hello")
    assert out.stop_reason == "stop"
    assert out.input_tokens == 120 and out.output_tokens == 40
    assert out.provider == "novita" and out.serving == "openrouter"
    assert out.quantization == "fp8"
    assert out.cost == pytest.approx(0.000123)

    call = engine.client.completions.calls[0]
    assert call["model"] == "deepseek/deepseek-v4-flash"
    assert call["extra_body"]["provider"] == {"order": ["Baidu"], "allow_fallbacks": False}
    assert call["extra_body"]["usage"] == {"include": True}
    # pipeline param mapped onto OpenRouter's max_tokens; temperature passes through
    assert call["max_tokens"] == 16384 and "max_completion_tokens" not in call
    assert call["temperature"] == 0.15
    assert call["response_format"]["type"] == "json_schema"
    assert out.response_format_mode == "json_schema"


def test_postprocessors_kwarg_applies_json_object(engine):
    # The nexus risk detector calls engine.generate(..., postprocessors=["json_object"]).
    engine.client.completions.calls.clear()

    class _JsonMessage:
        content = '{"risks": ["r1"]}'

    class _JsonChoice:
        message = _JsonMessage()
        finish_reason = "stop"

    class _JsonResponse(_FakeResponse):
        choices = [_JsonChoice()]

    engine.client.completions.create = lambda **kw: _JsonResponse()
    outs = engine.generate(["p"], postprocessors=["json_object"], verbose=True)
    assert outs[0].prediction == {"risks": ["r1"]}


def test_schema_errors_propagate_no_silent_downgrade(engine):
    # A provider that rejects json_schema must report the error rather than
    # silently degrading to unconstrained decoding.
    from openai import BadRequestError

    engine.client.completions.fail_first_with_schema = True
    with pytest.raises(BadRequestError):
        engine.generate(["hi"], response_format={"type": "object"})
    assert len(engine.client.completions.calls) == 1


def test_handler_integration_records_usage(engine, monkeypatch, tmp_path):
    handler = LLMHandler.__new__(LLMHandler)
    handler.engine_type = "OPENROUTER"
    handler.model_name = "deepseek-ai/DeepSeek-V4-Flash"
    handler.verbose = False
    handler.engine = engine

    log = tmp_path / "usage.jsonl"
    lh.set_usage_log_path(str(log))
    try:
        text, stop = handler.generate_with_meta("p", max_completion_tokens=2048)
    finally:
        lh.clear_usage_log_path()
    assert text == "hello world" and stop == "stop"  # think-tokens stripped
    # per-call override restored on the engine parameters
    assert engine.parameters["max_completion_tokens"] == 16384
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["serving"] == "openrouter" and rec["provider"] == "novita"
    assert rec["quantization"] == "fp8" and rec["cost"] == pytest.approx(0.000123)
    # the per-call override reached the wire as max_tokens
    assert engine.client.completions.calls[0]["max_tokens"] == 2048


def _rate_limit_error():
    import httpx
    from openai import RateLimitError

    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(429, request=req, text="rate limited")
    return RateLimitError("rate limited", response=resp, body=None)


def test_429_storm_degrades_to_slow_not_zero(engine, monkeypatch):
    # Failure class: a 429 burst turned into zero-emission cards because retries
    # exhausted in seconds. A burst must instead slow the run down and then succeed.
    monkeypatch.setattr(OpenRouterEngine, "_cooldown_until", 0.0)
    sleeps = []
    monkeypatch.setattr(lh.time, "sleep", sleeps.append)
    state = {"fails": 2}

    def create(**kwargs):
        engine.client.completions.calls.append(kwargs)
        if state["fails"] > 0:
            state["fails"] -= 1
            raise _rate_limit_error()
        return _FakeResponse()

    engine.client.completions.create = create
    outs = engine.generate(["hi"])
    assert outs[0].prediction.startswith("hello")
    assert len(engine.client.completions.calls) == 3
    # the two cooldown waits follow the schedule (5s then 10s, minus epsilon clock drift)
    assert len(sleeps) == 2
    assert 4.0 < sleeps[0] <= 5.0 and 9.0 < sleeps[1] <= 10.0
    monkeypatch.setattr(OpenRouterEngine, "_cooldown_until", 0.0)


def test_429_storm_exhausts_schedule_then_raises(engine, monkeypatch):
    from openai import RateLimitError

    monkeypatch.setattr(OpenRouterEngine, "_cooldown_until", 0.0)
    monkeypatch.setattr(lh.time, "sleep", lambda s: None)

    def create(**kwargs):
        engine.client.completions.calls.append(kwargs)
        raise _rate_limit_error()

    engine.client.completions.create = create
    with pytest.raises(RateLimitError):
        engine.generate(["hi"])
    assert len(engine.client.completions.calls) == len(lh._BACKOFF_SCHEDULE) + 1
    monkeypatch.setattr(OpenRouterEngine, "_cooldown_until", 0.0)


def test_composer_serving_engine_helper(monkeypatch):
    # Failure class: a composer-model component (the atomizer) constructed on the base
    # engine strands when a serving substitute is active; the helper is the single
    # reroute every such site must use.
    from auto_benchmarkcard import config as cfg

    monkeypatch.delenv("COMPOSER_ENGINE_TYPE", raising=False)
    assert cfg.composer_serving_engine() == cfg.Config.LLM_ENGINE_TYPE
    monkeypatch.setenv("COMPOSER_ENGINE_TYPE", "openrouter")
    assert cfg.composer_serving_engine() == "openrouter"


def test_get_llm_handler_reroutes_composer_only(monkeypatch):
    from auto_benchmarkcard import config as cfg

    created = []

    class _SpyHandler:
        def __init__(self, engine_type, model_name, parameters, verbose):
            created.append((engine_type, model_name))

    monkeypatch.setenv("COMPOSER_ENGINE_TYPE", "openrouter")
    monkeypatch.setattr(lh, "LLMHandler", _SpyHandler)
    monkeypatch.setattr(cfg, "_llm_cache", {})
    cfg.get_llm_handler(cfg.Config.COMPOSER_MODEL)
    cfg.get_llm_handler("meta-llama/Llama-3.3-70B-Instruct")
    assert created[0][0] == "openrouter"
    assert created[1][0] == cfg.Config.LLM_ENGINE_TYPE
