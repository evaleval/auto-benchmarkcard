"""FactReasoner's serving route is its own, and it has to reach a provider with logprobs."""

from auto_benchmarkcard.config import Config
from auto_benchmarkcard.tools.factreasoner import factreasoner_tool as T


def test_factreasoner_engine_can_differ_from_the_composer_engine(monkeypatch):
    """Failure class: factreasoner_route_bound_to_composer_engine. LLM_ENGINE_TYPE=rits in
    the .env forced the RITS branch, which needs RITS credentials that no longer exist,
    so the model-card final-claim pass could never build a backend on this machine even
    with a working OpenRouter route configured."""
    monkeypatch.setattr(Config, "LLM_ENGINE_TYPE", "rits")
    monkeypatch.delenv("FACTREASONER_ENGINE_TYPE", raising=False)
    assert T._active_factreasoner_engine() == "rits"
    monkeypatch.setenv("FACTREASONER_ENGINE_TYPE", "openai")
    assert T._active_factreasoner_engine() == "openai"


def test_the_route_pins_a_logprobs_provider_without_require_parameters(monkeypatch):
    """Failure class: logprobs_dropped_by_provider_routing. OpenRouter serves a model from
    its cheapest endpoint and most of them drop `logprobs`, so the NLI extractor's assert
    fired on the first atom. The provider is pinned by name with no fallback. It is NOT
    pinned with `require_parameters`: Mellea sends request fields the provider does not
    advertise, and with that flag OpenRouter answered 404 "Filter by Parameters"
    (2026-09-06, Parasail, then the same for any provider without `tools`)."""
    monkeypatch.delenv("FACTREASONER_PROVIDER", raising=False)
    assert T._provider_routing() is None
    monkeypatch.setenv("FACTREASONER_PROVIDER", "Novita")
    body = T._provider_routing()
    assert body == {"provider": {"order": ["Novita"], "allow_fallbacks": False}}
    assert "require_parameters" not in body["provider"]


def test_the_generic_backend_carries_the_routing_in_extra_body(monkeypatch):
    monkeypatch.setattr(Config, "LLM_ENGINE_TYPE", "rits")
    monkeypatch.setenv("FACTREASONER_ENGINE_TYPE", "openai")
    monkeypatch.setenv("FACTREASONER_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("FACTREASONER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("FACTREASONER_PROVIDER", "Novita")
    backend = T._create_factreasoner_backend("meta-llama/llama-3.3-70b-instruct")
    assert backend is not None
    options = backend.model_options
    assert options["extra_body"] == {"provider": {"order": ["Novita"], "allow_fallbacks": False}}
