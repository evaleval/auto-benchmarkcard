"""Thin wrapper around ai-atlas-nexus inference engines."""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from ai_atlas_nexus.blocks.inference import (
    HFInferenceEngine,
    OllamaInferenceEngine,
    RITSInferenceEngine,
    VLLMInferenceEngine,
    WMLInferenceEngine,
)
from ai_atlas_nexus.blocks.inference.params import (
    HFInferenceEngineParams,
    OllamaInferenceEngineParams,
    RITSInferenceEngineParams,
    VLLMInferenceEngineParams,
    WMLInferenceEngineParams,
)

load_dotenv()

_UNSET = object()

DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
DEFAULT_RITS_MODEL = os.getenv("RITS_MODEL", "meta-llama/Llama-3.1-80B-Instruct")
DEFAULT_VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_WML_MODEL = os.getenv("WML_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


# Optional sink for per-call LLM usage records. When set to a path, every LLM call
# appends one json line there. Unset means no telemetry. Writing is fail-soft.
_usage_log_path: Optional[str] = None
_usage_log_warned = False


def set_usage_log_path(path: str) -> None:
    """Route per-call LLM usage records to this jsonl file (append mode)."""
    global _usage_log_path, _usage_log_warned
    _usage_log_path = path
    _usage_log_warned = False


def clear_usage_log_path() -> None:
    """Stop recording LLM usage records."""
    global _usage_log_path
    _usage_log_path = None


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object in text, or None.

    Walks from the first '{' counting brace depth and skipping braces inside
    string literals. A non-greedy regex stops at the first '}' and so breaks on
    nested objects; this handles nesting.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _coerce_to_text(obj: Any) -> str:
    """Extract clean prompt text from a LangChain input.

    str() on a ChatPromptValue or BaseMessage yields a repr-like wrapper, not
    the prompt text. Prefer to_string()/.content, fall back to str() last.
    """
    if isinstance(obj, str):
        return obj
    to_string = getattr(obj, "to_string", None)
    if callable(to_string):
        return to_string()
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(obj, list):
        if not obj:
            return ""
        last = obj[-1]
        if isinstance(last, dict) and "content" in last:
            return last["content"]
        last_content = getattr(last, "content", None)
        if isinstance(last_content, str):
            return last_content
    return str(obj)


class OpenRouterEngine:
    """Minimal OpenAI-compatible engine for OpenRouter, mirroring the ai-atlas-nexus surface
    the handler uses: generate/chat return a list of outputs with prediction, stop_reason,
    input_tokens, output_tokens; `parameters` is a mutable dict (per-call token override);
    `client` is an openai client.

    Serving substitute for a retired RITS deployment (2026-07-04: DeepSeek-V4-Flash left
    RITS before the regen). The model IDENTITY stays the caller's model name, so
    card_info.llm is unchanged; OPENROUTER_MODEL_ID names the checkpoint on OpenRouter and
    OPENROUTER_PROVIDER_ORDER pins the serving provider(s) with fallbacks disabled, keeping
    provider and quantization constant across a run. Every output carries serving, provider,
    quantization (OPENROUTER_QUANTIZATION) and cost for the usage telemetry.
    """

    def __init__(self, model_name: str, parameters: Optional[Dict] = None):
        import httpx
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model_name = model_name
        self.model_id = os.getenv("OPENROUTER_MODEL_ID") or model_name
        self.parameters = dict(parameters or {})
        self.provider_order = [
            p.strip() for p in os.getenv("OPENROUTER_PROVIDER_ORDER", "").split(",") if p.strip()
        ]
        self.quantization = os.getenv("OPENROUTER_QUANTIZATION") or None
        # Same read-timeout/retry policy as the capped RITS client in LLMHandler.__init__.
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=httpx.Timeout(420.0, connect=10.0),
            max_retries=3,
        )

    @staticmethod
    def _schema_format(response_format):
        # Mirror the RITS engine's json_schema envelope.
        if not response_format:
            return None
        return {"type": "json_schema",
                "json_schema": {"name": "schema", "schema": response_format}}

    def _call(self, messages, response_format):
        extra_body: Dict[str, Any] = {"usage": {"include": True}}
        if self.provider_order:
            extra_body["provider"] = {"order": self.provider_order, "allow_fallbacks": False}
        kwargs = dict(self.parameters)
        # OpenRouter normalizes on max_tokens; the pipeline configures max_completion_tokens.
        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        rf = self._schema_format(response_format)
        if rf:
            kwargs["response_format"] = rf
        # Deliberately NO silent downgrade on schema errors: a provider that rejects or
        # mishandles json_schema must fail loudly (decoding parity with broad_run is part of
        # the generation regime; the pre-authorized response is a manual provider hop).
        response = self.client.chat.completions.create(
            model=self.model_id, messages=messages, extra_body=extra_body, **kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        extra = getattr(response, "model_extra", None) or {}

        class _Out:
            pass

        out = _Out()
        out.prediction = choice.message.content or ""
        out.stop_reason = choice.finish_reason
        # Honest OpenAI semantics here (unlike the RITS engine's total_tokens quirk).
        out.input_tokens = getattr(usage, "prompt_tokens", None)
        out.output_tokens = getattr(usage, "completion_tokens", None)
        out.provider = getattr(response, "provider", None) or extra.get("provider")
        out.cost = getattr(usage, "cost", None) if usage is not None else None
        if out.cost is None and usage is not None:
            out.cost = (getattr(usage, "model_extra", None) or {}).get("cost")
        out.serving = "openrouter"
        out.quantization = self.quantization
        out.response_format_mode = "json_schema" if rf else None
        return out

    @staticmethod
    def _apply_postprocessors(outs, postprocessors):
        # ai-atlas-nexus callers (risk detector) pass the raw engine around and request
        # prediction postprocessors (e.g. "json_object"); reuse the nexus registry verbatim
        # so behavior matches the native engines. Fail-soft per processor, like upstream.
        if not postprocessors:
            return outs
        from ai_atlas_nexus.blocks.inference.postprocessing import POSTPROCESSORS_REGISTRY
        for proc in postprocessors:
            try:
                for out in outs:
                    out.prediction = POSTPROCESSORS_REGISTRY[proc]().apply(out.prediction)
            except Exception:
                logger.debug("Postprocessor %s failed", proc)
        return outs

    def generate(self, prompts, response_format=None, postprocessors=None, verbose=False, **kwargs):
        outs = [self._call([{"role": "user", "content": p}], response_format) for p in prompts]
        return self._apply_postprocessors(outs, postprocessors)

    def chat(self, messages, response_format=None, postprocessors=None, verbose=False, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        outs = [self._call(messages, response_format)]
        return self._apply_postprocessors(outs, postprocessors)


class LLMHandler:
    """Unified interface to ai-atlas-nexus inference engines with structured output support."""

    def __init__(
        self,
        engine_type: str = "ollama",
        model_name: Optional[str] = None,
        credentials: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
        verbose: bool = False,
        **kwargs
    ):
        self.engine_type = engine_type.upper()
        self.verbose = verbose

        if self.engine_type == "HF":
            self.model_name = model_name or DEFAULT_HF_MODEL
            params_class = HFInferenceEngineParams
            engine_class = HFInferenceEngine
        elif self.engine_type == "OLLAMA":
            self.model_name = model_name or DEFAULT_OLLAMA_MODEL
            params_class = OllamaInferenceEngineParams
            engine_class = OllamaInferenceEngine
        elif self.engine_type == "RITS":
            self.model_name = model_name or DEFAULT_RITS_MODEL
            params_class = RITSInferenceEngineParams
            engine_class = RITSInferenceEngine
        elif self.engine_type == "VLLM":
            self.model_name = model_name or DEFAULT_VLLM_MODEL
            params_class = VLLMInferenceEngineParams
            engine_class = VLLMInferenceEngine
        elif self.engine_type == "WML":
            self.model_name = model_name or DEFAULT_WML_MODEL
            params_class = WMLInferenceEngineParams
            engine_class = WMLInferenceEngine
        elif self.engine_type == "OPENROUTER":
            # OpenAI-compatible serving substitute; no ai-atlas-nexus engine involved and
            # the client is already timeout-capped, so construction ends here.
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL_ID", "")
            self.engine = OpenRouterEngine(
                self.model_name,
                parameters if isinstance(parameters, dict) else None,
            )
            return
        else:
            raise ValueError(
                f"Unsupported engine type: {engine_type}. "
                f"Supported types: HF, OLLAMA, RITS, VLLM, WML, OPENROUTER"
            )

        if parameters:
            params = params_class(parameters) if isinstance(parameters, dict) else parameters
        else:
            params = params_class()

        self.engine = engine_class(
            model_name_or_path=self.model_name,
            credentials=credentials,
            parameters=params,
            **kwargs
        )

        # The RITS/openai client is built with no read timeout (httpx.Timeout(None, connect=1.0)
        # in ai_atlas_nexus), so a slow or stuck response hangs the worker forever instead of
        # failing. Cap the read timeout and allow retries so a hung request recovers on its own.
        # Best-effort: skip engines without an openai-style client.
        try:
            import httpx
            client = getattr(self.engine, "client", None)
            if client is not None and hasattr(client, "with_options"):
                self.engine.client = client.with_options(
                    timeout=httpx.Timeout(420.0, connect=10.0), max_retries=3
                )
        except Exception as exc:
            logger.warning("Could not cap LLM client timeout: %s", exc)

    @staticmethod
    def _strip_think_tokens(text: str) -> str:
        """Remove DeepSeek-style <think>…</think> reasoning traces from output.

        Also drop an unclosed <think> (no closing tag): a length-truncated response can
        cut off mid-reasoning, leaving a dangling open tag that would otherwise survive.
        """
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    def _record_usage(self, out: Any, method: str, wall: float) -> None:
        """Append one usage record for this call to the configured sink. Fail-soft.

        For the RITS engine input_tokens is populated from usage.total_tokens upstream
        (a quirk in ai-atlas-nexus); it is recorded raw without adjustment.
        """
        path = _usage_log_path
        if path is None:
            return
        global _usage_log_warned
        try:
            record = {
                "ts": time.time(),
                "model": self.model_name,
                "engine": self.engine_type,
                "method": method,
                "input_tokens": getattr(out, "input_tokens", None),
                "output_tokens": getattr(out, "output_tokens", None),
                "stop_reason": getattr(out, "stop_reason", None),
                "wall_s": round(wall, 3),
                # Serving substitution fields (OpenRouter); None on native engines.
                "serving": getattr(out, "serving", None),
                "provider": getattr(out, "provider", None),
                "quantization": getattr(out, "quantization", None),
                "response_format": getattr(out, "response_format_mode", None),
                "cost": getattr(out, "cost", None),
            }
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            if not _usage_log_warned:
                logger.warning("Could not write LLM usage log to %s: %s", path, exc)
                _usage_log_warned = True

    def generate(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Generate a text response from a single prompt."""
        t0 = time.monotonic()
        result = self.engine.generate([prompt], response_format=response_format, verbose=self.verbose)
        self._record_usage(result[0], "generate", time.monotonic() - t0)
        return self._strip_think_tokens(result[0].prediction)

    def generate_with_meta(self, prompt: str, response_format: Optional[Dict] = None,
                           max_completion_tokens: Optional[int] = None):
        """Like generate(), but returns (text, stop_reason) and allows a per-call token cap.

        stop_reason is the engine finish_reason ("length" when the output was truncated by
        the token cap). The optional max_completion_tokens temporarily overrides the engine
        parameter for this one call (Stage-B runs sequentially, so the mutate/restore is safe);
        the engine has no per-call override hook and lives in the separate ai-atlas-nexus repo.
        """
        params = getattr(self.engine, "parameters", None)
        override = max_completion_tokens is not None and isinstance(params, dict)
        prev = _UNSET
        if override:
            prev = params.get("max_completion_tokens", _UNSET)
            params["max_completion_tokens"] = max_completion_tokens
        try:
            t0 = time.monotonic()
            result = self.engine.generate([prompt], response_format=response_format, verbose=self.verbose)
            wall = time.monotonic() - t0
        finally:
            if override:
                if prev is _UNSET:
                    params.pop("max_completion_tokens", None)
                else:
                    params["max_completion_tokens"] = prev
        out = result[0]
        self._record_usage(out, "generate_with_meta", wall)
        return self._strip_think_tokens(out.prediction), getattr(out, "stop_reason", None)

    def chat(
        self,
        messages: Union[List[Dict[str, str]], str],
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send conversation messages and return the model's response."""
        t0 = time.monotonic()
        result = self.engine.chat(messages, response_format=response_format, verbose=self.verbose)
        self._record_usage(result[0], "chat", time.monotonic() - t0)
        return self._strip_think_tokens(result[0].prediction)

    def generate_structured(self, prompt: str, response_schema: BaseModel) -> Any:
        """Generate JSON output validated against a Pydantic schema."""
        schema = response_schema.model_json_schema()
        t0 = time.monotonic()
        result = self.engine.generate([prompt], response_format=schema, verbose=self.verbose)
        self._record_usage(result[0], "generate_structured", time.monotonic() - t0)
        raw = self._strip_think_tokens(result[0].prediction)

        try:
            parsed = json.loads(raw)
            return response_schema.model_validate(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            try:
                json_str = _extract_json_object(raw)
                if json_str:
                    parsed = json.loads(json_str)
                    return response_schema.model_validate(parsed)
            except (json.JSONDecodeError, ValueError):
                logger.debug("JSON fallback extraction also failed for: %.200s", raw)
            raise ValueError(f"Failed to parse structured output: {e}")

    def with_structured_output(self, schema: BaseModel):
        """Return a LangChain Runnable that validates output against a Pydantic schema."""

        class StructuredHandler(Runnable):

            def __init__(self, handler, schema):
                self.handler = handler
                self.schema = schema

            def invoke(self, input_data, config=None) -> Any:
                if isinstance(input_data, dict):
                    if "text" in input_data:
                        prompt = input_data["text"]
                    elif "query" in input_data:
                        prompt = input_data["query"]
                    elif "messages" in input_data:
                        prompt = _coerce_to_text(input_data["messages"])
                    else:
                        prompt = _coerce_to_text(input_data)
                else:
                    prompt = _coerce_to_text(input_data)

                return self.handler.generate_structured(prompt, self.schema)

        return StructuredHandler(self, schema)


def get_llm_handler(engine_type: str = "ollama", **kwargs) -> LLMHandler:
    """Factory function to create an LLM handler."""
    return LLMHandler(engine_type=engine_type, **kwargs)

