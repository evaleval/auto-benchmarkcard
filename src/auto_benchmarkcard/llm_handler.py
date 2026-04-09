"""Thin wrapper around ai-atlas-nexus inference engines.

Provides structured output and LangChain integration on top of the
ai-atlas-nexus inference backends.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

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

DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
DEFAULT_RITS_MODEL = os.getenv("RITS_MODEL", "meta-llama/Llama-3.1-80B-Instruct")
DEFAULT_VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_WML_MODEL = os.getenv("WML_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


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
        else:
            raise ValueError(
                f"Unsupported engine type: {engine_type}. "
                f"Supported types: HF, OLLAMA, RITS, VLLM, WML"
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

    @staticmethod
    def _strip_think_tokens(text: str) -> str:
        """Remove DeepSeek-style <think>…</think> reasoning traces from output."""
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def generate(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Generate a text response from a single prompt."""
        result = self.engine.generate([prompt], response_format=response_format, verbose=self.verbose)
        return self._strip_think_tokens(result[0].prediction)

    def chat(
        self,
        messages: Union[List[Dict[str, str]], str],
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send conversation messages and return the model's response."""
        result = self.engine.chat(messages, response_format=response_format, verbose=self.verbose)
        return self._strip_think_tokens(result[0].prediction)

    def generate_structured(self, prompt: str, response_schema: BaseModel) -> Any:
        """Generate JSON output validated against a Pydantic schema."""
        schema = response_schema.model_json_schema()
        result = self.engine.generate([prompt], response_format=schema, verbose=self.verbose)
        raw = self._strip_think_tokens(result[0].prediction)

        try:
            parsed = json.loads(raw)
            return response_schema.model_validate(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            try:
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return response_schema.model_validate(parsed)
            except Exception:
                pass
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
                        messages = input_data["messages"]
                        if isinstance(messages, list) and len(messages) > 0:
                            if isinstance(messages[-1], dict) and "content" in messages[-1]:
                                prompt = messages[-1]["content"]
                            else:
                                prompt = str(messages[-1])
                        else:
                            prompt = str(messages)
                    else:
                        prompt = str(input_data)
                else:
                    prompt = str(input_data)

                return self.handler.generate_structured(prompt, self.schema)

        return StructuredHandler(self, schema)


def get_llm_handler(engine_type: str = "ollama", **kwargs) -> LLMHandler:
    """Factory function to create an LLM handler."""
    return LLMHandler(engine_type=engine_type, **kwargs)

