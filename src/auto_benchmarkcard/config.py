"""Configuration management for benchmark metadata processing."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for the benchmark processing pipeline."""

    # src/auto_benchmarkcard/config.py -> project root is two levels up
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    EXTERNAL_DIR: Path = PROJECT_ROOT / "external"
    FACTREASONER_DIR: Path = EXTERNAL_DIR / "FactReasoner"
    MERLIN_BIN: Path = EXTERNAL_DIR / "merlin" / "bin" / "merlin"

    # LLM engine and model tiers
    LLM_ENGINE_TYPE: str = os.getenv("LLM_ENGINE_TYPE", "hf")

    COMPOSER_MODEL: str = (
        os.getenv(f"{LLM_ENGINE_TYPE.upper()}_COMPOSER_MODEL")
        or os.getenv("COMPOSER_MODEL")
        or "deepseek-ai/DeepSeek-V3.1"
    )
    FACTREASONER_MODEL: str = os.getenv(
        "FACTREASONER_MODEL", "llama-3.3-70b-instruct"
    )
    DEFAULT_EMBEDDING_MODEL: str = "bge-large"

    # Processing
    DEFAULT_FACTUALITY_THRESHOLD: float = 0.8
    DEFAULT_TOP_K: int = 4

    # Max chars of paper text sent to the LLM (25K chars ~ 6K tokens)
    PAPER_EXTRACTION_BUDGET: int = 25000
    PAPER_INTRO_CHARS: int = 4000

    # RAG
    ENABLE_LLM_RERANKING: bool = True
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_QUERY_EXPANSION: bool = True

    # Chunking
    PARENT_CHUNK_SIZE: int = 2048
    CHILD_CHUNK_SIZE: int = 512

    FACTREASONER_CACHE_DIR: str = "factreasoner_cache"

    JSON_EXTENSION: str = ".json"
    JSONL_EXTENSION: str = ".jsonl"
    TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M"

    TOOL_OUTPUT_DIR: str = "tool_output"
    BENCHMARK_CARD_DIR: str = "benchmarkcard"
    OUTPUT_DIR: str = "output"

    @classmethod
    def get_env_var(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default."""
        return os.getenv(key, default)

    _ENGINE_REQUIRED_VARS: dict = {
        "rits": ["RITS_API_KEY", "RITS_API_URL"],
        "hf": ["HF_TOKEN"],
        "ollama": ["OLLAMA_API_URL"],
        "vllm": ["VLLM_API_URL"],
        "wml": ["WML_API_URL"],
    }

    @classmethod
    def validate_config(cls) -> None:
        """Validate that required environment variables are set for the active engine."""
        engine = cls.LLM_ENGINE_TYPE.lower()
        required_env_vars = cls._ENGINE_REQUIRED_VARS.get(engine, [])

        missing_vars = [var for var in required_env_vars if not cls.get_env_var(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables for engine '{engine}': "
                f"{', '.join(missing_vars)}"
            )


_llm_cache: dict = {}


def get_llm_handler(model_name: Optional[str] = None):
    """Get or create a cached LLM handler for the given model."""
    import logging

    from auto_benchmarkcard.llm_handler import LLMHandler

    _log = logging.getLogger(__name__)

    key = model_name or Config.COMPOSER_MODEL
    if key not in _llm_cache:
        try:
            _log.info("Initializing LLM handler: %s", key)
            _llm_cache[key] = LLMHandler(
                engine_type=Config.LLM_ENGINE_TYPE,
                model_name=key,
                parameters={"temperature": 0.15},
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM handler ({key}): {e}") from e
    return _llm_cache[key]



