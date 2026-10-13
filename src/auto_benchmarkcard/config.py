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
    MERLIN_BIN: Path = Path(os.getenv("MERLIN_BIN", str(EXTERNAL_DIR / "merlin" / "bin" / "merlin")))

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

    # Max chars of paper text sent to the LLM (25K chars ~ 6K tokens)
    PAPER_EXTRACTION_BUDGET: int = 25000
    PAPER_INTRO_CHARS: int = 4000
    # Max chars of ranked retrieved excerpts shown per gap-fill sweep
    GAP_EXCERPT_BUDGET: int = 12000

    # RAG
    ENABLE_LLM_RERANKING: bool = True
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_QUERY_EXPANSION: bool = True

    # Resolve-time HF dataset-match verification, the counterpart of the paper-binding
    # verifier. When ON, run_hf verifies the mechanically bound repo before trusting it
    # (3-tier gate, LLM only for the ambiguous band) and rejects a confidently wrong match
    # to None. OFF keeps the unverified binding.
    HF_VERIFICATION_ENABLED: bool = os.getenv("HF_VERIFICATION_ENABLED", "true").lower() == "true"

    # Paper-binding subject-coherence verification (the mirror of the HF verifier). When ON,
    # run_paper_resolver verifies a paper_url bound by a NON-resolver path (an EEE/UnitXT pre-set
    # or an HF arxiv-tag extraction) through the same LLM verifier resolve_paper already uses, and
    # drops a confidently wrong binding so the search path can resolve the paper instead.
    # OFF keeps the unverified binding.
    PAPER_VERIFICATION_ENABLED: bool = os.getenv("PAPER_VERIFICATION_ENABLED", "true").lower() == "true"

    FACTREASONER_CACHE_DIR: str = "factreasoner_cache"

    JSON_EXTENSION: str = ".json"
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
        "openrouter": ["OPENROUTER_API_KEY"],
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


def composer_serving_engine() -> str:
    """Engine for composer-model calls: COMPOSER_ENGINE_TYPE when set, else the default.

    Every component that runs the composer model (Stage A/B, frame, EAV, the atomizer)
    must construct its engine through this, otherwise its calls stay on the base engine
    when the composer has been rerouted to another provider.
    """
    return os.getenv("COMPOSER_ENGINE_TYPE") or Config.LLM_ENGINE_TYPE


def get_llm_handler(model_name: Optional[str] = None):
    """Get or create a cached LLM handler for the given model."""
    import logging

    from auto_benchmarkcard.llm_handler import LLMHandler

    _log = logging.getLogger(__name__)

    key = model_name or Config.COMPOSER_MODEL
    if key not in _llm_cache:
        # COMPOSER_ENGINE_TYPE reroutes composer-model calls only; FactReasoner and any
        # other models keep the default engine.
        engine_type = (composer_serving_engine() if key == Config.COMPOSER_MODEL
                       else Config.LLM_ENGINE_TYPE)
        try:
            _log.info("Initializing LLM handler: %s (engine=%s)", key, engine_type)
            _llm_cache[key] = LLMHandler(
                engine_type=engine_type,
                model_name=key,
                # Generous output cap sized for the field-heaviest Stage-B group
                # (identity_purpose: benchmark_details + purpose_and_intended_users, 13
                # fields) plus DeepSeek <think> tokens. Without it the server default
                # truncates the group output and voids the whole group.
                parameters={"temperature": 0.15, "max_completion_tokens": 16384},
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM handler ({key}): {e}") from e
    return _llm_cache[key]



