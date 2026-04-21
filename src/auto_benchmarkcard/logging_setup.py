"""Logging and warning suppression for third-party libraries."""

import logging
import os
import warnings


def setup_logging_suppression(debug_mode: bool = False) -> None:
    """Configure log levels for third-party libraries and internal modules."""
    if not debug_mode:
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        os.environ["LITELLM_LOG"] = "ERROR"
        os.environ["LITELLM_DROP_PARAMS"] = "true"

        # tqdm progress bars conflict with CLI spinners
        os.environ["TQDM_DISABLE"] = "1"

        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.set_verbose = False
        except (ImportError, AttributeError):
            pass

        noisy_loggers = [
            "vllm",
            "vllm.config",
            "vllm.utils.import_utils",
            "vllm.importing",
            "transformers",
            "torch",
            "faiss.loader",
            "faiss",
            "ai_atlas_nexus",
            "AIAtlasNexus",
            "LiteLLM",
            "litellm",
            "litellm.llms",
            "litellm.utils",
            "litellm.cost_calculator",
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "asyncio",
            "fact_reasoner",
            "FactReasoner",
            "LLMHandler",
            "AtomExtractor",
            "AtomReviser",
            "NLIExtractor",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).propagate = False

        tool_loggers = [
            "auto_benchmarkcard.tools.unitxt.unitxt_tool",
            "auto_benchmarkcard.tools.hf.hf_tool",
            "auto_benchmarkcard.tools.docling.docling_tool",
            "auto_benchmarkcard.tools.composer.composer_tool",
            "auto_benchmarkcard.tools.rag.rag_retriever",
            "auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool",
            "auto_benchmarkcard.tools.factreasoner.factreasoner_tool",
        ]

        for logger_name in tool_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).propagate = False

        logging.getLogger("auto_benchmarkcard").setLevel(logging.WARNING)
    else:
        logging.getLogger("auto_benchmarkcard").setLevel(logging.DEBUG)


# Suppress noisy warnings from third-party libraries at import time.
# Centralized here so cli.py doesn't need its own copy.
warnings.filterwarnings("ignore", message=".*Triton.*")
warnings.filterwarnings("ignore", message=".*not installed.*")
warnings.filterwarnings("ignore", message=".*dummy decorators.*")
warnings.filterwarnings("ignore", message=".*Failed to load GPU.*")
warnings.filterwarnings("ignore", message=".*platform.*")
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")
warnings.filterwarnings("ignore", message=".*TOKENIZERS_PARALLELISM.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*LangChain.*deprecated.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*manual persistence.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
