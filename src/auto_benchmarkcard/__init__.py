"""Auto-BenchmarkCard: generate validated documentation cards for AI benchmarks.

Public helpers resolve to their original objects on first access. Configuration
stays eager to preserve its existing import-time environment snapshot; importing
a schema or sanitizer need not initialize the workflow and every optional tool.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

__version__ = "0.3.0.dev0"
__author__ = "Aris Hofmann"

from auto_benchmarkcard.config import Config

if _TYPE_CHECKING:
    from auto_benchmarkcard.state import GraphState
    from auto_benchmarkcard.output import OutputManager, sanitize_benchmark_name
    from auto_benchmarkcard.card_utils import extract_card, extract_missing_fields
    from auto_benchmarkcard.workflow import build_workflow
    from auto_benchmarkcard.eee_workflow import run_eee_pipeline, process_single_benchmark
    from auto_benchmarkcard.tools.unitxt import UnitxtMetadata, unitxt_benchmark_lookup
    from auto_benchmarkcard.tools.hf import hf_dataset_metadata
    from auto_benchmarkcard.tools.extractor import extract_ids
    from auto_benchmarkcard.tools.rag import RAGRetriever, MetadataIndexer, atomize_benchmark_card
    from auto_benchmarkcard.tools.composer import compose_benchmark_card
    from auto_benchmarkcard.tools.docling import extract_paper_with_docling

__all__ = [
    "Config",
    "GraphState",
    "build_workflow",
    "OutputManager",
    "sanitize_benchmark_name",
    "extract_card",
    "extract_missing_fields",
    "run_eee_pipeline",
    "process_single_benchmark",
    "unitxt_benchmark_lookup",
    "UnitxtMetadata",
    "hf_dataset_metadata",
    "extract_ids",
    "RAGRetriever",
    "MetadataIndexer",
    "atomize_benchmark_card",
    "compose_benchmark_card",
    "extract_paper_with_docling",
]

_LAZY_EXPORTS = {
    "GraphState": ("auto_benchmarkcard.state", "GraphState"),
    "build_workflow": ("auto_benchmarkcard.workflow", "build_workflow"),
    "OutputManager": ("auto_benchmarkcard.output", "OutputManager"),
    "sanitize_benchmark_name": ("auto_benchmarkcard.output", "sanitize_benchmark_name"),
    "extract_card": ("auto_benchmarkcard.card_utils", "extract_card"),
    "extract_missing_fields": ("auto_benchmarkcard.card_utils", "extract_missing_fields"),
    "run_eee_pipeline": ("auto_benchmarkcard.eee_workflow", "run_eee_pipeline"),
    "process_single_benchmark": ("auto_benchmarkcard.eee_workflow", "process_single_benchmark"),
    "unitxt_benchmark_lookup": ("auto_benchmarkcard.tools.unitxt", "unitxt_benchmark_lookup"),
    "UnitxtMetadata": ("auto_benchmarkcard.tools.unitxt", "UnitxtMetadata"),
    "hf_dataset_metadata": ("auto_benchmarkcard.tools.hf", "hf_dataset_metadata"),
    "extract_ids": ("auto_benchmarkcard.tools.extractor", "extract_ids"),
    "RAGRetriever": ("auto_benchmarkcard.tools.rag", "RAGRetriever"),
    "MetadataIndexer": ("auto_benchmarkcard.tools.rag", "MetadataIndexer"),
    "atomize_benchmark_card": ("auto_benchmarkcard.tools.rag", "atomize_benchmark_card"),
    "compose_benchmark_card": ("auto_benchmarkcard.tools.composer", "compose_benchmark_card"),
    "extract_paper_with_docling": ("auto_benchmarkcard.tools.docling", "extract_paper_with_docling"),
}

# Reload should resolve fresh exports from their modules, as the eager facade
# did, rather than retain an object cached before a child module was reloaded.
for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
