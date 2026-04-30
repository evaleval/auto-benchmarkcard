"""Auto-BenchmarkCard: generate validated documentation cards for AI benchmarks."""

__version__ = "0.1.0"
__author__ = "Aris Hofmann"

from auto_benchmarkcard.config import Config
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
