"""Benchmark metadata extraction pipeline built on LangGraph.

Orchestrates worker nodes through a conditional state machine:
  UnitXT → Extractor → HF → Docling → Composer → Risk → RAG → FactReasoner
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from auto_benchmarkcard.logging_setup import setup_logging_suppression
from auto_benchmarkcard.state import BenchmarkProcessingError, GraphState
from auto_benchmarkcard.output import OutputManager, sanitize_benchmark_name
from auto_benchmarkcard.card_utils import extract_card, extract_missing_fields
from auto_benchmarkcard.workers import (
    handle_error,
    run_unitxt,
    run_extractor,
    run_hf_extractor,
    run_docling,
    run_hf,
    run_composer,
    run_risk_identification,
    run_rag,
    run_factreasoner,
)

logger = logging.getLogger(__name__)

__all__ = [
    "setup_logging_suppression",
    "GraphState",
    "BenchmarkProcessingError",
    "OutputManager",
    "sanitize_benchmark_name",
    "build_workflow",
    "extract_card",
    "extract_missing_fields",
]


def orchestrator(state: GraphState) -> Dict[str, str]:
    """Determine the next workflow step based on current state."""
    completed = state.get("completed", [])
    completed_str = " ".join(completed)

    def _failed(step: str) -> bool:
        return f"{step} failed" in completed_str

    # EEE path bypasses UnitXT and extractor
    eee_metadata = state.get("eee_metadata")
    is_eee = eee_metadata is not None
    is_composite = is_eee and eee_metadata.get("benchmark_type") == "composite"

    if not is_eee:
        if state["unitxt_json"] is None and not _failed("unitxt"):
            return {"next": "unitxt_worker"}
        if state["extracted_ids"] is None and not _failed("id extraction"):
            return {"next": "extractor_worker"}

    if state["hf_repo"] is not None and state["hf_json"] is None and not _failed("huggingface"):
        return {"next": "hf_worker"}

    # Fall back to HF metadata for paper URL if UnitXT didn't provide one
    current_paper_url = state.get("extracted_ids", {}).get("paper_url")
    has_hf_data = state.get("hf_json") is not None
    hf_extraction_attempted = state.get("hf_extraction_attempted", False)
    needs_hf_extraction = not current_paper_url and has_hf_data and not hf_extraction_attempted

    if needs_hf_extraction:
        return {"next": "hf_extractor_worker"}

    paper_url = state.get("extracted_ids", {}).get("paper_url")
    if paper_url and state["docling_output"] is None and not _failed("docling"):
        return {"next": "docling_worker"}
    if state["composed_card"] is None and not _failed("composer"):
        return {"next": "composer_worker"}
    if state["risk_enhanced_card"] is None and not _failed("risk"):
        return {"next": "risk_worker"}

    # Composites have no source docs for RAG/FactReasoner — skip directly to END
    if is_composite:
        return {"next": "END"}

    if state["rag_results"] is None and not _failed("rag"):
        return {"next": "rag_worker"}
    if state["factuality_results"] is None and not _failed("factreasoner"):
        return {"next": "factreasoner_worker"}
    return {"next": "END"}


def build_workflow():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(GraphState)

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("unitxt_worker", run_unitxt)
    builder.add_node("extractor_worker", run_extractor)
    builder.add_node("hf_extractor_worker", run_hf_extractor)
    builder.add_node("docling_worker", run_docling)
    builder.add_node("hf_worker", run_hf)
    builder.add_node("composer_worker", run_composer)
    builder.add_node("risk_worker", run_risk_identification)
    builder.add_node("rag_worker", run_rag)
    builder.add_node("factreasoner_worker", run_factreasoner)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        lambda s, *_: s["next"],
        {
            "unitxt_worker": "unitxt_worker",
            "extractor_worker": "extractor_worker",
            "hf_extractor_worker": "hf_extractor_worker",
            "docling_worker": "docling_worker",
            "hf_worker": "hf_worker",
            "composer_worker": "composer_worker",
            "risk_worker": "risk_worker",
            "rag_worker": "rag_worker",
            "factreasoner_worker": "factreasoner_worker",
            "END": END,
        },
    )
    builder.add_edge("unitxt_worker", "orchestrator")
    builder.add_edge("extractor_worker", "orchestrator")
    builder.add_edge("hf_extractor_worker", "orchestrator")
    builder.add_edge("docling_worker", "orchestrator")
    builder.add_edge("hf_worker", "orchestrator")
    builder.add_edge("composer_worker", "orchestrator")
    builder.add_edge("risk_worker", "orchestrator")
    builder.add_edge("rag_worker", "orchestrator")
    builder.add_edge("factreasoner_worker", END)

    return builder.compile()
