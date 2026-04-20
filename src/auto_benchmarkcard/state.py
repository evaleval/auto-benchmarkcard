"""Workflow state and exception definitions."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from auto_benchmarkcard.output import OutputManager


class BenchmarkProcessingError(Exception):
    """Error raised during benchmark processing with operation context."""

    def __init__(self, message: str, operation: str, original_error: Exception = None):
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class GraphState(TypedDict):
    # Input: benchmark name or UnitXT catalog ID (e.g. "mmlu" or "cards.mmlu")
    query: str
    # Optional UnitXT catalog path override
    catalog_path: Optional[str]
    # Manages per-benchmark output directories and tool artifact saving
    output_manager: OutputManager

    # Phase 1 — extraction outputs (populated by workers, consumed by composer)
    unitxt_json: Optional[Dict[str, Any]]
    extracted_ids: Optional[Dict[str, Any]]  # hf_repo, paper_url, risk_tags
    hf_repo: Optional[str]
    hf_json: Optional[Dict[str, Any]]
    docling_output: Optional[Dict[str, Any]]  # paper text from PDF
    html_content: Optional[Dict[str, Any]]  # web page text from trafilatura
    eee_metadata: Optional[Dict[str, Any]]  # pre-aggregated EEE data (bypasses UnitXT)
    hf_extraction_attempted: Optional[bool]  # prevents re-running HF paper URL extraction

    # Phase 2 — composition and enrichment
    composed_card: Optional[Dict[str, Any]]  # LLM-generated card from composer
    risk_enhanced_card: Optional[Dict[str, Any]]  # card with AI Atlas Nexus risks

    # Phase 3 — validation
    rag_results: Optional[Dict[str, Any]]  # evidence retrieved for fact-checking
    factuality_results: Optional[Dict[str, Any]]  # FactReasoner probability scores
    final_card: Optional[Dict[str, Any]]  # flagged card after fact-checking

    # LangGraph reducer: each worker appends status strings (e.g. "unitxt done",
    # "composer failed"). The orchestrator reads these to decide the next step.
    completed: Annotated[list, operator.add]
    errors: Optional[List[str]]
