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
    query: str
    catalog_path: Optional[str]
    output_manager: OutputManager
    unitxt_json: Optional[Dict[str, Any]]
    extracted_ids: Optional[Dict[str, Any]]
    hf_repo: Optional[str]
    hf_json: Optional[Dict[str, Any]]
    docling_output: Optional[Dict[str, Any]]
    completed: Annotated[list, operator.add]
    errors: Optional[List[str]]
    composed_card: Optional[Dict[str, Any]]
    risk_enhanced_card: Optional[Dict[str, Any]]
    hf_extraction_attempted: Optional[bool]
    rag_results: Optional[Dict[str, Any]]
    factuality_results: Optional[Dict[str, Any]]
    final_card: Optional[Dict[str, Any]]
    eee_metadata: Optional[Dict[str, Any]]
