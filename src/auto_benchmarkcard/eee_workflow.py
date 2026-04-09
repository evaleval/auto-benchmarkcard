"""EEE-to-BenchmarkCard workflow.

Alternative pipeline entry point that starts from Every Eval Ever (EEE)
evaluation data instead of UnitXT. Scans EEE evaluation JSONs, resolves
HuggingFace repos, then feeds into the standard composition pipeline.

Flow:
  EEE data → scan & aggregate → resolve HF repos
    → [HF Worker] → [Docling Worker] → [Composer Worker]
    → [Risk Worker] → [RAG Worker] → [FactReasoner Worker]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from auto_benchmarkcard.config import Config
from auto_benchmarkcard.tools.eee.eee_tool import (
    scan_eee_folder,
    eee_to_pipeline_inputs,
    lookup_unitxt_paper,
)
from auto_benchmarkcard.workflow import (
    GraphState,
    OutputManager,
    build_workflow,
    sanitize_benchmark_name,
    setup_logging_suppression,
)

logger = logging.getLogger(__name__)


def build_eee_initial_state(
    benchmark_name: str,
    pipeline_inputs: Dict[str, Any],
    output_manager: OutputManager,
) -> Dict[str, Any]:
    """Build initial workflow state from EEE pipeline inputs.

    Pre-populates extracted_ids, hf_repo, and eee_metadata so the orchestrator
    skips UnitXT + extractor steps automatically.
    """
    return {
        "query": benchmark_name,
        "catalog_path": None,
        "output_manager": output_manager,
        "unitxt_json": None,
        "extracted_ids": pipeline_inputs["extracted_ids"],
        "hf_repo": pipeline_inputs["hf_repo"],
        "hf_json": None,
        "docling_output": None,
        "composed_card": None,
        "risk_enhanced_card": None,
        "completed": ["eee_scan done", "eee_resolve done"],
        "errors": [],
        "hf_extraction_attempted": False,
        "rag_results": None,
        "factuality_results": None,
        "eee_metadata": pipeline_inputs["eee_metadata"],
    }


_CARD_FIELD_ORDER = [
    "benchmark_details",
    "purpose_and_intended_users",
    "data",
    "methodology",
    "ethical_and_legal_considerations",
    "possible_risks",
    "flagged_fields",
    "missing_fields",
    "card_info",
]


def _reorder_card_fields(card: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder card fields to match canonical schema order."""
    ordered = {}
    for key in _CARD_FIELD_ORDER:
        if key in card:
            ordered[key] = card[key]
    for key in card:
        if key not in ordered:
            ordered[key] = card[key]
    return ordered


def _enrich_baseline_results(final_card: Dict[str, Any], eee_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in baseline_results from EEE evaluation data when missing."""
    eval_summary = eee_metadata.get("evaluation_summary", {})
    if not eval_summary:
        return final_card

    card = final_card.get("benchmark_card", final_card)

    methodology = card.get("methodology", {})
    baseline = methodology.get("baseline_results", "")
    if not baseline or baseline.lower() in ("not specified", "not specified."):
        top = eval_summary.get("top_performers", [])
        stats = eval_summary.get("score_statistics", {})
        metric = eval_summary.get("primary_metric", "score")
        n_models = eval_summary.get("total_models_evaluated", 0)

        if top and stats:
            top_str = ", ".join(
                f"{p['model']} ({p['score']:.4f})" for p in top[:3]
            )
            methodology["baseline_results"] = (
                f"Based on {n_models} model evaluations from Every Eval Ever: "
                f"mean {metric} = {stats['mean']:.4f} (std = {stats['std_dev']:.4f}). "
                f"Top performers: {top_str}."
            )
            card["methodology"] = methodology

    if "benchmark_card" in final_card:
        final_card["benchmark_card"] = card
    else:
        final_card = card

    return final_card


def process_single_benchmark(
    benchmark_name: str,
    pipeline_inputs: Dict[str, Any],
    base_output_path: Optional[str] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the full pipeline for a single EEE benchmark."""
    safe_name = sanitize_benchmark_name(benchmark_name)
    output_manager = OutputManager(safe_name, base_output_path)

    eee_metadata = pipeline_inputs.get("eee_metadata", {})
    output_manager.save_tool_output(eee_metadata, "eee", f"{safe_name}.json")

    hf_repo = pipeline_inputs.get("hf_repo")
    extracted_ids = pipeline_inputs.get("extracted_ids", {})
    if not extracted_ids.get("paper_url") and hf_repo:
        unitxt_paper = lookup_unitxt_paper(hf_repo)
        if unitxt_paper:
            extracted_ids["paper_url"] = unitxt_paper

    initial_state = build_eee_initial_state(benchmark_name, pipeline_inputs, output_manager)
    workflow = build_workflow()

    logger.info("Processing benchmark: %s (hf_repo=%s)", benchmark_name, pipeline_inputs.get("hf_repo"))

    try:
        final_state = workflow.invoke(initial_state)

        final_card = final_state.get("final_card")
        if final_card and eee_metadata:
            final_card = _enrich_baseline_results(final_card, eee_metadata)

            card = final_card.get("benchmark_card", final_card)
            ordered_card = _reorder_card_fields(card)
            if "benchmark_card" in final_card:
                final_card["benchmark_card"] = ordered_card
            else:
                final_card = ordered_card

            card_filename = f"benchmark_card_{safe_name}.json"
            output_manager.save_benchmark_card(final_card, card_filename)
            logger.info("Saved benchmark card with evaluation summary: %s", card_filename)

        completed = final_state.get("completed", [])
        errors = final_state.get("errors", [])
        logger.info("Completed steps: %s", completed)
        if errors:
            logger.warning("Errors: %s", errors)

        return final_card

    except Exception as e:
        logger.error("Failed to process %s: %s", benchmark_name, e, exc_info=debug)
        return None


def run_eee_pipeline(
    eee_path: str,
    output_path: Optional[str] = None,
    max_files_per_benchmark: int = 50,
    benchmarks_filter: Optional[List[str]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Scan EEE evaluation data, discover benchmarks, and generate cards for each."""
    setup_logging_suppression(debug_mode=debug)

    logger.info("Models — composer: %s | factreasoner: %s",
                Config.COMPOSER_MODEL, Config.FACTREASONER_MODEL)

    logger.info("Scanning EEE data at: %s", eee_path)
    scan_result = scan_eee_folder(eee_path, max_files_per_benchmark)

    if scan_result.errors:
        for err in scan_result.errors:
            logger.warning("Scan error: %s", err)

    benchmarks = scan_result.benchmarks
    logger.info("Found %d unique benchmarks in %d files", len(benchmarks), scan_result.total_eval_files)

    if benchmarks_filter:
        filter_set = {b.lower() for b in benchmarks_filter}
        benchmarks = {
            k: v for k, v in benchmarks.items()
            if k.lower() in filter_set
        }
        logger.info("Filtered to %d benchmarks: %s", len(benchmarks), list(benchmarks.keys()))

    logger.info("Resolving HuggingFace repos...")
    pipeline_inputs_map: Dict[str, Dict[str, Any]] = {}
    for name, bench in sorted(benchmarks.items()):
        inputs = eee_to_pipeline_inputs(bench)
        pipeline_inputs_map[name] = inputs
        hf = inputs.get("hf_repo", "None")
        logger.info("  %s -> hf_repo=%s (%d models)", name, hf, bench.num_models_evaluated)

    summary = {
        "total_benchmarks": len(pipeline_inputs_map),
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    for i, (name, inputs) in enumerate(sorted(pipeline_inputs_map.items()), 1):
        logger.info("[%d/%d] Processing: %s", i, len(pipeline_inputs_map), name)

        if not inputs.get("hf_repo"):
            logger.warning("Skipping %s: no HF repo resolved", name)
            summary["skipped"].append({"benchmark": name, "reason": "no_hf_repo"})
            continue

        card = process_single_benchmark(
            benchmark_name=name,
            pipeline_inputs=inputs,
            base_output_path=output_path,
            debug=debug,
        )

        if card:
            summary["successful"].append(name)
        else:
            summary["failed"].append(name)

    logger.info("EEE pipeline complete: success=%d, failed=%d, skipped=%d",
                len(summary["successful"]), len(summary["failed"]), len(summary["skipped"]))

    return summary
