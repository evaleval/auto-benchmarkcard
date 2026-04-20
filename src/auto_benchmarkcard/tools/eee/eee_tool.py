"""Tool for extracting benchmark metadata from Every Eval Ever (EEE) evaluation data.

Scans EEE evaluation JSONs, aggregates benchmark information across multiple
model evaluations, resolves HuggingFace dataset repositories, and produces
per-benchmark metadata compatible with the auto-benchmarkcard pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _normalize_benchmark_name(name: str) -> str:
    """Normalize benchmark name to merge variants like 'MMLU-PRO' and 'MMLU-Pro'."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


# Known mappings where HF search returns wrong results or needs disambiguation.
# key = EEE dataset_name (lowercased), value = correct HF repo
HF_REPO_OVERRIDES: Dict[str, str] = {
    "raft": "ought/raft",
    "quac": "allenai/quac",
    "math": "hendrycks/competition_math",
    "math level 5": "hendrycks/competition_math",
    "naturalquestions": "google-research-datasets/natural_questions",
    "naturalquestions (open-book)": "google-research-datasets/natural_questions",
    "naturalquestions (closed-book)": "google-research-datasets/natural_questions",
    "cnn/dailymail": "abisee/cnn_dailymail",
    "ms marco": "microsoft/ms_marco",
    "ms marco (trec)": "microsoft/ms_marco",
    "wmt 2014": None,  # No good HF equivalent
}

# Minimum downloads threshold to accept a HF search result as valid
MIN_HF_DOWNLOADS = 500


class EEEBenchmarkInfo(BaseModel):
    """Aggregated benchmark information extracted from EEE evaluation data."""

    name: str = Field(description="Benchmark/dataset name from EEE")
    eee_source_folder: str = Field(description="EEE top-level folder name")
    source_type: str = Field(description="'hf_dataset', 'url', or 'other'")
    hf_repo: Optional[str] = Field(None, description="HuggingFace repo if available")
    source_urls: List[str] = Field(default_factory=list, description="Source URLs if available")
    metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="metric_name -> metric_config from EEE",
    )
    model_scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {model, developer, score, metric} entries",
    )
    num_models_evaluated: int = 0
    eval_library: Optional[str] = Field(None, description="Evaluation framework used")


class EEEScanResult(BaseModel):
    """Result of scanning an EEE data directory."""

    benchmarks: Dict[str, EEEBenchmarkInfo] = Field(
        default_factory=dict,
        description="benchmark_name -> aggregated info",
    )
    scan_path: str = ""
    total_eval_files: int = 0
    errors: List[str] = Field(default_factory=list)


def scan_eee_folder(eee_path: str | Path, max_files_per_benchmark: int = 50) -> EEEScanResult:
    """Scan an EEE data directory and aggregate benchmark information.

    Reads evaluation JSONs from an EEE benchmark folder (or top-level data/ dir),
    extracts unique benchmarks, their metrics, and model scores.

    Args:
        eee_path: Path to EEE data directory (e.g., data/hfopenllm_v2 or data/).
        max_files_per_benchmark: Max eval files to read per benchmark folder
            to avoid reading thousands of files. Scores are sampled.

    Returns:
        EEEScanResult with aggregated benchmark information.
    """
    eee_path = Path(eee_path)
    result = EEEScanResult(scan_path=str(eee_path))

    if not eee_path.exists():
        result.errors.append(f"Path does not exist: {eee_path}")
        return result

    # Find all JSON files (not .jsonl)
    json_files = sorted(eee_path.rglob("*.json"))
    result.total_eval_files = len(json_files)
    logger.info("Found %d JSON files in %s", len(json_files), eee_path)

    # Group files by benchmark folder
    files_by_folder: Dict[str, list] = defaultdict(list)
    for f in json_files:
        # Determine the benchmark folder relative to eee_path
        rel = f.relative_to(eee_path)
        folder = rel.parts[0] if len(rel.parts) > 1 else eee_path.name
        files_by_folder[folder].append(f)

    for folder, files in files_by_folder.items():
        # Sample files if too many
        sample = files[:max_files_per_benchmark]
        for filepath in sample:
            try:
                data = json.loads(filepath.read_text())
            except (json.JSONDecodeError, OSError) as e:
                result.errors.append(f"Failed to read {filepath}: {e}")
                continue

            if "evaluation_results" not in data:
                continue

            _process_eval_file(data, folder, result)

    return result


def _process_eval_file(data: Dict[str, Any], folder: str, result: EEEScanResult) -> None:
    """Process a single EEE evaluation JSON and update the scan result."""
    model_info = data.get("model_info", {})
    model_name = model_info.get("name", "unknown")
    model_developer = model_info.get("developer", "unknown")
    eval_library = data.get("eval_library", {}).get("name")

    # Skip folder-level aggregate entries (helm_capabilities, helm_classic, etc.)
    # These have evaluation_name == folder name and contain only aggregate metrics
    _AGGREGATE_INDICATORS = {"mean win rate", "mean score"}

    for eval_result in data.get("evaluation_results", []):
        source_data = eval_result.get("source_data", {})
        dataset_name = source_data.get("dataset_name", "unknown")

        if dataset_name == "unknown":
            continue

        # Skip folder-level aggregates
        metric_desc = eval_result.get("metric_config", {}).get("evaluation_description", "").lower()
        eval_name = eval_result.get("evaluation_name", "").lower()
        if metric_desc in _AGGREGATE_INDICATORS or eval_name in _AGGREGATE_INDICATORS:
            continue

        # Normalize name to merge variants (MMLU-PRO vs MMLU-Pro)
        normalized = _normalize_benchmark_name(dataset_name)

        # Find existing benchmark by normalized name
        existing_key = None
        for k in result.benchmarks:
            if _normalize_benchmark_name(k) == normalized:
                existing_key = k
                break

        if existing_key is None:
            # Use the original casing for the display name
            result.benchmarks[dataset_name] = EEEBenchmarkInfo(
                name=dataset_name,
                eee_source_folder=folder,
                source_type=source_data.get("source_type", "other"),
                hf_repo=source_data.get("hf_repo"),
                eval_library=eval_library,
            )
            existing_key = dataset_name

        bench = result.benchmarks[existing_key]

        # Collect source URLs
        if source_data.get("url"):
            for url in source_data["url"]:
                if url not in bench.source_urls:
                    bench.source_urls.append(url)

        # Update hf_repo if found (prefer hf_dataset source over existing)
        if source_data.get("hf_repo"):
            if not bench.hf_repo or source_data.get("source_type") == "hf_dataset":
                bench.hf_repo = source_data["hf_repo"]
                bench.source_type = source_data.get("source_type", bench.source_type)

        # Collect metric config
        metric_config = eval_result.get("metric_config", {})
        metric_id = metric_config.get("metric_id") or eval_result.get("evaluation_name", "")
        if metric_id and metric_id not in bench.metrics:
            bench.metrics[metric_id] = metric_config

        # Collect model scores
        score = eval_result.get("score_details", {}).get("score")
        if score is not None:
            bench.model_scores.append({
                "model": model_name,
                "developer": model_developer,
                "score": score,
                "metric": metric_id,
                "evaluation_name": eval_result.get("evaluation_name", ""),
            })

        # Track unique models
        model_ids = {s["model"] for s in bench.model_scores}
        bench.num_models_evaluated = len(model_ids)

        if eval_library and not bench.eval_library:
            bench.eval_library = eval_library


def resolve_hf_repo(benchmark_name: str, existing_hf_repo: Optional[str] = None) -> Optional[str]:
    """Resolve a benchmark name to a HuggingFace dataset repository.

    First checks if there's already an hf_repo from the EEE data.
    Then checks manual overrides. Finally falls back to HF API search.

    Args:
        benchmark_name: The dataset/benchmark name from EEE.
        existing_hf_repo: HF repo already present in EEE source_data.

    Returns:
        HuggingFace repository ID or None if not resolvable.
    """
    # If EEE already has an hf_repo, use it
    if existing_hf_repo:
        return existing_hf_repo

    # Check manual overrides
    key = benchmark_name.lower().strip()
    if key in HF_REPO_OVERRIDES:
        override = HF_REPO_OVERRIDES[key]
        if override is None:
            logger.info("No HF repo for '%s' (known unmatchable)", benchmark_name)
        return override

    # Search HuggingFace
    try:
        api = HfApi()
        results = list(api.list_datasets(search=benchmark_name, sort="downloads", limit=5))

        if not results:
            logger.warning("No HF datasets found for '%s'", benchmark_name)
            return None

        # Filter: must have enough downloads and name should be related
        for ds in results:
            name_lower = ds.id.lower().split("/")[-1]
            search_lower = benchmark_name.lower().replace(" ", "").replace("-", "").replace("_", "")
            name_normalized = name_lower.replace(" ", "").replace("-", "").replace("_", "")

            # Check name similarity
            if search_lower in name_normalized or name_normalized in search_lower:
                if ds.downloads >= MIN_HF_DOWNLOADS:
                    logger.info("Resolved '%s' -> '%s' (%d downloads)", benchmark_name, ds.id, ds.downloads)
                    return ds.id

        # Fallback: top result if it has high downloads
        top = results[0]
        if top.downloads >= MIN_HF_DOWNLOADS * 10:
            logger.info(
                "Fallback: '%s' -> '%s' (%d downloads, name mismatch)",
                benchmark_name, top.id, top.downloads,
            )
            return top.id

        logger.warning("No confident HF match for '%s' (top: %s, %d downloads)", benchmark_name, top.id, top.downloads)
        return None

    except Exception as e:
        logger.warning("HF API search failed for '%s': %s", benchmark_name, e)
        return None


def build_evaluation_summary(bench: EEEBenchmarkInfo) -> Dict[str, Any]:
    """Build a deterministic evaluation summary from aggregated EEE scores.

    Computes statistics across all model evaluations for a benchmark.

    Args:
        bench: Aggregated benchmark info with model scores.

    Returns:
        Dictionary with evaluation summary statistics.
    """
    if not bench.model_scores:
        return {}

    # Group scores by metric
    scores_by_metric: Dict[str, List[float]] = defaultdict(list)
    models_by_metric: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in bench.model_scores:
        metric = entry.get("metric") or entry.get("evaluation_name", "score")
        scores_by_metric[metric].append(entry["score"])
        models_by_metric[metric].append(entry)

    # Pick the primary metric (most scores, or first)
    primary_metric = max(scores_by_metric, key=lambda m: len(scores_by_metric[m]))
    scores = scores_by_metric[primary_metric]

    # Sort models by score for top performers
    models_sorted = sorted(
        models_by_metric[primary_metric],
        key=lambda x: x["score"],
        reverse=True,
    )

    # Compute statistics
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    # Standard deviation
    if len(scores) > 1:
        variance = sum((s - avg_score) ** 2 for s in scores) / (len(scores) - 1)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0

    # Top performers (up to 5)
    top_performers = [
        {"model": m["model"], "developer": m.get("developer", ""), "score": round(m["score"], 4)}
        for m in models_sorted[:5]
    ]

    # Metric info
    metric_config = bench.metrics.get(primary_metric, {})

    summary = {
        "total_models_evaluated": bench.num_models_evaluated,
        "primary_metric": primary_metric,
        "metric_config": {
            "lower_is_better": metric_config.get("lower_is_better", False),
            "score_type": metric_config.get("score_type", "continuous"),
            "min_possible": metric_config.get("min_score"),
            "max_possible": metric_config.get("max_score"),
        },
        "score_statistics": {
            "mean": round(avg_score, 4),
            "std_dev": round(std_dev, 4),
            "min": round(min_score, 4),
            "max": round(max_score, 4),
        },
        "top_performers": top_performers,
        "all_metrics": list(scores_by_metric.keys()),
        "source": "Every Eval Ever (EEE)",
    }

    return summary


@lru_cache(maxsize=1)
def _build_unitxt_repo_index() -> Dict[str, Tuple[str, Any]]:
    """Build an index mapping HF repo IDs to UnitXT card arxiv tags.

    Scans the UnitXT catalog cards directory and returns a dict of
    hf_repo -> (card_id, arxiv_tags). Cached after first call (~0.2s).

    Returns:
        Dict mapping loader.path values to (card_id, __tags__.arxiv).
    """
    try:
        from unitxt.catalog import get_local_catalogs_paths
        catalog_paths = get_local_catalogs_paths()
    except Exception:
        return {}

    index: Dict[str, Tuple[str, Any]] = {}
    for catalog_path in catalog_paths:
        cards_dir = os.path.join(catalog_path, "cards")
        if not os.path.isdir(cards_dir):
            continue
        for root, _dirs, files in os.walk(cards_dir):
            for f in files:
                if not f.endswith(".json"):
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath) as fh:
                        card = json.load(fh)
                    loader_path = (card.get("loader") or {}).get("path")
                    arxiv = (card.get("__tags__") or {}).get("arxiv")
                    if loader_path and arxiv:
                        rel = os.path.relpath(filepath, cards_dir)
                        card_id = rel.replace(".json", "").replace("/", ".")
                        # Keep first match per repo (avoid overwriting)
                        if loader_path not in index:
                            index[loader_path] = (card_id, arxiv)
                except (json.JSONDecodeError, OSError):
                    continue
    return index


def lookup_unitxt_paper(hf_repo: str) -> Optional[str]:
    """Search UnitXT catalog for a paper URL matching an HF repo.

    Looks up UnitXT cards by their loader.path field, extracts __tags__.arxiv,
    and returns the first arxiv URL found.

    Matching strategy: exact match first, then name-only match (strip org prefix).

    Args:
        hf_repo: HuggingFace repository ID (e.g., 'cais/mmlu').

    Returns:
        arXiv URL string or None if not found in catalog.
    """
    index = _build_unitxt_repo_index()

    # Exact match
    match = index.get(hf_repo)

    # Fallback: match by dataset name only (strip org prefix)
    if not match:
        repo_name = hf_repo.split("/")[-1].lower()
        for loader_path, entry in index.items():
            if loader_path.split("/")[-1].lower() == repo_name:
                match = entry
                break

    if not match:
        return None

    card_id, arxiv = match
    # Convert arxiv ID(s) to URL
    if isinstance(arxiv, list) and arxiv:
        arxiv_id = arxiv[0]
    elif isinstance(arxiv, str):
        arxiv_id = arxiv
    else:
        return None

    url = f"https://arxiv.org/abs/{arxiv_id}"
    logger.info("Found paper via UnitXT catalog (%s): %s", card_id, url)
    return url


def eee_to_pipeline_inputs(
    bench: EEEBenchmarkInfo,
) -> Dict[str, Any]:
    """Convert an EEEBenchmarkInfo into pipeline-compatible inputs.

    Produces the extracted_ids and EEE metadata for the workflow.

    Args:
        bench: Single benchmark info from EEE scan.

    Returns:
        Dictionary with keys: extracted_ids, hf_repo, eee_metadata.
    """
    # Resolve HF repo
    hf_repo = resolve_hf_repo(bench.name, bench.hf_repo)

    # Build extracted_ids (same format as extractor_tool output)
    extracted_ids = {
        "hf_repo": hf_repo,
        "paper_url": None,  # Will be resolved from UnitXT catalog + HF metadata
        "risk_tags": None,
    }

    # Build evaluation summary
    eval_summary = build_evaluation_summary(bench)

    # Build EEE-specific metadata for the composer
    eee_metadata = {
        "benchmark_name": bench.name,
        "eee_source_folder": bench.eee_source_folder,
        "source_type": bench.source_type,
        "source_urls": bench.source_urls,
        "eval_library": bench.eval_library,
        "metrics": {k: v for k, v in bench.metrics.items()},
        "evaluation_summary": eval_summary,
        "num_models_evaluated": bench.num_models_evaluated,
    }

    return {
        "extracted_ids": extracted_ids,
        "hf_repo": hf_repo,
        "eee_metadata": eee_metadata,
    }


def scan_and_prepare(
    eee_path: str | Path,
    max_files_per_benchmark: int = 50,
) -> List[Dict[str, Any]]:
    """Scan EEE data and prepare pipeline inputs for all discovered benchmarks.

    This is the main entry point for the EEE adapter. It scans the data directory,
    discovers all unique benchmarks, resolves their HF repos, and returns
    pipeline-ready inputs for each one.

    Args:
        eee_path: Path to EEE data directory.
        max_files_per_benchmark: Max eval files to sample per folder.

    Returns:
        List of pipeline input dicts, one per benchmark.
    """
    scan = scan_eee_folder(eee_path, max_files_per_benchmark)

    if scan.errors:
        for err in scan.errors:
            logger.warning("EEE scan error: %s", err)

    logger.info(
        "EEE scan complete: %d benchmarks from %d files",
        len(scan.benchmarks), scan.total_eval_files,
    )

    results = []
    for name, bench in sorted(scan.benchmarks.items()):
        logger.info("Preparing: %s (hf_repo=%s, %d models)", name, bench.hf_repo, bench.num_models_evaluated)
        pipeline_input = eee_to_pipeline_inputs(bench)
        results.append(pipeline_input)

    return results
