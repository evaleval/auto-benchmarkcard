"""Tool for extracting benchmark metadata from Every Eval Ever (EEE) evaluation data.

Scans EEE evaluation JSONs, aggregates benchmark information across multiple
model evaluations, resolves HuggingFace dataset repositories, and produces
per-benchmark metadata compatible with the auto-benchmarkcard pipeline.

Composite benchmark handling:
  - A "composite" is a folder containing multiple distinct benchmarks (e.g.,
    helm_capabilities has BoolQ, HellaSwag, MMLU as separate benchmarks).
  - A "subject-composite" is a folder where all entries are variants of one
    benchmark (e.g., MMLU's 35 subjects). These get collapsed into a single
    benchmark with averaged scores.
  - Detection uses 4 heuristic signals (prefix, URLs, HF repo, name similarity).
    At least 2 must fire to classify as subject-composite.
  - True composites skip RAG/FactReasoner (no single source paper to check against).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

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
    "anthropic rlhf dataset": "Anthropic/hh-rlhf",
    "best chatgpt prompts": "fka/awesome-chatgpt-prompts",
    "koala test dataset": "HuggingFaceH4/Koala-test-set",
    "theory_of_mind": "grimulkan/theory-of-mind",
}

# Minimum downloads threshold to accept a HF search result as valid
MIN_HF_DOWNLOADS = 500


class EEEBenchmarkInfo(BaseModel):
    """Aggregated benchmark information extracted from EEE evaluation data."""

    name: str = Field(description="Benchmark/dataset name from EEE")
    eee_source_folders: List[str] = Field(
        default_factory=list, description="EEE top-level folder names this benchmark appears in",
    )
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

    @property
    def eee_source_folder(self) -> str:
        """First source folder (backward compat)."""
        return self.eee_source_folders[0] if self.eee_source_folders else ""


class CompositeInfo(BaseModel):
    """Metadata about a composite benchmark suite."""

    folder_name: str
    sub_benchmarks: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    eval_library: Optional[str] = None


class EEEScanResult(BaseModel):
    """Result of scanning an EEE data directory."""

    benchmarks: Dict[str, EEEBenchmarkInfo] = Field(
        default_factory=dict,
        description="benchmark_name -> aggregated info",
    )
    composites: Dict[str, CompositeInfo] = Field(
        default_factory=dict,
        description="folder_name -> composite info (only real suites, not subject-composites)",
    )
    folder_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="folder_name -> aggregate-level metadata (URLs, eval_library)",
    )
    scan_path: str = ""
    total_eval_files: int = 0
    errors: List[str] = Field(default_factory=list)


_HARNESS_PREFIXES = ["helm_", "hf_", "hfopenllm_"]


def _derive_benchmark_name(folder: str, bench_names: List[str]) -> str:
    """Derive a clean benchmark name for a collapsed subject-composite."""
    normalized = [_normalize_benchmark_name(n) for n in bench_names]
    prefix = os.path.commonprefix(normalized).rstrip("-").rstrip(" ")
    if len(prefix) >= 4:
        return prefix.replace("-", " ").title()
    # Fallback: strip harness prefix from folder name
    name = folder
    for p in _HARNESS_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    return name.replace("_", " ").replace("-", " ").title()


def _is_subject_composite(bench_names: List[str], scan_result: EEEScanResult) -> bool:
    """Heuristic: is this a subject-composite (collapse) or a real suite?

    Returns True if at least 2 of 4 signals fire (majority vote — 2/4 balances
    false positives vs false negatives):

    1. Shared name prefix >= 4 chars (shorter prefixes match noise like 'a-', 'b-')
    2. All sub-benchmarks share the same paper/source URLs (or all have none)
    3. All sub-benchmarks share the same HF repo
    4. Average pairwise name similarity > 70 (rapidfuzz 0-100; tuned on MMLU ~85
       vs HELM ~40, 70 separates cleanly)
    """
    signals = 0

    normalized = [_normalize_benchmark_name(n) for n in bench_names]

    # Signal 1: shared prefix (4+ chars gives meaningful shared roots)
    prefix = os.path.commonprefix(normalized)
    if len(prefix) >= 4:
        signals += 1

    # Signal 2: same paper source (only if at least one URL exists)
    all_urls: set[str] = set()
    for name in bench_names:
        bench = scan_result.benchmarks.get(name)
        if bench:
            all_urls.update(bench.source_urls)
    if all_urls and len(all_urls) == 1:
        signals += 1

    # Signal 3: same HF repo (only if repo is not None)
    repos = set()
    for name in bench_names:
        bench = scan_result.benchmarks.get(name)
        if bench:
            repos.add(bench.hf_repo)
    repos.discard(None)
    if repos and len(repos) == 1:
        signals += 1

    # Signal 4: high average pairwise name similarity
    # Deterministic sampling: sort then take evenly spaced pairs (20 is enough
    # for a stable average; more gives diminishing returns)
    pairs = list(itertools.combinations(normalized, 2))
    if len(pairs) > 20:
        pairs.sort()
        step = max(1, len(pairs) // 20)
        pairs = pairs[::step][:20]
    if pairs:
        avg_sim = sum(fuzz.ratio(a, b) for a, b in pairs) / len(pairs)
        if avg_sim > 70:
            signals += 1

    return signals >= 2


def _collapse_subject_composite(
    folder: str, bench_names: List[str], result: EEEScanResult,
) -> None:
    """Collapse a subject-composite into a single merged benchmark entry."""
    merged_name = _derive_benchmark_name(folder, bench_names)

    scores_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_metrics: Dict[str, Dict[str, Any]] = {}
    hf_repo: Optional[str] = None
    source_urls: List[str] = []

    for name in bench_names:
        bench = result.benchmarks.get(name)
        if not bench:
            continue
        all_metrics.update(bench.metrics)
        if bench.hf_repo and not hf_repo:
            hf_repo = bench.hf_repo
        source_urls.extend(u for u in bench.source_urls if u not in source_urls)
        for score_entry in bench.model_scores:
            scores_by_model[score_entry["model"]].append(score_entry)

    # Build aggregated model_scores: average across subjects per model
    merged_scores: List[Dict[str, Any]] = []
    for model, entries in scores_by_model.items():
        avg = sum(e["score"] for e in entries) / len(entries)
        merged_scores.append({
            "model": model,
            "developer": entries[0].get("developer", ""),
            "score": avg,
            "metric": "average_across_subjects",
            "evaluation_name": merged_name,
            "subject_scores": [
                {"subject": e.get("evaluation_name", ""), "score": e["score"]}
                for e in entries
            ],
        })

    # Remove individual sub-benchmark entries
    for name in bench_names:
        result.benchmarks.pop(name, None)

    # Create merged entry
    result.benchmarks[merged_name] = EEEBenchmarkInfo(
        name=merged_name,
        eee_source_folders=[folder],
        source_type="hf_dataset" if hf_repo else "other",
        hf_repo=hf_repo,
        source_urls=source_urls,
        metrics=all_metrics,
        model_scores=merged_scores,
        num_models_evaluated=len(scores_by_model),
        eval_library=result.folder_metadata.get(folder, {}).get("eval_library"),
    )

    logger.info(
        "Collapsed %d subjects in '%s' into single benchmark '%s'",
        len(bench_names), folder, merged_name,
    )


def _detect_intra_benchmark_subjects(result: EEEScanResult) -> None:
    """Detect benchmarks that contain multiple subjects under one dataset_name.

    When a single benchmark (e.g., helm_mmlu) has scores with many distinct
    evaluation_name values (e.g., "Abstract Algebra", "Anatomy"), aggregate them
    into per-model averages with subject_scores detail lists.

    This handles cases like MMLU (35 subjects) and global-mmlu-lite (19 languages)
    where dataset_name is the same but evaluation_name varies.
    """
    for name, bench in list(result.benchmarks.items()):
        eval_names = set(s.get("evaluation_name", "") for s in bench.model_scores)
        # 3+ subjects needed: 2 could be coincidence (e.g., train/test split)
        if len(eval_names) < 3:
            continue

        # Group scores by model, then aggregate
        scores_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for score_entry in bench.model_scores:
            scores_by_model[score_entry["model"]].append(score_entry)

        merged_scores: List[Dict[str, Any]] = []
        for model, entries in scores_by_model.items():
            avg = sum(e["score"] for e in entries) / len(entries)
            merged_scores.append({
                "model": model,
                "developer": entries[0].get("developer", ""),
                "score": avg,
                "metric": "average_across_subjects",
                "evaluation_name": bench.name,
                "subject_scores": [
                    {"subject": e.get("evaluation_name", ""), "score": e["score"]}
                    for e in entries
                ],
            })

        bench.model_scores = merged_scores
        bench.num_models_evaluated = len(scores_by_model)
        logger.info(
            "Aggregated %d subjects for '%s' into per-model averages (%d models)",
            len(eval_names), name, len(scores_by_model),
        )


def _detect_composites(result: EEEScanResult, no_collapse: bool = False) -> None:
    """Classify folders as composite suites or subject-composites.

    Folders with 2+ benchmarks are either real composite suites (distinct benchmarks)
    or subject-composites (variants of one benchmark, e.g., MMLU subjects).
    Subject-composites get collapsed into a single merged benchmark entry.
    """
    folder_benchmarks: Dict[str, List[str]] = defaultdict(list)
    for name, bench in result.benchmarks.items():
        for folder in bench.eee_source_folders:
            folder_benchmarks[folder].append(name)

    for folder, bench_names in folder_benchmarks.items():
        if len(bench_names) < 2:
            continue

        if not no_collapse and _is_subject_composite(bench_names, result):
            _collapse_subject_composite(folder, bench_names, result)
        else:
            meta = result.folder_metadata.get(folder, {})
            result.composites[folder] = CompositeInfo(
                folder_name=folder,
                sub_benchmarks=sorted(bench_names),
                source_urls=meta.get("source_urls", []),
                eval_library=meta.get("eval_library"),
            )


def scan_eee_folder(
    eee_path: str | Path,
    max_files_per_benchmark: int = 50,
    no_collapse: bool = False,
) -> EEEScanResult:
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

    # Group files by source config.
    # If files are in subfolders (data/helm_capabilities/model/eval.json), use subfolder name.
    # If files are flat (eee_samples/helm_capabilities.json), use filename stem.
    files_by_folder: Dict[str, list] = defaultdict(list)
    for f in json_files:
        rel = f.relative_to(eee_path)
        if len(rel.parts) > 1:
            folder = rel.parts[0]
        else:
            folder = f.stem  # e.g., "helm_capabilities" from helm_capabilities.json
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

    if not no_collapse:
        _detect_intra_benchmark_subjects(result)

    _detect_composites(result, no_collapse=no_collapse)

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

        # Capture folder-level aggregate metadata before skipping
        metric_desc = eval_result.get("metric_config", {}).get("evaluation_description", "").lower()
        eval_name = eval_result.get("evaluation_name", "").lower()
        if metric_desc in _AGGREGATE_INDICATORS or eval_name in _AGGREGATE_INDICATORS:
            if folder not in result.folder_metadata:
                result.folder_metadata[folder] = {"source_urls": [], "eval_library": None}
            fm = result.folder_metadata[folder]
            if source_data.get("url"):
                for url in source_data["url"]:
                    if url not in fm["source_urls"]:
                        fm["source_urls"].append(url)
            if eval_library and not fm["eval_library"]:
                fm["eval_library"] = eval_library
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
                eee_source_folders=[folder],
                source_type=source_data.get("source_type", "other"),
                hf_repo=source_data.get("hf_repo"),
                eval_library=eval_library,
            )
            existing_key = dataset_name

        bench = result.benchmarks[existing_key]

        # Track all folders this benchmark appears in
        if folder not in bench.eee_source_folders:
            bench.eee_source_folders.append(folder)

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
    # Check manual overrides first (they take priority over EEE defaults)
    key = benchmark_name.lower().strip()
    if key in HF_REPO_OVERRIDES:
        override = HF_REPO_OVERRIDES[key]
        if override is None:
            logger.info("No HF repo for '%s' (known unmatchable)", benchmark_name)
        return override

    # If EEE already has a valid hf_repo, use it
    if existing_hf_repo:
        return existing_hf_repo

    # Search HuggingFace
    try:
        api = HfApi()
        results = list(api.list_datasets(search=benchmark_name, sort="downloads", direction=-1, limit=5))

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
    benchmark_type: str = "single",
    appears_in: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert an EEEBenchmarkInfo into pipeline-compatible inputs.

    Produces the extracted_ids and EEE metadata for the workflow.

    Args:
        bench: Single benchmark info from EEE scan.
        benchmark_type: "single" or "composite".
        appears_in: List of composite folder names this benchmark belongs to.

    Returns:
        Dictionary with keys: extracted_ids, hf_repo, eee_metadata.
    """
    # Resolve HF repo
    hf_repo = resolve_hf_repo(bench.name, bench.hf_repo)

    # Build extracted_ids (same format as extractor_tool output)
    # Paper URL resolution happens later via paper_resolver (Semantic Scholar + LLM)
    extracted_ids = {
        "hf_repo": hf_repo,
        "paper_url": None,
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
        "benchmark_type": benchmark_type,
        "appears_in": appears_in or [],
    }

    return {
        "extracted_ids": extracted_ids,
        "hf_repo": hf_repo,
        "eee_metadata": eee_metadata,
    }


def composite_to_pipeline_inputs(
    composite: CompositeInfo,
    scan_result: EEEScanResult,
) -> Dict[str, Any]:
    """Build pipeline inputs for a composite benchmark suite card.

    Aggregates metrics and scores from all sub-benchmarks in the composite.

    Args:
        composite: Composite suite info.
        scan_result: Full scan result for looking up sub-benchmark data.

    Returns:
        Dictionary with keys: extracted_ids, hf_repo, eee_metadata.
    """
    all_scores: List[Dict[str, Any]] = []
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for name in composite.sub_benchmarks:
        bench = scan_result.benchmarks.get(name)
        if bench:
            all_scores.extend(bench.model_scores)
            all_metrics.update(bench.metrics)

    # Paper URL resolution happens later via paper_resolver (Semantic Scholar + LLM)
    eee_metadata = {
        "benchmark_name": composite.folder_name,
        "eee_source_folder": composite.folder_name,
        "source_type": "url",
        "source_urls": list(composite.source_urls),
        "eval_library": composite.eval_library,
        "metrics": all_metrics,
        "evaluation_summary": {},
        "num_models_evaluated": len({s["model"] for s in all_scores}),
        "benchmark_type": "composite",
        "contains": composite.sub_benchmarks,
    }

    return {
        "extracted_ids": {"hf_repo": None, "paper_url": None, "risk_tags": None},
        "hf_repo": None,
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
