"""Extract metadata from HuggingFace datasets (README, configs, stats)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Dict, List, Union

# Suppress verbose logging
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from huggingface_hub import HfApi, hf_hub_download
from langchain.tools import tool

# handle different versions of huggingface_hub
try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError

try:
    from huggingface_hub import list_dataset_configs
except ImportError:
    list_dataset_configs = None

logger = logging.getLogger(__name__)
api = HfApi()


def _clean(obj: Any) -> Any:
    """Convert dates to strings for JSON serialization.

    Args:
        obj: Object to clean for JSON serialization.

    Returns:
        Cleaned object with dates converted to ISO format strings.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


@lru_cache(maxsize=64)
def _collect_hf_metadata(repo_id: str, repo_type: str = "dataset",
                         revision: str | None = None) -> Dict[str, Any]:
    """Get all metadata for a HuggingFace dataset, or a model repo when repo_type="model".

    The dataset path uses the default arguments. The model path reads model_info at
    the pinned revision, the README and config.json at that revision, and the file list;
    dataset-only extras (dataset_infos.json, configs) are skipped, and the tags that carry
    lineage (base_model:*) plus the card's model-index are surfaced as plain keys.

    Args:
        repo_id: HuggingFace repository ID in format 'username/dataset-name'.

    Returns:
        Dictionary containing dataset metadata including info, README, files, and configs.

    Raises:
        ValueError: If dataset is not found on HuggingFace Hub.
    """
    if repo_type == "model":
        return _collect_model_metadata(repo_id, revision)

    # Get basic info
    try:
        info = api.dataset_info(repo_id)
    except HfHubHTTPError as exc:
        raise ValueError(f"Dataset '{repo_id}' not found: {exc}") from exc

    meta: Dict[str, Any] = _clean(asdict(info))

    # try to get readme
    try:
        readme_path = hf_hub_download(
            repo_id, "README.md", repo_type="dataset", resume_download=True
        )
        with open(readme_path, encoding="utf-8") as fh:
            meta["readme_markdown"] = fh.read()
    except Exception as exc:
        logger.warning("Could not fetch README for %s: %s", repo_id, exc)

    # list files
    try:
        meta["file_list"] = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning("Could not list files for %s: %s", repo_id, exc)
        meta["file_list"] = []

    # get builder metadata if available
    if "dataset_infos.json" in meta["file_list"]:
        try:
            local = hf_hub_download(
                repo_id, "dataset_infos.json", repo_type="dataset", resume_download=True
            )
            with open(local, encoding="utf-8") as fh:
                meta["builder_metadata"] = json.load(fh)
        except Exception as exc:
            logger.warning("Could not parse dataset_infos.json for %s: %s", repo_id, exc)

    # get all configs
    configs: List[str] = []
    if list_dataset_configs is not None:
        try:
            configs = list_dataset_configs(repo_id)
        except Exception:
            configs = []

    # fallback to builder metadata
    if not configs and "builder_metadata" in meta:
        configs = list(meta["builder_metadata"].keys())

    if configs:
        meta["all_configs"] = configs

    return meta


def _collect_model_metadata(repo_id: str, revision: str | None) -> Dict[str, Any]:
    """Model-repo metadata at a pinned revision: model_info, README, config.json, files."""
    try:
        info = api.model_info(repo_id, revision=revision)
    except HfHubHTTPError as exc:
        raise ValueError(f"Model '{repo_id}' not found: {exc}") from exc

    meta: Dict[str, Any] = _clean(asdict(info))
    meta["repo_type"] = "model"
    meta["requested_revision"] = revision
    meta["resolved_revision"] = meta.get("sha")

    for name, key in (("README.md", "readme_markdown"), ("config.json", "config")):
        try:
            path = hf_hub_download(repo_id, name, repo_type="model", revision=revision,
                                   resume_download=True)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            meta[key] = json.loads(text) if name == "config.json" else text
        except Exception as exc:
            logger.warning("Could not fetch %s for %s: %s", name, repo_id, exc)

    try:
        meta["file_list"] = api.list_repo_files(repo_id, repo_type="model", revision=revision)
    except Exception as exc:
        logger.warning("Could not list files for %s: %s", repo_id, exc)
        meta["file_list"] = []

    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    meta["base_model_tags"] = [t.split(":", 1)[1] for t in tags
                               if isinstance(t, str) and t.startswith("base_model:")]
    card_data = meta.get("card_data") if isinstance(meta.get("card_data"), dict) else {}
    meta["model_index"] = card_data.get("model-index") or card_data.get("model_index")
    return meta


@tool("hf_model_metadata")
def hf_model_metadata(repo_id: str, revision: str | None = None) -> Dict[str, Any]:
    """Get metadata for one HuggingFace MODEL repo at an optional pinned revision.

    Returns the model_info fields (tags, license, downloads, likes, pipeline_tag,
    safetensors, card_data with its model-index) plus readme_markdown, config (parsed
    config.json), file_list, base_model_tags and model_index.
    """
    logger.debug("Fetching HuggingFace metadata for model: %s@%s", repo_id, revision)
    return _collect_hf_metadata(repo_id, "model", revision)


@tool("hf_dataset_metadata")
def hf_dataset_metadata(repo_id: Union[str, List[str]]) -> Dict[str, Any]:
    """Get metadata for one or more HuggingFace datasets.

    Args:
        repo_id: Single repository ID string or list of repository IDs.

    Returns:
        Dictionary containing metadata for requested dataset(s). For single repo,
        returns metadata directly. For multiple repos, returns dict mapping repo ID
        to metadata (or error message if fetch failed).
    """
    if isinstance(repo_id, str):
        logger.debug(f"Fetching HuggingFace metadata for dataset: {repo_id}")
        # single repo
        result = _collect_hf_metadata(repo_id)
        logger.debug(f"Successfully retrieved HuggingFace metadata for {repo_id}")
        return result

    # multiple repos
    logger.debug(f"Fetching HuggingFace metadata for {len(repo_id)} datasets")
    results: Dict[str, Any] = {}
    for rid in repo_id:
        try:
            logger.debug(f"Processing dataset: {rid}")
            results[rid] = _collect_hf_metadata(rid)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", rid, exc)
            results[rid] = {"error": str(exc)}

    logger.debug(f"Completed HuggingFace metadata retrieval for {len(repo_id)} datasets")

    return results
