"""hf_tool repo_type parametrization: the dataset path is byte-identical, the model path
reads model_info at a pinned revision plus README/config.json.

Failure class: dataset hardwiring (repo_type="dataset" in three places) made model
repos unreadable; the fix must not touch what dataset callers get.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import auto_benchmarkcard.tools.hf.hf_tool as H


@dataclass
class _DatasetInfo:
    id: str = "acme/bench"
    tags: List[str] = field(default_factory=lambda: ["license:mit", "size_categories:n<1K"])
    card_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ModelInfo:
    id: str = "allenai/OLMo-2-1124-7B"
    sha: str = "a" * 40
    tags: List[str] = field(default_factory=lambda: ["base_model:allenai/OLMo-2-1124-7B-stage1",
                                                     "license:apache-2.0", "transformers"])
    downloads: int = 1234
    likes: int = 5
    pipeline_tag: str = "text-generation"
    card_data: Dict[str, Any] = field(default_factory=lambda: {
        "model-index": [{"name": "OLMo-2-1124-7B", "results": [{"dataset": {"name": "MMLU"},
                                                                 "metrics": [{"type": "acc", "value": 0.63}]}]}]})


class _Api:
    def __init__(self):
        self.calls = []

    def dataset_info(self, repo_id):
        self.calls.append(("dataset_info", repo_id))
        return _DatasetInfo(id=repo_id)

    def model_info(self, repo_id, revision=None):
        self.calls.append(("model_info", repo_id, revision))
        return _ModelInfo(id=repo_id)

    def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
        self.calls.append(("list_repo_files", repo_id, repo_type, revision))
        return ["README.md", "config.json"] if repo_type == "model" else ["README.md", "data.parquet"]


def _fake_download(tmp_path):
    def download(repo_id, name, repo_type="dataset", revision=None, resume_download=True):
        p = tmp_path / f"{repo_type}-{name}"
        if name == "config.json":
            p.write_text(json.dumps({"architectures": ["Olmo2ForCausalLM"], "max_position_embeddings": 4096}))
        else:
            p.write_text(f"# README for {repo_id} ({repo_type})")
        download.calls.append((repo_id, name, repo_type, revision))
        return str(p)
    download.calls = []
    return download


def test_dataset_path_unchanged(monkeypatch, tmp_path):
    api = _Api()
    dl = _fake_download(tmp_path)
    monkeypatch.setattr(H, "api", api)
    monkeypatch.setattr(H, "hf_hub_download", dl)
    monkeypatch.setattr(H, "list_dataset_configs", None)
    H._collect_hf_metadata.cache_clear()
    meta = H.hf_dataset_metadata.func("acme/bench")
    assert meta["id"] == "acme/bench" and meta["readme_markdown"].endswith("(dataset)")
    assert ("dataset_info", "acme/bench") in api.calls
    assert dl.calls == [("acme/bench", "README.md", "dataset", None)]
    assert meta["file_list"] == ["README.md", "data.parquet"]
    assert "model_index" not in meta and "config" not in meta


def test_model_path_reads_pinned_revision_and_structured_keys(monkeypatch, tmp_path):
    api = _Api()
    dl = _fake_download(tmp_path)
    monkeypatch.setattr(H, "api", api)
    monkeypatch.setattr(H, "hf_hub_download", dl)
    H._collect_hf_metadata.cache_clear()
    rev = "b" * 40
    meta = H.hf_model_metadata.func("allenai/OLMo-2-1124-7B", revision=rev)
    assert ("model_info", "allenai/OLMo-2-1124-7B", rev) in api.calls
    assert ("list_repo_files", "allenai/OLMo-2-1124-7B", "model", rev) in api.calls
    assert sorted(c[1:] for c in dl.calls) == [("README.md", "model", rev), ("config.json", "model", rev)]
    assert meta["repo_type"] == "model" and meta["requested_revision"] == rev
    assert meta["resolved_revision"] == "a" * 40
    assert meta["config"]["max_position_embeddings"] == 4096
    assert meta["base_model_tags"] == ["allenai/OLMo-2-1124-7B-stage1"]
    assert meta["model_index"][0]["results"][0]["metrics"][0]["value"] == 0.63
    assert meta["downloads"] == 1234 and meta["pipeline_tag"] == "text-generation"
    assert meta["readme_markdown"].endswith("(model)")
