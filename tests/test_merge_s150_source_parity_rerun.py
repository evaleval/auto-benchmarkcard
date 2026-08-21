import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from merge_s150_source_parity_rerun import (  # noqa: E402
    MergeError,
    merge_source_parity_verdicts,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _hash(path, algorithm):
    digest = hashlib.new(algorithm)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _input(name, source_text):
    return {
        "name": name,
        "source_text": source_text,
        "fields": [{"path": "benchmark_details.overview", "is_ns": False, "value": name}],
        "risks": ["Hallucination"],
    }


def _verdict(name, note):
    return {
        "name": name,
        "field_verdicts": [
            {
                "path": "benchmark_details.overview",
                "status": "supported",
                "specificity": "specific",
                "info_in_source": "na",
                "note": note,
            }
        ],
        "risk_verdicts": [
            {
                "category": "Hallucination",
                "relevant_and_grounded": "yes",
                "note": note,
            }
        ],
    }


def _run_meta(names):
    return {
        "model": "claude-sonnet-4-6",
        "transport": "api",
        "execution": "anthropic-api-sync",
        "sdk_version": "test",
        "request_params": {
            "model": "claude-sonnet-4-6",
            "max_tokens": 16000,
            "temperature": 0.0,
            "thinking": "omitted (always-on for this model)",
            "stream": False,
        },
        "prompt_md5": "prompt",
        "schema_md5": "schema",
        "prompt_version": "v2-eee",
        "input_mode": "json",
        "total_cost_est_usd": 1.25,
        "batches": [
            {
                "started": "2026-07-17T10:00:00+00:00",
                "finished": "2026-07-17T10:01:00+00:00",
                "cards": [{"name": name, "ok": True} for name in names],
            }
        ],
    }


def _tree_digest(path):
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def artifacts(tmp_path):
    names = ["a", "b", "c"]
    changed = ["b"]
    original_inputs = tmp_path / "original_inputs"
    repaired_inputs = tmp_path / "repaired_inputs"
    original_verdicts = tmp_path / "original_verdicts"
    rerun_verdicts = tmp_path / "rerun_verdicts"
    for name in names:
        _write_json(original_inputs / f"{name}.json", _input(name, f"old evidence {name}"))
        repaired_source = "complete corrected evidence b" if name == "b" else f"old evidence {name}"
        _write_json(repaired_inputs / f"{name}.json", _input(name, repaired_source))
        _write_json(original_verdicts / f"{name}.json", _verdict(name, f"old {name}"))
    _write_json(rerun_verdicts / "b.json", _verdict("b", "corrected b"))

    original_manifest_path = tmp_path / "original_manifest.json"
    original_cards = {
        name: {
            "bytes": (original_inputs / f"{name}.json").stat().st_size,
            "md5": _hash(original_inputs / f"{name}.json", "md5"),
        }
        for name in names
    }
    _write_json(original_manifest_path, {"cards": original_cards})

    repaired_manifest_path = tmp_path / "repaired_manifest.json"
    repaired_cards = {
        name: {
            "bytes": (repaired_inputs / f"{name}.json").stat().st_size,
            "md5": _hash(repaired_inputs / f"{name}.json", "md5"),
        }
        for name in names
    }
    _write_json(
        repaired_manifest_path,
        {
            "parent_manifest": str(original_manifest_path),
            "parent_manifest_sha256": _hash(original_manifest_path, "sha256"),
            "cards": repaired_cards,
        },
    )

    audit_path = tmp_path / "repair_audit.json"
    original_b = original_inputs / "b.json"
    repaired_b = repaired_inputs / "b.json"
    _write_json(
        audit_path,
        {
            "original_inputs_dir": str(original_inputs),
            "original_manifest": str(original_manifest_path),
            "output_inputs_dir": str(repaired_inputs),
            "repaired_manifest": str(repaired_manifest_path),
            "n_sample_cards": 3,
            "n_changed_cards": 1,
            "n_unchanged_cards_byte_identical": 2,
            "changed_card_names": changed,
            "parity_gate": {"passed": True},
            "changed_cards": [
                {
                    "name": "b",
                    "original_input_md5": _hash(original_b, "md5"),
                    "original_input_sha256": _hash(original_b, "sha256"),
                    "repaired_input_md5": _hash(repaired_b, "md5"),
                    "repaired_input_sha256": _hash(repaired_b, "sha256"),
                }
            ],
        },
    )

    original_run_meta = tmp_path / "original_run_meta.json"
    rerun_run_meta = tmp_path / "rerun_run_meta.json"
    _write_json(original_run_meta, _run_meta(names))
    _write_json(rerun_run_meta, _run_meta(changed))
    return {
        "repo": REPO,
        "original_inputs": original_inputs,
        "original_manifest": original_manifest_path,
        "original_verdicts": original_verdicts,
        "original_run_meta": original_run_meta,
        "repaired_inputs": repaired_inputs,
        "repaired_manifest": repaired_manifest_path,
        "repair_audit": audit_path,
        "rerun_verdicts": rerun_verdicts,
        "rerun_run_meta": rerun_run_meta,
        "output": tmp_path / "merged",
    }


def _merge(artifacts):
    return merge_source_parity_verdicts(**artifacts)


def test_validated_merge_replaces_exact_changed_set_and_preserves_sources(artifacts):
    source_digests = {
        key: _tree_digest(artifacts[key])
        for key in ("original_inputs", "original_verdicts", "repaired_inputs", "rerun_verdicts")
    }

    summary = _merge(artifacts)

    assert summary["n_cards"] == 3
    assert summary["n_corrected"] == 1
    assert summary["n_retained"] == 2
    results = json.loads((artifacts["output"] / "results_s150.json").read_text())
    assert results["a"]["field_verdicts"][0]["note"] == "old a"
    assert results["b"]["field_verdicts"][0]["note"] == "corrected b"
    manifest = json.loads((artifacts["output"] / "verdicts_sha256.json").read_text())
    assert manifest["cards"]["a"]["source"] == "retained_original"
    assert manifest["cards"]["b"]["source"] == "source_corrected_rerun"
    run_meta = json.loads((artifacts["output"] / "run_meta.json").read_text())
    assert len(run_meta["batches"]) == 2
    assert set(path.stem for path in (artifacts["output"] / "verdicts").glob("*.json")) == {
        "a",
        "b",
        "c",
    }
    for key, digest in source_digests.items():
        assert _tree_digest(artifacts[key]) == digest


def test_refuses_partial_rerun_set_without_creating_output(artifacts):
    (artifacts["rerun_verdicts"] / "b.json").unlink()

    with pytest.raises(MergeError, match="rerun verdict set mismatch"):
        _merge(artifacts)

    assert not artifacts["output"].exists()


def test_refuses_extra_rerun_verdict(artifacts):
    _write_json(artifacts["rerun_verdicts"] / "a.json", _verdict("a", "extra"))

    with pytest.raises(MergeError, match=r"extra=\['a'\]"):
        _merge(artifacts)


def test_refuses_invalid_corrected_verdict_against_repaired_input(artifacts):
    invalid = _verdict("b", "bad")
    invalid["field_verdicts"][0]["path"] = "wrong.path"
    _write_json(artifacts["rerun_verdicts"] / "b.json", invalid)

    with pytest.raises(MergeError, match="corrected verdict invalid for b"):
        _merge(artifacts)


def test_refuses_changed_set_that_disagrees_with_manifest_diff(artifacts):
    audit = json.loads(artifacts["repair_audit"].read_text())
    audit["changed_card_names"] = []
    audit["n_changed_cards"] = 0
    audit["n_unchanged_cards_byte_identical"] = 3
    audit["changed_cards"] = []
    _write_json(artifacts["repair_audit"], audit)

    with pytest.raises(MergeError, match="repair-audit changed-card set mismatch"):
        _merge(artifacts)


def test_refuses_existing_destination(artifacts):
    artifacts["output"].mkdir()

    with pytest.raises(MergeError, match="destination already exists"):
        _merge(artifacts)


def test_validated_merge_can_replace_an_invalid_original_schema_verdict(artifacts):
    invalid = _verdict("c", "old invalid c")
    invalid["field_verdicts"].append(dict(invalid["field_verdicts"][0]))
    _write_json(artifacts["original_verdicts"] / "c.json", invalid)

    schema_verdicts = artifacts["output"].parent / "schema_repair_verdicts"
    schema_run_meta = artifacts["output"].parent / "schema_repair_run_meta.json"
    _write_json(schema_verdicts / "c.json", _verdict("c", "schema-corrected c"))
    _write_json(schema_run_meta, _run_meta(["c"]))
    artifacts.update(
        {
            "schema_repair_verdicts": schema_verdicts,
            "schema_repair_run_meta": schema_run_meta,
            "schema_repair_names": ["c"],
        }
    )

    summary = _merge(artifacts)

    assert summary["n_corrected"] == 2
    assert summary["n_source_corrected"] == 1
    assert summary["n_schema_corrected"] == 1
    assert summary["n_retained"] == 1
    results = json.loads((artifacts["output"] / "results_s150.json").read_text())
    assert results["c"]["field_verdicts"][0]["note"] == "schema-corrected c"
    manifest = json.loads((artifacts["output"] / "verdicts_sha256.json").read_text())
    assert manifest["cards"]["c"]["source"] == "schema_corrected_rerun"
    run_meta = json.loads((artifacts["output"] / "run_meta.json").read_text())
    assert len(run_meta["batches"]) == 3
