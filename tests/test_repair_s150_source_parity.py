import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from repair_s150_source_parity import RepairError, repair_inputs  # noqa: E402


def _write_json(path, value, *, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=indent), encoding="utf-8")


def _prepared_input(name, source_text="[WEBPAGE]\nExisting source"):
    return {
        "name": name,
        "fields": [{"path": "purpose.goal", "value": "A goal", "is_ns": False}],
        "risks": ["privacy"],
        "source_text": source_text,
    }


def _abstract_record(text):
    return {
        "paper_abstract": text,
        "paper_abstract_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "paper_title": "Example paper",
        "recovery_source_url": "https://example.test/paper",
        "recovered_at": "2026-07-17T12:00:00Z",
        "recovery_method": "test fixture",
    }


def _fixture(tmp_path, cards, *, abstract_manifest_cards=None):
    repo = tmp_path / "repo"
    inputs = tmp_path / "original_inputs"
    inputs.mkdir(parents=True)
    sample_cards = []
    manifest_cards = {}
    default_abstract_cards = {}
    for card in cards:
        name = card["name"]
        input_json = card["input"]
        provenance = card["provenance"]
        raw = json.dumps(input_json).encode("utf-8")
        (inputs / f"{name}.json").write_bytes(raw)
        manifest_cards[name] = {"bytes": len(raw), "md5": hashlib.md5(raw).hexdigest()}
        run_dir = repo / "runs" / name
        _write_json(
            run_dir / "tool_output" / "composer" / f"provenance_{name}.json",
            provenance,
            indent=2,
        )
        _write_json(
            run_dir / "tool_output" / "composer" / f"docling_telemetry_{name}.json",
            {"degraded_to_abstract_only": card.get("abstract_only", False)},
            indent=2,
        )
        if "github" in card:
            _write_json(
                run_dir / "tool_output" / "github" / f"{name}.json",
                card["github"],
                indent=2,
            )
        source_run_dir = f"runs/{name}"
        sample_cards.append({"name": name, "source_run_dir": source_run_dir})
        if card.get("abstract_only") and "abstract" in card:
            default_abstract_cards[name] = {
                **card["abstract"],
                "source_run_dir": source_run_dir,
            }
    sample_path = repo / "eval" / "s150" / "sample.json"
    manifest_path = repo / "eval" / "s150" / "judge" / "inputs_md5.json"
    abstract_manifest_path = repo / "eval" / "s150" / "abstracts.json"
    _write_json(sample_path, {"n": len(sample_cards), "cards": sample_cards}, indent=2)
    _write_json(manifest_path, {"cards": manifest_cards}, indent=2)
    _write_json(
        abstract_manifest_path,
        {
            "version": "s150-paper-abstracts-v1",
            "cards": (
                default_abstract_cards
                if abstract_manifest_cards is None
                else abstract_manifest_cards
            ),
        },
        indent=2,
    )
    return {
        "repo": repo,
        "sample_path": sample_path,
        "inputs_dir": inputs,
        "original_manifest_path": manifest_path,
        "abstract_manifest_path": abstract_manifest_path,
        "output_dir": tmp_path / "repaired_inputs",
        "repaired_manifest_path": repo / "eval" / "repair" / "inputs_md5.json",
        "audit_path": repo / "eval" / "repair" / "audit.json",
    }


def test_repairs_complete_abstract_and_github_and_preserves_unaffected_bytes(tmp_path):
    abstract_changed = _prepared_input("abstract-card")
    github_changed = _prepared_input("github-card")
    unchanged = _prepared_input("unchanged")
    args = _fixture(
        tmp_path,
        [
            {
                "name": "abstract-card",
                "input": abstract_changed,
                "abstract_only": True,
                "abstract": _abstract_record(
                    "Leading context. Exact abstract sentence. Trailing context."
                ),
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "abstract",
                            "evidence": "Exact abstract sentence.",
                            "evidence_ids": ["E01"],
                        },
                    }
                },
            },
            {
                "name": "github-card",
                "input": github_changed,
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "github",
                            "evidence": "Exact GitHub sentence.",
                            "evidence_ids": ["G01"],
                        }
                    }
                },
                "github": {
                    "success": True,
                    "text": "README start. Exact GitHub sentence. README end.",
                    "url": "https://github.com/example/repo",
                    "http_status": 200,
                },
            },
            {
                "name": "unchanged",
                "input": unchanged,
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "html",
                            "evidence": "Existing source",
                            "evidence_ids": ["E03"],
                        }
                    }
                },
            },
        ],
    )

    audit = repair_inputs(**args)

    assert audit["changed_card_names"] == ["abstract-card", "github-card"]
    abstract_repaired = json.loads(
        (args["output_dir"] / "abstract-card.json").read_text()
    )
    assert abstract_repaired["fields"] == abstract_changed["fields"]
    assert abstract_repaired["risks"] == abstract_changed["risks"]
    assert "[PRIMARY SOURCE: COMPLETE PAPER TITLE AND ABSTRACT]" in abstract_repaired[
        "source_text"
    ]
    assert "Leading context. Exact abstract sentence." in abstract_repaired["source_text"]
    github_repaired = json.loads((args["output_dir"] / "github-card.json").read_text())
    assert "[PRIMARY SOURCE: COMPLETE PERSISTED GITHUB README]" in github_repaired[
        "source_text"
    ]
    assert "README start. Exact GitHub sentence." in github_repaired["source_text"]
    assert (args["output_dir"] / "unchanged.json").read_bytes() == (
        args["inputs_dir"] / "unchanged.json"
    ).read_bytes()
    assert audit["parity_gate"]["abstract_required_cards"] == ["abstract-card"]
    assert audit["parity_gate"]["github_required_cards"] == ["github-card"]
    assert audit["parity_gate"]["all_required_full_payloads_present_after_repair"] is True
    manifest = json.loads(args["repaired_manifest_path"].read_text())
    for name in ("abstract-card", "github-card", "unchanged"):
        output = (args["output_dir"] / f"{name}.json").read_bytes()
        assert manifest["cards"][name] == {
            "bytes": len(output),
            "md5": hashlib.md5(output).hexdigest(),
        }


def test_fails_when_assembled_source_marker_and_evidence_are_both_absent(tmp_path):
    input_json = _prepared_input("card", source_text="[EEE EVALUATION DATA]\n{}")
    args = _fixture(
        tmp_path,
        [
            {
                "name": "card",
                "input": input_json,
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "html",
                            "evidence": "Persisted webpage evidence.",
                            "evidence_ids": ["E01"],
                        }
                    }
                },
            }
        ],
    )

    with pytest.raises(RepairError, match="assembled-source parity failure"):
        repair_inputs(**args)


def test_fails_closed_on_unclassified_source_label(tmp_path):
    args = _fixture(
        tmp_path,
        [
            {
                "name": "card",
                "input": _prepared_input("card"),
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "new_remote_source",
                            "evidence": "Evidence.",
                            "evidence_ids": [],
                        }
                    }
                },
            }
        ],
    )

    with pytest.raises(RepairError, match="unclassified provenance source label"):
        repair_inputs(**args)
    assert not args["output_dir"].exists()


def test_fails_closed_on_original_input_hash_mismatch(tmp_path):
    args = _fixture(
        tmp_path,
        [
            {
                "name": "card",
                "input": _prepared_input("card"),
                "abstract_only": True,
                "abstract": _abstract_record("Evidence."),
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "abstract",
                            "evidence": "Evidence.",
                            "evidence_ids": [],
                        }
                    }
                },
            }
        ],
    )
    (args["inputs_dir"] / "card.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RepairError, match="original input guard failed"):
        repair_inputs(**args)
    assert not args["output_dir"].exists()


def test_fails_without_dropping_source_when_cap_would_be_exceeded(tmp_path, monkeypatch):
    input_json = _prepared_input("card", source_text="1234567890")
    args = _fixture(
        tmp_path,
        [
            {
                "name": "card",
                "input": input_json,
                "abstract_only": True,
                "abstract": _abstract_record("Evidence."),
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "abstract",
                            "evidence": "Evidence.",
                            "evidence_ids": [],
                        }
                    }
                },
            }
        ],
    )
    monkeypatch.setattr("repair_s150_source_parity.REPAIRED_SOURCE_CAP", 20)

    with pytest.raises(RepairError, match="pre-existing source will not be dropped"):
        repair_inputs(**args)
    assert not args["output_dir"].exists()


def test_records_nonliteral_composer_evidence_without_rejecting_source_identity(tmp_path):
    args = _fixture(
        tmp_path,
        [
            {
                "name": "card",
                "input": _prepared_input("card"),
                "abstract_only": True,
                "abstract": _abstract_record("A different abstract."),
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "abstract",
                            "evidence": "Required evidence sentence.",
                            "evidence_ids": [],
                        }
                    }
                },
            }
        ],
    )

    audit = repair_inputs(**args)
    supplement = audit["changed_cards"][0]["supplements"][0]
    assert supplement["provenance_evidence_nonliteral_or_missing"] == ["purpose.goal"]


def test_fails_closed_on_missing_abstract_manifest_card(tmp_path):
    args = _fixture(
        tmp_path,
        [
            {
                "name": "card",
                "input": _prepared_input("card"),
                "abstract_only": True,
                "provenance": {
                    "purpose": {
                        "goal": {
                            "source": "abstract",
                            "evidence": "Evidence.",
                            "evidence_ids": [],
                        }
                    }
                },
            }
        ],
        abstract_manifest_cards={},
    )

    with pytest.raises(RepairError, match="card set must exactly match"):
        repair_inputs(**args)
