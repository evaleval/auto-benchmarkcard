"""Risk stage retrieve-then-rerank. The detector over-fetches candidates, one LLM
call selects 0-5 with benchmark-specific justifications, LLM failure falls back to
the detector order. Network-free: fake handler, fake detector, fake engine.
"""

import json

import pytest

import auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool as tool
from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import (
    create_usecase_from_benchmark_card,
    identify_and_integrate_risks,
    identify_risks_from_benchmark_metadata,
    select_risks_with_justifications,
)


class RiskStub:
    def __init__(self, rid, name, description="A generic description."):
        self.id = rid
        self.name = name
        self.description = description
        self.tag = rid.split("-")[-1]
        self.type = "Risk"
        self.concern = None
        self.url = f"https://example.org/{rid}"
        self.isDefinedByTaxonomy = "ibm-risk-atlas"


class FakeHandler:
    def __init__(self, response=None, exc=None):
        self.prompts = []
        self.response = response
        self.exc = exc

    def generate(self, prompt, response_format=None):
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return self.response


CANDIDATES = [
    RiskStub("atlas-data-privacy", "Personal information in data",
             "Personal or sensitive information may be exposed."),
    RiskStub("atlas-data-bias", "Data bias",
             "Unrepresentative data skews evaluation results."),
    RiskStub("atlas-eval-gaming", "Metric gaming",
             "Models can overfit the metric without real capability."),
]


def _card():
    return {
        "benchmark_details": {
            "name": "HealthChat",
            "overview": "A benchmark of real user health conversations for evaluating medical chat assistants.",
            "data_type": "text",
            "domains": ["healthcare"],
            "languages": ["English"],
        },
        "purpose_and_intended_users": {
            "goal": "evaluate the safety of medical advice",
            "audience": ["Model developers"],
            "tasks": ["open-ended medical question answering"],
        },
        "data": {
            "source": "Conversations collected from a public health forum",
            "size": "10k dialogues",
        },
        "methodology": {},
        "ethical_and_legal_considerations": {},
    }


def _frame():
    return {
        "target_id": "target",
        "nodes": [
            {"id": "target", "type": "target", "name": "HealthChat", "aliases": []},
            {"id": "sibling:medqa", "type": "sibling", "name": "MedQA", "aliases": []},
            {"id": "split:test", "type": "split", "name": "test", "aliases": []},
        ],
    }


@pytest.fixture
def wired(monkeypatch):
    """Patch the network-touching pieces; returns the dict the fake detector records
    into. Tests going through identify_and_integrate_risks (no llm_handler kwarg)
    set seen["handler"] so the rerank seam yields their fake."""
    seen = {}

    def fake_detect(ai_atlas_nexus, usecases, inference_engine, taxonomy, max_risk):
        seen["usecases"] = usecases
        seen["max_risk"] = max_risk
        return [list(CANDIDATES)]

    monkeypatch.setattr(tool, "create_inference_engine", lambda: object())
    monkeypatch.setattr(tool, "_get_nexus", lambda: object())
    monkeypatch.setattr(tool, "identify_risks_with_benchmark_detector", fake_detect)
    monkeypatch.setattr(tool, "_get_rerank_handler", lambda: seen["handler"])
    return seen


def _selection_json(*pairs):
    return json.dumps({"selected": [{"id": i, "justification": j} for i, j in pairs]})


def test_conditioning_block_has_card_fields_and_frame_names(wired):
    handler = FakeHandler(response=_selection_json())
    identify_risks_from_benchmark_metadata(_card(), frame=_frame(), llm_handler=handler)

    assert len(handler.prompts) == 1
    prompt = handler.prompts[0]
    # card conditioning: overview, domain, task, data type, source, audience
    assert "real user health conversations" in prompt
    assert "healthcare" in prompt
    assert "open-ended medical question answering" in prompt
    assert "Data type: text" in prompt
    assert "Conversations collected from a public health forum" in prompt
    assert "Intended audience: Model developers" in prompt
    # frame node names
    assert "MedQA" in prompt
    # candidate risks are listed with id, name, description
    assert "atlas-data-privacy | Personal information in data" in prompt
    # the detector saw the same conditioned usecase, and over-fetched
    assert "MedQA" in wired["usecases"][0]
    assert wired["max_risk"] == tool.RISK_CANDIDATE_COUNT


def test_selection_and_justification_propagate_to_card(wired):
    handler = FakeHandler(response=_selection_json(
        ("atlas-data-privacy", "collects real user health conversations -> privacy exposure"),
        ("atlas-data-bias", "single public forum source skews the population"),
    ))
    wired["handler"] = handler
    telemetry = {}
    card = identify_and_integrate_risks(_card(), frame=_frame(), telemetry_out=telemetry)

    risks = card["possible_risks"]
    assert [r["category"] for r in risks] == ["Personal information in data", "Data bias"]
    assert risks[0]["justification"] == (
        "collects real user health conversations -> privacy exposure")
    assert all(isinstance(r["description"], str) for r in risks)
    assert telemetry["selected_count"] == 2
    assert telemetry["fallback"] is False
    assert telemetry["candidate_count"] == 3
    assert telemetry["selections"][0]["id"] == "atlas-data-privacy"


def test_zero_selection_is_valid_and_leaves_card_unchanged(wired):
    handler = FakeHandler(response=_selection_json())
    wired["handler"] = handler
    telemetry = {}
    card = identify_and_integrate_risks(_card(), frame=None, telemetry_out=telemetry)

    assert "possible_risks" not in card
    assert telemetry["selected_count"] == 0
    assert telemetry["fallback"] is False
    assert telemetry["candidate_count"] == 3


def test_llm_exception_falls_back_to_detector_order(wired):
    handler = FakeHandler(exc=RuntimeError("boom"))
    risks, telemetry = identify_risks_from_benchmark_metadata(
        _card(), frame=None, llm_handler=handler)

    assert [r["id"] for r in risks] == [c.id for c in CANDIDATES]
    assert all("justification" not in r for r in risks)
    assert telemetry["fallback"] is True
    assert telemetry["fallback_reason"].startswith("llm_error")
    assert telemetry["selected_count"] == 3


def test_unparseable_response_falls_back(wired):
    handler = FakeHandler(response="I cannot answer in JSON, sorry.")
    risks, telemetry = identify_risks_from_benchmark_metadata(
        _card(), frame=None, llm_handler=handler)

    assert len(risks) == 3
    assert telemetry["fallback"] is True
    assert telemetry["fallback_reason"] == "unparseable_response"


def test_selection_parsing_drops_unknown_ids_dedupes_and_caps():
    candidates = [tool._format_risk(c) for c in CANDIDATES]
    items = [{"id": "atlas-data-privacy", "justification": "a"},
             {"id": "atlas-data-privacy", "justification": "dup"},
             {"id": "no-such-risk", "justification": "x"},
             {"id": "Metric gaming", "justification": "matched by name"},
             {"id": "atlas-data-bias", "justification": "b"}]
    handler = FakeHandler(response=json.dumps({"selected": items}))
    result = select_risks_with_justifications(handler, "summary", candidates, max_risk=2)

    ids = [item["candidate"]["id"] for item in result["selected"]]
    assert ids == ["atlas-data-privacy", "atlas-eval-gaming"]
    assert result["invalid_items"] == 1
    assert result["selected"][1]["justification"] == "matched by name"


def test_bare_list_and_prose_wrapped_json_both_parse():
    candidates = [tool._format_risk(c) for c in CANDIDATES]
    bare = FakeHandler(response='[{"id": "atlas-data-bias", "justification": "j"}]')
    result = select_risks_with_justifications(bare, "s", candidates)
    assert [i["candidate"]["id"] for i in result["selected"]] == ["atlas-data-bias"]

    wrapped = FakeHandler(response='Here is my answer:\n```json\n'
                                   '{"selected": [{"id": "atlas-eval-gaming", "justification": "j"}]}\n```')
    result = select_risks_with_justifications(wrapped, "s", candidates)
    assert [i["candidate"]["id"] for i in result["selected"]] == ["atlas-eval-gaming"]


def test_no_candidates_short_circuits_without_llm_call(monkeypatch):
    monkeypatch.setattr(tool, "create_inference_engine", lambda: object())
    monkeypatch.setattr(tool, "_get_nexus", lambda: object())
    monkeypatch.setattr(tool, "identify_risks_with_benchmark_detector",
                        lambda **kwargs: [[]])
    handler = FakeHandler(response=_selection_json())
    risks, telemetry = identify_risks_from_benchmark_metadata(
        _card(), frame=None, llm_handler=handler)

    assert risks == []
    assert handler.prompts == []
    assert telemetry["candidate_count"] == 0
    assert telemetry["fallback"] is False


def test_usecase_builder_without_frame_stays_frame_free():
    usecase = create_usecase_from_benchmark_card(_card())
    assert "MedQA" not in usecase
    assert "Named entities" not in usecase
    assert "Data type: text" in usecase


def test_worker_passes_frame_and_saves_telemetry(monkeypatch):
    import auto_benchmarkcard.workers as workers

    captured = {}

    def fake_identify(card, frame=None, telemetry_out=None):
        captured["frame"] = frame
        captured["card"] = card
        if isinstance(telemetry_out, dict):
            telemetry_out.update({"candidate_count": 3, "selected_count": 1,
                                  "fallback": False})
        return dict(card, possible_risks=[
            {"category": "Data bias", "description": "d", "url": "http://x",
             "justification": "j"}])

    saved = []

    class FakeOutputManager:
        def save_tool_output(self, data, tool_name, filename):
            saved.append((tool_name, filename, data))
            return f"/fake/{tool_name}/{filename}"

    monkeypatch.setattr(workers, "identify_and_integrate_risks", fake_identify)

    frame = _frame()
    state = {
        "query": "HealthChat",
        "composed_card": {"benchmark_card": _card(), "frame": frame},
        "output_manager": FakeOutputManager(),
    }
    result = workers.run_risk_identification(state)

    assert captured["frame"] is frame
    assert captured["card"] == _card()
    telemetry_writes = [s for s in saved if s[0] == "ai_atlas_nexus"
                        and s[1].startswith("risk_telemetry_")]
    assert len(telemetry_writes) == 1
    assert telemetry_writes[0][2]["selected_count"] == 1
    card_out = result["risk_enhanced_card"]["benchmark_card"]
    assert card_out["possible_risks"][0]["justification"] == "j"
