"""Unit tests for verify_hf_match: the single-candidate HF dataset-match LLM verifier.

Network-free: the LLM handler is monkeypatched to return canned JSON or raise. The checks are
general mechanisms (parse, retry-once, error fallback), never keyed on a benchmark-name list.
"""

import json

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr


class _FakeLLM:
    """Returns queued responses in order; a callable item is invoked (to raise)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        item = self._responses.pop(0)
        if callable(item):
            return item()
        return item


def _patch_llm(monkeypatch, responses):
    fake = _FakeLLM(responses)
    monkeypatch.setattr(pr, "get_llm_handler", lambda *a, **k: fake)
    return fake


_IDENTITY = {
    "suite_name": "WidgetBench",
    "full_name": "Widget Benchmark",
    "subject_text": "WidgetBench measures widget assembly reasoning over factory schematics",
    "metrics": ["accuracy"],
    "eval_library": "lm-eval",
}
_CANDIDATE = {
    "repo_id": "acme/widgetbench",
    "readme_lead": "WidgetBench is a dataset of factory schematics for widget assembly reasoning.",
    "pretty_name": "WidgetBench",
    "description": "Widget assembly reasoning",
    "task_categories": ["question-answering"],
    "tags": ["arxiv:2401.00001"],
}


def test_accept_matching_identity(monkeypatch):
    _patch_llm(monkeypatch, [json.dumps(
        {"is_match": True, "confidence": 0.92, "reasoning": "subject and domain align"})])
    v = pr.verify_hf_match(_IDENTITY, _CANDIDATE)
    assert v["is_match"] is True and v["confidence"] == 0.92
    assert "error" not in v


def test_reject_name_collision_different_domain(monkeypatch):
    # ActionBench identity vs an ActionMesh (3D-mesh) README: a name collision, not a match.
    identity = {"suite_name": "ActionBench",
                "subject_text": "ActionBench evaluates text-to-image generation of human actions",
                "metrics": ["fid"], "eval_library": None}
    candidate = {"repo_id": "facebook/actionbench",
                 "readme_lead": "ActionMesh: reconstructing 3D human body meshes from monocular video.",
                 "pretty_name": "ActionMesh", "description": "3D mesh from video",
                 "task_categories": ["video"], "tags": ["arxiv:2601.16148"]}
    _patch_llm(monkeypatch, [json.dumps(
        {"is_match": False, "confidence": 0.9, "reasoning": "3D-mesh domain, not text-to-image"})])
    v = pr.verify_hf_match(identity, candidate)
    assert v["is_match"] is False and v["confidence"] == 0.9


def test_retry_once_on_parse_failure(monkeypatch):
    fake = _patch_llm(monkeypatch, [
        "not json at all",
        json.dumps({"is_match": True, "confidence": 0.8, "reasoning": "ok on retry"})])
    v = pr.verify_hf_match(_IDENTITY, _CANDIDATE)
    assert v["is_match"] is True and fake.calls == 2


def test_error_fallback_on_llm_exception(monkeypatch):
    def _boom():
        raise RuntimeError("LLM down")
    _patch_llm(monkeypatch, [_boom])
    v = pr.verify_hf_match(_IDENTITY, _CANDIDATE)
    assert v.get("error") is True
    assert v["is_match"] is False and v["confidence"] == 0.0


def test_second_parse_failure_falls_through_to_error(monkeypatch):
    fake = _patch_llm(monkeypatch, ["still not json", "again not json"])
    v = pr.verify_hf_match(_IDENTITY, _CANDIDATE)
    assert v.get("error") is True and fake.calls == 2
