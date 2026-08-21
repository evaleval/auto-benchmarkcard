"""Tests for the EEE model-graded -> methodology.judge_uses_llm injection (#4).

Mirrors the interpretation injection. Verifies the model-graded signal derivation and the
conservative guards (already-True, genuine cluster, text-grounded, aboutness exec demotion).
Network-free.
"""

import pytest

from auto_benchmarkcard.eee_workflow import (
    _derive_judge_uses_llm,
    _inject_eee_judge_llm,
)
from auto_benchmarkcard.output import OutputManager

# aa-lcr HF "Scoring Approach": an unnamed-checker sentence, the named-model sentence, and the
# checker's own instruction (the text V4-Flash misread as the model-under-test's answer format).
_HF_AALCR = {"readme_markdown": (
    "## Scoring Approach\n"
    "We use an LLM-based equality checker to evaluate responses. "
    "Qwen3 235B A22B 2507 Non-reasoning is used as the equality checker model. "
    "The checker replies only with CORRECT or INCORRECT.")}
_MISREAD_DENIAL = {"methodology": {"judge_uses_llm": {
    "source": "hf", "status": "stated", "evidence_ids": ["E1"],
    "evidence": "Reply only with CORRECT or INCORRECT."}}}

# Verbatim d6 EEE metric evaluation_description fragments.
_BIOLP_DESC = "BioLP-Bench is a model-graded evaluation measuring ability to find and correct mistakes"
_BIGCODE_DESC = "A benchmark that challenges LLMs to invoke multiple function calls as tools"

_EEE_DERIVED = {"source": "eee", "evidence": "", "status": "derived",
                "verified": True, "evidence_ids": []}


def _eee(desc):
    return {"metrics": {"m": {"evaluation_description": desc}}}


def _ns_cluster_card(judge=False):
    return {"methodology": {"judge_uses_llm": judge,
                            "judge_num": "Not specified",
                            "judge_models": ["Not specified"]}}


def test_derive_signal():
    assert _derive_judge_uses_llm(_eee(_BIOLP_DESC)) is True
    assert _derive_judge_uses_llm(_eee("Scoring is graded by an LLM judge.")) is True
    assert _derive_judge_uses_llm(_eee("llm-as-a-judge equality check")) is True
    assert _derive_judge_uses_llm(_eee(_BIGCODE_DESC)) is None
    assert _derive_judge_uses_llm({"metrics": {}}) is None


def test_injects_true_when_no_cluster_and_no_provenance(tmp_path):
    om = OutputManager("demo-biolp", base_path=str(tmp_path))
    card = _ns_cluster_card(judge=False)
    entry = _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, om.benchmark_name)
    assert entry == _EEE_DERIVED
    assert card["methodology"]["judge_uses_llm"] is True


def test_no_signal_is_noop(tmp_path):
    om = OutputManager("demo-bigcode", base_path=str(tmp_path))
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, _eee(_BIGCODE_DESC), om, om.benchmark_name) is None
    assert card["methodology"]["judge_uses_llm"] is False


def test_noop_when_already_true(tmp_path):
    om = OutputManager("demo-true", base_path=str(tmp_path))
    card = {"methodology": {"judge_uses_llm": True}}
    assert _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, om.benchmark_name) is None


def test_noop_when_genuine_stage_b_cluster(tmp_path):
    om = OutputManager("demo-cluster", base_path=str(tmp_path))
    card = {"methodology": {"judge_uses_llm": False, "judge_num": 2, "judge_models": ["GPT-4"]}}
    assert _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, om.benchmark_name) is None
    assert card["methodology"]["judge_uses_llm"] is False


def test_respects_aboutness_exec_demotion(tmp_path):
    om = OutputManager("demo-demoted", base_path=str(tmp_path))
    safe = om.benchmark_name
    # aboutness branch-a signature: derived False citing real execution-scoring evidence
    base = {"methodology": {"judge_uses_llm": {
        "source": "derived", "status": "derived", "evidence_ids": ["E07"],
        "evidence": "scoring runs the generated code against unit tests"}}}
    om.save_tool_output(base, "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False


def test_overrides_derived_with_empty_evidence_ids(tmp_path):
    om = OutputManager("demo-empty-eids", base_path=str(tmp_path))
    safe = om.benchmark_name
    base = {"methodology": {"judge_uses_llm": {
        "source": "derived", "status": "derived", "evidence_ids": []}}}
    om.save_tool_output(base, "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, safe) is not None
    assert card["methodology"]["judge_uses_llm"] is True


def test_respects_text_grounded_value(tmp_path):
    om = OutputManager("demo-textgrounded", base_path=str(tmp_path))
    safe = om.benchmark_name
    base = {"methodology": {"judge_uses_llm": {
        "source": "docling/stated", "status": "stated", "evidence_ids": ["E03"],
        "evidence": "scored by exact string match, no judge model"}}}
    om.save_tool_output(base, "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False


def test_clears_stale_missing_and_flagged_entries(tmp_path):
    om = OutputManager("demo-stale", base_path=str(tmp_path))
    card = _ns_cluster_card(judge="Not specified")
    card["missing_fields"] = ["methodology.judge_uses_llm", "data.source"]
    card["flagged_fields"] = {"methodology.judge_uses_llm": "x", "data.size": "y"}
    assert _inject_eee_judge_llm(card, _eee(_BIOLP_DESC), om, om.benchmark_name) is not None
    assert card["missing_fields"] == ["data.source"]
    assert "methodology.judge_uses_llm" not in card["flagged_fields"]
    assert "data.size" in card["flagged_fields"]


# J1/J2 -- HF-readme signal + named-model fill ----------------------------------

def test_derive_signal_from_hf_readme():
    # the signal lives ONLY in the HF readme (EEE has no metric description)
    assert _derive_judge_uses_llm({"metrics": {}}, _HF_AALCR) is True
    assert _derive_judge_uses_llm({"metrics": {}}, {"readme_markdown": "exact match accuracy"}) is None


def test_hf_readme_signal_injects_and_fills_cluster(tmp_path):
    om = OutputManager("demo-aalcr", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output(_HF_AALCR, "hf", f"{safe}.json")
    card = _ns_cluster_card(judge=False)
    entry = _inject_eee_judge_llm(card, {"metrics": {}}, om, safe)
    assert entry == _EEE_DERIVED
    meth = card["methodology"]
    assert meth["judge_uses_llm"] is True
    assert meth["judge_models"] == ["Qwen3 235B A22B 2507 Non-reasoning"]
    assert meth["judge_num"] == 1


def test_unnamed_signal_sets_boolean_only(tmp_path):
    om = OutputManager("demo-unnamed", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output({"readme_markdown": "We use an LLM-based equality checker to evaluate responses."},
                        "hf", f"{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) == _EEE_DERIVED
    assert card["methodology"]["judge_uses_llm"] is True
    assert card["methodology"]["judge_models"] == ["Not specified"]
    assert card["methodology"]["judge_num"] == "Not specified"


# J3 -- named judge overrides the misread text-grounded denial (but not exec) ---

def test_named_judge_overrides_misread_text_grounded_denial(tmp_path):
    om = OutputManager("demo-misread", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output(_HF_AALCR, "hf", f"{safe}.json")
    om.save_tool_output(_MISREAD_DENIAL, "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) is not None
    assert card["methodology"]["judge_uses_llm"] is True
    assert card["methodology"]["judge_models"] == ["Qwen3 235B A22B 2507 Non-reasoning"]


def test_unnamed_signal_does_not_override_text_grounded_denial(tmp_path):
    om = OutputManager("demo-unnamed-denial", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output({"readme_markdown": "We use an LLM-based equality checker to evaluate responses."},
                        "hf", f"{safe}.json")
    om.save_tool_output(_MISREAD_DENIAL, "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False


def test_named_judge_does_not_override_exec_demotion(tmp_path):
    om = OutputManager("demo-exec", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output(_HF_AALCR, "hf", f"{safe}.json")  # named-judge signal present...
    exec_demotion = {"methodology": {"judge_uses_llm": {
        "source": "derived", "status": "derived", "evidence_ids": ["E07"],
        "evidence": "scoring runs the generated code against unit tests"}}}
    om.save_tool_output(exec_demotion, "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    # ...but real execution-scoring evidence keeps judge_uses_llm False (bigcode class)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False


@pytest.mark.parametrize("readme", [
    "Answers are scored by exact match against the reference answer.",
    "Correctness is measured by exact string match of the predicted API call.",
    "The benchmark reports Pass@1 over unit tests executed in a sandbox.",
])
def test_exact_match_scorers_stay_not_judge(tmp_path, readme):
    om = OutputManager("demo-exactmatch", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output({"readme_markdown": readme}, "hf", f"{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False
    assert card["methodology"]["judge_models"] == ["Not specified"]
    assert card["methodology"]["judge_num"] == "Not specified"


def test_construction_use_model_does_not_override_exact_match_denial(tmp_path):
    """A model NAMED for dataset construction (not output scoring) must NOT flip a genuine
    exact-match FALSE: the construction veto keeps the override from firing."""
    om = OutputManager("demo-construction", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output({"readme_markdown": (
        "We use GPT-4 to evaluate the difficulty of each task during dataset construction. "
        "Model answers are scored by exact match against the gold answer.")},
        "hf", f"{safe}.json")
    om.save_tool_output({"methodology": {"judge_uses_llm": {
        "source": "hf", "status": "stated", "evidence_ids": ["E1"],
        "evidence": "scored by exact match against the gold answer"}}},
        "composer", f"provenance_{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False
    assert card["methodology"]["judge_models"] == ["Not specified"]


@pytest.mark.parametrize("readme", [
    "We use three human evaluators to judge the responses.",
    "We use Mechanical Turk workers to score each answer.",
])
def test_human_evaluation_does_not_inject(tmp_path, readme):
    om = OutputManager("demo-human", base_path=str(tmp_path))
    safe = om.benchmark_name
    om.save_tool_output({"readme_markdown": readme}, "hf", f"{safe}.json")
    card = _ns_cluster_card(judge=False)
    assert _inject_eee_judge_llm(card, {"metrics": {}}, om, safe) is None
    assert card["methodology"]["judge_uses_llm"] is False
