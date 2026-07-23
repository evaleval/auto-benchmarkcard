"""Tests for the human-baseline relocation (#2): the deterministic detector and
_relocate_human_baseline. Uses the verbatim d6_repilot_batchA baseline_results values so the
detector is exercised against real ground truth. Network-free.
"""

import json
import os

from auto_benchmarkcard.eee_workflow import (
    _enrich_baseline_results,
    _is_human_baseline_text,
    _relocate_human_baseline,
)
from auto_benchmarkcard.output import OutputManager

# Verbatim d6_repilot_batchA methodology.baseline_results values.
_AALCR = (
    "Human performance revealed the benchmark's challenging nature - individual evaluators "
    "achieved modest accuracy rates, typically answering 40-60% of questions correctly on "
    "the first attempt."
)
_BIOLP = (
    "Only OpenAI o1-preview scored similarly to the performance of human experts, while "
    "other language models demonstrated substantially worse performance, and in most cases "
    "couldn’t correctly identify the mistake."
)
_BIGCODE = (
    "An extensive evaluation of 60 LLMs shows that LLMs are not yet capable of following "
    "complex instructions to use function calls precisely, with scores up to 60%, "
    "significantly lower than the human performance of 97%."
)


def _eee_eval(n=13):
    return {
        "evaluation_summary": {
            "primary_metric": "score",
            "total_models_evaluated": n,
            "top_performers": [
                {"model": "m1", "score": 0.7120},
                {"model": "m2", "score": 0.7000},
            ],
            "score_statistics": {"mean": 0.5695, "std_dev": 0.1958},
        }
    }


def test_detector_fires_only_on_human_text():
    assert _is_human_baseline_text(_AALCR) is True
    # model baselines that merely reference humans must NOT match (negative guard)
    assert _is_human_baseline_text(_BIOLP) is False
    assert _is_human_baseline_text(_BIGCODE) is False
    assert _is_human_baseline_text("Not specified") is False
    assert _is_human_baseline_text(None) is False


def test_relocate_resets_baseline_then_eee_injects(tmp_path):
    om = OutputManager("demo-aalcr", base_path=str(tmp_path))
    safe = om.benchmark_name
    # the d6 reality: human_baseline already populated -> move-branch skipped, reset only
    card = {"benchmark_card": {"methodology": {
        "baseline_results": _AALCR,
        "human_baseline": "Individual human evaluators achieved 40-60%.",
    }}}
    assert _relocate_human_baseline(card, om, safe) is True
    meth = card["benchmark_card"]["methodology"]
    assert meth["baseline_results"] == "Not specified"
    assert meth["human_baseline"] == "Individual human evaluators achieved 40-60%."
    # now the EEE model scores inject authoritatively
    entry = _enrich_baseline_results(card, _eee_eval())
    assert entry is not None
    assert meth["baseline_results"].startswith(
        "Based on 13 model evaluations from Every Eval Ever"
    )


def test_relocate_moves_into_empty_human_baseline_carrying_provenance(tmp_path):
    om = OutputManager("demo-move", base_path=str(tmp_path))
    safe = om.benchmark_name
    base = {"methodology": {"baseline_results": {
        "source": "hf_readme/stated", "evidence": _AALCR, "evidence_ids": ["E14"]}}}
    om.save_tool_output(base, "composer", f"provenance_{safe}.json")

    card = {"methodology": {"baseline_results": _AALCR, "human_baseline": "Not specified"}}
    assert _relocate_human_baseline(card, om, safe) is True
    assert card["methodology"]["human_baseline"] == _AALCR
    assert card["methodology"]["baseline_results"] == "Not specified"

    path = os.path.join(om.get_tool_output_path("composer"), f"provenance_{safe}.json")
    with open(path) as f:
        prov = json.load(f)["methodology"]
    assert prov["human_baseline"]["source"] == "hf_readme/stated"  # carried over from baseline


def test_does_not_clobber_a_genuine_model_baseline(tmp_path):
    om = OutputManager("demo-bigcode", base_path=str(tmp_path))
    safe = om.benchmark_name
    card = {"methodology": {
        "baseline_results": _BIGCODE,
        "human_baseline": "Human performance on the benchmark is 97%.",
    }}
    assert _relocate_human_baseline(card, om, safe) is False
    assert card["methodology"]["baseline_results"] == _BIGCODE  # untouched


def test_no_relocation_leaves_baseline_for_genuine_value(tmp_path):
    om = OutputManager("demo-model", base_path=str(tmp_path))
    safe = om.benchmark_name
    card = {"methodology": {"baseline_results": "GPT-4 scores 0.81 (exact match).",
                            "human_baseline": "Not specified"}}
    assert _relocate_human_baseline(card, om, safe) is False
    assert card["methodology"]["baseline_results"] == "GPT-4 scores 0.81 (exact match)."
