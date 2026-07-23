import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_s_judge import validate_verdict  # noqa: E402


SCHEMA = {
    "type": "object",
    "required": ["name", "field_verdicts", "risk_verdicts"],
    "properties": {
        "name": {"type": "string"},
        "field_verdicts": {"type": "array"},
        "risk_verdicts": {"type": "array"},
    },
}


def field(path):
    return {
        "path": path,
        "status": "supported",
        "specificity": "specific",
        "info_in_source": "na",
        "note": "evidence",
    }


def test_duplicate_field_path_fails_even_when_set_coverage_matches():
    input_json = {
        "name": "card",
        "fields": [{"path": "a"}, {"path": "b"}],
        "risks": ["privacy"],
    }
    verdict = {
        "name": "card",
        "field_verdicts": [field("a"), field("a"), field("b")],
        "risk_verdicts": [{"category": "privacy", "relevant_and_grounded": "yes", "note": ""}],
    }

    problems = validate_verdict(verdict, input_json, SCHEMA)

    assert "duplicate field verdicts: ['a']" in problems


def test_duplicate_risk_category_fails_exact_coverage():
    input_json = {"name": "card", "fields": [{"path": "a"}], "risks": ["privacy"]}
    verdict = {
        "name": "card",
        "field_verdicts": [field("a")],
        "risk_verdicts": [
            {"category": "privacy", "relevant_and_grounded": "yes", "note": ""},
            {"category": "privacy", "relevant_and_grounded": "yes", "note": ""},
        ],
    }

    problems = validate_verdict(verdict, input_json, SCHEMA)

    assert any(problem.startswith("risk coverage mismatch:") for problem in problems)
