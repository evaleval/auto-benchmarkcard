"""Code-benchmark detection in extract_hf_tags (contract S4.3). A code dataset carries no
HF code-modality tag, so an explicit code signal (bare "code", code/programming task tag, or
a language:code value) must pin data_type="code" and languages=["not-applicable"]. Modeled on
the real bigcodebench tags (language:code + modality:text + bare code). Network-free.
"""

from auto_benchmarkcard.card_utils import extract_hf_tags


def _hf(tags):
    return {"tags": tags}


def test_code_signals_pin_code_shape_over_modality_and_language():
    # the real bigcodebench shape: a text modality and a language:code value that must lose.
    out = extract_hf_tags(_hf([
        "language:code", "modality:text", "license:apache-2.0", "code",
    ]))
    assert out["benchmark_details.data_type"] == "code"
    assert out["benchmark_details.languages"] == ["not-applicable"]


def test_non_code_bench_untouched():
    out = extract_hf_tags(_hf([
        "language:en", "modality:text", "task_categories:question-answering",
    ]))
    assert out["benchmark_details.data_type"] == "text"
    assert out["benchmark_details.languages"] == ["English"]
