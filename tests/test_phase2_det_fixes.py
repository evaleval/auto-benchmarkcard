"""Phase-2 screen deterministic-channel fixes (synthetic fixtures only; no network).

Four defects from the regen-20 screen: OpenAlex authors garbled by entity disambiguation
(artifactsbench), code-benchmark programming languages invisible behind the not-applicable
languages shape (repobench / swe-bench-multimodal), and a multimodal benchmark stuck at
data_type=text because its images are URL strings (swe-bench-multimodal). Fixture values are
taken from the real run sidecars but encode no name-keyed logic.
"""

from auto_benchmarkcard.card_utils import extract_code_languages, extract_hf_tags
from auto_benchmarkcard.tools.composer import composer_tool as C
from auto_benchmarkcard.tools.eee.paper_resolver import _natural_author_order, _normalize_openalex_paper


# ---------------------------------------------------------------------------------------------
# Item 1 -- OpenAlex authors: raw byline beats the disambiguated entity name


def _work(authorships):
    return {"title": "T", "authorships": authorships, "ids": {}, "locations": []}


def test_openalex_authors_prefer_raw_byline():
    # 2507.04952: the entity linker maps "Wu, Dengpeng" to a different person "Dan Wu".
    work = _work([
        {"raw_author_name": "Wu, Dengpeng", "author": {"display_name": "Dan Wu"}},
        {"raw_author_name": "Yi, Qi", "author": {"display_name": "Qi Long Yi"}},
        {"raw_author_name": "Xu, Zenan", "author": {"display_name": "Z. P. Xu"}},
    ])
    assert _normalize_openalex_paper(work)["authors"] == ["Dengpeng Wu", "Qi Yi", "Zenan Xu"]


def test_openalex_authors_fall_back_to_display_name():
    work = _work([{"author": {"display_name": "Jane Doe"}}, {"raw_author_name": ""}])
    assert _normalize_openalex_paper(work)["authors"] == ["Jane Doe"]


def test_natural_author_order():
    assert _natural_author_order("Zhang, Chenchen") == "Chenchen Zhang"
    assert _natural_author_order("Chenchen Zhang") == "Chenchen Zhang"
    assert _natural_author_order("Smith, Jr.") == "Smith, Jr."
    assert _natural_author_order("Doe, Jane, Extra") == "Doe, Jane, Extra"


# ---------------------------------------------------------------------------------------------
# Item 2 -- programming-language harvest and the data.format surface


def test_code_languages_from_repo_id_and_tags():
    # the real repobench repos: bare "code" tag, languages only in the repo id.
    hf = {"id": "tianyang/repobench_python_v1.1",
          "tags": ["task_categories:text-generation", "language:en", "code"]}
    assert extract_code_languages(hf) == ["Python"]
    hf_java = {"id": "tianyang/repobench_java_v1.1", "tags": ["code"]}
    assert extract_code_languages(hf_java) == ["Java"]


def test_code_languages_from_readme_languages_section():
    # swe-bench-multimodal shape: languages live only in the README Languages section; the
    # lowercase harness command outside the section must not add Python.
    hf = {"id": "SWE-bench/SWE-bench_Multimodal", "tags": ["modality:text"],
          "readme_markdown": (
              "---\ndataset_info:\n  features: []\n---\n\n**Languages**\n\n"
              "Source code is JavaScript and TypeScript. Issue text is primarily English.\n\n"
              "**Usage**\n\n```bash\npython -m swebench.harness.run_evaluation\n```\n")}
    assert extract_code_languages(hf) == ["JavaScript", "TypeScript"]


def test_code_languages_empty_without_signals():
    assert extract_code_languages(None) == []
    assert extract_code_languages({"id": "org/wordle-arena", "tags": ["modality:text"]}) == []
    # "go to" prose and a Google mention never trip the case-sensitive section scan.
    hf = {"id": "org/x", "tags": [],
          "readme_markdown": "## Languages\n\nPlease go to the Google docs page for details.\n"}
    assert extract_code_languages(hf) == []


def test_c3_tier3_appends_code_languages_to_format():
    card = {"benchmark_details": {"languages": ["not-applicable"], "overview": "software engineering"},
            "methodology": {"metrics": [], "methods": []},
            "data": {"format": "parquet (HuggingFace hosting format)"}}
    hf = {"id": "SWE-bench/SWE-bench_Multimodal", "tags": ["modality:text"],
          "readme_markdown": "**Languages**\n\nSource code is JavaScript and TypeScript.\n"}
    C.apply_coverage_backstops(card, {}, {}, hf)
    assert card["data"]["format"] == \
        "parquet (HuggingFace hosting format); code in JavaScript, TypeScript"
    # idempotent: a second pass never double-appends.
    C.apply_coverage_backstops(card, {}, {}, hf)
    assert card["data"]["format"].count("code in") == 1


def test_c3_tier3_fills_empty_format():
    card = {"benchmark_details": {"languages": ["not-applicable"], "overview": "code completion"},
            "methodology": {"metrics": ["pass@1"], "methods": []},
            "data": {"format": "Not specified"}}
    hf = {"id": "tianyang/repobench_python_v1.1", "tags": ["code"]}
    C.apply_coverage_backstops(card, {}, {}, hf)
    assert card["data"]["format"] == "Code in Python"


def test_c3_tier3_noop_without_signals_or_off_code():
    # code benchmark, no PL signals: byte-identical.
    card = {"benchmark_details": {"languages": ["not-applicable"], "overview": "code completion"},
            "methodology": {"metrics": ["pass@1"], "methods": []},
            "data": {"format": "parquet (HuggingFace hosting format)"}}
    C.apply_coverage_backstops(card, {}, {}, {"id": "org/somebench", "tags": []})
    assert card["data"]["format"] == "parquet (HuggingFace hosting format)"
    # non-code benchmark with PL-looking id: never touched.
    card2 = {"benchmark_details": {"languages": ["English"], "overview": "emotion classification"},
             "methodology": {"metrics": [], "methods": []},
             "data": {"format": "parquet (HuggingFace hosting format)"}}
    C.apply_coverage_backstops(card2, {}, {}, {"id": "google-research-datasets/go_emotions",
                                               "tags": []})
    assert card2["data"]["format"] == "parquet (HuggingFace hosting format)"


# ---------------------------------------------------------------------------------------------
# Item 4 -- EEE-declared modalities extend data_type


def _eee(raw_modality, raw_categories):
    return {"metrics": {"llm_stats.x.score": {"additional_details": {
        "raw_modality": raw_modality, "raw_categories": raw_categories}}}}


def test_c3_tier4_eee_multimodal_adds_image():
    # swe-bench-multimodal: HF modality tags say text (images are URL strings), EEE says
    # multimodal with a vision category.
    card = {"benchmark_details": {"data_type": "text", "languages": ["not-applicable"],
                                  "overview": "software engineering"},
            "methodology": {"metrics": [], "methods": []}, "data": {}}
    C.apply_coverage_backstops(
        card, {}, _eee("multimodal", '["vision","agents","code","multimodal","reasoning"]'))
    assert card["benchmark_details"]["data_type"] == "image, text"


def test_c3_tier4_video_subsumes_vision():
    # perceptiontest: audio/text/video already present; the ambiguous vision category must
    # not add a still-image claim.
    card = {"benchmark_details": {"data_type": "audio, text, video", "languages": ["English"],
                                  "overview": "video benchmark"},
            "methodology": {"metrics": [], "methods": []}, "data": {}}
    C.apply_coverage_backstops(card, {}, _eee("multimodal", '["video","vision","multimodal"]'))
    assert card["benchmark_details"]["data_type"] == "audio, text, video"


def test_c3_tier4_noop_without_multimodal_signal():
    # repobench: raw_modality text, category code -> no modality claim.
    card = {"benchmark_details": {"data_type": "text", "languages": ["not-applicable"],
                                  "overview": "code completion"},
            "methodology": {"metrics": [], "methods": []}, "data": {}}
    C.apply_coverage_backstops(card, {}, _eee("text", '["code","reasoning"]'))
    assert card["benchmark_details"]["data_type"] == "text"


def test_c3_tier4_never_fills_ns():
    card = {"benchmark_details": {"data_type": "Not specified", "languages": ["Not specified"],
                                  "overview": "captioning"},
            "methodology": {"metrics": [], "methods": []}, "data": {}}
    C.apply_coverage_backstops(card, {}, _eee("multimodal", '["audio"]'))
    assert card["benchmark_details"]["data_type"] == "Not specified"


def test_extract_hf_tags_output_unchanged():
    # frozen instruments consume extract_hf_tags: the harvest lives outside it, so a repo
    # with code + PL signals still yields the exact S4.3 shape and untouched data.format.
    out = extract_hf_tags({"id": "tianyang/repobench_python_v1.1",
                           "tags": ["code", "modality:text", "format:parquet"]})
    assert out["benchmark_details.data_type"] == "code"
    assert out["benchmark_details.languages"] == ["not-applicable"]
    assert out["data.format"] == "parquet (HuggingFace hosting format)"
