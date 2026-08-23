"""Tests for the referent frame builder (tools/composer/frame.py).

Synthetic fixtures mirror the real cached shapes: EEE metadata with contains/
appears_in/metrics/evaluation_summary, HF json whose card_data is {} and whose
configs and pretty_name live in the README frontmatter (the shape of every
cached tool_output/hf dump). No network: the LLM pass is exercised with fake
handler objects only.
"""

import pytest

from auto_benchmarkcard.tools.composer import frame as F


def make_eee(**overrides):
    eee = {
        "benchmark_name": "FibbleBench",
        "benchmark_type": "single",
        "appears_in": ["helm_lite"],
        "metrics": {
            "llm_stats.fibblebench.score": {
                "metric_id": "llm_stats.fibblebench.score",
                "metric_name": "FibbleBench score",
                "metric_unit": "proportion",
            },
            "llm_stats.fibblebench.accuracy": {
                "metric_id": "llm_stats.fibblebench.accuracy",
                "metric_name": "Accuracy",
            },
        },
        "evaluation_summary": {"primary_metric": "llm_stats.fibblebench.score"},
    }
    eee.update(overrides)
    return eee


def make_hf():
    return {
        "id": "acme/fibble-bench",
        "card_data": {},
        "readme_markdown": (
            "---\n"
            "license: mit\n"
            "pretty_name: Fibble Bench\n"
            "configs:\n"
            "- config_name: heldout\n"
            "  data_files: heldout.csv\n"
            "- config_name: hard\n"
            "  data_files: hard.csv\n"
            "dataset_info:\n"
            "  splits:\n"
            "  - name: train\n"
            "  - name: test\n"
            "---\n\n# Fibble Bench\n"
        ),
    }


class FakeHandler:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, response_format=None):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def node_ids(fr):
    return [n["id"] for n in fr["nodes"]]


def get_node(fr, node_id):
    return F.node_lookup(fr)[node_id]


def test_target_node_aliases():
    fr = F.build_frame("fibblebench", make_eee(), make_hf())
    target = get_node(fr, "target")
    assert fr["target_id"] == "target"
    assert target["type"] == "target"
    assert target["name"] == "fibblebench"
    # HF basename and frontmatter pretty_name survive as distinct spellings;
    # case variants of the resolved name collapse
    assert "fibble-bench" in target["aliases"]
    assert "Fibble Bench" in target["aliases"]
    assert "FibbleBench" not in target["aliases"]  # case variant of the name


def test_target_only_frame_on_empty_inputs():
    fr = F.build_frame("solobench", None, None)
    assert node_ids(fr) == ["target"]
    assert fr["telemetry"]["llm_pass"] == "skipped"
    assert fr["telemetry"]["total_nodes"] == 1


def test_eee_sibling_nodes_contains_and_appears_in():
    eee = make_eee(contains=["MMLU", "HellaSwag"])
    fr = F.build_frame("fibblebench", eee, None)
    siblings = {n["name"]: n for n in fr["nodes"] if n["type"] == "sibling"}
    assert set(siblings) == {"MMLU", "HellaSwag", "helm_lite"}
    assert "contains" in siblings["MMLU"]["description"]
    assert "appears_in" in siblings["helm_lite"]["description"]
    assert siblings["MMLU"]["id"] == "sibling:mmlu"


def test_subset_nodes_from_frontmatter_configs():
    fr = F.build_frame("fibblebench", make_eee(), make_hf())
    subsets = sorted(n["name"] for n in fr["nodes"] if n["type"] == "subset")
    assert subsets == ["hard", "heldout"]
    assert get_node(fr, "subset:heldout")["description"] == "HF dataset config."


def test_subset_nodes_from_card_data_and_size_breakdown():
    eee = make_eee(size_breakdown={"easy": 100, "hard": 50})
    hf = {"id": "a/b", "card_data": {"configs": [{"config_name": "heldout"}, "plain"]},
          "readme_markdown": ""}
    fr = F.build_frame("fibblebench", eee, hf)
    subsets = sorted(n["name"] for n in fr["nodes"] if n["type"] == "subset")
    assert subsets == ["easy", "hard", "heldout", "plain"]


def test_subset_filters_split_words_and_default():
    hf = make_hf()
    hf["readme_markdown"] = (
        "---\nconfigs:\n- config_name: default\n- config_name: test\n"
        "- config_name: ok_subset\n---\n"
    )
    fr = F.build_frame("fibblebench", None, hf)
    subsets = [n["name"] for n in fr["nodes"] if n["type"] == "subset"]
    assert subsets == ["ok_subset"]


def test_metric_nodes_role_and_raw_id_alias():
    fr = F.build_frame("fibblebench", make_eee(), None)
    score = get_node(fr, "metric:fibblebench-score")
    acc = get_node(fr, "metric:accuracy")
    assert score["role"] == "primary"
    assert "llm_stats.fibblebench.score" in score["aliases"]
    assert acc["role"] == "unknown"
    assert acc["role"] in F.METRIC_ROLES


def test_suite_member_metric_keys_do_not_become_aliases():
    # helm-lite shape: some metrics keys are suite-MEMBER benchmark names
    eee = make_eee(contains=["MMLU"])
    eee["metrics"] = {
        "MMLU": {"metric_name": "EM"},
        "llm_stats.mmlu.score": {"metric_name": "MMLU score"},
    }
    fr = F.build_frame("fibblebench", eee, None)
    em = get_node(fr, "metric:em")
    assert em["aliases"] == []  # "MMLU" names the sibling, not the metric
    score = get_node(fr, "metric:mmlu-score")
    assert score["aliases"] == ["llm_stats.mmlu.score"]
    assert F.match_mention(fr, "Scores on MMLU improved.") == ["sibling:mmlu"]


def test_metric_primary_via_alphaxiv_flag():
    eee = make_eee(evaluation_summary={})
    eee["metrics"]["llm_stats.fibblebench.accuracy"]["additional_details"] = {
        "alphaxiv_is_primary": True}
    fr = F.build_frame("fibblebench", eee, None)
    assert get_node(fr, "metric:accuracy")["role"] == "primary"


def test_version_near_benchmark_name():
    paper = "We introduce FibbleBench v2.0, a revision of the original benchmark."
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper)
    node = get_node(fr, "version:2.0")
    assert node["name"] == "v2.0"


def test_version_word_form_and_dedup_with_token_form():
    paper = "FibbleBench version 2.0 supersedes v1 entirely. The v2.0 release fixes leakage."
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper)
    versions = [n for n in fr["nodes"] if n["type"] == "version"]
    assert sorted(n["id"] for n in versions) == ["version:1", "version:2.0"]
    v2 = get_node(fr, "version:2.0")
    assert "version 2.0" in v2["aliases"]


def test_version_not_from_urls_or_model_names():
    paper = (
        "FibbleBench evaluates models. Code at https://example.com/fibble/v3/docs "
        "and doi.org/10.1234/v5.2 near FibbleBench. We evaluate GPT v4 and Llama v2 "
        "on FibbleBench."
    )
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper)
    assert [n for n in fr["nodes"] if n["type"] == "version"] == []


def test_version_without_context_is_ignored():
    paper = "Appendix table v7 lists hyperparameters for the baseline models only."
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper)
    assert [n for n in fr["nodes"] if n["type"] == "version"] == []


def test_split_nodes_from_paper_context_only():
    paper = "We test the models on the held-out test set; the train split has 5k items."
    fr = F.build_frame("fibblebench", None, None, paper_text=paper)
    splits = sorted(n["id"] for n in fr["nodes"] if n["type"] == "split")
    assert splits == ["split:test", "split:train"]


def test_bare_split_verbs_do_not_count():
    paper = "We test several models and train probes on their activations."
    fr = F.build_frame("fibblebench", None, None, paper_text=paper)
    assert [n for n in fr["nodes"] if n["type"] == "split"] == []


def test_split_nodes_from_hf_frontmatter():
    fr = F.build_frame("fibblebench", None, make_hf())
    splits = sorted(n["id"] for n in fr["nodes"] if n["type"] == "split")
    assert splits == ["split:test", "split:train"]


def test_training_canonicalizes_to_train_with_alias():
    paper = "The training set has 5k items."
    fr = F.build_frame("fibblebench", None, None, paper_text=paper)
    node = get_node(fr, "split:train")
    assert node["name"] == "train"
    assert node["aliases"] == ["training"]


def test_dedup_across_builders():
    # a suite member that equals an HF config merges into one node, first builder wins
    eee = make_eee(contains=["heldout"])
    fr = F.build_frame("fibblebench", eee, make_hf())
    ids = node_ids(fr)
    assert "sibling:heldout" in ids
    assert "subset:heldout" not in ids
    assert fr["telemetry"]["merged_duplicates"] >= 1


def test_subset_equal_to_target_is_dropped():
    hf = make_hf()
    hf["readme_markdown"] = "---\nconfigs:\n- config_name: FibbleBench\n---\n"
    fr = F.build_frame("fibblebench", None, hf)
    assert [n for n in fr["nodes"] if n["type"] == "subset"] == []


def test_llm_pass_skipped_without_handler_or_paper():
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text="text here")
    assert fr["telemetry"]["llm_pass"] == "skipped"
    fr = F.build_frame("fibblebench", make_eee(), None, llm_handler=FakeHandler("[]"))
    assert fr["telemetry"]["llm_pass"] == "skipped"


def test_llm_pass_fail_soft_on_exception():
    paper = "FibbleBench compares against MMLU."
    handler = FakeHandler(RuntimeError("engine down"))
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper, llm_handler=handler)
    assert fr["telemetry"]["llm_pass"] == "failed"
    assert "engine down" in fr["telemetry"]["llm_pass_error"]
    # deterministic frame is intact
    assert "target" in node_ids(fr)
    assert all(n["description"] != "Benchmark or dataset named in the paper (LLM sibling pass)."
               for n in fr["nodes"])


def test_llm_pass_validates_names_against_paper():
    paper = "FibbleBench compares against MMLU and HellaSwag on held-out data."
    handler = FakeHandler('["MMLU", "HellaSwag", "TotallyFakeBench", "FibbleBench"]')
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper, llm_handler=handler)
    assert fr["telemetry"]["llm_pass"] == "ok"
    llm_nodes = [n for n in fr["nodes"]
                 if n["description"].endswith("(LLM sibling pass).")]
    assert sorted(n["name"] for n in llm_nodes) == ["HellaSwag", "MMLU"]
    assert all(n["type"] == "sibling" for n in llm_nodes)
    # the target name returned as a "sibling" merges into the target, not a new node
    assert node_ids(fr).count("target") == 1


def test_llm_pass_parses_fenced_json():
    paper = "FibbleBench compares against MMLU."
    handler = FakeHandler('Here you go:\n```json\n["MMLU"]\n```')
    fr = F.build_frame("fibblebench", make_eee(), None, paper_text=paper, llm_handler=handler)
    assert fr["telemetry"]["llm_pass"] == "ok"
    assert "sibling:mmlu" in node_ids(fr)


def test_llm_prompt_contains_section_titles():
    class Sec:
        def __init__(self, title):
            self.title = title

    class Index:
        sections = [Sec("Introduction"), Sec("Related Work")]

    handler = FakeHandler("[]")
    F.build_frame("fibblebench", make_eee(), None, doc_index=Index(),
                  paper_text="FibbleBench text", llm_handler=handler)
    assert "Related Work" in handler.prompts[0]


def test_node_lookup():
    fr = F.build_frame("fibblebench", make_eee(), make_hf())
    lookup = F.node_lookup(fr)
    assert set(lookup) == set(node_ids(fr))
    assert lookup["target"]["type"] == "target"


def test_match_mention_whole_token():
    fr = F.build_frame("fibblebench", make_eee(contains=["MMLU"]), None)
    assert F.match_mention(fr, "Results on MMLU are strong.") == ["sibling:mmlu"]
    # substring inside a longer word does not match
    assert F.match_mention(fr, "The MMLUX suite is different.") == []
    # case-insensitive
    assert F.match_mention(fr, "results on mmlu") == ["sibling:mmlu"]


def test_match_mention_ignores_urls():
    fr = F.build_frame("fibblebench", make_eee(contains=["MMLU"]), None)
    assert F.match_mention(fr, "See https://hf.co/datasets/MMLU for data.") == []


def test_match_mention_longest_alias_first():
    fr = F.build_frame("fibblebench", make_eee(contains=["Fibble"]), make_hf())
    # "fibble-bench" (target alias) contains "Fibble"; the longer alias claims the
    # span so the sibling does not fire on the same characters
    assert F.match_mention(fr, "We ship fibble-bench today.") == ["target"]
    assert F.match_mention(fr, "The older Fibble corpus differs.") == ["sibling:fibble"]


def test_match_mention_ignores_very_short_aliases():
    fr = {"target_id": "target", "nodes": [
        {"id": "sibling:go", "type": "sibling", "name": "GO", "aliases": [],
         "description": "x"}], "telemetry": {}}
    assert F.match_mention(fr, "We go to the store. GO is a benchmark.") == []


def test_match_mention_orders_by_first_occurrence():
    fr = F.build_frame("fibblebench", make_eee(contains=["MMLU", "HellaSwag"]), None)
    out = F.match_mention(fr, "HellaSwag then MMLU then HellaSwag again.")
    assert out == ["sibling:hellaswag", "sibling:mmlu"]


def test_builder_counts_telemetry():
    fr = F.build_frame("fibblebench", make_eee(contains=["MMLU"]), make_hf(),
                       paper_text="FibbleBench v2.0 has a test set.")
    counts = fr["telemetry"]["builder_counts"]
    assert counts["target"] == 1
    assert counts["siblings_eee"] == 2  # MMLU + helm_lite
    assert counts["subsets"] == 2  # heldout + hard
    assert counts["metrics"] == 2
    assert counts["versions"] == 1
    assert counts["splits"] == 2  # test (paper) + train (frontmatter)
    assert fr["telemetry"]["total_nodes"] == len(fr["nodes"])
