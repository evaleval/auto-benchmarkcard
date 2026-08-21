"""Tests for post_process_card's resources heuristic (BUG 3b).

Only a clean owner/name repo id gets the HF datasets prefix; prose is never mangled — a
real URL is extracted from it, otherwise the value is dropped.
"""

from auto_benchmarkcard.tools.composer.composer_tool import post_process_card


def _resources(values):
    return post_process_card({"benchmark_details": {"resources": values}})["benchmark_details"]["resources"]


def test_clean_repo_id_gets_hf_prefix():
    assert _resources(["owner/dataset-name"]) == ["https://huggingface.co/datasets/owner/dataset-name"]


def test_prose_with_url_extracts_url():
    assert _resources(["see https://github.com/x/y for details"]) == ["https://github.com/x/y"]


def test_pure_prose_no_url_dropped():
    assert _resources(["a study / recommendation about safety"]) == []


def test_existing_url_kept():
    assert _resources(["https://huggingface.co/datasets/a/b"]) == ["https://huggingface.co/datasets/a/b"]


def test_leaked_evidence_id_dropped():
    assert _resources(["E01"]) == []


def test_typo_host_fixed():
    assert _resources(["https://hugdingface.co/datasets/a/b"]) == ["https://huggingface.co/datasets/a/b"]


def test_llm_stats_dropped_and_paper_mirrors_deduped():
    out = _resources([
        "https://arxiv.org/abs/2406.15877",
        "https://alphaxiv.org/abs/2406.15877",        # same paper -> deduped
        "https://github.com/bigcode-project/bigcodebench",
        "https://llm-stats.com/models/foo",           # furniture -> dropped
        "https://api.llm-stats.com/v1/bar",            # furniture -> dropped
    ])
    assert out == [
        "https://arxiv.org/abs/2406.15877",
        "https://github.com/bigcode-project/bigcodebench",
    ]


def test_paper_mirror_dedup_handles_versions_and_old_style_ids():
    # version-suffix mismatch across mirrors still dedups to one paper
    assert _resources([
        "https://arxiv.org/abs/2406.15877v1",
        "https://alphaxiv.org/abs/2406.15877v2",
    ]) == ["https://arxiv.org/abs/2406.15877v1"]
    # canonical (no version) vs versioned mirror also dedups
    assert _resources([
        "https://arxiv.org/abs/2406.15877",
        "https://alphaxiv.org/abs/2406.15877v1",
    ]) == ["https://arxiv.org/abs/2406.15877"]
    # distinct old-style category/number ids must NOT collapse (no silent data loss)
    assert _resources([
        "https://arxiv.org/abs/cs/0501001",
        "https://arxiv.org/abs/cs/0601042",
    ]) == ["https://arxiv.org/abs/cs/0501001", "https://arxiv.org/abs/cs/0601042"]


def test_nested_foreign_domain_dropped():
    # F4: ace's malformed/injected URL nests a foreign domain as a HF dataset owner
    assert _resources(["https://huggingface.co/datasets/mercor.com/ace-leaderboard"]) == []
    assert _resources(["https://github.com/org/repo.com/x"]) == []


def test_furniture_org_url_dropped():
    # F4: the hallucinated Exgentic/Ergon pipeline-org repos (a cross-card leak) are dropped
    assert _resources(["https://github.com/Exgentic/exgentic"]) == []
    # word-boundary guard: a legit repo merely containing the substring is kept
    assert _resources(
        ["https://github.com/ergonomics-lab/bench"]) == ["https://github.com/ergonomics-lab/bench"]


def test_legit_hf_doi_githubio_kept():
    # the guard is narrow: legit owner/name HF paths and other hosts are untouched
    keep = [
        "https://huggingface.co/datasets/allenai/reward-bench",
        "https://doi.org/10.1609/aaai.v33i01.3301590",
        "https://texttron.github.io/BrowseComp-Plus/",
        "https://github.com/bigcode-project/bigcodebench",
    ]
    assert _resources(keep) == keep
