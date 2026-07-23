"""Network-free smoke test for scripts/hf_verification_dryrun.py::_classify.

The harness is a standalone script (not imported by the package), so it is loaded via importlib and
its module-level resolve_hf_repo / hf_dataset_metadata are monkeypatched to stay offline. The test
pins the deterministic tier buckets, including the EEE Tier-0 exemption (an aggregate-basename repo
that is EEE-provided must NOT tier0_reject, while a search-path aggregate must)."""

import importlib.util
from pathlib import Path

import pytest


def _load_dryrun():
    p = Path(__file__).resolve().parents[1] / "scripts" / "hf_verification_dryrun.py"
    spec = importlib.util.spec_from_file_location("hf_verification_dryrun", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch(monkeypatch, mod, *, resolved, readme):
    """resolved: name -> repo_id (or None). readme: repo_id -> README text."""
    monkeypatch.setattr(mod, "resolve_hf_repo",
                        lambda name, existing=None: resolved.get(name))

    class _T:
        @staticmethod
        def func(repo_id):
            return {"id": repo_id, "card_data": {}, "readme_markdown": readme.get(repo_id, ""),
                    "tags": []}
    monkeypatch.setattr(mod, "hf_dataset_metadata", _T)


def _eee_meta(name, desc):
    return {"benchmark_name": name,
            "metrics": {"m": {"evaluation_description": desc}}}


def test_dryrun_buckets_classify_with_eee_exemption(monkeypatch):
    mod = _load_dryrun()

    # A clean coined repo whose README strongly overlaps the identity subject -> tier1_accept.
    widget_desc = "widget assembly reasoning over factory schematics mechanical parts gears"
    widget_readme = ("WidgetBench is a benchmark of factory schematics for widget assembly reasoning "
                     "over mechanical parts and gears, measuring assembly accuracy.")
    # An EEE-provided abbreviation: basename 'bbh' bears no token run of 'BIG-Bench Hard'
    # (aggregate_repo), but because it IS the EEE-provided repo it must NOT tier0_reject.
    # A SEARCH-path aggregate (no EEE repo) with the same shape MUST tier0_reject.
    resolved = {
        "WidgetBench": "acme/WidgetBench",
        "BIG-Bench Hard": "lukaemon/bbh",
        "AggSuite": "someorg/multi_benchmark_suite",
        "GhostBench": None,                       # no candidate -> no_repo
    }
    readme = {
        "acme/WidgetBench": widget_readme,
        "lukaemon/bbh": "BIG-Bench Hard: 23 challenging BIG-Bench tasks.",
        "someorg/multi_benchmark_suite": "An aggregate of many unrelated benchmarks.",
    }
    _patch(monkeypatch, mod, resolved=resolved, readme=readme)

    def bucket(name, existing, meta):
        return mod._classify(name, existing, meta)[0]

    # (name, existing_hf_repo, eee_metadata) -> expected bucket
    assert bucket("WidgetBench", None, _eee_meta("WidgetBench", widget_desc)) == "tier1_accept"
    # EEE-provided (existing == resolved repo) aggregate basename -> exempt from Tier-0 -> tier2_bound
    assert bucket("BIG-Bench Hard", "lukaemon/bbh",
                  _eee_meta("BIG-Bench Hard", "23 hard reasoning tasks")) == "tier2_bound"
    # Search-path aggregate (no existing repo) -> tier0_reject
    assert bucket("AggSuite", None, _eee_meta("AggSuite", "many tasks")) == "tier0_reject"
    # No candidate -> no_repo
    assert bucket("GhostBench", None, _eee_meta("GhostBench", "x")) == "no_repo"


def test_dryrun_eee_detail_carries_repo_source(monkeypatch):
    mod = _load_dryrun()
    _patch(monkeypatch, mod,
           resolved={"BIG-Bench Hard": "lukaemon/bbh"},
           readme={"lukaemon/bbh": "BIG-Bench Hard tasks."})
    _, detail = mod._classify("BIG-Bench Hard", "lukaemon/bbh",
                              _eee_meta("BIG-Bench Hard", "hard tasks"))
    assert detail["repo_source"] == "eee"
    # The same repo arriving via search (no existing) is classified as search.
    _, detail2 = mod._classify("BIG-Bench Hard", None,
                               _eee_meta("BIG-Bench Hard", "hard tasks"))
    assert detail2["repo_source"] == "search"
