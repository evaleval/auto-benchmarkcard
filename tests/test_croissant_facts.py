"""fetch_croissant_facts: deterministic structural facts from the HF dataset-viewer API.

All tests are network-free: requests.get is monkeypatched with canned bodies trimmed from
the live 2026-08-24 responses for openai/gsm8k (multi-config) and yahma/alpaca-cleaned
(single config, single split). Covers multi-config preference, single-config split
fallback, license mapping and passthrough, total-size formatting, hosting-format
labelling, per-endpoint failure isolation, the /info fallback, partial-response
abstention, and the full-failure shape.
"""

import requests

import pytest

from auto_benchmarkcard.tools.hf import croissant


# Trimmed from the live openai/gsm8k responses.
GSM8K_CROISSANT = {
    "@type": "sc:Dataset",
    "name": "gsm8k",
    "license": "https://choosealicense.com/licenses/mit/",
    "distribution": [
        {"@type": "cr:FileObject", "name": "repo", "encodingFormat": "git+https"},
        {"@type": "cr:FileSet", "encodingFormat": "application/x-parquet"},
        {"@type": "cr:FileSet", "encodingFormat": "application/x-parquet"},
    ],
}

GSM8K_SIZE = {
    "size": {
        "dataset": {"dataset": "openai/gsm8k", "num_rows": 17584},
        "configs": [
            {"dataset": "openai/gsm8k", "config": "main", "num_rows": 8792},
            {"dataset": "openai/gsm8k", "config": "socratic", "num_rows": 8792},
        ],
        "splits": [
            {"dataset": "openai/gsm8k", "config": "main", "split": "train", "num_rows": 7473},
            {"dataset": "openai/gsm8k", "config": "main", "split": "test", "num_rows": 1319},
            {"dataset": "openai/gsm8k", "config": "socratic", "split": "train", "num_rows": 7473},
            {"dataset": "openai/gsm8k", "config": "socratic", "split": "test", "num_rows": 1319},
        ],
    },
    "pending": [],
    "failed": [],
    "partial": False,
}

# The gsm8k "main" config alone: the single-config shape with two splits.
SINGLE_CONFIG_SIZE = {
    "size": {
        "dataset": {"dataset": "x/y", "num_rows": 8792},
        "configs": [{"dataset": "x/y", "config": "main", "num_rows": 8792}],
        "splits": [
            {"dataset": "x/y", "config": "main", "split": "train", "num_rows": 7473},
            {"dataset": "x/y", "config": "main", "split": "test", "num_rows": 1319},
        ],
    },
    "pending": [],
    "failed": [],
    "partial": False,
}

# Trimmed from the live yahma/alpaca-cleaned /size: single config, single split.
ALPACA_SIZE = {
    "size": {
        "dataset": {"dataset": "yahma/alpaca-cleaned", "num_rows": 51760},
        "configs": [{"dataset": "yahma/alpaca-cleaned", "config": "default", "num_rows": 51760}],
        "splits": [
            {"dataset": "yahma/alpaca-cleaned", "config": "default", "split": "train", "num_rows": 51760},
        ],
    },
    "pending": [],
    "failed": [],
    "partial": False,
}

# The live /info shape for gsm8k (dataset_info keyed by config, splits keyed by name).
GSM8K_INFO = {
    "dataset_info": {
        "main": {"splits": {"train": {"num_examples": 7473}, "test": {"num_examples": 1319}}},
        "socratic": {"splits": {"train": {"num_examples": 7473}, "test": {"num_examples": 1319}}},
    },
    "pending": [],
    "failed": [],
    "partial": False,
}


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def _route(monkeypatch, croissant_r=None, size_r=None, info_r=None):
    """Route the three endpoint URLs to canned responses. A value may be a _Resp, an
    Exception to raise, or None for a plain 404."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/croissant"):
            name, r = "croissant", croissant_r
        elif url.endswith("/size"):
            name, r = "size", size_r
        elif url.endswith("/info"):
            name, r = "info", info_r
        else:
            raise AssertionError(f"unexpected url {url}")
        calls.append(name)
        if isinstance(r, Exception):
            raise r
        return r if r is not None else _Resp(status=404, body={"error": "Not Found"})

    monkeypatch.setattr(croissant.requests, "get", fake_get)
    return calls


def test_multi_config_prefers_per_config_totals(monkeypatch):
    _route(monkeypatch, croissant_r=_Resp(body=GSM8K_CROISSANT), size_r=_Resp(body=GSM8K_SIZE))
    facts = croissant.fetch_croissant_facts("openai/gsm8k")
    assert facts["data.size"] == "17,584 examples"
    assert facts["data.size_breakdown"] == {"main": 8792, "socratic": 8792}


def test_single_config_falls_back_to_splits(monkeypatch):
    _route(monkeypatch, size_r=_Resp(body=SINGLE_CONFIG_SIZE))
    facts = croissant.fetch_croissant_facts("x/y")
    assert facts["data.size"] == "8,792 examples"
    assert facts["data.size_breakdown"] == {"train": 7473, "test": 1319}


def test_single_split_breakdown_is_dropped(monkeypatch):
    # A one-entry breakdown restates the total, mirroring the README parser's >= 2 floor.
    _route(monkeypatch, size_r=_Resp(body=ALPACA_SIZE))
    facts = croissant.fetch_croissant_facts("yahma/alpaca-cleaned")
    assert facts["data.size"] == "51,760 examples"
    assert "data.size_breakdown" not in facts


def test_license_url_maps_to_display_name(monkeypatch):
    _route(monkeypatch, croissant_r=_Resp(body=GSM8K_CROISSANT))
    facts = croissant.fetch_croissant_facts("openai/gsm8k")
    assert facts["ethical_and_legal_considerations.data_licensing"] == "MIT License"


@pytest.mark.parametrize(
    "raw,expected",
    [
        # unmappable choosealicense slug passes through as the slug
        ("https://choosealicense.com/licenses/bigscience-openrail-m/", "bigscience-openrail-m"),
        # bare unmappable string passes through raw
        ("LDC User Agreement", "LDC User Agreement"),
        # bare mappable id maps regardless of case
        ("CC-BY-4.0", "Creative Commons Attribution 4.0"),
        # list form: first string wins
        (["apache-2.0", "mit"], "Apache License 2.0"),
    ],
)
def test_license_mapping_and_passthrough(monkeypatch, raw, expected):
    body = dict(GSM8K_CROISSANT, license=raw)
    _route(monkeypatch, croissant_r=_Resp(body=body))
    facts = croissant.fetch_croissant_facts("x/y")
    assert facts["ethical_and_legal_considerations.data_licensing"] == expected


@pytest.mark.parametrize("raw", ["other", "unknown", "https://choosealicense.com/licenses/other/", ""])
def test_placeholder_license_yields_nothing(monkeypatch, raw):
    body = dict(GSM8K_CROISSANT, license=raw)
    _route(monkeypatch, croissant_r=_Resp(body=body))
    facts = croissant.fetch_croissant_facts("x/y")
    assert "ethical_and_legal_considerations.data_licensing" not in facts


def test_hosting_format_labelled_like_deterministic_channel(monkeypatch):
    _route(monkeypatch, croissant_r=_Resp(body=GSM8K_CROISSANT))
    facts = croissant.fetch_croissant_facts("openai/gsm8k")
    assert facts["data.format"] == "parquet (HuggingFace hosting format)"


def test_endpoint_failures_are_isolated(monkeypatch):
    # croissant answers; size times out and info 404s: license/format survive, size absent.
    _route(
        monkeypatch,
        croissant_r=_Resp(body=GSM8K_CROISSANT),
        size_r=requests.exceptions.ConnectTimeout("boom"),
    )
    facts = croissant.fetch_croissant_facts("openai/gsm8k")
    assert facts["ethical_and_legal_considerations.data_licensing"] == "MIT License"
    assert "data.size" not in facts and "data.size_breakdown" not in facts
    eps = facts["_croissant_meta"]["endpoints"]
    assert eps["croissant"] == 200
    assert eps["size"] == "error: ConnectTimeout"
    assert eps["info"] == 404


def test_info_fallback_when_size_fails(monkeypatch):
    _route(monkeypatch, size_r=_Resp(status=500, body={"error": "boom"}), info_r=_Resp(body=GSM8K_INFO))
    facts = croissant.fetch_croissant_facts("openai/gsm8k")
    assert facts["data.size"] == "17,584 examples"
    assert facts["data.size_breakdown"] == {"main": 8792, "socratic": 8792}
    eps = facts["_croissant_meta"]["endpoints"]
    assert eps["size"] == 500 and eps["info"] == 200


def test_info_not_fetched_when_size_answers(monkeypatch):
    calls = _route(monkeypatch, size_r=_Resp(body=GSM8K_SIZE))
    croissant.fetch_croissant_facts("openai/gsm8k")
    assert "info" not in calls


def test_partial_size_contributes_nothing(monkeypatch):
    body = dict(GSM8K_SIZE, partial=True)
    info_body = dict(GSM8K_INFO, partial=True)
    _route(monkeypatch, size_r=_Resp(body=body), info_r=_Resp(body=info_body))
    facts = croissant.fetch_croissant_facts("openai/gsm8k")
    assert "data.size" not in facts and "data.size_breakdown" not in facts
    assert facts["_croissant_meta"]["partial"] is True


def test_malformed_bodies_never_raise(monkeypatch):
    _route(
        monkeypatch,
        croissant_r=_Resp(body=None),  # 200 but not JSON
        size_r=_Resp(body={"size": "nonsense"}),
    )
    facts = croissant.fetch_croissant_facts("x/y")
    assert set(facts) == {"_croissant_meta"}
    assert facts["_croissant_meta"]["endpoints"]["croissant"] == "malformed"


def test_full_failure_returns_meta_only(monkeypatch):
    err = requests.exceptions.ConnectionError("no network")
    _route(monkeypatch, croissant_r=err, size_r=err, info_r=err)
    facts = croissant.fetch_croissant_facts("x/y")
    assert set(facts) == {"_croissant_meta"}
    eps = facts["_croissant_meta"]["endpoints"]
    assert eps == {
        "croissant": "error: ConnectionError",
        "size": "error: ConnectionError",
        "info": "error: ConnectionError",
    }
    assert "fetched_at" in facts["_croissant_meta"]
