"""OpenAlex auth wiring: Bearer header, status mapping, and batch-lived caches.

All network-free: requests.get is monkeypatched and time.sleep is stubbed so _http_get's
retries don't delay the suite. An autouse fixture clears every batch-lived cache and the
per-call _SOURCE_STATUS map so cases don't leak into each other.
"""

import pytest

from auto_benchmarkcard.tools.eee import paper_resolver as pr


class _Resp:
    def __init__(self, *, text="", json_data=None, status=200):
        self._text, self._json, self.status_code = text, json_data, status

    @property
    def text(self):
        return self._text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _no_sleep_clean_caches(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda *a, **k: None)
    pr._openalex_search_cache.clear()
    pr._openalex_doi_cache.clear()
    pr._OPENALEX_DOI_STATUS.clear()
    pr._SOURCE_STATUS.clear()
    pr._author_cache.clear()


def _capture_get(monkeypatch, resp, calls):
    """Stub requests.get to record (headers, params) per call and return `resp`."""
    def fake_get(url, **kwargs):
        calls.append({"url": url, "headers": kwargs.get("headers"), "params": kwargs.get("params")})
        return resp() if callable(resp) else resp
    monkeypatch.setattr(pr.requests, "get", fake_get)


_OK_SEARCH = {"results": [{"title": "SWE-bench", "id": "W1"}]}
_OK_DOI = {"title": "A Paper", "authorships": [{"author": {"display_name": "Jane Doe"}}],
           "publication_year": 2024, "ids": {}}


# 1 -- search sends Bearer when key set + mailto retained
def test_search_sends_bearer_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key")
    calls = []
    _capture_get(monkeypatch, _Resp(json_data=_OK_SEARCH), calls)
    pr._search_openalex("SWE-bench")
    assert calls[0]["headers"] == {"Authorization": "Bearer secret-key"}
    assert calls[0]["params"]["mailto"] == pr.OPENALEX_MAILTO
    assert calls[0]["params"]["search"] == "SWE-bench"


# 2 -- no key: no Authorization header, mailto still sent, no crash
def test_search_no_key_anonymous(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    _capture_get(monkeypatch, _Resp(json_data=_OK_SEARCH), calls)
    results = pr._search_openalex("SWE-bench")
    assert calls[0]["headers"] == {}
    assert calls[0]["params"]["mailto"] == pr.OPENALEX_MAILTO
    assert results == _OK_SEARCH["results"]


# 3 -- a non-empty result is cached: the 2nd identical query issues no HTTP call
def test_search_caches_nonempty_result(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    _capture_get(monkeypatch, _Resp(json_data=_OK_SEARCH), calls)
    first = pr._search_openalex("SWE-bench")
    second = pr._search_openalex("SWE-bench")
    assert first == second == _OK_SEARCH["results"]
    assert len(calls) == 1  # second served from cache


# 4 -- an empty result is NOT cached: the 2nd query re-fetches
def test_search_does_not_cache_empty(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    _capture_get(monkeypatch, _Resp(json_data={"results": []}), calls)
    assert pr._search_openalex("nothing") == []
    assert pr._search_openalex("nothing") == []
    assert len(calls) == 2  # nothing cached, both queries hit the network


# 5 -- an error path is NOT cached: a 500 then a good response returns the good result
def test_search_does_not_cache_error(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    seq = [_Resp(status=500), _Resp(json_data=_OK_SEARCH)]
    _capture_get(monkeypatch, lambda: seq.pop(0), calls)
    # _http_get retries 500 within one _search_openalex call; force only one attempt by
    # exhausting the budget via a fresh sequence -- instead, drive two separate calls.
    monkeypatch.setattr(pr, "HTTP_RETRIES", 1)
    assert pr._search_openalex("swe") == []  # 500 -> raise_for_status -> except -> []
    second = pr._search_openalex("swe")
    assert second == _OK_SEARCH["results"]


# 6 -- 401 maps to auth_error in _SOURCE_STATUS, returns [], logs a warning
def test_search_401_auth_error(monkeypatch, caplog):
    monkeypatch.setenv("OPENALEX_API_KEY", "bad-key")
    monkeypatch.setattr(pr, "HTTP_RETRIES", 1)
    calls = []
    _capture_get(monkeypatch, _Resp(status=401), calls)
    with caplog.at_level("WARNING"):
        results = pr._search_openalex("SWE-bench")
    assert results == []
    assert pr._SOURCE_STATUS["openalex"] == "auth_error"
    assert any("auth" in r.message.lower() for r in caplog.records)


# 7 -- 429 still maps to rate_limited (regression for the prior special-case)
def test_search_429_rate_limited(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setattr(pr, "HTTP_RETRIES", 1)
    calls = []
    _capture_get(monkeypatch, _Resp(status=429), calls)
    assert pr._search_openalex("SWE-bench") == []
    assert pr._SOURCE_STATUS["openalex"] == "rate_limited"


# 8 -- a cache hit does not touch _SOURCE_STATUS
def test_search_cache_hit_does_not_touch_status(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    _capture_get(monkeypatch, _Resp(json_data=_OK_SEARCH), calls)
    pr._search_openalex("SWE-bench")  # populates cache
    pr._SOURCE_STATUS.clear()
    pr._search_openalex("SWE-bench")  # cache hit
    assert pr._SOURCE_STATUS == {}
    assert len(calls) == 1


# 9 -- DOI fetch sends Bearer + caches; a clean success records nothing degraded
def test_doi_fetch_sends_bearer_and_caches(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key")
    calls = []
    _capture_get(monkeypatch, _Resp(json_data=_OK_DOI), calls)
    first = pr._fetch_openalex_by_doi("10.1162/tacl_a_00276")
    assert first["authors"] == ["Jane Doe"]
    assert calls[0]["headers"] == {"Authorization": "Bearer secret-key"}
    second = pr._fetch_openalex_by_doi("10.1162/tacl_a_00276")
    assert second == first and len(calls) == 1  # served from cache
    assert "10.1162/tacl_a_00276" not in pr._OPENALEX_DOI_STATUS  # clean success, no degraded status


# 10 -- DOI 401 records auth_error in _OPENALEX_DOI_STATUS, returns empty triple, warns,
#       and leaves _SOURCE_STATUS untouched (the DOI path is outside the search window)
def test_doi_fetch_401_auth_error(monkeypatch, caplog):
    monkeypatch.setenv("OPENALEX_API_KEY", "bad-key")
    monkeypatch.setattr(pr, "HTTP_RETRIES", 1)
    calls = []
    _capture_get(monkeypatch, _Resp(status=401), calls)
    with caplog.at_level("WARNING"):
        result = pr._fetch_openalex_by_doi("10.1162/tacl_a_00276")
    assert result == {"authors": [], "title": "", "year": None}
    assert pr._OPENALEX_DOI_STATUS["10.1162/tacl_a_00276"] == "auth_error"
    assert pr._SOURCE_STATUS == {}
    assert any("auth" in r.message.lower() for r in caplog.records)


# 11 -- DOI fetch with no key is safe (anonymous, no crash, empty Authorization header)
def test_doi_fetch_no_key_safe(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    _capture_get(monkeypatch, _Resp(json_data=_OK_DOI), calls)
    result = pr._fetch_openalex_by_doi("10.1162/tacl_a_00276")
    assert result["authors"] == ["Jane Doe"]
    assert calls[0]["headers"] == {}
