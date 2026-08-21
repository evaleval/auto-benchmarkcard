"""Unit tests for paper-URL normalization, redirect resolution, accessibility gating,
and the run_docling routing that feeds Docling.

Network-free: requests.head/get are monkeypatched on the workers module.
"""

import auto_benchmarkcard.workers as W


# --- _normalize_paper_url / _biorxiv_full_pdf (pure, no network) ------------------

def test_arxiv_abs_passthrough():
    u = "https://arxiv.org/abs/2401.00001"
    assert W._normalize_paper_url(u) == u


def test_aclanthology_appends_pdf():
    assert (W._normalize_paper_url("https://aclanthology.org/2023.acl-long.1")
            == "https://aclanthology.org/2023.acl-long.1.pdf")


def test_aclanthology_already_pdf_unchanged():
    u = "https://aclanthology.org/2023.acl-long.1.pdf"
    assert W._normalize_paper_url(u) == u


def test_openreview_passthrough():
    u = "https://openreview.net/forum?id=abc123"
    assert W._normalize_paper_url(u) == u


def test_semanticscholar_returns_none():
    assert W._normalize_paper_url("https://www.semanticscholar.org/paper/x/abc") is None


def test_biorxiv_versioned_landing_to_full_pdf():
    assert (W._normalize_paper_url("https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1")
            == "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full.pdf")


def test_biorxiv_unversioned_landing_to_full_pdf():
    assert (W._normalize_paper_url("https://www.biorxiv.org/content/10.1101/2024.08.21.608694")
            == "https://www.biorxiv.org/content/10.1101/2024.08.21.608694.full.pdf")


def test_medrxiv_landing_to_full_pdf():
    assert (W._normalize_paper_url("https://www.medrxiv.org/content/10.1101/2024.01.02.99999v2")
            == "https://www.medrxiv.org/content/10.1101/2024.01.02.99999v2.full.pdf")


def test_biorxiv_already_full_pdf_unchanged():
    u = "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full.pdf"
    assert W._normalize_paper_url(u) == u


def test_biorxiv_trailing_full_to_pdf():
    assert (W._normalize_paper_url("https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full")
            == "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full.pdf")


def test_biorxiv_strips_query_and_fragment():
    assert (W._normalize_paper_url(
        "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1?rss=1#fig2")
        == "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full.pdf")


def test_doi_passthrough_pure():
    # The pure normalize does not resolve the redirect; run_docling does that.
    u = "https://doi.org/10.1101/2024.08.21.608694"
    assert W._normalize_paper_url(u) == u


# --- _resolve_landing_url (mock requests.get) ------------------------------------

class _FakeResp:
    def __init__(self, url, status_code=200, content_type="application/pdf"):
        self.url = url
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def close(self):
        pass


def test_resolve_follows_doi_to_biorxiv(monkeypatch):
    landing = "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1"
    monkeypatch.setattr(W.requests, "get", lambda url, **k: _FakeResp(landing))
    assert W._resolve_landing_url("https://doi.org/10.1101/2024.08.21.608694") == landing


def test_resolve_network_error_returns_original(monkeypatch):
    def boom(url, **k):
        raise W.requests.exceptions.ConnectionError("no net")
    monkeypatch.setattr(W.requests, "get", boom)
    u = "https://doi.org/10.1101/2024.08.21.608694"
    assert W._resolve_landing_url(u) == u


def test_resolve_passes_timeout(monkeypatch):
    captured = {}

    def fake_get(url, **k):
        captured.update(k)
        return _FakeResp(url)
    monkeypatch.setattr(W.requests, "get", fake_get)
    W._resolve_landing_url("https://doi.org/x")
    assert captured.get("timeout") == 10


# --- _check_paper_accessible (mock requests.head) --------------------------------

class _FakeHead:
    def __init__(self, status_code=200, content_type="application/pdf"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}


def test_pdf_url_accepted_despite_html_head(monkeypatch):
    # Core fix: a normalized .full.pdf whose HEAD returns text/html is still trusted.
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "text/html"))
    ok, reason, status = W._check_paper_accessible(
        "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full.pdf")
    assert ok is True and reason == "ok"


def test_blocked_403_rejected_even_for_pdf(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(403, "text/html"))
    ok, reason, status = W._check_paper_accessible(
        "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1.full.pdf")
    assert ok is False and reason == "blocked" and status == 403


def test_html_landing_rejected(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "text/html"))
    ok, reason, status = W._check_paper_accessible("https://example.com/paper")
    assert ok is False and reason == "html_only"


def test_arxiv_html_still_accepted(monkeypatch):
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "text/html"))
    ok, reason, status = W._check_paper_accessible("https://arxiv.org/abs/2401.00001")
    assert ok is True and reason == "ok"


def test_head_exception_optimistic(monkeypatch):
    def boom(url, **k):
        raise W.requests.exceptions.Timeout("slow")
    monkeypatch.setattr(W.requests, "head", boom)
    ok, reason, status = W._check_paper_accessible("https://example.com/x")
    assert ok is True and reason == "ok" and status is None


# --- run_docling integration -----------------------------------------------------

class _FakeOM:
    def save_tool_output(self, data, tool, filename):
        return f"/tmp/{tool}/{filename}"


def _state(paper_url):
    return {
        "query": "biolp-bench",
        "extracted_ids": {"paper_url": paper_url},
        "output_manager": _FakeOM(),
    }


def test_run_docling_biorxiv_doi_reaches_docling_with_full_pdf(monkeypatch):
    landing = "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1"
    expected_pdf = landing + ".full.pdf"
    monkeypatch.setattr(W.requests, "get", lambda url, **k: _FakeResp(landing))
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(200, "application/pdf"))
    captured = {}

    class _FakeTool:
        @staticmethod
        def func(paper_url):
            captured["url"] = paper_url
            return {"success": True, "filtered_text": "Full paper body here.",
                    "metadata": {"title": "T"}}
    monkeypatch.setattr(W, "extract_paper_with_docling", _FakeTool)

    out = W.run_docling(_state("https://doi.org/10.1101/2024.08.21.608694"))
    assert captured["url"] == expected_pdf
    assert out["docling_output"]["success"] is True
    assert "docling done" in out["completed"]
    assert out["docling_telemetry"]["full_text"] is True
    assert out["docling_telemetry"]["reason"] == "ok"
    assert out["docling_telemetry"]["degraded_to_abstract_only"] is False


def test_run_docling_biorxiv_cloudflare_blocked_is_degraded(monkeypatch):
    # The real biolp case: doi -> biorxiv landing, .full.pdf HEAD 403 (Cloudflare).
    landing = "https://www.biorxiv.org/content/10.1101/2024.08.21.608694v1"
    monkeypatch.setattr(W.requests, "get", lambda url, **k: _FakeResp(landing))
    monkeypatch.setattr(W.requests, "head", lambda url, **k: _FakeHead(403, "text/html"))

    out = W.run_docling(_state("https://doi.org/10.1101/2024.08.21.608694"))
    assert out["docling_output"] is None
    tel = out["docling_telemetry"]
    assert tel["reason"] == "fetch_blocked" and tel["http_status"] == 403
    assert tel["degraded_to_abstract_only"] is True
    # the normalized .full.pdf is promoted to website_url for the html fallback
    assert out["extracted_ids"]["website_url"].endswith(".full.pdf")
