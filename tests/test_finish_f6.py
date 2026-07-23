"""F6: deterministic extractors -- HF-readme BibTeX authors, docling run-on normalize, and the
benchmark-identity-vs-paper-title name preference. Synthetic inputs only (no network)."""

from auto_benchmarkcard.tools.composer import composer_tool as C


def test_parse_bibtex_authors_both_name_orders():
    bib = "@article{x, author = {Zhuo, Terry Yue and Vu, Minh Chien and others}, year=2024}"
    assert C._parse_bibtex_authors(bib) == ["Terry Yue Zhuo", "Minh Chien Vu"]
    quoted = 'cite as author = "Alex Wang and Yada Pruksachatkun" rest'
    assert C._parse_bibtex_authors(quoted) == ["Alex Wang", "Yada Pruksachatkun"]


def test_parse_bibtex_authors_none_when_absent():
    assert C._parse_bibtex_authors("no bibtex here") is None
    assert C._parse_bibtex_authors(None) is None


def test_bibtex_org_handle_rejected_by_person_guard():
    # a single org token is not a person list -> the caller's _looks_like_person_list guard drops it
    parsed = C._parse_bibtex_authors("@misc{x, author = {bigcode}}")
    assert parsed == ["bigcode"]
    assert C._looks_like_person_list(parsed) is False


def test_normalize_docling_runon_and_idempotent():
    raw = "the loop ends here.)Skip the boilerplate."
    out = C._normalize_docling_text(raw)
    assert out == "the loop ends here.) Skip the boilerplate."
    assert C._normalize_docling_text(out) == out                  # idempotent
    assert C._normalize_docling_text("normal text. Next line.") == "normal text. Next line."
    assert C._normalize_docling_text("") == ""


def test_name_is_paper_title_prefers_identity():
    title = "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators"
    # the card name is the title's pre-subtitle head -> it is the paper title, not the identity
    assert C._name_is_paper_title("Length-Controlled AlpacaEval", title) is True
    # a genuine short identity is not the paper title
    assert C._name_is_paper_title("GSM8K", title) is False
    assert C._name_is_paper_title("AnythingElse", title) is False
