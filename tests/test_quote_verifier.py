"""Adversarial unit tests for the Stage-A quote verifier (the security property).

No LLM calls: synthetic sources and items only. Covers positive matches, whitespace
variance, paraphrase / hallucination / case-drift rejection, the 300c cap, evidence_id
assignment, digest grouping, telemetry, the canonical-field-source wiring, and the
robust JSON parser.
"""

import pytest

from auto_benchmarkcard.tools.composer import evidence as ev


def _items(*tuples):
    """Build raw item dicts from (field, quote[, kind[, note]]) tuples."""
    out = []
    for t in tuples:
        field, quote = t[0], t[1]
        kind = t[2] if len(t) > 2 else "stated"
        note = t[3] if len(t) > 3 else ""
        out.append({"field": field, "quote": quote, "kind": kind, "note": note})
    return out


# --- verify_quote / normalization -------------------------------------------------

def test_positive_exact_match():
    src = "The benchmark contains 1140 hand-written programming problems."
    recs, telem = ev.verify_items(
        _items(("data.size", "1140 hand-written programming problems")), src, ev.DOC_PAPER)
    assert len(recs) == 1
    r = recs[0]
    assert r["field"] == "data.size"
    assert r["doc"] == "docling"
    assert r["char_start"] == src.find("1140")
    assert telem["verified"] == 1 and telem["rejected"] == 0


def test_whitespace_variance_accepted():
    # Source has a newline, a tab, a double space and a non-breaking space inside the span.
    src = "Models are evaluated\tusing  the\npass@1\xa0metric."
    quote = "evaluated using the pass@1 metric"
    recs, telem = ev.verify_items(_items(("methodology.metrics", quote)), src, ev.DOC_PAPER)
    assert len(recs) == 1
    norm_src = ev.normalize_ws(src)
    assert recs[0]["char_start"] == norm_src.find("evaluated using the pass@1 metric")
    assert recs[0]["quote"] == quote
    assert telem["verified"] == 1


def test_paraphrase_rejected():
    src = "The dataset was collected from public GitHub repositories."
    # synonym swap: "gathered" / "open-source" never appear verbatim
    recs, telem = ev.verify_items(
        _items(("data.source", "The dataset was gathered from open-source repositories")),
        src, ev.DOC_PAPER)
    assert recs == []
    assert telem["rejected"] == 1 and telem["reject_reasons"]["no_match"] == 1


def test_hallucinated_proper_noun_rejected():
    # The classic failure mode: a confident-sounding fact the model invented from memory.
    src = "The benchmark evaluates code generation across several languages."
    recs, telem = ev.verify_items(
        _items(("data.source", "Data was sourced from the Upwork freelancing platform")),
        src, ev.DOC_PAPER)
    assert recs == []
    assert telem["reject_reasons"]["no_match"] == 1


def test_case_drift_rejected():
    src = "Accuracy is reported as the primary metric."
    recs, telem = ev.verify_items(
        _items(("methodology.metrics", "ACCURACY IS REPORTED")), src, ev.DOC_PAPER)
    assert recs == []
    assert telem["reject_reasons"]["no_match"] == 1


def test_empty_and_whitespace_quote_rejected():
    src = "Some source text."
    recs, telem = ev.verify_items(
        _items(("data.size", ""), ("data.size", "   \n\t  ")), src, ev.DOC_PAPER)
    assert recs == []
    assert telem["reject_reasons"]["empty"] == 2


def test_bad_field_rejected():
    src = "Model achieved 85% accuracy on the test set."
    recs, telem = ev.verify_items(
        _items(("methodology.result", "85% accuracy")), src, ev.DOC_PAPER)
    assert recs == []
    assert telem["reject_reasons"]["bad_field"] == 1


# --- 300c cap ---------------------------------------------------------------------

def test_over_cap_quote_truncated_after_verify():
    sentence = ("The benchmark was constructed by " + "carefully curating examples "
                + "from many diverse sources ") * 4  # > 300 chars, all verbatim
    sentence = sentence.strip()
    assert len(sentence) > ev.QUOTE_CAP
    src = "Intro. " + sentence + " End."
    recs, telem = ev.verify_items(_items(("data.source", sentence)), src, ev.DOC_PAPER)
    assert len(recs) == 1
    stored = recs[0]["quote"]
    assert len(stored) <= ev.QUOTE_CAP
    assert not stored.endswith(" ")
    # still verbatim: the stored (truncated) quote is a substring of the normalized source
    assert ev.normalize_ws(src).find(stored) == recs[0]["char_start"]
    assert telem["truncated"] == 1 and telem["verified"] == 1


def test_over_cap_hallucination_still_rejected():
    # A >300c quote that is NOT in the source must fail the substring check, not be capped.
    src = "Short authentic source sentence."
    fake = "x" * 500
    recs, telem = ev.verify_items(_items(("data.source", fake)), src, ev.DOC_PAPER)
    assert recs == []
    assert telem["truncated"] == 0 and telem["reject_reasons"]["no_match"] == 1


# --- kind / note coercion ---------------------------------------------------------

def test_derived_kind_preserved_bad_kind_coerced():
    src = "Lower scores indicate better calibration on this benchmark."
    recs, _ = ev.verify_items(
        _items(("methodology.interpretation", "Lower scores indicate better calibration",
                "derived", "premise for higher_is_better=False"),
               ("methodology.metrics", "better calibration on this benchmark", "garbage")),
        src, ev.DOC_PAPER)
    assert recs[0]["kind"] == "derived"
    assert recs[0]["note"] == "premise for higher_is_better=False"
    assert recs[1]["kind"] == "stated"  # coerced


# --- evidence_id assignment + digest ---------------------------------------------

def test_evidence_id_sequencing_across_calls():
    src_p = "Paper says the dataset has 500 examples and uses F1 as the metric."
    src_h = "The web page describes accuracy as the headline number."
    paper, _ = ev.verify_items(
        _items(("data.size", "500 examples"), ("methodology.metrics", "F1 as the metric")),
        src_p, ev.DOC_PAPER)
    gap, _ = ev.verify_items(
        _items(("data.size", "the dataset has 500 examples")), src_p, ev.DOC_PAPER)
    html, _ = ev.verify_items(
        _items(("methodology.metrics", "accuracy as the headline number")), src_h, ev.DOC_HTML)
    items, digest = ev.finalize_evidence(paper, gap, html, hf=[])
    ids = [it["evidence_id"] for it in items]
    assert ids == ["E01", "E02", "E03", "E04"]
    # order is paper, gap, html, hf
    assert items[0]["doc"] == "docling" and items[3]["doc"] == "html"
    # every record has exactly the 7 contract keys
    for it in items:
        assert set(it.keys()) == {
            "evidence_id", "field", "quote", "char_start", "doc", "kind", "note"}
    # digest grouped by field, in evidence_id order within a field
    assert [it["evidence_id"] for it in digest["data.size"]] == ["E01", "E03"]
    assert [it["evidence_id"] for it in digest["methodology.metrics"]] == ["E02", "E04"]


def test_id_format_zero_padded():
    src = "A. B. C."
    recs, _ = ev.verify_items(_items(("data.size", "A")), src, ev.DOC_PAPER)
    items, _ = ev.finalize_evidence(recs, [], [], [])
    assert items[0]["evidence_id"] == "E01"


# --- telemetry --------------------------------------------------------------------

def test_card_telemetry_merge_and_by_doc():
    src = "The corpus has 1000 items collected from forums."
    card = ev.new_card_telemetry()
    paper, t1 = ev.verify_items(
        _items(("data.size", "1000 items"), ("data.source", "nope not here")),
        src, ev.DOC_PAPER)
    gap, t2 = ev.verify_items(
        _items(("data.source", "collected from forums")), src, ev.DOC_PAPER)
    ev.merge_telemetry(card, t1)
    ev.merge_telemetry(card, t2)
    assert card["emitted"] == 3
    assert card["verified"] == 2
    assert card["rejected"] == 1
    assert card["reject_reasons"]["no_match"] == 1
    # paper and gap share doc=docling -> additive in by_doc
    assert card["by_doc"]["docling"] == {
        "emitted": 3, "verified": 2, "rejected": 1, "truncated": 0}
    assert ev.verify_rate(card) == pytest.approx(2 / 3)


def test_verify_rate_empty_card():
    assert ev.verify_rate(ev.new_card_telemetry()) == 0.0


# --- canonical field source (drop-in extensibility) ------------------------------

def test_whitelist_equals_curator_fields():
    assert ev.EVIDENCE_FIELD_WHITELIST == frozenset(ev.CURATOR_FIELDS)


def test_derivations_are_subsets_of_whitelist():
    for src in ("paper", "html", "hf"):
        for path in ev.fields_for_source(src):
            assert path in ev.EVIDENCE_FIELD_WHITELIST
    for path in ev.gap_fields():
        assert path in ev.EVIDENCE_FIELD_WHITELIST


def test_always_override_fields_whitelisted_but_not_extracted():
    for path in ("benchmark_details.languages",
                 "ethical_and_legal_considerations.data_licensing"):
        assert path in ev.EVIDENCE_FIELD_WHITELIST
        assert path not in ev.fields_for_source("paper")
        assert path not in ev.gap_fields()


def test_finalized_content_fields_present_and_extractable():
    new_paths = [
        "data.size_breakdown", "data.collection_date", "data.contamination_controls",
        "methodology.human_baseline", "methodology.judge_setup",
        "methodology.validity_justification",
        "purpose_and_intended_users.audience",
    ]
    for path in new_paths:
        assert path in ev.EVIDENCE_FIELD_WHITELIST
        # every new content field is paper-extractable and has a paper gap query
        assert "paper" in ev.CURATOR_FIELDS[path]["sources"]
        assert path in ev.gap_fields()
    # a verified quote for a new field is accepted end to end
    src = "The training data was collected in March 2023 from public sources."
    recs, telem = ev.verify_items(
        _items(("data.collection_date", "collected in March 2023")), src, ev.DOC_PAPER)
    assert len(recs) == 1 and recs[0]["field"] == "data.collection_date"


def test_contamination_reroute_documented_in_hints():
    # privacy hint points canary/leakage at contamination_controls (reroute)
    assert "contamination_controls" in (
        ev.CURATOR_FIELDS["ethical_and_legal_considerations.privacy_and_anonymity"]["hint"])
    assert "canary" in ev.CURATOR_FIELDS["data.contamination_controls"]["hint"].lower()


def test_new_field_drops_into_all_three_consumers():
    dummy = "data.dummy_axis"
    ev.CURATOR_FIELDS[dummy] = {
        "sources": frozenset({"paper", "html"}),
        "gap_query": "dummy axis query terms",
        "hint": "a dummy field",
    }
    # whitelist must be recomputed from the table (it is a frozenset snapshot at import);
    # this test exercises the derivation helpers, which read CURATOR_FIELDS live.
    try:
        assert dummy in ev.fields_for_source("paper")
        assert dummy in ev.fields_for_source("html")
        assert dummy not in ev.fields_for_source("hf")
        assert ev.gap_fields()[dummy] == "dummy axis query terms"
        assert "- data.dummy_axis: a dummy field" in ev.render_field_list([dummy])
    finally:
        del ev.CURATOR_FIELDS[dummy]


# --- robust JSON parsing ----------------------------------------------------------

def test_parse_clean_json_object():
    raw = '{"evidence": [{"field": "data.size", "quote": "500 examples", "kind": "stated"}]}'
    parsed = ev.parse_curator_extraction(raw)
    assert len(parsed.evidence) == 1
    assert parsed.evidence[0].field == "data.size"


def test_parse_json_in_markdown_fence():
    raw = ('Here is the evidence:\n```json\n'
           '{"evidence": [{"field": "data.size", "quote": "x"}]}\n```\nDone.')
    parsed = ev.parse_curator_extraction(raw)
    assert len(parsed.evidence) == 1


def test_parse_nested_braces_tag0_case():
    # The non-greedy handler regex would stop at the first '}'. Balanced matcher must not.
    raw = ('{"evidence": [{"field": "methodology.metrics", "quote": "uses {brace} token", '
           '"note": "nested {and} more"}]}')
    parsed = ev.parse_curator_extraction(raw)
    assert len(parsed.evidence) == 1
    assert parsed.evidence[0].quote == "uses {brace} token"


def test_parse_bare_list_and_single_item():
    assert len(ev.parse_curator_extraction(
        '[{"field": "data.size", "quote": "a"}]').evidence) == 1
    assert len(ev.parse_curator_extraction(
        '{"field": "data.size", "quote": "a"}').evidence) == 1


def test_parse_garbage_returns_empty():
    assert ev.parse_curator_extraction("not json at all").evidence == []
    assert ev.parse_curator_extraction("").evidence == []


def test_extract_json_object_string_aware():
    raw = 'prefix {"a": "}{ braces in string", "b": {"c": 1}} suffix'
    obj = ev.extract_json_object(raw)
    assert obj == '{"a": "}{ braces in string", "b": {"c": 1}}'


# --- abstract-only source token (fix c) -------------------------------------------

def test_doc_abstract_constant():
    assert ev.DOC_ABSTRACT == "abstract"
    assert ev.DOC_ABSTRACT != ev.DOC_PAPER


def test_verify_items_tags_abstract_doc():
    src = "The benchmark contains 1140 hand-written programming problems."
    recs, _ = ev.verify_items(
        _items(("data.size", "1140 hand-written programming problems")), src, ev.DOC_ABSTRACT)
    assert len(recs) == 1
    assert recs[0]["doc"] == "abstract"


# --- typographic normalization (near-verbatim recall, false-positive safe) --------

def test_smart_quote_and_dash_span_verifies():
    src = "The model couldn’t solve the “gold” task with the held–out split."
    quote = "couldn't solve the \"gold\" task with the held-out split"   # ASCII quotes + hyphen
    recs, telem = ev.verify_items(_items(("methodology.validation", quote)), src, ev.DOC_PAPER)
    assert len(recs) == 1 and telem["verified"] == 1


def test_case_drift_still_rejected_after_typo_norm():
    src = "Accuracy is reported as the primary metric."
    recs, telem = ev.verify_items(
        _items(("methodology.metrics", "ACCURACY IS REPORTED")), src, ev.DOC_PAPER)
    assert recs == [] and telem["reject_reasons"]["no_match"] == 1


def test_paraphrase_still_rejected_after_typo_norm():
    src = "The dataset was collected from public GitHub repositories."
    recs, _ = ev.verify_items(
        _items(("data.source", "The dataset was gathered from open repositories")), src, ev.DOC_PAPER)
    assert recs == []


# --- drop-only truncation (no marker; stays a verbatim substring) -----------------

def test_truncate_helper_drops_trailing_function_words():
    body = "x" * (ev.QUOTE_CAP - 30) + " modified protocols to an apple"
    out = ev._truncate_to_word_boundary(body)
    assert out.endswith("protocols")          # "to", "an" dropped; "apple" cut by the cap
    assert not out.endswith(" ")
    assert len(out) <= ev.QUOTE_CAP


def test_over_cap_quote_drops_dangling_tail_and_stays_substring():
    base = ("The annotators carefully reviewed and corrected each of the modified protocols "
            "across many distinct biological procedures and then presented these to an ")
    quote = (base * 2).strip()
    assert len(quote) > ev.QUOTE_CAP
    src = "Intro. " + quote + " expert reviewer."
    recs, telem = ev.verify_items(_items(("data.annotation", quote)), src, ev.DOC_PAPER)
    assert len(recs) == 1
    stored = recs[0]["quote"]
    assert len(stored) <= ev.QUOTE_CAP
    last = stored.split(" ")[-1].strip(".,;:").lower()
    assert last not in {"to", "an", "a", "the", "of", "and"}      # no dangling tail
    # no marker is appended -> still a verbatim substring at the recorded char_start
    assert ev.normalize_ws(src).find(stored) == recs[0]["char_start"]
    assert telem["truncated"] == 1


# --- curator hint sharpening (recall wiring) + generality guard -------------------

def test_curator_hints_sharpened_for_recall():
    cf = ev.CURATOR_FIELDS
    assert "pass rate" in cf["methodology.calculation"]["hint"]
    vh = cf["methodology.validation"]["hint"].lower()
    assert "validate the questions" in vh or "expert review" in vh
    cg = cf["data.contamination_controls"]["gap_query"].lower()
    assert "n-gram" in cg and "decontamination" in cg and "memorization" in cg
    oh = cf["purpose_and_intended_users.out_of_scope_uses"]["hint"].lower()
    assert "misuse" in oh or "warns against" in oh or "should not be used" in oh
    mh = cf["methodology.methods"]["hint"].lower()
    assert "scoring mechanism" in mh and "cli command" in mh
    jh = cf["methodology.judge_setup"]["hint"].lower()
    assert "equality-checker" in jh or "grader" in jh or "verifier" in jh


def test_no_curator_hint_or_query_names_a_specific_benchmark():
    banned = ("bigcode", "biolp", "aa-lcr", "humaneval", "mbpp", "swe-bench")
    for spec in ev.CURATOR_FIELDS.values():
        blob = ((spec.get("hint") or "") + " " + (spec.get("gap_query") or "")).lower()
        assert not any(b in blob for b in banned)
