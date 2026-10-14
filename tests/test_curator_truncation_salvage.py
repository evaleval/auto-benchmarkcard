"""A reply cut off by the token cap still yields the quotes it finished.

Failure class: truncated_extraction_yields_nothing. The evidence array never closes, so
the document fails to parse and a call that emitted dozens of good, verifiable quotes
produced none of them. A single truncated README call could cost a card its whole
training_context section.
"""

from __future__ import annotations

import json

from auto_benchmarkcard.tools.composer.evidence import parse_curator_extraction


def _item(field, quote):
    return {"field": field, "quote": quote, "referent": "target", "claim_role": "primary_result"}


def test_complete_items_survive_a_cut_off_array():
    complete = [_item("identity.summary", "OLMo 2 7B is a fully open model."),
                _item("training_context.training_data", "Trained on the Dolma mix.")]
    full = json.dumps({"evidence": complete})
    # cut the array open right after the second item, the way a token cap does
    truncated = full[: full.rindex("}", 0, full.rindex("]"))+ 1] + ', {"field": "identity.na'
    parsed = parse_curator_extraction(truncated)
    assert [(i.field, i.quote) for i in parsed.evidence] == [
        ("identity.summary", "OLMo 2 7B is a fully open model."),
        ("training_context.training_data", "Trained on the Dolma mix.")]


def test_a_brace_inside_a_quote_does_not_split_an_item():
    truncated = ('{"evidence": [{"field": "links.citation", "quote": "@misc{olmo2, title={2 OLMo '
                 '2 Furious}}", "referent": "target"}, {"field": "identity.na')
    parsed = parse_curator_extraction(truncated)
    assert [i.field for i in parsed.evidence] == ["links.citation"]
    assert parsed.evidence[0].quote.startswith("@misc{olmo2")


def test_clean_output_is_unchanged_and_garbage_is_still_empty():
    clean = json.dumps({"evidence": [_item("identity.name", "OLMo-2-1124-7B")]})
    assert [i.field for i in parse_curator_extraction(clean).evidence] == ["identity.name"]
    assert parse_curator_extraction("no json here at all").evidence == []
    assert parse_curator_extraction("").evidence == []


def test_the_incomplete_tail_is_discarded_not_guessed():
    truncated = '{"evidence": [{"field": "identity.summary", "quote": "A model that'
    assert parse_curator_extraction(truncated).evidence == []
