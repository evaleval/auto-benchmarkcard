"""Tests for the document structure index (docstructure.py).

The coordinate contract is the load-bearing part: offsets produced by
evidence.verify_quote against the normalized document must land in the right
section via DocIndex.locate, even though heading/table detection runs on the
raw (newline-carrying) text.
"""

from auto_benchmarkcard.tools.composer.docstructure import build_index
from auto_benchmarkcard.tools.composer.evidence import normalize_ws, verify_quote

FIXTURE = """Preamble text before any heading appears.

## 1 Introduction

The benchmark contains 204 tasks in total.

## 2 Related Work

Prior work includes HELM and BIG-bench.

## 3 Method

We construct prompts from templates.

### 3.1 Data collection

Annotators wrote 500 examples with    multiple   spaces.

## 4 Experiments

Table 2: Main results on the test set.

| Model | Accuracy | F1 |
|---|---|---|
| GPT-4 | 86.4 | 88.1 |
| Llama | 71.2 | 70.3 |

### 4.1 Ablation study

Removing retrieval drops accuracy by 12 points.

## 5 Ethics and broader impact

The dataset may contain harmful content.

## A Appendix details

### A.1 Extra tables

| Setting | Score |
|---|---|
| base | 1.0 |

More appendix prose here.
"""


def _by_title(index, title):
    matches = [s for s in index.sections if s.title == title]
    assert len(matches) == 1, title
    return matches[0]


def test_sections_and_nesting():
    index = build_index(FIXTURE)
    titles = [s.title for s in index.sections]
    assert titles == [
        "1 Introduction", "2 Related Work", "3 Method", "3.1 Data collection",
        "4 Experiments", "4.1 Ablation study", "5 Ethics and broader impact",
        "A Appendix details", "A.1 Extra tables",
    ]
    sub = _by_title(index, "3.1 Data collection")
    assert sub.path == ["3 Method", "3.1 Data collection"]
    parent = _by_title(index, "3 Method")
    assert parent.char_start < sub.char_start
    assert sub.char_end <= parent.char_end
    # parent span covers the subsection, next top-level section closes it
    nxt = _by_title(index, "4 Experiments")
    assert parent.char_end == nxt.char_start


def test_number_refined_nesting_at_same_md_level():
    # docling flattens most headings to '##'; the number path must still nest
    text = "## 4 Experiments\n\nBody four.\n\n## 4.1 Ablation study\n\nBody.\n\n## 5 Conclusion\n\nDone.\n"
    index = build_index(text)
    sub = _by_title(index, "4.1 Ablation study")
    assert sub.path == ["4 Experiments", "4.1 Ablation study"]
    top = _by_title(index, "5 Conclusion")
    assert top.path == ["5 Conclusion"]


def test_region_typing():
    index = build_index(FIXTURE)
    expected = {
        "1 Introduction": "intro",
        "2 Related Work": "related_work",
        "3 Method": "method",
        "3.1 Data collection": "data",
        "4 Experiments": "results",
        "4.1 Ablation study": "ablation",
        "5 Ethics and broader impact": "ethics",
        "A Appendix details": "appendix",
        "A.1 Extra tables": "appendix",
    }
    got = {s.title: s.region for s in index.sections}
    assert got == expected


def test_region_falls_back_to_ancestor():
    text = "## 4 Experiments\n\nBody.\n\n### Setup details\n\nMore body.\n"
    index = build_index(text)
    assert _by_title(index, "Setup details").region == "results"


def test_article_a_is_not_appendix():
    # no digit-numbered headings anywhere, so 'A' reads as the article
    text = "## A Case Study of Robustness\n\nBody text.\n\n## Discussion\n\nMore.\n"
    index = build_index(text)
    sec = _by_title(index, "A Case Study of Robustness")
    assert sec.region != "appendix"
    assert sec.path == ["A Case Study of Robustness"]


def test_coordinate_contract_quote_lands_in_right_section():
    # the pinned end-to-end case: a quote verified against the normalized text
    # locates the correct section, across collapsed newlines and multi-spaces
    index = build_index(FIXTURE)
    quote = normalize_ws("Annotators wrote 500 examples with    multiple   spaces.")
    pos = verify_quote(quote, index.normalized_text)
    assert pos is not None
    sec = index.locate(pos)
    assert sec is not None
    assert sec.title == "3.1 Data collection"
    # a quote from the appendix prose lands in the innermost appendix section
    pos2 = verify_quote(normalize_ws("More appendix prose here."), index.normalized_text)
    assert pos2 is not None
    assert index.locate(pos2).title == "A.1 Extra tables"
    # and one from the intro
    pos3 = verify_quote(normalize_ws("The benchmark contains 204 tasks in total."), index.normalized_text)
    assert pos3 is not None
    assert index.locate(pos3).title == "1 Introduction"


def test_locate_before_first_heading_is_none():
    index = build_index(FIXTURE)
    pos = verify_quote(normalize_ws("Preamble text before any heading appears."), index.normalized_text)
    assert pos is not None
    assert index.locate(pos) is None


def test_tables():
    index = build_index(FIXTURE)
    assert len(index.tables) == 2
    main = index.tables[0]
    assert main.table_id == "t2"  # caption number wins over document order
    assert main.caption.startswith("Table 2:")
    assert main.header_row == ["Model", "Accuracy", "F1"]
    assert main.row_labels == ["GPT-4", "Llama"]
    extra = index.tables[1]
    assert extra.table_id == "t1"  # uncaptioned, first free sequential id
    assert extra.caption == ""
    assert extra.header_row == ["Setting", "Score"]
    assert extra.row_labels == ["base"]


def test_table_at_via_verified_quote():
    index = build_index(FIXTURE)
    pos = verify_quote(normalize_ws("| GPT-4 | 86.4 | 88.1 |"), index.normalized_text)
    assert pos is not None
    tab = index.table_at(pos)
    assert tab is not None and tab.table_id == "t2"
    # the table sits in section 4, before the 4.1 subsection
    assert index.locate(pos).title == "4 Experiments"
    # prose outside any table
    pos2 = verify_quote(normalize_ws("Removing retrieval drops accuracy"), index.normalized_text)
    assert index.table_at(pos2) is None


def test_empty_and_headingless_input():
    empty = build_index("")
    assert empty.sections == [] and empty.tables == []
    assert empty.locate(0) is None
    plain = build_index("Just prose, no structure at all.")
    assert plain.sections == []
    assert plain.locate(3) is None
