"""Deterministic document structure index over docling markdown.

Phase 2 foundation: every verified quote carries a char_start into the
whitespace-normalized full document (see evidence.normalize_ws). This module maps
those offsets back to the section and table they came from, so assignment checks
can reason about WHERE evidence sits (a related-work sentence must not fill a
headline field; a number from an ablation table must not become the benchmark's
score).

Coordinate contract: build_index() normalizes the input with evidence.normalize_ws
and every char_start/char_end in the index refers to THAT normalized string, so
offsets compose directly with EvidenceItem.char_start. Never index the raw text.
"""

from dataclasses import dataclass, field
from typing import List, Optional

REGION_TYPES = (
    "intro", "method", "data", "results", "ablation", "related_work",
    "appendix", "ethics", "other",
)


@dataclass
class Section:
    title: str
    path: List[str]
    region: str
    char_start: int
    char_end: int


@dataclass
class TableAnchor:
    table_id: str
    caption: str
    char_start: int
    char_end: int
    header_row: List[str] = field(default_factory=list)
    row_labels: List[str] = field(default_factory=list)


@dataclass
class DocIndex:
    sections: List[Section] = field(default_factory=list)
    tables: List[TableAnchor] = field(default_factory=list)
    normalized_text: str = ""

    def locate(self, char_offset: int) -> Optional[Section]:
        """The innermost section containing char_offset, or None before any heading."""
        raise NotImplementedError

    def table_at(self, char_offset: int) -> Optional[TableAnchor]:
        """The table whose span contains char_offset, or None."""
        raise NotImplementedError


def build_index(markdown_text: str) -> DocIndex:
    """Build the section/table index from docling markdown (filtered_text).

    Sections come from markdown headings (#..####); numbered headings (1, 4.1)
    and appendix letters (A, F.2) define nesting; consecutive same-level headings
    are siblings. Region typing is keyword-based over the title and its ancestor
    titles (e.g. 'related work' -> related_work, 'ablation' -> ablation,
    'experiments'/'results'/'evaluation' -> results, 'ethics'/'broader impact'
    -> ethics, appendix letters -> appendix) with 'other' as the honest default.
    Tables are contiguous runs of markdown pipe rows; the caption is the nearest
    preceding 'Table N' line when present, the header row and leading row labels
    are parsed from the pipe grid.
    """
    raise NotImplementedError
