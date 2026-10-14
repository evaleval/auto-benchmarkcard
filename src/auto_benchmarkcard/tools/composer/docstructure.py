"""Deterministic document structure index over docling markdown.

Every verified quote carries a char_start into the
whitespace-normalized full document (see evidence.normalize_ws). This module maps
those offsets back to the section and table they came from, so assignment checks
can reason about WHERE evidence sits (a related-work sentence must not fill a
headline field; a number from an ablation table must not become the benchmark's
score).

Coordinate contract: build_index() normalizes the input with evidence.normalize_ws
and every char_start/char_end in the index refers to THAT normalized string, so
offsets compose directly with EvidenceItem.char_start. Never index the raw text.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from auto_benchmarkcard.tools.composer.evidence import normalize_ws

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
        # Sections are in document order and properly nested, so the last
        # containing span is the innermost one.
        best = None
        for sec in self.sections:
            if sec.char_start <= char_offset < sec.char_end:
                best = sec
        return best

    def table_at(self, char_offset: int) -> Optional[TableAnchor]:
        """The table whose span contains char_offset, or None."""
        for tab in self.tables:
            if tab.char_start <= char_offset < tab.char_end:
                return tab
        return None


_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
_DIGIT_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?(?:\s+|$)")
_LETTER_NUM_RE = re.compile(r"^([A-Z])((?:\.\d+)+)?\.?(?:\s+|$)")
_CAPTION_RE = re.compile(r"^\s*Table\s+(\d+)\b", re.IGNORECASE)
_SEP_CELL_RE = re.compile(r"^:?-+:?$")

# How many raw lines above a table to search for its 'Table N' caption line.
_CAPTION_LOOKBACK = 5


@dataclass
class _RawHeading:
    raw_start: int
    level: int
    title: str
    line: str = ""
    nump: Optional[Tuple[str, ...]] = None
    letter: str = ""
    letter_dotted: bool = False
    norm_start: int = -1


@dataclass
class _RawTable:
    raw_start: int
    block: str
    rows: List[str]
    caption: str = ""
    caption_num: str = ""
    norm_start: int = -1
    norm_end: int = -1


def _scan_raw(raw: str) -> Tuple[List[_RawHeading], List[_RawTable]]:
    """Line scan over the raw text: headings and contiguous pipe-row runs.

    Runs on the RAW text because normalize_ws collapses newlines, which erases
    line structure. Lines inside ``` fences are ignored so code comments never
    become phantom headings.
    """
    headings: List[_RawHeading] = []
    tables: List[_RawTable] = []
    lines = raw.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    in_fence = False
    run_start: Optional[int] = None  # line index of the current pipe run
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        is_pipe = not in_fence and stripped.startswith("|")
        if is_pipe:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            _close_table_run(lines, offsets, run_start, i, tables)
            run_start = None
        if in_fence:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            lead_ws = len(line) - len(line.lstrip())
            h = _RawHeading(
                raw_start=offsets[i] + lead_ws,
                level=len(m.group(1)),
                title=m.group(2),
                line=stripped,
            )
            _parse_number(h)
            headings.append(h)
    if run_start is not None:
        _close_table_run(lines, offsets, run_start, len(lines), tables)
    return headings, tables


def _parse_number(h: _RawHeading) -> None:
    """Extract a leading number path ('4.1' -> ('4','1'), 'F.2' -> ('F','2'))."""
    m = _DIGIT_NUM_RE.match(h.title)
    if m:
        h.nump = tuple(m.group(1).split("."))
        return
    m = _LETTER_NUM_RE.match(h.title)
    if m:
        h.letter = m.group(1)
        h.letter_dotted = bool(m.group(2))
        parts = (m.group(2) or "").strip(".").split(".") if m.group(2) else []
        h.nump = (m.group(1),) + tuple(parts)


def _resolve_letters(headings: List[_RawHeading]) -> None:
    """Decide which bare-letter headings really are appendix numbers.

    'A Case Study' can start with an article, so a bare letter only counts when
    digit-numbered headings came earlier at the same md level, or the title says
    appendix, or a dotted heading with the same letter (A.1) exists. Dotted
    letters are unambiguous. Rejected candidates become plain unnumbered titles.
    """
    dotted_letters = {h.letter for h in headings if h.letter and h.letter_dotted}
    seen_digit_levels = set()
    for h in headings:
        if h.nump and not h.letter:
            seen_digit_levels.add(h.level)
            continue
        if not h.letter or h.letter_dotted:
            continue
        ok = (
            h.level in seen_digit_levels
            or "appendix" in h.title.lower()
            or h.letter in dotted_letters
        )
        if not ok:
            h.nump = None
            h.letter = ""


def _close_table_run(lines: List[str], offsets: List[int], start: int, end: int,
                     tables: List[_RawTable]) -> None:
    """Finish a contiguous pipe-row run covering lines[start:end]."""
    rows = [lines[j] for j in range(start, end)]
    if len(rows) < 2:
        return
    raw_start = offsets[start] + (len(lines[start]) - len(lines[start].lstrip()))
    block = "\n".join(lines[start:end])
    tab = _RawTable(raw_start=raw_start, block=block, rows=rows)
    for j in range(start - 1, max(-1, start - 1 - _CAPTION_LOOKBACK), -1):
        line = lines[j]
        if not line.strip():
            continue
        m = _CAPTION_RE.match(line)
        if m:
            tab.caption = line.strip()
            tab.caption_num = m.group(1)
            break
    tables.append(tab)


def _parse_cells(row: str) -> List[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(_SEP_CELL_RE.match(c) for c in cells if c) and any(cells)


def _map_anchor(raw: str, norm: str, raw_start: int, chunk: str,
                cursor: int) -> Optional[Tuple[int, int]]:
    """Map a line-start raw anchor to its span in the normalized text.

    normalize_ws folds typographic chars 1:1 and collapses whitespace runs to
    single spaces, so the normalized prefix length pins the exact offset: the
    anchor starts right after a newline, which collapses into the single joining
    space. A slice check guards the assumption, falling back to an in-order find.
    """
    norm_chunk = normalize_ws(chunk)
    if not norm_chunk:
        return None
    prefix = normalize_ws(raw[:raw_start])
    start = len(prefix) + 1 if prefix else 0
    if norm[start:start + len(norm_chunk)] != norm_chunk:
        start = norm.find(norm_chunk, cursor)
        if start < 0:
            return None
    return start, start + len(norm_chunk)


def _keyword_region(text: str) -> Optional[str]:
    t = text.lower()
    if "appendix" in t:
        return "appendix"
    if "related work" in t:
        return "related_work"
    if "ablation" in t:
        return "ablation"
    if "ethic" in t or "broader impact" in t:
        return "ethics"
    if "experiment" in t or "result" in t or "evaluation" in t:
        return "results"
    if "introduction" in t:
        return "intro"
    if "data" in t:
        return "data"
    if "method" in t:
        return "method"
    return None


def _region_for(title: str, ancestors_nearest_first: List[str],
                in_appendix: bool) -> str:
    if in_appendix:
        return "appendix"
    for text in [title] + ancestors_nearest_first:
        region = _keyword_region(text)
        if region:
            return region
    return "other"


@dataclass
class _OpenSection:
    section: Section
    level: int
    nump: Optional[Tuple[str, ...]]
    is_appendix: bool


def _is_child(h: _RawHeading, top: _OpenSection) -> bool:
    """Nesting rule: when both sides carry number paths the numbers decide
    (4.1 nests under 4 even at the same md level); otherwise md level does."""
    if h.nump and top.nump:
        return len(h.nump) > len(top.nump) and h.nump[:len(top.nump)] == top.nump
    return h.level > top.level


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
    raw = markdown_text or ""
    norm = normalize_ws(raw)
    index = DocIndex(normalized_text=norm)
    if not norm:
        return index

    headings, raw_tables = _scan_raw(raw)
    _resolve_letters(headings)

    # Map every anchor into normalized coordinates in raw document order so the
    # fallback find() cursor only ever moves forward.
    anchors = sorted(
        [("h", h) for h in headings] + [("t", t) for t in raw_tables],
        key=lambda a: a[1].raw_start,
    )
    cursor = 0
    for kind, item in anchors:
        chunk = item.line if kind == "h" else item.block
        span = _map_anchor(raw, norm, item.raw_start, chunk, cursor)
        if span is None:
            continue
        item.norm_start = span[0]
        if kind == "t":
            item.norm_end = span[1]
        cursor = span[1]

    stack: List[_OpenSection] = []
    for h in headings:
        if h.norm_start < 0:
            continue
        while stack and not _is_child(h, stack[-1]):
            stack.pop().section.char_end = h.norm_start
        ancestors = [open_sec.section.title for open_sec in stack]
        in_appendix = (
            bool(h.letter)
            or "appendix" in h.title.lower()
            or any(open_sec.is_appendix for open_sec in stack)
        )
        sec = Section(
            title=h.title,
            path=ancestors + [h.title],
            region=_region_for(h.title, list(reversed(ancestors)), in_appendix),
            char_start=h.norm_start,
            char_end=len(norm),
        )
        index.sections.append(sec)
        stack.append(_OpenSection(sec, h.level, h.nump, in_appendix))
    for open_sec in stack:
        open_sec.section.char_end = len(norm)

    used_ids = set()
    seq = 1
    for t in raw_tables:
        if t.norm_start < 0:
            continue
        table_id = f"t{t.caption_num}" if t.caption_num else ""
        if not table_id or table_id in used_ids:
            while f"t{seq}" in used_ids:
                seq += 1
            table_id = f"t{seq}"
        used_ids.add(table_id)
        parsed = [_parse_cells(r) for r in t.rows]
        header_row = parsed[0] if parsed else []
        body = [c for c in parsed[1:] if not _is_separator_row(c)]
        index.tables.append(TableAnchor(
            table_id=table_id,
            caption=t.caption,
            char_start=t.norm_start,
            char_end=t.norm_end,
            header_row=header_row,
            row_labels=[c[0] for c in body if c],
        ))
    return index
