"""Evidence Items for the Stage-A curator: Python-verified verbatim quotes.

The LLM emits candidate evidence (field + quote). Python verifies every quote as an exact
substring of the source after whitespace normalization, records char_start, assigns a
sequential evidence_id, and drops anything it cannot verify. A verified quote cannot be a
model memory, which is the security property that replaces check_cross_contamination.

Record contract (composer_contract_v1 section 1), exactly these 7 keys:
    {evidence_id, field, quote, char_start, doc, kind, note}
The LLM never sets evidence_id, char_start or doc; Python sets those.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

QUOTE_CAP = 300  # max characters stored per quote (contract section 1)

# doc tokens (contract section 1: "docling | abstract | html | hf_readme | eee"). [A]
# produces the first four; eee provenance is deterministic and owned by [0].
# DOC_ABSTRACT is used when docling produced nothing and only the paper abstract was
# available, so abstract-derived quotes are not falsely labelled docling.
DOC_PAPER = "docling"
DOC_ABSTRACT = "abstract"
DOC_HTML = "html"
DOC_HF = "hf_readme"
# A GitHub README is web-style text, so it reuses the html field-routing; a distinct
# token keeps its provenance honest instead of mislabelling quotes as a web page.
DOC_GITHUB = "github_readme"


# CURATOR_FIELDS is the single canonical source for three things: the verifier whitelist,
# the per-source extraction-prompt field lists, and the gap-fill queries. Adding a field
# here makes it drop into all three. It is the finalized v2 content set: the contract
# section-1 paths plus the eval-cards content fields (size_breakdown, collection_date,
# contamination_controls, human_baseline, judge_setup, validity_justification,
# audience). The new paths must also be added to the
# atomizer whitelist (atomizer.py) so they flag instead of mis-routing (trap S3); once [0]
# ships a canonical field_specs, feed this table from it instead of maintaining it twice.
#
# Keys are dotted card paths. "sources" lists which extraction calls ask for the field
# (paper / html / hf). languages, data_type and data_licensing are _ALWAYS_OVERRIDE
# deterministic fields owned by [0]; they stay whitelisted but are not extracted by [A]
# (sources empty), so [A] never competes with [0] for a value [0] will overwrite.
# authors/logo/org_url/appears_in/benchmark_type are out entirely (display / EEE-injected).
# "gap_query" (paper only) drives _fill_paper_gaps. The new content fields are
# paper-extracted judged content, NOT deterministic -> they do not enter
# DETERMINISTIC_FIELDS or JUDGE_SKIP (orchestrator confirmation 2026-06-18).
CURATOR_FIELDS: Dict[str, Dict] = {
    "benchmark_details.name": {
        "sources": frozenset({"paper"}), "gap_query": None,
        "hint": "Official benchmark name and any acronym expansion",
    },
    "benchmark_details.overview": {
        "sources": frozenset({"paper", "html", "hf"}), "gap_query": None,
        "hint": ("What it measures; how many tasks/datasets; what makes it distinctive. "
                 "Also capture the verbatim sentence naming the literal measured object "
                 "(primary metric + item/sample type + count or domain breadth) if one "
                 "exists; do not infer it"),
    },
    "benchmark_details.data_type": {
        "sources": frozenset({"paper"}), "gap_query": None,
        "hint": "Primary data modality (text, image, audio, multimodal, tabular)",
    },
    "benchmark_details.domains": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "domain area subject field topic application area research discipline",
        "hint": "Research domains or subject areas covered",
    },
    "benchmark_details.languages": {
        "sources": frozenset(), "gap_query": None,
        "hint": "",  # deterministic ([0] _ALWAYS_OVERRIDE); whitelisted, not extracted by [A]
    },
    "benchmark_details.similar_benchmarks": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("related work comparison compared to versus unlike existing prior benchmark "
                      "dataset builds on extends improves over outperforms baseline alternative"),
        "hint": ("Other named benchmarks/datasets this source explicitly compares itself to, "
                 "contrasts with, or names as related/prior work (e.g. 'unlike X', 'compared to "
                 "X', 'builds on X'). Only NAMED benchmarks, not generic categories"),
    },
    "benchmark_details.resources": {
        "sources": frozenset({"paper"}), "gap_query": None,
        "hint": "URLs mentioned (homepage, leaderboard, GitHub)",
    },
    "purpose_and_intended_users.goal": {
        "sources": frozenset({"paper", "html", "hf"}), "gap_query": None,
        "hint": "Primary research objective or purpose",
    },
    "purpose_and_intended_users.tasks": {
        "sources": frozenset({"paper", "html", "hf"}), "gap_query": None,
        "hint": "Specific evaluation tasks or sub-tasks",
    },
    "purpose_and_intended_users.limitations": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "limitation weakness constraint bias shortcoming caveat",
        "hint": "Limitations, biases, or constraints acknowledged",
    },
    "purpose_and_intended_users.out_of_scope_uses": {
        "sources": frozenset({"paper"}),
        "gap_query": ("not designed for should not be used inappropriate misuse out of scope "
                      "does not measure not a substitute caution against"),
        "hint": ("Concrete uses the source warns AGAINST or says it does NOT support (e.g. "
                 "'not designed to evaluate agents', 'should not be used to certify safety', "
                 "'does not measure X'). A specific misuse to avoid, NOT a restatement of what "
                 "it IS for"),
    },
    "purpose_and_intended_users.audience": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "who runs evaluates uses interprets benchmark evaluator practitioner intended user community stakeholder",
        "hint": "Who the benchmark is designed for -- who runs it and who uses its results",
    },
    "data.source": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "data collection source origin corpus created gathered compiled",
        "hint": ("Where the ORIGINAL data came from and how it was collected; NOT how a "
                 "subset/variant (e.g. a 'Hard' split) was selected or filtered"),
    },
    "data.size": {
        "sources": frozenset({"paper", "html", "hf"}), "gap_query": None,
        "hint": "Number of examples and train/dev/test splits",
    },
    "data.format": {
        "sources": frozenset({"paper"}), "gap_query": None,
        "hint": "How the data instances are structured",
    },
    "data.annotation": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "annotation annotator label crowdworker human judge agreement quality control",
        "hint": "How labeling was done, who annotated, quality control",
    },
    "data.size_breakdown": {
        "sources": frozenset({"paper", "hf"}),
        "gap_query": "subset split breakdown per-domain per-category item counts size distribution",
        "hint": "Per-subset or per-domain item counts (e.g. '{subset}: {count}' breakdowns)",
    },
    "data.collection_date": {
        "sources": frozenset({"paper", "hf"}),
        "gap_query": "data collection date period when collected cutoff timeframe gathered released",
        "hint": "When the data was collected, or its cutoff/knowledge date",
    },
    "data.contamination_controls": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("contamination canary string held-out private hidden test split encryption "
                      "decontamination n-gram overlap memorization data leakage prevention "
                      "training cutoff gated"),
        "hint": ("Contamination controls: canary strings, held-out/private/hidden or gated test "
                 "splits, encryption, decontamination or n-gram overlap checks, memorization or "
                 "training-cutoff safeguards, leakage prevention"),
    },
    "methodology.methods": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("evaluation method zero-shot few-shot fine-tune prompt protocol scored graded "
                      "correct response checked compared"),
        "hint": ("How a model's answer is turned into a result: the evaluation setting (zero-shot, "
                 "few-shot, fine-tuning) AND the scoring mechanism (exact match, unit tests, an "
                 "equality/grader check). NOT the CLI command used to launch a run"),
    },
    "methodology.metrics": {
        "sources": frozenset({"paper", "html", "hf"}), "gap_query": None,
        "hint": "Metrics used (list each by name)",
    },
    "methodology.calculation": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "overall score calculation aggregate average weighted macro pass rate percentage correct",
        "hint": ("How the score is computed from per-item results: aggregation (average, weighted, "
                 "macro), pass rate / percentage correct, or any 'we compute/report the score as "
                 "...' or 'we use ... to evaluate/check responses' statement"),
    },
    "methodology.interpretation": {
        "sources": frozenset({"paper"}), "gap_query": None,
        "hint": "What score ranges indicate strong vs weak performance",
    },
    "methodology.baseline_results": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "baseline result score performance accuracy model evaluation",
        "hint": ("Specific model results reported (model names and scores), plus the "
                 "baseline's conditions (time limit, tool/web access) where reported"),
    },
    "methodology.validation": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("quality assurance reliability reproducibility expert review verified checked "
                      "validated questions answers inter-annotator agreement pilot study manually "
                      "filtered curated double-checked"),
        "hint": ("Quality assurance steps taken on the benchmark itself: reliability or "
                 "reproducibility procedures, expert review, or having people write/check/validate "
                 "the questions or answers before release"),
    },
    "methodology.human_baseline": {
        "sources": frozenset({"paper", "hf"}),
        "gap_query": "human baseline human performance human expert reference accuracy time limit",
        "hint": ("Human or reference performance and its conditions (time limit, tool/web "
                 "access, expertise level), separate from model scores"),
    },
    "methodology.judge_setup": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("LLM judge evaluator GPT automatic grading rubric judge model consolidated "
                      "equality checker grader verifier scored by model"),
        "hint": ("LLM-as-judge details ONLY about how MODEL OUTPUTS are scored/graded by a model: "
                 "whether an LLM/grader/equality-checker/verifier model judges responses, how many, "
                 "which models, how their scores are consolidated. NOT how the dataset was built or "
                 "annotated, and NOT human annotators"),
    },
    "methodology.validity_justification": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("construct validity justification why metric measures rationale valid measure "
                      "reflects real-world realistic"),
        "hint": ("Where the source argues WHY its tasks/metrics genuinely measure the intended "
                 "ability (e.g. 'realistic', 'reflects real-world use', 'designed to capture'), as "
                 "opposed to merely describing QA/reproducibility (that goes to validation)"),
    },
    "ethical_and_legal_considerations.privacy_and_anonymity": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("privacy anonymized de-identified personally identifiable information PII "
                      "personal data removed redacted faces blurred names sensitive patient "
                      "synthetic data no personal information"),
        "hint": ("How PII is handled; whether data is anonymized. Canary strings, "
                 "held-out/private splits and leakage prevention go to "
                 "data.contamination_controls, not here"),
    },
    "ethical_and_legal_considerations.data_licensing": {
        "sources": frozenset(), "gap_query": None,
        "hint": "",  # deterministic ([0] _ALWAYS_OVERRIDE); whitelisted, not extracted by [A]
    },
    "ethical_and_legal_considerations.consent_procedures": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": "crowdworker consent compensation IRB ethical approval platform",
        "hint": "How annotators were compensated; platform; consent",
    },
    "ethical_and_legal_considerations.compliance_with_regulations": {
        "sources": frozenset({"paper", "html", "hf"}),
        "gap_query": ("IRB institutional review board ethics approval ethical review GDPR HIPAA "
                      "regulatory compliance data protection regulation informed consent waiver "
                      "approved by"),
        "hint": "IRB approval, GDPR compliance, ethical review",
    },
}

# The verifier whitelist (contract section 1: field must be one of the 26 paths).
EVIDENCE_FIELD_WHITELIST = frozenset(CURATOR_FIELDS)


class EvidenceItem(TypedDict):
    """A verified evidence record (the 7-key contract output).

    char_start indexes the WHITESPACE-NORMALIZED doc (runs of whitespace collapsed to one
    space, stripped), not the raw source. A downstream consumer that wants to slice the
    span must normalize the doc the same way (normalize_ws) before using char_start.
    """
    evidence_id: str
    field: str
    quote: str
    char_start: int
    doc: str
    kind: str
    note: str


class EvidenceItemIn(BaseModel):
    """What the LLM emits per item. No evidence_id / char_start / doc (Python sets those).

    kind is a plain str (not Literal) so a stray value does not invalidate the whole
    extraction; verify_items coerces anything outside {stated, derived} to stated.
    """
    field: str = ""
    quote: str = ""
    kind: str = "stated"
    note: str = ""


class CuratorExtraction(BaseModel):
    """Top-level LLM output container (a single JSON object, brace-repair friendly)."""
    evidence: List[EvidenceItemIn] = Field(default_factory=list)


def fields_for_source(source: str) -> List[str]:
    """Dotted paths the given extraction call (paper/html/hf) should ask for."""
    return [p for p, spec in CURATOR_FIELDS.items() if source in spec["sources"]]


def gap_fields() -> Dict[str, str]:
    """Paper gap-fill targets: dotted path -> retrieval query."""
    return {p: spec["gap_query"] for p, spec in CURATOR_FIELDS.items() if spec["gap_query"]}


def render_field_list(paths: List[str]) -> str:
    """Render allowed fields as prompt bullets with their hints."""
    lines = []
    for p in paths:
        hint = CURATOR_FIELDS.get(p, {}).get("hint", "")
        lines.append(f"- {p}: {hint}" if hint else f"- {p}")
    return "\n".join(lines)


EXTRACTION_SYSTEM = (
    "You are a precise extraction assistant. You read one source document about a named "
    "benchmark and return verbatim supporting quotes as JSON. You never paraphrase a quote, "
    "never use knowledge from outside the document, and only extract facts about the target "
    "benchmark, not other benchmarks mentioned in the same document."
)


def build_extraction_prompt(
    source_label: str,
    fields: List[str],
    benchmark_name: str,
    identity_anchor: str = "",
    source_text: str = "",
    name_alias_line: str = "",
) -> str:
    """Assemble the full extraction prompt (system + body) for one source call.

    Built by concatenation rather than str.format so the JSON example braces need no
    escaping. The model is asked for verbatim quotes only; Python verifies them.
    """
    header = (
        f'Read this {source_label} about the benchmark "{benchmark_name}" and extract '
        f"verbatim supporting quotes."
    )
    schema_line = (
        'Return a JSON object of this form: '
        '{"evidence": [{"field": "<dotted.path>", "quote": "<verbatim text>", '
        '"kind": "stated|derived", "note": "<optional>"}]}'
    )
    rules = (
        "For each item:\n"
        "- \"field\": one of the allowed dotted paths listed below (use the exact path).\n"
        "- \"quote\": copy the EXACT supporting sentence VERBATIM from the source text below. "
        "Do NOT paraphrase, summarize, translate, or fix typos. Keep it under 300 characters. "
        "The quote must appear word for word in the source.\n"
        "- \"kind\": \"stated\" if the quote directly states the fact; \"derived\" if the field "
        "is a conclusion you infer, in which case \"quote\" must be the premise sentence(s).\n"
        "- \"note\": optional short gloss (never used as evidence; may be empty).\n"
        "\n"
        "Rules:\n"
        f"- Only extract facts about \"{benchmark_name}\". Ignore other benchmarks or datasets "
        "discussed in the source.\n"
        "- If there is no verbatim supporting sentence for a field, emit NO item for it. "
        "Never invent a quote.\n"
        "- Use ONLY the source text below. Do not use outside knowledge.\n"
        "- Multiple items per field are allowed; emit the strongest supporting sentences."
    )
    parts = [header]
    if identity_anchor:
        parts.append(identity_anchor)
    if name_alias_line:
        parts.append(name_alias_line)
    parts.extend([schema_line, rules, "Allowed fields:", render_field_list(fields),
                  f"\nSOURCE TEXT ({source_label}):", source_text])
    body = "\n".join(parts)
    return f"{EXTRACTION_SYSTEM}\n\n{body}"


_WS_RE = re.compile(r"\s+")

# Typographic variants a model often substitutes when copying a span: curly quotes and
# unicode dashes fold to their ASCII forms. Both quote and source pass through this, so it
# only ever lets a near-verbatim span verify; it never changes a content word and never folds
# case, so it cannot make a paraphrase or a different sentence match.
_TYPOGRAPHIC = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "´": "'", "ʼ": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
}
_TYPO_RE = re.compile("|".join(re.escape(c) for c in _TYPOGRAPHIC))


def normalize_ws(text: str) -> str:
    """Normalize for verbatim matching: fold typographic quote/dash variants to ASCII, then
    collapse runs of whitespace to a single space and strip. Unicode-aware (NBSP etc.).

    Case is deliberately NOT folded: a case-drifted quote must still be rejected, since the
    model rewriting case is a sign it did not copy the span verbatim.
    """
    text = _TYPO_RE.sub(lambda m: _TYPOGRAPHIC[m.group(0)], text)
    return _WS_RE.sub(" ", text).strip()


# Function words that read as dangling if a truncated quote ends on them; dropped from the
# tail so the stored prefix ends on a content word. No marker is appended -- the result stays
# a verbatim prefix (a substring) of the normalized source.
_DANGLING_TAIL = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "were", "that", "which", "this", "these",
    "we", "they", "it", "its", "their", "into", "than", "then", "so",
})


def _truncate_to_word_boundary(quote: str, cap: int = QUOTE_CAP) -> str:
    """Truncate an over-cap quote to the last word boundary <= cap (prefix stays verbatim).

    After the boundary cut, trailing function words are dropped so the prefix never ends on a
    dangling article/preposition. The result is still a verbatim prefix of the (normalized)
    source -- no marker is appended -- so it stays substring-checkable.
    """
    if len(quote) <= cap:
        return quote
    head = quote[:cap]
    sp = head.rfind(" ")
    head = head[:sp] if sp > 0 else head
    words = head.split(" ")
    while len(words) > 1 and words[-1].strip(".,;:").lower() in _DANGLING_TAIL:
        words.pop()
    return " ".join(words)


def verify_quote(norm_quote: str, norm_source: str) -> Optional[int]:
    """Return the char_start of norm_quote in norm_source, or None if it is not a substring."""
    if not norm_quote:
        return None
    idx = norm_source.find(norm_quote)
    return idx if idx >= 0 else None


def empty_call_telemetry(doc: str) -> Dict:
    """Zero-count telemetry for a call that emitted nothing (e.g. missing source)."""
    return {
        "doc": doc, "emitted": 0, "verified": 0, "rejected": 0, "truncated": 0,
        "reject_reasons": {"no_match": 0, "empty": 0, "bad_field": 0},
    }


def verify_items(raw_items, source_text: str, doc: str) -> Tuple[List[Dict], Dict]:
    """Verify candidate items against the source. Returns (id-less records, call telemetry).

    Each record has six keys (field, quote, char_start, doc, kind, note); evidence_id is
    assigned later across all calls. Fail-closed: anything not verifiable is dropped and
    counted. The source is whitespace-normalized once per call.
    """
    telem = empty_call_telemetry(doc)
    norm_source = normalize_ws(source_text or "")
    records: List[Dict] = []

    for raw in raw_items or []:
        telem["emitted"] += 1
        d = raw.model_dump() if isinstance(raw, BaseModel) else dict(raw)

        field = str(d.get("field", "") or "").strip()
        if field not in EVIDENCE_FIELD_WHITELIST:
            telem["rejected"] += 1
            telem["reject_reasons"]["bad_field"] += 1
            continue

        norm_quote = normalize_ws(str(d.get("quote", "") or ""))
        if not norm_quote:
            telem["rejected"] += 1
            telem["reject_reasons"]["empty"] += 1
            continue

        char_start = verify_quote(norm_quote, norm_source)
        if char_start is None:
            telem["rejected"] += 1
            telem["reject_reasons"]["no_match"] += 1
            continue

        if len(norm_quote) > QUOTE_CAP:
            norm_quote = _truncate_to_word_boundary(norm_quote)
            telem["truncated"] += 1

        kind = d.get("kind", "stated")
        if kind not in ("stated", "derived"):
            kind = "stated"

        records.append({
            "field": field,
            "quote": norm_quote,
            "char_start": char_start,
            "doc": doc,
            "kind": kind,
            "note": str(d.get("note", "") or ""),
        })
        telem["verified"] += 1

    return records, telem


def new_card_telemetry() -> Dict:
    """Empty card-level quote-verify telemetry (feeds the GO-gate quote-verify-rate)."""
    return {
        "emitted": 0, "verified": 0, "rejected": 0, "truncated": 0,
        "reject_reasons": {"no_match": 0, "empty": 0, "bad_field": 0},
        "by_doc": {},
    }


def merge_telemetry(card: Dict, call: Dict) -> None:
    """Fold one call's telemetry into the card-level accumulator (in place)."""
    for k in ("emitted", "verified", "rejected", "truncated"):
        card[k] += call[k]
    for reason, count in call["reject_reasons"].items():
        card["reject_reasons"][reason] += count
    bucket = card["by_doc"].setdefault(
        call["doc"], {"emitted": 0, "verified": 0, "rejected": 0, "truncated": 0}
    )
    for k in ("emitted", "verified", "rejected", "truncated"):
        bucket[k] += call[k]


def verify_rate(card: Dict) -> float:
    """Verified / emitted as a fraction (0.0 when nothing was emitted)."""
    emitted = card.get("emitted", 0)
    return (card.get("verified", 0) / emitted) if emitted else 0.0


def assign_evidence_ids(record_lists: List[List[Dict]]) -> List[EvidenceItem]:
    """Assign sequential E01.. ids across all calls, in the given list order."""
    items: List[EvidenceItem] = []
    n = 0
    for lst in record_lists:
        for rec in lst or []:
            n += 1
            items.append({"evidence_id": f"E{n:02d}", **rec})  # type: ignore[typeddict-item]
    return items


def build_digest(items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    """Group verified items by field (insertion order, each field in evidence_id order)."""
    digest: Dict[str, List[EvidenceItem]] = {}
    for it in items:
        digest.setdefault(it["field"], []).append(it)
    return digest


# Field paths used by the Stage-A evidence router (route_evidence).
_TASKS_FIELD = "purpose_and_intended_users.tasks"
_DATA_TYPE_FIELD = "benchmark_details.data_type"
_ANNOTATION_FIELD = "data.annotation"
_OUT_OF_SCOPE_FIELD = "purpose_and_intended_users.out_of_scope_uses"


def route_evidence(records: List[Dict]) -> List[Dict]:
    """Re-file or drop verified evidence by quote topic before ids are assigned.

    Model-independent (regex-only) backstop for Stage-A field mis-tags:
      - a benchmark_details.data_type quote that is a metric/methodology description with no
        modality term is dropped, so Stage-B abstains instead of emitting a junk modality;
      - a purpose_and_intended_users.tasks quote that actually describes dataset construction
        is also surfaced as data.annotation; one that frames what the benchmark is NOT for is
        also surfaced as out_of_scope_uses. The original tasks record is always kept.
    Derived records copy the verified span (quote/char_start/doc) so no re-verification is
    needed; they are deduped against existing and prior-derived (field, char_start, doc).
    """
    from auto_benchmarkcard.tools.composer import aboutness

    kept: List[Dict] = []
    seen = set()  # (field, char_start, doc) already present; dedups derived items
    for rec in records or []:
        field = rec.get("field", "")
        quote = rec.get("quote", "") or ""
        if field == _DATA_TYPE_FIELD and aboutness.is_metric_only_data_type(quote):
            continue  # drop the metric/methodology quote mis-tagged as a modality
        kept.append(rec)
        seen.add((field, rec.get("char_start"), rec.get("doc")))

    derived: List[Dict] = []
    for rec in kept:
        if rec.get("field") != _TASKS_FIELD:
            continue
        quote = rec.get("quote", "") or ""
        if aboutness.is_construction_quote(quote):
            target = _ANNOTATION_FIELD
        elif aboutness.is_scope_framing(quote):
            target = _OUT_OF_SCOPE_FIELD
        else:
            continue
        key = (target, rec.get("char_start"), rec.get("doc"))
        if key in seen:
            continue
        seen.add(key)
        derived.append({**rec, "field": target})

    return kept + derived


def finalize_evidence(
    paper: List[Dict], gap: List[Dict], html: List[Dict], hf: List[Dict],
    github: Optional[List[Dict]] = None,
) -> Tuple[List[EvidenceItem], Dict[str, List[EvidenceItem]]]:
    """Route by topic, assign ids in canonical order (paper, gap, html, github, hf), build the digest.

    route_evidence runs before id assignment so the surviving ids stay dense (E01..En) and the
    derived re-filed records get their own ids at the tail, matching the digest exactly.
    """
    routed = route_evidence([*paper, *gap, *html, *(github or []), *hf])
    items = assign_evidence_ids([routed])
    return items, build_digest(items)


def extract_json_object(raw: str) -> Optional[str]:
    """Return the first balanced {...} block in raw (string-aware), or None.

    Robust to nested braces, which the handler's own json-repair (llm_handler.py:122,
    non-greedy until the Tag-0 fix lands) cannot handle. We do not depend on that fix.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def parse_curator_extraction(raw: str) -> CuratorExtraction:
    """Parse LLM output into a CuratorExtraction. Fail-soft to empty on any error.

    Handles clean JSON, JSON wrapped in prose or markdown fences (balanced-brace
    extraction), a bare top-level list, and a single bare item.
    """
    if not raw or not raw.strip():
        return CuratorExtraction()

    candidates = [raw.strip()]
    obj = extract_json_object(raw)
    if obj and obj != raw.strip():
        candidates.append(obj)

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            data = {"evidence": data}
        elif isinstance(data, dict) and "evidence" not in data and "field" in data:
            data = {"evidence": [data]}
        try:
            return CuratorExtraction.model_validate(data)
        except Exception as e:  # pydantic ValidationError and friends
            logger.debug("Curator extraction validation failed: %s", e)
            continue

    logger.warning("Could not parse curator extraction from %d chars; 0 items", len(raw))
    return CuratorExtraction()
