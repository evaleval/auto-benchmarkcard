"""Compose structured benchmark cards from heterogeneous metadata via LLM synthesis.

Architecture: Extract (isolated) -> Merge -> Compose -> Override -> Post-process
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import requests
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field, create_model, field_validator

from auto_benchmarkcard.card_utils import _label_hosting_format, extract_hf_tags, map_benchmark_type
from auto_benchmarkcard.config import Config, get_llm_handler
from auto_benchmarkcard.output import sanitize_chroma_collection_name
from auto_benchmarkcard.tools.composer import aboutness
from auto_benchmarkcard.tools.composer import evidence
from auto_benchmarkcard.tools.composer import field_spec as fs
from auto_benchmarkcard.tools.composer import validator as _validator

logger = logging.getLogger(__name__)

# Stage-A extraction prompts now live in evidence.build_extraction_prompt (verbatim-quote
# Evidence Items). _EXTRACTOR_SYSTEM below is retained for the sub-benchmark path only.

_EXTRACTOR_SYSTEM = (
    "You are a precise research assistant that reads source material and describes what it says. "
    "You rephrase information in your own words — you do NOT copy text verbatim. "
    "You NEVER use your own knowledge about benchmarks or AI — only what the provided text says. "
    "If something is not in the text, you say 'No information found'. "
    "CRITICAL: You must ONLY extract facts about the specific benchmark named in the prompt. "
    "Do NOT confuse it with other benchmarks, datasets, or methods discussed in the same paper. "
    "If the paper discusses multiple benchmarks, only describe facts about the TARGET benchmark."
)

SECTION_QUERIES = {
    "benchmark_details": [
        "benchmark name overview introduction contribution",
        "related work similar benchmarks comparison",
        "resources homepage leaderboard repository URL",
    ],
    "data": [
        "dataset collection source corpus sub-task data origin",
        "dataset size examples training test split statistics",
        "annotation crowdsource label annotator agreement quality",
    ],
    "methodology": [
        "evaluation method metrics accuracy F1 score measurement",
        "baseline results performance human comparison model scores",
        "diagnostic analysis validation quality assurance",
    ],
    "purpose_and_intended_users": [
        "goal objective motivation purpose research question",
        "tasks sub-tasks evaluation individual task description",
        "limitations bias constraints scope out-of-scope",
    ],
    "ethical_and_legal_considerations": [
        "ethics privacy anonymity personal information",
        "license consent crowdworker compensation IRB",
    ],
}

ALL_SECTIONS = [
    "benchmark_details",
    "purpose_and_intended_users",
    "data",
    "methodology",
    "ethical_and_legal_considerations",
]

LANG_MAP = {
    "en": "English", "zh": "Chinese", "de": "German", "fr": "French",
    "es": "Spanish", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "multilingual": "Multilingual",
}


def _get_benchmark_identity(
    benchmark_name: str,
    hf_metadata: Optional[Dict[str, Any]] = None,
    eee_metadata: Optional[Dict[str, Any]] = None,
    paper_title: str = "",
) -> str:
    """Build a short identity anchor so the LLM doesn't confuse benchmarks."""
    signals = []

    if paper_title and paper_title.lower() != benchmark_name.lower():
        signals.insert(0, f"Also known as: {paper_title}")

    if hf_metadata:
        meta = _get_hf_meta(hf_metadata)
        tags = meta.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("task_categories:"):
                    signals.append(tag.split(":", 1)[1].strip().replace("-", " "))
            for tag in tags:
                if isinstance(tag, str) and ":" not in tag and len(tag) > 2:
                    signals.append(tag.replace("-", " "))

        card_data = meta.get("card_data", {})
        if isinstance(card_data, dict):
            summary = card_data.get("dataset_summary", "")
            if summary and len(summary) < 300:
                signals.append(summary)

        desc = meta.get("description", "")
        if desc and len(desc) < 200:
            signals.append(desc)

    if not signals:
        return ""

    seen = set()
    unique = []
    for s in signals:
        sl = s.lower().strip()
        if sl not in seen and sl:
            seen.add(sl)
            unique.append(s.strip())

    identity = ", ".join(unique[:6])
    return (
        f'BENCHMARK IDENTITY: "{benchmark_name}" is about: {identity}. '
        f"Only extract facts that are relevant to this topic."
    )


def _run_extraction(prompt: str, verify_source: str, doc: str, label: str):
    """Run one Stage-A call: LLM -> robust JSON parse -> quote verification.

    Returns (verified records without evidence_id, call telemetry). Fail-soft: an LLM or
    parse failure yields zero items, never an exception. Every quote is verified as an
    exact substring of verify_source; the LLM is never trusted.
    """
    try:
        llm_handler = get_llm_handler()
        raw = llm_handler.generate(
            prompt, response_format=evidence.CuratorExtraction.model_json_schema()
        )
    except Exception as e:
        logger.warning("%s extraction call failed: %s", label, e)
        return [], evidence.empty_call_telemetry(doc)

    parsed = evidence.parse_curator_extraction(raw)
    records, telem = evidence.verify_items(parsed.evidence, verify_source, doc)
    logger.info("%s extraction: %d/%d quotes verified", label, telem["verified"], telem["emitted"])
    return records, telem


def extract_facts_from_paper(
    paper_content: str, benchmark_name: str, identity_anchor: str = "",
    paper_title: str = "", verify_source: str = "", doc: str = "",
):
    """Extract verified Evidence Items from paper text in isolation (no other sources).

    The model reads paper_content; quotes are verified against verify_source (the full
    paper text) so paper and gap-fill items share one docling reference. Returns
    (records, telemetry). doc is the provenance source token (DOC_PAPER for docling
    full text, DOC_ABSTRACT when only the abstract was available).
    """
    doc = doc or evidence.DOC_PAPER
    if not paper_content or paper_content == "Not available":
        return [], evidence.empty_call_telemetry(doc)

    name_alias_line = (
        f'This benchmark may also be referred to as: "{paper_title}"'
        if paper_title else ""
    )
    prompt = evidence.build_extraction_prompt(
        source_label="research paper",
        fields=evidence.fields_for_source("paper"),
        benchmark_name=benchmark_name,
        identity_anchor=identity_anchor,
        source_text=paper_content,
        name_alias_line=name_alias_line,
    )
    return _run_extraction(prompt, verify_source or paper_content, doc, "Paper")


def extract_facts_from_hf_readme(
    hf_metadata: Dict[str, Any],
    benchmark_name: str,
    hf_retriever: Any = None,
    identity_anchor: str = "",
):
    """Extract verified Evidence Items from the HF README (capped at 15K chars)."""
    readme = _get_hf_readme(hf_metadata)
    if not readme or len(readme) < 100:
        return [], evidence.empty_call_telemetry(evidence.DOC_HF)

    hf_content = readme[:15000]
    prompt = evidence.build_extraction_prompt(
        source_label="HuggingFace dataset page",
        fields=evidence.fields_for_source("hf"),
        benchmark_name=benchmark_name,
        identity_anchor=identity_anchor,
        source_text=hf_content,
    )
    return _run_extraction(prompt, hf_content, evidence.DOC_HF, "HF README")


def extract_facts_from_html(
    html_content: Dict[str, Any],
    benchmark_name: str,
    identity_anchor: str = "",
):
    """Extract verified Evidence Items from HTML web page content."""
    text = html_content.get("text", "")
    if not text or len(text) < 50:
        return [], evidence.empty_call_telemetry(evidence.DOC_HTML)

    html_text = text[:15000]
    prompt = evidence.build_extraction_prompt(
        source_label="web page",
        fields=evidence.fields_for_source("html"),
        benchmark_name=benchmark_name,
        identity_anchor=identity_anchor,
        source_text=html_text,
    )
    return _run_extraction(prompt, html_text, evidence.DOC_HTML, "HTML")


def extract_facts_from_github_readme(
    github_readme: Dict[str, Any],
    benchmark_name: str,
    identity_anchor: str = "",
):
    """Extract verified Evidence Items from a GitHub repo README.

    A README is web-style text, so it reuses the html field-routing but keeps its own
    DOC_GITHUB provenance token.
    """
    text = github_readme.get("text", "")
    if not text or len(text) < 50:
        return [], evidence.empty_call_telemetry(evidence.DOC_GITHUB)

    readme_text = text[:15000]
    prompt = evidence.build_extraction_prompt(
        source_label="GitHub repository README",
        fields=evidence.fields_for_source("html"),
        benchmark_name=benchmark_name,
        identity_anchor=identity_anchor,
        source_text=readme_text,
    )
    return _run_extraction(prompt, readme_text, evidence.DOC_GITHUB, "GitHub README")


SUB_BENCHMARK_EXTRACTION_PROMPT = """This paper describes the benchmark "{parent_name}".
{name_alias_line}
Search the text for ANY information specifically about the variant, sub-benchmark, category, or split called "{sub_name}".
Also look for references to just "{sub_label}" — e.g., a section, category, task type, or evaluation split called "{sub_label}".

Look for:
- Sections, subsections, or paragraphs describing "{sub_label}" specifically
- Data splits, sizes, or sources unique to this variant
- Metrics or evaluation methodology that differs from the parent
- Specific results or baselines reported for this variant
- Task categories or domains unique to this variant
- Any definition or description of what "{sub_label}" means in the context of this benchmark

If you find specific information, describe it using the fields below.
If the paper does NOT mention "{sub_label}" or "{sub_name}" and has NO information specific to it beyond what applies to the parent benchmark as a whole, respond with EXACTLY: NO SPECIFIC INFORMATION FOUND

{identity_anchor}

Fields to describe (ONLY if paper has sub-benchmark-specific information):

## benchmark_details
- overview: What specifically distinguishes this sub-benchmark from the parent?
- domains: Any specific domains or categories this variant focuses on

## data
- source: Data sources unique to this sub-benchmark
- size: Size information specific to this sub-benchmark

## methodology
- metrics: Metrics specific to this sub-benchmark
- baseline_results: Results reported specifically for this variant

PAPER TEXT:
{paper_content}"""

HF_SUB_BENCHMARK_EXTRACTION_PROMPT = """This HuggingFace dataset README describes the benchmark "{parent_name}".
Search the text for ANY information specifically about the variant, sub-benchmark, category, or split called "{sub_name}".
Also look for references to just "{sub_label}" — e.g., a section, category, task type, or split called "{sub_label}".

If you find specific information about what "{sub_label}" is or how it differs from the parent benchmark, describe it.
If the README does NOT mention "{sub_label}" or "{sub_name}", respond with EXACTLY: NO SPECIFIC INFORMATION FOUND

Fields to describe (ONLY if README has sub-benchmark-specific information):

## benchmark_details
- overview: What specifically distinguishes this sub-benchmark from the parent?
- domains: Any specific domains or categories this variant focuses on

## data
- source: Data sources unique to this sub-benchmark
- size: Size information specific to this sub-benchmark

README TEXT:
{readme_content}"""

SUB_BENCHMARK_FALLBACK_PROMPT = """You are writing a brief, factual description for a sub-benchmark card.

Parent benchmark: "{parent_name}"
Parent overview: "{parent_overview}"
Sub-benchmark name: "{sub_name}"
Distinguishing label: "{sub_label}"

Based on the sub-benchmark name and the parent benchmark context, write exactly 1-2 sentences describing what this sub-benchmark likely evaluates or measures, and how it relates to the parent benchmark. Be specific about the "{sub_label}" component.

Keep it factual and concise. Do NOT invent specific numbers, dates, or paper citations."""


def _extract_sub_label(sub_name: str, parent_name: str) -> str:
    """Extract the distinguishing label from a sub-benchmark name.

    E.g., 'rewardbench_2_safety' with parent 'rewardbench_2' → 'safety'
          'bfcl_multi_turn' with parent 'bfcl' → 'multi turn'
          'arc_agi_v2_public_eval' with parent 'arc_agi' → 'v2 public eval'
    """
    # Simple normalization (avoid importing from eee_tool to prevent circular imports)
    def _norm(n):
        return n.strip().lower().replace("_", "-").replace(" ", "-")

    parent_norm = _norm(parent_name)
    sub_norm = _norm(sub_name)

    # Try to strip the parent prefix
    if sub_norm.startswith(parent_norm):
        label = sub_norm[len(parent_norm):].strip().strip("-").strip()
        if label:
            return label.replace("-", " ")

    # Fallback: just replace underscores/dashes with spaces
    return sub_name.replace("_", " ").replace("-", " ")


def _targeted_extract(paper_content: str, sub_label: str, context_chars: int = 2000) -> str:
    """Extract paragraphs from paper that mention the sub-benchmark label.

    Returns the most relevant portions of the paper text containing the label,
    plus the first section for context. Caps total at context_chars * number of matches.
    """
    paragraphs = paper_content.split("\n\n")
    label_lower = sub_label.lower()
    # Also search for individual words if label is multi-word
    label_words = label_lower.split()

    relevant = []
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        if label_lower in para_lower:
            # Include surrounding paragraphs for context
            start = max(0, i - 1)
            end = min(len(paragraphs), i + 2)
            for j in range(start, end):
                if paragraphs[j] not in relevant:
                    relevant.append(paragraphs[j])
        elif len(label_words) > 1 and all(w in para_lower for w in label_words):
            if para not in relevant:
                relevant.append(para)

    if relevant:
        # Include first 2 paragraphs for paper context + all relevant paragraphs
        header = "\n\n".join(paragraphs[:2])
        body = "\n\n".join(relevant)
        return f"{header}\n\n...\n\n{body}"

    return ""


def extract_sub_benchmark_facts(
    paper_content: str,
    parent_name: str,
    sub_name: str,
    identity_anchor: str = "",
    paper_title: str = "",
    hf_readme_content: str = "",
) -> Optional[str]:
    """Search paper text and HF README for information specific to a sub-benchmark.

    Returns extracted facts string if found, None if nothing specific was found.
    Tries: (1) targeted paper extraction, (2) full paper extraction, (3) HF README.
    """
    sub_label = _extract_sub_label(sub_name, parent_name)

    name_alias_line = (
        f'This benchmark may also be referred to as: "{paper_title}"'
        if paper_title else ""
    )

    from auto_benchmarkcard.config import get_llm_handler
    llm_handler = get_llm_handler()

    # Strategy 1: Targeted extraction — find paragraphs mentioning sub_label
    if paper_content:
        targeted_text = _targeted_extract(paper_content, sub_label)
        search_text = targeted_text if targeted_text else paper_content[:30000]

        prompt = SUB_BENCHMARK_EXTRACTION_PROMPT.format(
            parent_name=parent_name,
            sub_name=sub_name,
            sub_label=sub_label,
            paper_content=search_text,
            identity_anchor=identity_anchor,
            name_alias_line=name_alias_line,
        )

        try:
            result = llm_handler.generate(f"{_EXTRACTOR_SYSTEM}\n\n{prompt}")

            if "NO SPECIFIC INFORMATION FOUND" not in result.upper():
                logger.info("Found sub-benchmark facts for %s (paper): %d chars", sub_name, len(result))
                return result

            logger.info("No specific info in paper for sub-benchmark %s (label: %s)", sub_name, sub_label)
        except Exception as e:
            logger.warning("Paper sub-benchmark extraction failed for %s: %s", sub_name, e)

    # Strategy 2: HF README fallback
    if hf_readme_content:
        prompt = HF_SUB_BENCHMARK_EXTRACTION_PROMPT.format(
            parent_name=parent_name,
            sub_name=sub_name,
            sub_label=sub_label,
            readme_content=hf_readme_content[:15000],
        )

        try:
            result = llm_handler.generate(f"{_EXTRACTOR_SYSTEM}\n\n{prompt}")

            if "NO SPECIFIC INFORMATION FOUND" not in result.upper():
                logger.info("Found sub-benchmark facts for %s (HF README): %d chars", sub_name, len(result))
                return result

            logger.info("No specific info in HF README for sub-benchmark %s", sub_name)
        except Exception as e:
            logger.warning("HF README sub-benchmark extraction failed for %s: %s", sub_name, e)

    return None


def generate_sub_benchmark_fallback_description(
    sub_name: str,
    parent_name: str,
    parent_overview: str,
) -> Optional[str]:
    """Generate a brief distinguishing description for a sub-benchmark as last resort.

    Used when neither paper nor HF README contain specific sub-benchmark info.
    Returns a 1-2 sentence description, or None if generation fails.
    """
    sub_label = _extract_sub_label(sub_name, parent_name)

    prompt = SUB_BENCHMARK_FALLBACK_PROMPT.format(
        parent_name=parent_name,
        parent_overview=parent_overview[:500],
        sub_name=sub_name,
        sub_label=sub_label,
    )

    try:
        from auto_benchmarkcard.config import get_llm_handler
        llm_handler = get_llm_handler()
        result = llm_handler.generate(prompt)
        # Clean up — take just the first 1-2 sentences
        sentences = result.strip().split(". ")
        description = ". ".join(sentences[:2])
        if not description.endswith("."):
            description += "."
        logger.info("Generated fallback description for %s: %s", sub_name, description[:100])
        return description
    except Exception as e:
        logger.warning("Fallback description generation failed for %s: %s", sub_name, e)
        return None


def compose_sub_benchmark_card(
    parent_card: Dict[str, Any],
    sub_facts: Optional[str],
    sub_name: str,
    eee_metadata: Optional[Dict[str, Any]] = None,
    fallback_description: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose a sub-benchmark card by inheriting from parent and specializing.

    If sub_facts is None (no specific info found), uses fallback_description
    to at least distinguish the overview. If neither is available, inherits verbatim.
    """
    import copy

    parent_bc = parent_card.get("benchmark_card", parent_card)
    card = copy.deepcopy(parent_bc)

    # Always override identity fields
    if "benchmark_details" not in card:
        card["benchmark_details"] = {}
    card["benchmark_details"]["name"] = sub_name
    card["benchmark_details"]["benchmark_type"] = map_benchmark_type("sub-benchmark")
    if eee_metadata:
        card["benchmark_details"]["appears_in"] = eee_metadata.get("appears_in", [])

    if sub_facts:
        # Use LLM to specialize the inherited card with sub-benchmark-specific facts
        _specialize_card_with_facts(card, sub_facts, sub_name)
    elif fallback_description:
        # No grounded facts found, but inject a distinguishing overview
        parent_name = parent_bc.get("benchmark_details", {}).get("name", "unknown")
        sub_label = _extract_sub_label(sub_name, parent_name)
        parent_overview = card["benchmark_details"].get("overview", "")
        card["benchmark_details"]["overview"] = (
            f"{fallback_description}\n\n"
            f"This is the \"{sub_label}\" sub-component of {parent_name}. "
            f"{parent_overview}"
        )

    card = post_process_card(card)

    return {
        "benchmark_card": card,
        "generation_metadata": {
            "generation_method": "sub_benchmark_inheritance",
            "parent_benchmark": parent_bc.get("benchmark_details", {}).get("name", "unknown"),
            "has_specific_facts": bool(sub_facts),
            "has_fallback_description": bool(fallback_description) and not bool(sub_facts),
            "timestamp": datetime.now().isoformat(),
        },
    }


def _specialize_card_with_facts(card: Dict[str, Any], sub_facts: str, sub_name: str):
    """Apply sub-benchmark-specific facts to inherited card fields."""
    # Sections that should NOT be overridden (always inherit from parent)
    _INHERIT_SECTIONS = {"ethical_and_legal_considerations"}

    prompt = f"""You are specializing a benchmark card for the sub-benchmark "{sub_name}".

Below is the PARENT benchmark card (inherited values) and FACTS specific to this sub-benchmark extracted from the paper.

For each section and field:
- If the sub-benchmark facts provide SPECIFIC information for that field, update the value to reflect the sub-benchmark specifics.
- If the facts say nothing specific about a field, keep the parent value UNCHANGED.
- Do NOT remove or empty any parent values — only add or refine.

PARENT CARD:
{json.dumps(card, indent=2)}

SUB-BENCHMARK FACTS:
{sub_facts}

Return the updated card as valid JSON. Keep the exact same structure as the parent card."""

    try:
        from auto_benchmarkcard.config import get_llm_handler
        llm_handler = get_llm_handler()
        result = llm_handler.generate(prompt)

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            specialized = json.loads(json_match.group())

            # Apply specializations but protect inherited sections
            for section_key, section_value in specialized.items():
                if section_key in _INHERIT_SECTIONS:
                    continue
                if section_key in card and isinstance(section_value, dict):
                    card[section_key].update(section_value)
                else:
                    card[section_key] = section_value
        else:
            logger.warning("Could not parse specialized card JSON for %s", sub_name)
    except Exception as e:
        logger.warning("Card specialization failed for %s: %s", sub_name, e)


def _get_hf_readme(hf_metadata: Optional[Dict[str, Any]]) -> str:
    """Get README text from HF metadata, handling nested dict layouts."""
    if not hf_metadata or not isinstance(hf_metadata, dict):
        return ""
    readme = hf_metadata.get("readme_markdown", "")
    if not readme:
        for v in hf_metadata.values():
            if isinstance(v, dict):
                readme = v.get("readme_markdown", "")
                if readme:
                    break
    return readme


def _get_hf_meta(hf_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Navigate to the HF metadata dict, which may be nested by dataset ID."""
    if not hf_metadata or not isinstance(hf_metadata, dict):
        return {}
    if "tags" in hf_metadata:
        return hf_metadata
    for v in hf_metadata.values():
        if isinstance(v, dict) and "tags" in v:
            return v
    return hf_metadata


# Canonical metric-name vocabulary (contract S4.5). The canonical name is the card VALUE;
# the raw EEE id stays in provenance only. Conservative: only well-known patterns map, and
# everything else keeps its raw form behind an `other:` tag -- we never guess.
_METRIC_SUBSTR_MAP = [
    ("exact_match", "Exact Match"),
    ("exactmatch", "Exact Match"),
    ("win_rate", "win-rate"),
    ("winrate", "win-rate"),
    ("rouge", "ROUGE"),
    ("bleu", "BLEU"),
    ("ndcg", "NDCG@k"),
    ("f1", "F1"),
    ("accuracy", "Accuracy"),
]

# Also canonicalize the "pass_at_1" / "pass at 1" spelling (an "at" between pass and k), not
# just "pass_1" / "pass@1", so both fold to pass@<k> instead of falling to other:pass_at_k.
_PASS_AT_K_RE = re.compile(r"pass[_@ ]*(?:at[_@ ]*)?(\d+)\b")

# Metric phrases stated in an EEE evaluation_description (only canonical tokens the gate
# recognises). Matched against the description, never the benchmark name.
_DESC_METRIC_PHRASES = [
    ("length-controlled win", "win-rate"),
    ("win-rate", "win-rate"),
    ("win rate", "win-rate"),
    ("auroc", "AUC"),
]
# Answer-correctness signal -> the score is an accuracy: multiple-choice / QA / explicit
# "accuracy" wording only, never bare "solve"/"questions" (those leak into open-ended tasks).
_ANSWER_CORRECTNESS_RE = re.compile(
    r"multiple[- ]choice|\bmcq\b|exactly correct|correct answer|answers? correctly"
    r"|\baccurac(?:y|ies)\b", re.I)
# Code / execution / completion markers. These OVERRIDE the answer-correctness signal: a
# code-run benchmark is pass-rate scored, so it must never be relabelled "Accuracy" (a silent
# wrong-but-canonical metric the gate cannot see). Keyed on task-shape words, not names.
_CODE_EXEC_RE = re.compile(
    r"\bcode\b|coding|program|exercism|exercise|function call|test case|test error"
    r"|compil|execut|software|unit test|repositor", re.I)


def _metric_from_cfg(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    """Derive a canonical metric from an EEE metric cfg's kind / description / unit, or None
    when the source states none. Emits only tokens the gate recognises (Accuracy, win-rate,
    pass@k, AUC). Conservative: a code/execution task never maps to Accuracy, and nothing is
    ever guessed from the benchmark name."""
    cfg = cfg if isinstance(cfg, dict) else {}
    kind = str(cfg.get("metric_kind", "")).strip().lower()
    low = str(cfg.get("evaluation_description", "")).lower()
    unit = str(cfg.get("metric_unit", "")).strip().lower()
    score_type = str(cfg.get("score_type", "")).strip().lower()

    # An explicit scorer kind names the metric directly.
    if kind == "pass_rate":
        k = _PASS_AT_K_RE.search(low)
        return f"pass@{int(k.group(1))}" if k else "pass@1"
    if kind == "win_rate":
        return "win-rate"
    if kind == "accuracy":
        return "Accuracy"

    # An explicit metric phrase stated in the description.
    k = _PASS_AT_K_RE.search(low)
    if k:
        return f"pass@{int(k.group(1))}"
    for needle, canon in _DESC_METRIC_PHRASES:
        if needle in low:
            return canon

    # A proportion/binary score whose description shows answer-correctness is an accuracy --
    # but never for a code/execution task (pass-rate scored; the guard wins).
    if unit in ("proportion", "binary") or score_type in ("proportion", "binary"):
        if _CODE_EXEC_RE.search(low):
            return None
        if _ANSWER_CORRECTNESS_RE.search(low):
            return "Accuracy"
    return None


def _canonicalize_metric(raw_id: Any, cfg: Optional[Dict[str, Any]] = None) -> str:
    """Map one raw EEE metric id onto the contract S4.5 canonical vocabulary.

    The id is tried first; when it names no metric the cfg's kind/description/unit is consulted
    (_metric_from_cfg) so a degenerate `*.score` proxy still yields the real metric the source
    states. A bare `*.score` whose score_type is proportion/binary resolves to "Accuracy".
    Unmapped ids keep their raw form as `other:<id>` (the noisy `llm_stats.` prefix is dropped in
    the display value only; the full raw id is preserved in provenance).
    """
    rid = str(raw_id)
    low = rid.lower()

    # An error / loss metric (lower_is_better) is never an accuracy. Suppress an Accuracy
    # label the id or cfg would otherwise yield so the metric can't contradict its own
    # direction; the raw id then survives as other:<id> (honest, demoted/cleaned downstream).
    lower_is_better = bool((cfg or {}).get("lower_is_better"))

    def _guard(canon: Optional[str]) -> Optional[str]:
        return None if (canon == "Accuracy" and lower_is_better) else canon

    m = _PASS_AT_K_RE.search(low)
    if m:
        return f"pass@{int(m.group(1))}"

    for needle, canon in _METRIC_SUBSTR_MAP:
        if needle in low:
            guarded = _guard(canon)
            if guarded:
                return guarded
            break

    from_cfg = _guard(_metric_from_cfg(cfg))
    if from_cfg:
        return from_cfg

    if low == "score" or low.endswith(".score") or low.endswith("_score"):
        score_type = str((cfg or {}).get("score_type", "")).strip().lower()
        if score_type in ("proportion", "binary") and not lower_is_better:
            return "Accuracy"

    display = rid[len("llm_stats."):] if low.startswith("llm_stats.") else rid
    return f"other:{display}"


def canonicalize_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Canonicalize an EEE metrics dict (id -> config) to S4.5 names, order-preserving.

    No dedup: distinct raw ids stay distinct entries even when two map to the same canonical
    name (they are different EEE metrics). Raw ids are carried in provenance, not here.
    """
    return [
        _canonicalize_metric(rid, cfg if isinstance(cfg, dict) else {})
        for rid, cfg in metrics.items()
    ]


def _is_degenerate_metric(raw_id: Any, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True for the llm_stats aggregate-wrapper metric that masks the real scorer.

    The clean structural key is metric_kind == "benchmark_score" (an llm-stats proxy, not a
    real scorer). The <slug>.score / llm_stats.<slug>.score raw shape only corroborates it --
    a legitimate scorer can also end in `.score`, so the shape is never the sole trigger.
    """
    return str((cfg or {}).get("metric_kind", "")).strip().lower() == "benchmark_score"


# Pipeline furniture that must never reach prose: internal rollout-card / Ergon names. These are
# LLM hallucinations (absent from the codebase), so only a literal backstop can catch them.
_PIPELINE_JARGON_RE = re.compile(r"ergon-native|rollout-card|sharded rollout", re.I)
# A stray metric-namespace prefix that survived id-replacement.
_LEAK_PREFIX_RE = re.compile(r"\b(?:llm_stats|openeval)\.", re.I)
# The internal aggregation-source label "Every Eval Ever" leaks into user-facing prose when the
# Stage-B EEE prefill block (build_eee_prefill_block) is echoed into baseline_results. Strip the
# phrase as a substring -- keeping the surrounding model scores -- with a leading "from"/"From"
# and a flanking comma. Only the full phrase is targeted, never a bare "EEE" (would hit "IEEE").
_SOURCE_LABEL_RE = re.compile(r"\s*(?:,\s*)?(?:from\s+)?Every Eval Ever(?:\s*,)?", re.I)


def _scrub_pipeline_jargon(text: str) -> str:
    """Strip pipeline furniture from prose. A sentence carrying furniture jargon is dropped whole
    (so the result stays grammatical); a stray llm_stats./openeval. namespace prefix and the
    "Every Eval Ever" source-label phrase are removed in place. Idempotent; clean prose untouched."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    if _PIPELINE_JARGON_RE.search(out):
        kept = [s for s in re.split(r"(?<=[.!?])\s+", out) if not _PIPELINE_JARGON_RE.search(s)]
        joined = " ".join(kept).strip()
        # never blank the field outright: if every sentence carried jargon, drop just the tokens
        out = joined if joined else _PIPELINE_JARGON_RE.sub("", out)
    out = _LEAK_PREFIX_RE.sub("", out)
    if _SOURCE_LABEL_RE.search(out):
        out = _SOURCE_LABEL_RE.sub(" ", out)
        # the strip can leave a dangling " ," / " :" before punctuation -- close it up.
        out = re.sub(r"\s+([,;:.])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


# M3: V4-Flash generation artifacts. Keyed on the artifact TOKEN PATTERN, never the card; each
# pattern requires a marker so clean prose is untouched, and each repair is idempotent.

# Pattern 1: a stray control-word suffix glued mid-prose ('years)Skip. The' from a sampler/decoder
# control token that leaked into text). The control word is one of a small closed set; an optional
# ')' that got glued before it is dropped with it. Repair: drop ')Skip.' / 'Skip.', keeping the
# sentence boundary ('years)Skip. The' -> 'years. The').
_CONTROL_WORD_SUFFIX_RE = re.compile(
    r"\)?(?:Skip|Continue|Stop|End|Pause)\.(?=\s+[A-Z]|\s*$)")

# Pattern 3: a field-final clause ending on a dangling conjunction/preposition with no terminal
# punctuation ('...However, it is deficient if'). The clause is incomplete -> trim to the last
# complete sentence. A clause that DOES end on terminal punctuation ('...deficient if X.') is kept.
_DANGLING_TAIL_RE = re.compile(
    r"\b(?:if|and|but|or|because|when|while|which|that|with|for|to|of|in|on|as)\s*$", re.I)

# Pattern 2: a known stem glued to a short non-word fragment ('problemhol' = 'problem' + 'hol').
# A small closed set of common stems is enough to catch the V4-Flash glue without a wordlist; the
# fragment is 2-4 trailing letters that leave no real compound. Detected and FLAGGED, never repaired
# (auto-repair without a dictionary risks mangling a legitimate compound).
_GLUED_STEMS = ("problem", "question", "example", "sample", "answer", "dataset", "benchmark",
                "instance", "response", "prompt", "task")
_GLUED_WORD_RE = re.compile(
    r"\b(" + "|".join(_GLUED_STEMS) + r")([a-z]{2,4})\b(?=[,.\s])", re.I)
# A trailing fragment that is a real English derivational/inflectional suffix means the token is a
# legitimate word (problematic, questionable, prompted, sampled), not an artifact. Precision guard:
# only a fragment OUTSIDE this set ('hol') is flagged, so clean derived prose is never touched.
_LEGIT_WORD_SUFFIXES = frozenset({
    "ed", "ing", "er", "ers", "es", "ly", "al", "ic", "atic", "able", "ible", "ive", "ity",
    "ous", "ant", "ent", "ary", "ory", "ful", "less", "ness", "ment", "ation", "ize", "ise",
    "ist", "ism", "ide", "ine", "ate", "ling", "lly",
})


def _scrub_generation_artifacts(text: str) -> str:
    """Auto-repair two V4-Flash artifact classes in a prose string: a stray control-word suffix
    glued mid-prose (pattern 1) and a field-final dangling-conjunction truncation (pattern 3). Each
    repair fires only on its marker, leaves clean prose untouched, and is idempotent. The glued-word
    fragment (pattern 2) is NOT repaired here -- it is flagged via _glued_word_fragments."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    if _CONTROL_WORD_SUFFIX_RE.search(out):
        out = _CONTROL_WORD_SUFFIX_RE.sub(".", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
    stripped = out.rstrip()
    if stripped and stripped[-1] not in ".!?:;" and _DANGLING_TAIL_RE.search(stripped):
        sentences = _SENTENCE_SPLIT_RE.split(stripped)
        if len(sentences) > 1:
            kept = " ".join(sentences[:-1]).strip()
            if kept:
                out = kept
    return out


def _glued_word_fragments(text: Any) -> List[str]:
    """The glued-word artifacts in a prose string (pattern 2), e.g. ['problemhol']. A known stem
    fused to a 2-4 char trailing fragment with no separating space. Empty when none -- clean prose
    yields []. Flagged, not repaired."""
    if not isinstance(text, str) or not text:
        return []
    out: List[str] = []
    for m in _GLUED_WORD_RE.finditer(text):
        if m.group(2).lower() in _LEGIT_WORD_SUFFIXES:
            continue  # a real derived word (problematic / prompted), not an artifact
        token = m.group(0)
        if token.lower() not in {o.lower() for o in out}:
            out.append(token)
    return out


# Prose fields the artifact scrub covers, beyond the metric-id prose fields: data.source and the
# out_of_scope_uses list (where the pilot artifacts landed).
_ARTIFACT_SCRUB_FIELDS = [
    ("benchmark_details", "overview"),
    ("methodology", "calculation"), ("methodology", "methods"),
    ("methodology", "interpretation"), ("methodology", "baseline_results"),
    ("methodology", "validation"),
    ("data", "source"),
    ("purpose_and_intended_users", "out_of_scope_uses"),
]


def _apply_generation_artifact_scrub(card: Dict[str, Any]) -> None:
    """Auto-repair control-word and truncation artifacts across the prose fields, and FLAG (never
    repair) glued-word fragments. Mutates card in place. String and list-of-string fields are both
    handled; flags land in flagged_fields keyed by the dotted field path."""
    flags = card.setdefault("flagged_fields", {})
    for sec, field in _ARTIFACT_SCRUB_FIELDS:
        section = card.get(sec)
        if not isinstance(section, dict) or field not in section:
            continue
        value = section[field]
        if isinstance(value, str):
            repaired = _scrub_generation_artifacts(value)
            if repaired != value:
                section[field] = repaired
            if _glued_word_fragments(section[field]):
                flags[f"{sec}.{field}"] = "[Probable generation artifact] glued/garbled token"
        elif isinstance(value, list):
            new_list = []
            glued_any = False
            for item in value:
                if isinstance(item, str):
                    item = _scrub_generation_artifacts(item)
                    if _glued_word_fragments(item):
                        glued_any = True
                new_list.append(item)
            section[field] = new_list
            if glued_any:
                flags[f"{sec}.{field}"] = "[Probable generation artifact] glued/garbled token"


def _scrub_leaked_metric_ids(card: Dict[str, Any],
                             eee_metadata: Optional[Dict[str, Any]]) -> None:
    """Rewrite any raw EEE metric id the model echoed into prose, and strip pipeline jargon.

    A degenerate benchmark_score wrapper id becomes its human metric_name (or is dropped) --
    never its other:<id> canonical, which would re-inject the very proxy F5-1 suppresses. A
    real scorer's leaked id becomes its canonical display form. The pipeline-jargon backstop
    runs regardless of whether EEE carries metrics. Idempotent; leaves clean prose untouched."""
    metrics = (eee_metadata or {}).get("metrics", {})
    repl: Dict[str, str] = {}
    if isinstance(metrics, dict):
        for rid, cfg in metrics.items():
            cfg = cfg if isinstance(cfg, dict) else {}
            rid = str(rid)
            # Only a leaked RAW id (namespaced / dotted / underscored) is scrubbed from prose. A
            # bare word or benchmark name that doubles as a metric id -- "accuracy", "score",
            # "GPQA", "BoolQ" -- is left alone so the scrub never rewrites ordinary prose (e.g.
            # "65% accuracy" or a composite overview's sub-benchmark list).
            if "." not in rid and "_" not in rid:
                continue
            if _is_degenerate_metric(rid, cfg):
                repl[rid] = str(cfg.get("metric_name", "")).strip()
            else:
                canon = _canonicalize_metric(rid, cfg)
                # And only rewrite to a REAL canonical -- never to an other:<id> form, which is
                # uglier than the original token.
                if not canon.startswith("other:"):
                    repl[rid] = canon
    # Longest ids first so a short id never matches inside a longer one.
    rids = sorted(repl, key=len, reverse=True)
    prose_fields = [
        ("methodology", "calculation"), ("methodology", "methods"),
        ("methodology", "interpretation"), ("methodology", "baseline_results"),
        ("methodology", "validation"), ("benchmark_details", "overview"),
    ]
    for sec, field in prose_fields:
        s = card.get(sec)
        if not isinstance(s, dict) or not isinstance(s.get(field), str):
            continue
        new = s[field]
        for rid in rids:
            if rid not in new:
                continue
            if repl[rid]:
                new = new.replace(rid, repl[rid])
            else:
                new = re.sub(r"\s*\(?" + re.escape(rid) + r"\)?", "", new)
                new = re.sub(r"\s{2,}", " ", new).strip()
        new = _scrub_pipeline_jargon(new)
        if new != s[field]:
            s[field] = new


# A STATED-score token in baseline prose: a number that carries SCORE CONTEXT, i.e. a trailing '%'
# (group 2) OR a decimal value (group 1 has a '.'). A bare integer or a name-glued version digit
# ("Claude Fable 5", "GPT-5", "Gemini 3.1" -> 5 / 5 / 3.1) is excluded by the score-context test in
# _stated_score_numbers, so a model-name fragment is never read as a score. Commas are thousands-only.
_STATED_SCORE_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*%)?")
# A four-digit year (1900-2099) is not a score -- excluded so "in 2024" never reads as a baseline.
_YEAR_LIKE_RE = re.compile(r"^(?:19|20)\d\d$")
# Baseline-prose sentence splitter (distinct name -- the module already has a loose _SENTENCE_SPLIT_RE
# at module scope; this one additionally requires the next sentence to start with a capital/quote/paren
# so a decimal like 91.50 is never split mid-number). Kept local to the baseline guard.
_BASELINE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
# A genuine reformat (0.4140 -> 41.40%, or 0.7184 -> 71.84) is EXACT after scale-normalizing, so the
# cross-scale match uses a TIGHT tolerance -- the larger of a tiny absolute floor (rounding noise, e.g.
# 0.8874 reported as "88.70%"=0.887) and 0.5% of the magnitude. A loose band (e.g. 0.05 at the fraction
# scale = 5 percentage points) would silently "ground" a fabricated 91.50% against a real 88.7%.
_SCORE_MATCH_ABS_EPS = 0.005
_SCORE_MATCH_REL_EPS = 0.005


def _stated_score_numbers(text: str) -> List[float]:
    """Every STATED score in baseline prose: a number that has SCORE CONTEXT -- a trailing '%' or a
    decimal point. A bare integer ("5", "200") or a name-glued version digit ("Fable 5", "GPT-5",
    "Gemini 3.1") is NOT a score, so a model-name fragment never enters the comparison. A trailing-'%'
    value is normalized to its fraction (91.50% -> 0.9150) so both sides compare on one scale; a plain
    decimal is taken as-is. Four-digit years are excluded. Returns [] for prose with no stated score."""
    out: List[float] = []
    if not isinstance(text, str) or not text:
        return out
    for m in _STATED_SCORE_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        has_pct = m.group(2) is not None
        if _YEAR_LIKE_RE.match(raw):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if has_pct:
            out.append(val / 100.0)                    # "91.50%" -> 0.9150 (percent has score context)
            continue
        # No '%': a number is a stated score ONLY if it is a fraction in [0, 1] (a decimal score like
        # 0.4140). A bare integer ("5", "200") or a name-glued version decimal ("Gemini 3.1" -> 3.1,
        # >1) carries no score context and is NOT a score -- so a model-name fragment never compares.
        if "." in raw and 0.0 <= val <= 1.0:
            out.append(val)
    return out


def _grounded_score_pool(evidence_items: Optional[List[Dict[str, Any]]],
                         det_facts: Optional[Dict[str, Any]]) -> List[float]:
    """The score VALUES the composer was actually given, as structured numbers -- never re-parsed from
    rendered model:score text (which would extract version digits out of model names). Sources:
      - EEE evaluation_summary: top_performers[].score, score_statistics {mean,std_dev,min,max};
      - explicit STATED scores (percent/decimal, score-context) inside verified evidence-item quotes.
    A baseline number absent from this pool was not in any source the composer saw. Structural; the
    pool can never contain a model-name digit because it reads numeric score fields, not names."""
    pool: List[float] = []
    es = (det_facts or {}).get("evaluation_summary") if isinstance(det_facts, dict) else None
    if isinstance(es, dict):
        for p in es.get("top_performers", []) or []:
            if isinstance(p, dict):
                sc = p.get("score")
                if isinstance(sc, (int, float)) and not isinstance(sc, bool):
                    pool.append(float(sc))
        stats = es.get("score_statistics")
        if isinstance(stats, dict):
            for _k in ("mean", "std_dev", "min", "max"):
                sv = stats.get(_k)
                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                    pool.append(float(sv))
    for it in (evidence_items or []):
        if isinstance(it, dict):
            pool.extend(_stated_score_numbers(str(it.get("quote", "") or "")))
    return pool


def _score_is_grounded(n: float, grounded: List[float]) -> bool:
    """True when a stated score n (already normalized: a '%' value is its fraction) matches some
    grounded score g at a common scale -- as-is, or one is the other scaled by 100 (a percent vs a
    fraction reformat, e.g. n=0.9150 vs g=91.50, or n=0.7184 vs g=71.84). The match is EXACT to a tiny
    absolute epsilon after normalizing, so a genuine reformat passes while a fabricated value absent at
    every scale fails. Structural number-matching, never a model/benchmark name."""
    for g in grounded:
        for candidate in (g, g * 100.0, g / 100.0):
            if abs(n - candidate) <= max(_SCORE_MATCH_ABS_EPS, abs(candidate) * _SCORE_MATCH_REL_EPS):
                return True
    return False


def _drop_ungrounded_baselines(card: Dict[str, Any],
                               evidence_items: Optional[List[Dict[str, Any]]],
                               det_facts: Optional[Dict[str, Any]]) -> None:
    """Drop a fabricated baseline leaderboard from methodology.baseline_results: any sentence whose
    STATED scores are NONE of them traceable to a score the composer was given (the EEE evaluation
    summary's structured scores + explicit scores in the verified evidence quotes). Mutates card in
    place and FLAGS the change (descriptive, never silent). Structural -- keys on 'is this score
    present in a source', never on a model-name list.

      - Granularity is the SENTENCE, so re-joined prose stays grammatical.
      - 'Score' means a number with score context (trailing '%' or a decimal); a bare integer or a
        name-glued version digit ('Fable 5', 'GPT-5', 'Gemini 3.1') is NOT a score and never grounds.
      - A sentence with NO stated score is qualitative context -> kept untouched.
      - A MIXED sentence (some scores grounded, some not) is ambiguous -> KEPT + flagged, never dropped
        (a real baseline must not be lost to a co-located stray number).
      - Only a sentence whose every stated score is ungrounded is dropped.
      - Surviving grounded + qualitative sentences are KEPT (whole sentences -> grammatical, no mangled
        remnant); the WHOLE field is set to 'Not specified' only when NO grounded sentence survives a
        real drop (e.g. mmlu-pro's all-fabricated leaderboard -> whole-field NS)."""
    meth = card.get("methodology")
    if not isinstance(meth, dict):
        return
    text = meth.get("baseline_results")
    if not isinstance(text, str) or not text.strip() or _is_ns_str(text):
        return

    grounded = _grounded_score_pool(evidence_items, det_facts)
    sentences = [s for s in _BASELINE_SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return

    kept: List[str] = []
    dropped = False
    mixed = False
    any_grounded_kept = False
    for s in sentences:
        nums = _stated_score_numbers(s)
        if not nums:
            kept.append(s)                                   # qualitative -> keep (no score to judge)
            continue
        grounded_hits = [n for n in nums if _score_is_grounded(n, grounded)]
        if not grounded_hits:
            dropped = True                                   # every stated score ungrounded -> drop
            continue
        if len(grounded_hits) < len(nums):
            mixed = True                                     # mixed -> keep, but flag
        any_grounded_kept = True
        kept.append(s)

    if not dropped and not mixed:
        return                                               # nothing to change

    flags = card.setdefault("flagged_fields", {})
    surviving = " ".join(kept).strip()
    # Whole-field NS ONLY when a real drop left no grounded sentence behind (nothing trustworthy to
    # keep). Otherwise keep the surviving whole sentences (grammatical) -- never over-drop a real
    # baseline just because the fabricated sentences were the majority.
    if dropped and (not any_grounded_kept or not surviving):
        meth["baseline_results"] = "Not specified"
        flags["methodology.baseline_results"] = (
            "[Ungrounded baseline] all reported scores not traceable to a retrieved source; "
            "field set to Not specified")
    elif dropped:
        meth["baseline_results"] = surviving
        flags["methodology.baseline_results"] = (
            "[Ungrounded baseline] dropped score claim(s) not traceable to a retrieved source")
    elif mixed:
        flags["methodology.baseline_results"] = (
            "[Ungrounded baseline] one or more reported scores not traceable to a retrieved source")


def build_metric_labels(eee_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map each EEE metric to its human label {token_or_raw_id: {name, unit, raw_id}}, keyed
    by both the canonical token (via the unchanged _canonicalize_metric) and the raw id. For
    the provenance sidecar only -- the card's methodology.metrics value is never changed."""
    labels: Dict[str, Any] = {}
    metrics = eee_metadata.get("metrics") if isinstance(eee_metadata, dict) else None
    if not isinstance(metrics, dict):
        return labels
    for rid, cfg in metrics.items():
        if not isinstance(cfg, dict):
            continue
        info = {"name": cfg.get("metric_name"), "unit": cfg.get("metric_unit"), "raw_id": rid}
        labels[_canonicalize_metric(rid, cfg)] = info
        labels[rid] = info
    return labels


# Controlled metric vocabulary for the prose->structured promotion (S4.5 names the gate
# recognises). A term is promoted only when it is NAMED in a metric-definition context, so an
# incidental mention never invents a metric. Keyed on the vocabulary, never on a benchmark name.
# mAP is case-sensitive ("\bmAP\b") so it never fires on the common word "map".
_PROSE_METRIC_VOCAB = [
    (re.compile(r"length[- ]controlled\s+win[- ]?rate", re.I), "length-controlled win-rate"),
    (re.compile(r"win[- ]?rate", re.I), "win-rate"),
    (re.compile(r"\bauroc\b|\bauc\b|area under the (?:receiver operating characteristic|roc)", re.I), "AUC"),
    (re.compile(r"\bf1(?:[- ]score)?\b", re.I), "F1"),
    (re.compile(r"success[- ]rate", re.I), "success-rate"),
    (re.compile(r"\baccurac(?:y|ies)\b", re.I), "Accuracy"),
    (re.compile(r"\brecall\b", re.I), "recall"),
    (re.compile(r"mean average precision"), "mAP"),
    (re.compile(r"\bmAP\b"), "mAP"),
    (re.compile(r"\brouge\b", re.I), "ROUGE"),
    (re.compile(r"\bbleu\b", re.I), "BLEU"),
    (re.compile(r"exact[- ]match", re.I), "Exact Match"),
    (re.compile(r"\bndcg\b", re.I), "NDCG@k"),
]
_METRIC_DEF_CUE_RE = re.compile(
    r"calculat|comput|defin|measur|evaluat|report|present|score|metric|using|employ"
    r"|\buse[ds]?\b|follows|primary metric", re.I)


def _metric_from_prose(*texts: str) -> List[str]:
    """Canonical metric tokens NAMED in methodology prose, in order, de-duped. A controlled-vocab
    term counts only when its sentence also carries a metric-definition cue (so an incidental
    mention is ignored). pass@k is read first (it is the most specific). Returns [] when the prose
    names no recognised metric -- never guesses, never reads a benchmark name."""
    found: List[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if not _METRIC_DEF_CUE_RE.search(sentence):
                continue
            k = _PASS_AT_K_RE.search(sentence.lower())
            if k:
                tok = f"pass@{int(k.group(1))}"
                if tok not in found:
                    found.append(tok)
            for rx, canon in _PROSE_METRIC_VOCAB:
                if rx.search(sentence) and canon not in found:
                    found.append(canon)
    return found


def _deterministic_field_provenance(
    dotted: str,
    doc: str,
    card: Dict[str, Any],
    det_facts: Dict[str, Any],
    eee_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the source=deterministic provenance entry for one deterministically-filled
    field, or None when the deterministic write did not win (card value != det fact).

    The emission decision is gated solely on `card_value == det_facts[dotted]` (contract S10
    no-op invariant) -- unchanged by metric canonicalization. For methodology.metrics the
    stored quote/evidence is the RAW EEE ids (S4.5: canonical names are the card value; raw
    ids live in provenance only); every other field quotes its own structured value.
    """
    if dotted not in det_facts:
        return None
    sec, _, field = dotted.partition(".")
    section = card.get(sec)
    if not isinstance(section, dict):
        return None
    value = section.get(field)
    if value is None or value != det_facts[dotted]:
        return None
    quote = value if isinstance(value, str) else ", ".join(str(v) for v in value)
    if dotted == "methodology.metrics" and isinstance(eee_metadata, dict):
        raw_ids = list((eee_metadata.get("metrics") or {}).keys())
        if raw_ids:
            quote = ", ".join(str(r) for r in raw_ids)
    return {
        "source": "deterministic",
        "evidence": quote,
        "quote": quote,
        "quote_loc": {"doc": doc, "char_start": None},
        "status": "stated",
        "verified": True,
        "evidence_ids": [],
    }


_DATA_EXT_FORMAT = {
    ".parquet": "parquet", ".csv": "csv", ".json": "json",
    ".jsonl": "jsonl", ".tsv": "tsv", ".arrow": "arrow",
}
_INT_CELL_RE = re.compile(r"^[\d,]+$")


def _format_from_files(file_list: Any) -> Optional[str]:
    """Derive data.format from the dominant data-file extension when no HF format: tag
    exists. Structural and general; honest-None when the repo has no data files. The
    extension is a storage token, so it is labelled as the HuggingFace hosting format."""
    if not isinstance(file_list, list):
        return None
    counts: Counter = Counter()
    for f in file_list:
        if isinstance(f, str):
            fmt = _DATA_EXT_FORMAT.get(os.path.splitext(f)[1].lower())
            if fmt:
                counts[fmt] += 1
    return _label_hosting_format(counts.most_common(1)[0][0]) if counts else None


_YEAR_RE = re.compile(r"(19|20)\d\d")


def _has_year(value: Any) -> bool:
    """True when a string contains a plausible 4-digit year. Used to reject non-year prose
    as a collection_date."""
    return bool(isinstance(value, str) and _YEAR_RE.search(value))


_RELATIVE_DATE_RE = re.compile(
    r"\b(?:today|now|current(?:ly)?|present|recently|yesterday|tomorrow)\b", re.I)


def _coerce_collection_date(value: Any) -> Any:
    """A collection_date must be a year, never prose: a set, non-NS string is dropped to Not
    specified when it carries a relative-time reference (today/now/current -- a pipeline-relative
    date, not a real one) or has no 4-digit year. Any other value is returned unchanged."""
    if isinstance(value, str) and value.strip() and value != "Not specified":
        if _RELATIVE_DATE_RE.search(value) or not _has_year(value):
            return "Not specified"
    return value


_DOCLING_RUNON_RE = re.compile(r"([.!?;:,)\]])([A-Z][a-z])")


def _normalize_docling_text(text: Any) -> str:
    """Conservative pre-clean for docling markdown run-on artifacts: insert a missing space
    after sentence/clause punctuation that abuts a capitalized word (e.g. ')Skip' -> ') Skip').
    Idempotent. Applied identically to the model-read text AND the quote-verify source, so the
    exact-substring grounding check is preserved."""
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    return _DOCLING_RUNON_RE.sub(r"\1 \2", text)


def _is_ns_str(value: Any) -> bool:
    """True for an empty/missing/Not-specified string value."""
    return (not isinstance(value, str)) or (not value.strip()) or value.strip() == "Not specified"


def _dedup_preserve_order(items: List[Any]) -> List[Any]:
    """Drop exact-duplicate strings from a list, preserving order. Non-string items are kept
    as-is (compared by repr)."""
    seen: set = set()
    out: List[Any] = []
    for it in items:
        key = it if isinstance(it, str) else repr(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _size_breakdown_from_readme(markdown: Any) -> Optional[Dict[str, int]]:
    """Parse {subset_or_domain: count} from the first README markdown table that pairs a
    text label with a single integer count. Conservative: integer counts only (so model
    score tables with decimals are ignored); needs >= 2 rows. A row carrying an explicit
    '%' in any cell is skipped -- those are proportions, never stored as counts.
    Honest-None otherwise."""
    if not isinstance(markdown, str) or "|" not in markdown:
        return None
    out: Dict[str, int] = {}
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or _INT_CELL_RE.match(cells[0]):
            continue
        # Percentage rows are proportions, not counts: an explicit '%' anywhere in the row
        # (label or value) disqualifies it. No sum-to-100 inference (it false-rejects real
        # count splits such as train:60 / test:40).
        if any("%" in c for c in cells):
            continue
        label, count = cells[0], None
        for c in cells[1:]:
            cc = c.replace(",", "")
            if _INT_CELL_RE.match(c) and cc.isdigit():
                count = int(cc)
                break
        if label and count is not None:
            out[label] = count
    return out if len(out) >= 2 else None


_SPLIT_NAME_RE = re.compile(r"^(train|test|validation|valid|dev|eval|evaluation)$", re.I)


def _is_split_name(label: Any) -> bool:
    """True when a breakdown key is a dataset split name (train/test/validation/dev/eval)."""
    return bool(isinstance(label, str) and _SPLIT_NAME_RE.match(label.strip()))


def _size_count(size: Any) -> Optional[int]:
    """A precise integer count from data.size, e.g. '805 instruction-following tasks' -> 805.
    None for an HF size-category bucket (a range like '1K<n<10K') or any non-count string."""
    if isinstance(size, int) and not isinstance(size, bool):
        return size
    if not isinstance(size, str):
        return None
    s = size.strip()
    if any(ch in s for ch in "<>≤≥"):  # a bucket range, not a count
        return None
    m = re.match(r"^([\d,]+)\b", s)
    if m:
        digits = m.group(1).replace(",", "")
        if digits.isdigit():
            return int(digits)
    return None


def _size_breakdown_is_consistent(
    breakdown: Optional[Dict[str, int]], total_examples: Any, size: Any
) -> bool:
    """Whether a parsed size_breakdown may be emitted.

    Only a split-named breakdown (keys train/test/validation/dev/eval) is a summable total,
    so it must agree with the dataset's example count -- the HF dataset_info split sum
    (total_examples), else a clean integer from data.size -- within 1%; a mismatched or
    train-only split breakdown is dropped. Heterogeneous statistic breakdowns (keys like
    '# Tasks', '# Domains', 'libraries', 'combinations') are not summable totals and are
    always kept. When no authoritative total is available the breakdown cannot be disproven
    and is kept."""
    if not isinstance(breakdown, dict) or not breakdown:
        return False
    if not all(_is_split_name(k) for k in breakdown):
        return True  # heterogeneous statistics, not a summable split total
    total = total_examples if (isinstance(total_examples, int)
                               and not isinstance(total_examples, bool)) else _size_count(size)
    if not isinstance(total, int) or total <= 0:
        return True  # cannot disprove without an authoritative total
    split_total = sum(v for v in breakdown.values() if isinstance(v, int) and not isinstance(v, bool))
    return abs(split_total - total) <= max(1, total // 100)


# C2: numeric internal-consistency guard. Never ship a size / size_breakdown that the card's own
# prose contradicts -- abstain (NS / drop) instead. All signals are structural, never a name.

# Inverse of the frozen card_utils _SIZE_CATEGORY_MAP (standard HF size_categories vocabulary).
_SIZE_BUCKET_RANGES = {
    "Less than 1K examples": (0, 1000),
    "1K to 10K examples": (1000, 10000),
    "10K to 100K examples": (10000, 100000),
    "100K to 1M examples": (100000, 1000000),
    "1M to 10M examples": (1000000, 10000000),
    "10M to 100M examples": (10000000, 100000000),
    "100M to 1B examples": (100000000, 1000000000),
    "1B to 10B examples": (1000000000, 10000000000),
    "10B to 100B examples": (10000000000, 100000000000),
    "100B to 1T examples": (100000000000, 1000000000000),
}


def _size_bucket_range(size: Any) -> Optional[tuple]:
    """The (low, high) example range of a coarse HF size bucket string, or None when size is not
    a bucket (an explicit count string or NS)."""
    return _SIZE_BUCKET_RANGES.get(size) if isinstance(size, str) else None


def _size_bucket_unit(size: Any) -> Optional[str]:
    """The trailing unit word of a coarse size bucket string ('Less than 1K examples' -> 'examples'),
    so a prose-magnitude counted noun that matches the bucket's own unit is accepted even when it is
    not in the generic content-noun list. None when size is not a known bucket."""
    if not isinstance(size, str) or size not in _SIZE_BUCKET_RANGES:
        return None
    words = size.strip().split()
    return words[-1] if words else None


# Example-level nouns only, so a structural count ("8 primary tasks", "11 harm categories",
# "139 libraries") is never mistaken for the dataset's example count.
# The number must not be glued to a preceding identifier (e.g. the '0613' in 'GPT-4-0613'); it is a
# plain integer or thousands-grouped, with at most two filler words before an example-level noun.
# Reliable eval-example nouns only -- corpus/ambiguous nouns (documents/samples/images/videos/
# sentences) are excluded so a retrieval corpus ("100K web documents") is not read as the size.
_EXAMPLE_NOUN_RE = re.compile(
    r"(?<![\w-])(\d{1,3}(?:,\d{3})+|\d+)\s+(?:[A-Za-z][\w-]*\s+){0,2}?"
    r"(?:examples?|questions?|instances?|prompts?|trios?|pairs?|problems?|queries|"
    r"dialogues?|conversations?|episodes?|utterances?)\b", re.I)
# A measure/time unit right after the matched noun means the noun is adjectival ("849 video hours",
# "30 image frames") -- a magnitude, not an example count.
_MEASURE_AFTER_RE = re.compile(
    r"\s+(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|frames?|tokens?|words?|chars?|characters?"
    r"|bytes?|[kmgt]b|hours|years?)\b", re.I)


def _prose_example_count(text: Any) -> Optional[int]:
    """The largest explicit example-level count stated in prose, e.g. '2,294 ... problems' -> 2294.
    Example-level nouns only; a 'per <unit>' rate or a noun used adjectivally before a measure unit
    ('849 video hours') is skipped. None when no such count is stated."""
    if not isinstance(text, str) or not text:
        return None
    counts: List[int] = []
    for m in _EXAMPLE_NOUN_RE.finditer(text):
        rest = text[m.end():]
        if rest[:6].lower().startswith(" per ") or _MEASURE_AFTER_RE.match(rest):
            continue
        digits = m.group(1).replace(",", "")
        if digits.isdigit():
            counts.append(int(digits))
    return max(counts) if counts else None


# A breakdown key that names a statistic (not a data subset/split) -> not a summable partition.
_HETERO_KEY_RE = re.compile(
    r"^#|number of|\bdomain|\blibrar|\bcombo|\bfunction|\bcall\b|standard|3rd party"
    r"|\bmodul|\bclass(?:es)?\b|categor", re.I)


def _is_heterogeneous_stat_key(key: Any) -> bool:
    """True for a breakdown key like '# Tasks' / '# Domains' / 'Library Combo' -- a statistic, not
    a data subset, so such a breakdown is never summed against the example total."""
    return bool(isinstance(key, str) and _HETERO_KEY_RE.search(key.strip()))


_ARCHIVE_EXT = {".tar", ".gz", ".tgz", ".zip", ".7z", ".rar", ".bz2", ".xz"}
# HF-native row formats: their presence means the size_categories tag counts examples, not shards.
_ROW_DATA_EXT = {".parquet", ".arrow"}
_SPLIT_ARCHIVE_RE = re.compile(r"\.\d{2,}$")


def _hf_size_is_shard_count(meta: Optional[Dict[str, Any]]) -> bool:
    """True when an HF size_categories bucket reflects shard/archive count, not an example count:
    no authoritative dataset_info row count, NO HF-native row files (parquet/arrow), and the repo
    is hosted as archives (.tar/.gz/.zip or split-archive parts .00/.01/...). Then the size tag is
    unreliable -> abstain. A parquet/arrow- or dataset_info-backed dataset (e.g. blink, 28 parquet)
    stays trusted; an archive-hosted media corpus (e.g. activitynet) does not. Structural, no name."""
    if not isinstance(meta, dict):
        return False
    di = meta.get("dataset_info")
    if isinstance(di, dict) and di.get("splits"):
        return False
    fl = meta.get("file_list")
    if not isinstance(fl, list):
        return False
    exts = [os.path.splitext(f)[1].lower() for f in fl if isinstance(f, str)]
    if any(e in _ROW_DATA_EXT for e in exts):
        return False
    return any(e in _ARCHIVE_EXT or _SPLIT_ARCHIVE_RE.match(e) for e in exts)


# M2 Rule A2: a size_breakdown can encode a MULTIPLICATIVE total (203 classes x 137 videos/class).
# A key carrying a per-token ("per", "average ... per") names one multiplicand and its partner the
# other, so their product is the real example count -- which can disprove a too-small bucket. Keyed
# on the per-token + integer structure, never the noun (so "videos per class" generalizes).
_PER_KEY_RE = re.compile(r"\bper\b|\baverage\b.*\bper\b|/", re.I)


def _breakdown_product(breakdown: Dict[str, Any]) -> Optional[int]:
    """The product of a per-structured size_breakdown's two integer multiplicands, e.g. {'203
    classes': 203, 'avg videos per class': 137} -> 27811. Requires exactly one key carrying a
    per-token AND a partner integer key; total/average-rate keys (value 1 or named total/hours) are
    skipped so only the two real multiplicands are multiplied. None when the structure is absent."""
    if not isinstance(breakdown, dict):
        return None

    def _usable(k: Any, v: Any) -> bool:
        if not (isinstance(v, int) and not isinstance(v, bool)) or v <= 1:
            return False
        kl = str(k).lower()
        return "total" not in kl and "hour" not in kl

    per_vals = [v for k, v in breakdown.items() if _usable(k, v) and _PER_KEY_RE.search(str(k))]
    base_vals = [v for k, v in breakdown.items() if _usable(k, v) and not _PER_KEY_RE.search(str(k))]
    if len(per_vals) != 1 or len(base_vals) != 1:
        return None
    return per_vals[0] * base_vals[0]


# The COUNTED noun of a per-rate ('average of N <noun> per <unit>') tells whether N*M is the example
# count. A measure / sub-item noun (frames per video, tokens per doc, annotations per item) is NOT an
# example yield -- multiplying it false-drops a correct size (a multimodal "30 frames per video" x M
# videos). The composer is deliberately MORE conservative than the descriptive gate: compute the
# magnitude ONLY when the counted noun is an example-ish content noun (or equals the size bucket's own
# unit), never on a sub-unit/measure rate.
_MAGNITUDE_MEASURE_NOUNS = frozenset({
    "frame", "frames", "token", "tokens", "word", "words", "character", "characters", "char", "chars",
    "byte", "bytes", "pixel", "pixels", "hour", "hours", "minute", "minutes", "second", "seconds",
    "annotation", "annotations", "label", "labels", "rating", "ratings", "review", "reviews",
    "turn", "turns", "step", "steps", "sentence", "sentences"})
_MAGNITUDE_CONTENT_NOUNS = frozenset({
    "video", "videos", "image", "images", "document", "documents", "question", "questions",
    "example", "examples", "problem", "problems", "instance", "instances", "sample", "samples",
    "prompt", "prompts", "dialogue", "dialogues", "conversation", "conversations", "pair", "pairs",
    "item", "items", "episode", "episodes", "task", "tasks", "clip", "clips", "utterance", "utterances",
    "query", "queries", "trio", "trios", "record", "records", "case", "cases", "scene", "scenes"})


def _prose_magnitude(text: Any, bucket_unit: Optional[str] = None) -> Optional[int]:
    """A dataset magnitude implied by an 'M <unit>s ... average of N <noun> per <unit>' phrase in
    PROSE, returned as M*N, or None. The PROSE sibling of _breakdown_product (which reads a structured
    size_breakdown dict): activitynet states its multiplicative magnitude in the overview text ("203
    activity classes with an average of 137 untrimmed videos per class") where _breakdown_product can
    never see it and _prose_example_count excludes the videos/classes nouns.

    Conservative (intentionally diverges from the descriptive gate's byte-shape -- the composer must
    never false-drop a correct size): the COUNTED noun (N's noun) must be an example-ish CONTENT noun
    or equal the size bucket's own unit; a measure/sub-item noun (frames/tokens/annotations/hours per
    unit) yields None. Needs BOTH the rate AND a matching '<M> <unit>(s)' count whose number differs
    from the rate's own number (so "average of 3 annotations per item" never grabs the rate's 3 as M).
    """
    if not isinstance(text, str) or not text:
        return None
    bu = (bucket_unit or "").strip().lower().rstrip("s")
    best = None
    # group(2) is the COUNTED noun: the word HEAD immediately before 'per' (adjectives/fillers between
    # the number and the noun are skipped by the lazy [\w\s-]*?), e.g. "137 untrimmed videos per class"
    # -> counted = "videos", per-unit = "class".
    for m in re.finditer(r"average of\s+([\d,]+)\s+(?:[\w-]+\s+)*?([a-z][\w-]*)\s+per\s+([a-z]+)",
                         text, re.IGNORECASE):
        n = int(m.group(1).replace(",", ""))
        counted = m.group(2).lower()
        counted_sing = counted.rstrip("s")
        # Only an example-ish content noun (or the bucket's own unit) is a real example yield; a
        # measure/sub-item noun is not -> abstain (return None for this match, never the magnitude).
        is_content = counted in _MAGNITUDE_CONTENT_NOUNS or (bu and counted_sing == bu)
        if counted in _MAGNITUDE_MEASURE_NOUNS or not is_content:
            continue
        unit = re.escape(m.group(3).lower())
        # The M-count of the magnitude unit, anchored to a number that is NOT the rate's own number
        # (avoid grabbing N as M when they coincide / when the rate precedes the base count).
        cm = None
        for c in re.finditer(r"([\d,]+)\s+(?:[a-z]+\s+){0,2}" + unit + r"(?:es|s)?\b",
                             text, re.IGNORECASE):
            if c.start() == m.start(1):   # this is the rate's own "<N> ... per <unit>" number
                continue
            cm = c
            break
        if cm:
            best = max(best or 0, int(cm.group(1).replace(",", "")) * n)
    return best


def _prose_count_phrase(text: Any) -> Optional[str]:
    """The largest example-level count in prose WITH its noun, e.g. '448 questions', so a
    reconciled size value is grounded in the prose, not a guessed noun. None when no count phrase is
    stated (same matching as _prose_example_count, which supplies the count itself)."""
    if not isinstance(text, str) or not text:
        return None
    best_n, best_phrase = -1, None
    for m in _EXAMPLE_NOUN_RE.finditer(text):
        rest = text[m.end():]
        if rest[:6].lower().startswith(" per ") or _MEASURE_AFTER_RE.match(rest):
            continue
        digits = m.group(1).replace(",", "")
        if digits.isdigit() and int(digits) > best_n:
            best_n = int(digits)
            num = m.group(1)
            noun = m.group(0)[m.group(0).rfind(num) + len(num):].strip()
            best_phrase = f"{int(digits)} {noun}" if noun else str(int(digits))
    return best_phrase


def _enforce_numeric_consistency(
    card: Dict[str, Any], det_facts: Dict[str, Any], provenance: Optional[Dict[str, Any]]
) -> None:
    """Drop a size / size_breakdown that the card's own prose contradicts (abstain, never a false
    number). Mutates card in place.
      A  a coarse HF size bucket whose range excludes an explicit example count in the prose;
      A2 a coarse HF size bucket disproved by a MULTIPLICATIVE size_breakdown product (base x per);
      A3 a SPECIFIC size count contradicted by a different definitional-lead prose count (wrong
         subset): prefer the lead count when cleanly extractable, else abstain;
      C  a size_breakdown that is really percentages ('%' in its provenance, or values summing to
         ~100): convert when an authoritative total is known, else drop;
      D  a subset-partition size_breakdown whose value-sum OVERSHOOTS an authoritative total (a
         real partition can never exceed the total -- a token-length column read as counts can)."""
    data = card.get("data")
    if not isinstance(data, dict):
        return
    bd = card.get("benchmark_details") or {}
    overview = bd.get("overview") if isinstance(bd.get("overview"), str) else ""
    source = data.get("source") if isinstance(data.get("source"), str) else ""
    prose_count = _prose_example_count(overview)
    if prose_count is None:
        prose_count = _prose_example_count(source)

    # Rule A: a coarse bucket clearly contradicted by an explicit prose count, or a shard-count
    # bucket the prose cannot confirm. The 1.5x margin absorbs boundary noise and intermediate
    # counts (e.g. "1,005 pairs from the previous stage, 830 passed" must not unseat an n<1K tag),
    # while a real over/under-count (swe-bench's 2,294 vs n<1K) still trips it.
    rng = _size_bucket_range(data.get("size"))
    if rng:
        lo, hi = rng
        if prose_count is not None:
            if prose_count >= hi * 1.5 or (lo > 0 and prose_count < lo / 1.5):
                data["size"] = "Not specified"
        elif det_facts.get("_size_is_shard_host"):
            data["size"] = "Not specified"
        else:
            # Rule A2: a per-structured breakdown product disproves a too-small bucket. Same 1.5x
            # boundary margin as Rule A (203 x 137 = 27811 >> 1500 unseats an n<1K tag).
            product = _breakdown_product(data.get("size_breakdown"))
            # Rule A2b: a card-internal MULTIPLICATIVE magnitude in the OVERVIEW PROSE ("203 classes
            # ... average of 137 videos per class" -> 27811) disproves a too-small bucket where the
            # structured breakdown is absent and _prose_example_count excludes the videos/classes
            # nouns. Conservative (more than the descriptive gate): only an example-ish content counted
            # noun (or the bucket's own unit) yields a magnitude, so a measure/sub-item rate ("30
            # frames per video") never false-drops a correct size. Abstain-only, over-count, 1.5x.
            if product is None:
                bucket_unit = _size_bucket_unit(data.get("size"))
                product = (_prose_magnitude(overview, bucket_unit)
                           or _prose_magnitude(source, bucket_unit))
            if product is not None and product >= hi * 1.5:
                data["size"] = "Not specified"

    # Rule A3: a SPECIFIC size count (explicit, not a bucket) contradicted by a different
    # definitional-lead prose count is the wrong subset (gpqa "546 questions" Extended vs the
    # overview's "448 multiple-choice questions" main set). Prefer the lead count when it parses to
    # a clean count string, else abstain. Tolerance is the same ~1% the breakdown checks use, so two
    # counts that are meant to be the same number must match, not merely sit within a 1.5x band.
    elif _size_count(data.get("size")) is not None:
        sc = _size_count(data.get("size"))
        lead = _first_sentences(overview, 2)
        lead_count = _prose_example_count(lead)
        if (lead_count is not None and lead_count > 0
                and abs(sc - lead_count) > max(1, lead_count // 100)):
            phrase = _prose_count_phrase(lead)
            data["size"] = phrase if phrase else "Not specified"

    # Rules C/D: size_breakdown sanity.
    sb = data.get("size_breakdown")
    if not (isinstance(sb, dict) and sb):
        return
    vals = [v for k, v in sb.items()
            if isinstance(v, int) and not isinstance(v, bool) and "total" not in str(k).lower()]
    if not vals:
        return
    det_total = det_facts.get("data.total_examples")
    det_total = det_total if (isinstance(det_total, int) and not isinstance(det_total, bool)) else None

    # Rule E: when a size_breakdown's parts OVERSHOOT its own stated Total, the object is internally
    # contradictory -- a real partition's parts can never exceed the whole (e.g. mmlu-pro Total=6810
    # while the per-discipline parts sum to 12032). Only the OVERSHOOT direction is a contradiction:
    # parts summing BELOW the stated Total is legitimate (the parts are an incomplete enumeration, or
    # the keys are dimension/subgroup counts whose Total counts something the parts don't fully list --
    # e.g. BOLD {Profession:18, Gender:2, Race:4, Total:43}). This mirrors Rule D's overshoot-only
    # asymmetry. ABSTAIN by default: DROP the contradictory Total key (leaving the clean per-part
    # breakdown), because recomputing it to sum(parts) would itself assert a NEW false fact when the
    # parts are an INCOMPLETE partition. Recompute-to-sum ONLY when an authoritative HF example total
    # corroborates the parts are a complete partition (det_total == sum(parts)). Skip heterogeneous-stat
    # breakdowns (not a summable partition) and percentage breakdowns (Rule C owns those). The total key
    # is matched by NORMALIZED EQUALITY to "total" (not a substring), so a "Total Tokens"/"Subtotal"
    # stat key is never mistaken for the partition total; the compared key IS the mutated key.
    # Structural, no name. (size_breakdown values are Dict[str, int], so abstain = drop the key.)
    if not any(_is_heterogeneous_stat_key(k) for k in sb):
        is_pct_like = len(vals) >= 3 and all(0 < v <= 100 for v in vals) and abs(sum(vals) - 100) <= 1
        total_key = next((k for k in sb
                          if re.sub(r"[^a-z]", "", str(k).lower()) == "total"
                          and isinstance(sb[k], int) and not isinstance(sb[k], bool)), None)
        parts_sum = sum(vals)
        if total_key is not None and not is_pct_like and parts_sum > 0:
            stated_total = sb[total_key]
            if parts_sum - stated_total > max(1, stated_total // 100):   # overshoot only
                if det_total is not None and abs(det_total - parts_sum) <= max(1, parts_sum // 100):
                    sb[total_key] = parts_sum        # corroborated complete partition -> recompute
                else:
                    del sb[total_key]                # uncorroborated -> abstain (drop), never assert

    prov_sb = ((provenance or {}).get("data") or {}).get("size_breakdown") or {}
    prov_text = " ".join(str(prov_sb.get(_k, "")) for _k in ("evidence", "quote"))
    looks_pct = "%" in prov_text or (
        len(vals) >= 3 and all(0 < v <= 100 for v in vals) and abs(sum(vals) - 100) <= 1
        and (det_total is None or abs(det_total - sum(vals)) > max(1, det_total // 100)))
    if looks_pct:
        # Rule C: convert only against an authoritative HF total; otherwise abstain.
        if det_total and det_total > 0:
            data["size_breakdown"] = {
                k: round(v / 100 * det_total) for k, v in sb.items()
                if isinstance(v, int) and not isinstance(v, bool) and "total" not in str(k).lower()}
        else:
            data["size_breakdown"] = "Not specified"
        return

    # Rule D: a subset partition that overshoots an authoritative total cannot be example counts.
    if any(_is_heterogeneous_stat_key(k) for k in sb):
        return
    total = det_total if det_total is not None else prose_count
    if isinstance(total, int) and total > 0 and sum(vals) > total * 1.01 + 1:
        data["size_breakdown"] = "Not specified"


# A meta-statement about missing documentation is not a benchmark description: do not surface it
# as an overview (the card stays honestly thin instead).
_NO_DOC_META_RE = re.compile(
    r"no (?:official )?(?:academic |peer[- ]reviewed )?(?:documentation|sources?|paper)"
    r"|no documentation found|requires? verification|could not (?:find|locate)"
    r"|no peer[- ]reviewed|yielded no", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_sentences(text: str, n: int) -> str:
    """The first n sentences of text, whitespace-normalised. Conservative splitter."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return " ".join(p for p in parts[:n]).strip()


def _eee_overview_text(eee_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """A benchmark-level overview drawn from EEE, to fill a husk card's Not-specified overview.

    Composite suites get a declarative sentence from `contains`. A single benchmark uses the
    benchmark_score wrapper's evaluation_description (the benchmark-level blurb), read from the
    UNMUTATED metrics dict so the degenerate-wrapper drop cannot blank it; otherwise the longest
    metric description. A meta-statement about missing documentation is not a description -> None.
    Honest-None when there is no usable text."""
    if not isinstance(eee_metadata, dict):
        return None
    if eee_metadata.get("benchmark_type") == "composite":
        contains = [c for c in (eee_metadata.get("contains") or [])
                    if isinstance(c, str) and c.strip()]
        return f"A composite benchmark suite comprising: {', '.join(contains)}." if contains else None
    metrics = eee_metadata.get("metrics")
    if not isinstance(metrics, dict):
        return None
    wrapper_desc, longest = "", ""
    for rid, cfg in metrics.items():
        cfg = cfg if isinstance(cfg, dict) else {}
        desc = str(cfg.get("evaluation_description", "")).strip()
        if not desc:
            continue
        if _is_degenerate_metric(rid, cfg) and not wrapper_desc:
            wrapper_desc = desc
        if len(desc) > len(longest):
            longest = desc
    desc = wrapper_desc or longest
    if len(desc) < 40 or _NO_DOC_META_RE.search(desc):
        return None
    return _first_sentences(desc, 3)


# C1: a husk-fill EEE description can describe a DIFFERENT benchmark than the resolved card, because
# the llm-stats slug (e.g. "arc") can collide with another benchmark of the same short name (ARC-AGI
# vs the AI2 Reasoning Challenge). Detect a definitional `Full Name (ACRONYM)` head whose acronym is
# the card's short name but whose full-name words do not match the resolved identity. Keyed on the
# acronym-expansion mismatch, never on a benchmark name.
_OVERVIEW_DEF_RE = re.compile(r"^\s*(?:The\s+)?([A-Z][A-Za-z0-9&/\- ]+?)\s*\(([A-Za-z][A-Za-z0-9\-]{1,8})\)")
_IDENTITY_STOPWORDS = {"the", "a", "an", "of", "and", "for", "to", "in", "on", "is", "are",
                       "with", "by", "benchmark", "dataset"}


def _content_words(text: str) -> set:
    """Lower-cased content words (>2 chars, minus function-word/generic stopwords)."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in _IDENTITY_STOPWORDS}


def _overview_subject_conflicts(overview_text: str, identity_text: str) -> bool:
    """True when an EEE-derived overview's definitional subject collides with the resolved
    paper/card identity. Conservative: fires only on an explicit `Full Name (ACRONYM)` head whose
    acronym appears in the identity but whose full-name content words barely overlap it (< 0.5),
    with at least one distinctive full-name word absent. Returns False without an identity to
    compare against. Never reads a benchmark name."""
    if not isinstance(overview_text, str) or not isinstance(identity_text, str) or not identity_text.strip():
        return False
    m = _OVERVIEW_DEF_RE.match(overview_text)
    if not m:
        return False
    full_name, acronym = m.group(1).strip(), m.group(2).strip()
    id_words = _content_words(identity_text)
    if not id_words:
        return False
    # the acronym must actually be the identity's short name (else this paren is unrelated)
    ak = _alnum_key(acronym)
    if ak not in {_alnum_key(w) for w in id_words} and ak not in _alnum_key(identity_text):
        return False
    fn_words = _content_words(full_name)
    if not fn_words:
        return False
    overlap = len(fn_words & id_words) / len(fn_words)
    return overlap < 0.5 and bool(fn_words - id_words)


# M1: HF/source subject-coherence. A slug can match an HF repo describing a DIFFERENT benchmark of
# the same short name (actionbench-the-slug matched facebook/actionbench = ActionMesh's 3D-mesh
# benchmark). The corroboration guard (workers _repo_corroboration) cannot see this: the basename
# IS the distinctive name and the repo's arxiv tag became the paper, so it reads as corroborated.
# This guard compares the README's leading description against the EEE-identity subject by
# content-word overlap; a near-disjoint pair means the repo describes a different entity. Structural
# (overlap of content words), never keyed on a benchmark name. Replicates the resolver's
# _distinctive_base PATTERN locally. The resolver's accept-time subject veto reuses _content_words
# from here (name-blind, untruncated variant in paper_resolver._subject_veto_overlap).

# README markup that carries no subject content: YAML frontmatter, html/markdown, image/link refs.
_README_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.S)
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_MARKUP_RE = re.compile(r"[#*`>|=_-]{1,}")

# A subject comparison needs enough words on both sides to be meaningful; a thin README lead or a
# bare identity is undecidable -> never fire.
_SUBJECT_MIN_WORDS = 8
# Near-disjoint content-word overlap (best of the two directions) below this means the README
# describes a different subject than the identity. Grounded on the pilot: actionbench best-overlap
# 0.05 (wrong repo) sits far below gaia 0.40 / charxiv 0.24 / mbpp 0.93 (correct repos).
_SUBJECT_OVERLAP_FLOOR = 0.10


def _readme_lead(readme_text: str, n: int = 600) -> str:
    """The leading prose of an HF README, with YAML frontmatter and markdown/html markup stripped.
    The first ~n chars are where a dataset card states what the benchmark is."""
    if not isinstance(readme_text, str) or not readme_text:
        return ""
    t = _README_FRONTMATTER_RE.sub("", readme_text)
    t = _MARKDOWN_LINK_RE.sub(" ", t)
    t = _HTML_TAG_RE.sub(" ", t)
    t = _MARKDOWN_MARKUP_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n]


def _eee_identity_subject(eee_metadata: Optional[Dict[str, Any]]) -> str:
    """The EEE-identity subject text: benchmark_name joined with each metric's
    evaluation_description. This is what the card IS, independent of any bound repo."""
    if not isinstance(eee_metadata, dict):
        return ""
    parts: List[str] = []
    name = eee_metadata.get("benchmark_name")
    if isinstance(name, str) and name.strip():
        parts.append(name)
    metrics = eee_metadata.get("metrics")
    if isinstance(metrics, dict):
        for cfg in metrics.values():
            if isinstance(cfg, dict):
                desc = cfg.get("evaluation_description")
                if isinstance(desc, str) and desc.strip():
                    parts.append(desc)
    return " ".join(parts)


def _subject_overlap(readme_text: str, identity_subject_text: str) -> Optional[float]:
    """Content-word overlap between an HF README's leading description and the EEE-identity subject,
    or None when either side is too thin to judge (fewer than _SUBJECT_MIN_WORDS content words).
    Overlap is the best of the two directions, so a long README vs a short identity (or vice versa)
    is judged by how well the shorter is covered. Structural, never reads a benchmark name.

    Factored out of _hf_subject_conflicts so the resolve-time HF verifier (run_hf Tier-1) can apply a
    higher ACCEPT floor than the conflict floor while sharing this single computation. None means
    undecidable: the conflict guard reads it as 'no conflict', the verifier reads it as 'route to LLM'."""
    lead = _readme_lead(readme_text)
    lead_words = _content_words(lead)
    id_words = _content_words(identity_subject_text)
    if len(lead_words) < _SUBJECT_MIN_WORDS or len(id_words) < _SUBJECT_MIN_WORDS:
        return None
    inter = len(lead_words & id_words)
    return max(inter / len(lead_words), inter / len(id_words))


def _hf_subject_conflicts(readme_text: str, identity_subject_text: str) -> bool:
    """True when an HF README's leading description and the EEE-identity subject are near-disjoint,
    i.e. the matched repo describes a DIFFERENT benchmark than the card's identity. Returns False on
    any thin/missing input (too few content words on either side) so it never fires on undecidable
    cases. Overlap is the best of the two directions, so a long README vs a short identity (or vice
    versa) is judged by how well the shorter is covered. Structural, never reads a benchmark name."""
    overlap = _subject_overlap(readme_text, identity_subject_text)
    return overlap is not None and overlap < _SUBJECT_OVERLAP_FLOOR


def extract_deterministic_facts(
    eee_metadata: Optional[Dict[str, Any]] = None,
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    unitxt_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract ground-truth facts from structured sources (no LLM involved)."""
    facts: Dict[str, Any] = {}

    if hf_metadata:
        meta = _get_hf_meta(hf_metadata)
        tags = meta.get("tags", [])
        if isinstance(tags, list):
            # Canonical HF-tag values come from extract_hf_tags, the single formatter.
            # This keeps the deterministic override at the composition site byte-identical
            # to the workers override (S1: divergent ad-hoc formatting here used to clobber
            # languages/license with unsorted/unmapped values). extract_hf_tags supplies
            # languages, data_type, data.format, data.size, data_licensing in canonical form.
            hf_canonical = extract_hf_tags(hf_metadata)
            for _k in (
                "benchmark_details.languages",
                "benchmark_details.data_type",
                "data.format",
                "data.size",
                "ethical_and_legal_considerations.data_licensing",
            ):
                if _k in hf_canonical:
                    facts[_k] = hf_canonical[_k]
            # C2 Rule B signal: flag a size bucket that reflects shard/archive count (no parquet/
            # arrow row files, no dataset_info). The abstain decision is deferred to
            # _enforce_numeric_consistency, which drops the bucket only when the prose does NOT
            # independently confirm it (so a real small archive-hosted set like aa-lcr is kept).
            if "data.size" in facts and _hf_size_is_shard_count(meta):
                facts["_size_is_shard_host"] = True

            task_cats = []
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("task_categories:"):
                    task_cats.append(tag.split(":", 1)[1].strip().replace("-", " "))
            if task_cats:
                facts["purpose_and_intended_users.tasks_hf"] = task_cats

            domain_tags = []
            for tag in tags:
                if isinstance(tag, str) and ":" not in tag and len(tag) > 2:
                    domain_tags.append(tag.replace("-", " "))
            if domain_tags:
                facts["benchmark_details.domain_tags"] = domain_tags

        # No meta/card_data license fallback: extract_hf_tags is the single license
        # formatter (license: tag -> _LICENSE_MAP, other/unknown skipped). A raw,
        # un-mapped fallback would write a non-Tier-1 value (e.g. "mit") that the workers
        # override never produces -> breaks the S1 byte-identity invariant (contract S4.4/S10).

        card_data = meta.get("card_data", {})
        if isinstance(card_data, dict):
            pretty_name = card_data.get("pretty_name", "")
            if pretty_name:
                facts["benchmark_details.pretty_name"] = pretty_name
            annot_creators = card_data.get("annotations_creators", [])
            if annot_creators and isinstance(annot_creators, list):
                facts["data.annotation_creators"] = [
                    c.replace("-", " ") for c in annot_creators
                ]
            source_datasets = card_data.get("source_datasets", [])
            if source_datasets and isinstance(source_datasets, list):
                facts["data.source_datasets"] = source_datasets

        dataset_info = meta.get("dataset_info", {})
        if isinstance(dataset_info, dict):
            splits = dataset_info.get("splits", [])
            if isinstance(splits, list) and splits:
                split_info = []
                total = 0
                for s in splits:
                    if isinstance(s, dict) and "name" in s and "num_examples" in s:
                        split_info.append(f"{s['name']}: {s['num_examples']}")
                        total += s.get("num_examples", 0)
                if split_info:
                    facts["data.split_info"] = split_info
                    facts["data.total_examples"] = total

        # A9 data.format fallback: no HF format: tag -> dominant data-file extension.
        if "data.format" not in facts:
            fmt = _format_from_files(meta.get("file_list"))
            if fmt:
                facts["data.format"] = fmt

        # A9 data.size_breakdown: per-subset counts parsed from the README counts table.
        # A split-named breakdown must be consistent with the dataset's example count (else
        # dropped); heterogeneous statistic breakdowns are kept as-is.
        size_breakdown = _size_breakdown_from_readme(meta.get("readme_markdown"))
        if size_breakdown and _size_breakdown_is_consistent(
            size_breakdown, facts.get("data.total_examples"), facts.get("data.size")
        ):
            facts["data.size_breakdown"] = size_breakdown

    if eee_metadata:
        metrics = eee_metadata.get("metrics", {})
        if metrics:
            # Drop the degenerate llm_stats benchmark_score wrapper that masks the real
            # scorer; keep it only when dropping it would blank the metrics field (a
            # proxy-only benchmark is degenerate-source, flagged elsewhere, not blanked).
            real = {rid: cfg for rid, cfg in metrics.items()
                    if not _is_degenerate_metric(rid, cfg if isinstance(cfg, dict) else {})}
            metrics = real or metrics
            facts["methodology.metrics"] = canonicalize_metrics(metrics)
            facts["methodology.metric_configs"] = {
                name: {
                    "lower_is_better": cfg.get("lower_is_better", False),
                    "score_type": cfg.get("score_type", ""),
                    "description": cfg.get("evaluation_description", ""),
                }
                for name, cfg in list(metrics.items())[:10]
            }

        eval_summary = eee_metadata.get("evaluation_summary", {})
        if eval_summary:
            facts["evaluation_summary"] = eval_summary

        source_urls = eee_metadata.get("source_urls", [])
        if source_urls:
            facts["benchmark_details.eee_source_urls"] = source_urls[:5]

        if eee_metadata.get("eval_library"):
            facts["methodology.eval_library"] = eee_metadata["eval_library"]

        # Composite context in deterministic facts: structured data for merging.
        # A second composite block exists in the prompt (~line 1744) for LLM guidance.
        # Both are needed: this one feeds the facts dict, that one shapes generation.
        if eee_metadata.get("benchmark_type") == "composite":
            contains = eee_metadata.get("contains", [])
            facts["benchmark_type"] = "composite"
            facts["composite_context"] = (
                f"This is a composite benchmark suite containing the following "
                f"sub-benchmarks: {', '.join(contains)}. Describe what the suite "
                f"measures as a whole, not individual sub-benchmarks."
            )

        # Overview from the EEE benchmark-level description, so a wrapper-only / degenerate-source
        # card is not a bare NS husk. Fill-only-NS downstream; honest-None for undocumented cards.
        overview_text = _eee_overview_text(eee_metadata)
        if overview_text:
            # C1: suppress the husk fill when the EEE description's definitional subject conflicts
            # with the resolved paper/card identity (a slug colliding with a different benchmark of
            # the same short name). An honest NS husk beats a wrong-benchmark description.
            _hf_meta = _get_hf_meta(hf_metadata) if isinstance(hf_metadata, dict) else {}
            _pretty = _hf_meta.get("card_data", {}).get("pretty_name") if isinstance(
                _hf_meta.get("card_data"), dict) else None
            _identity = " ".join(str(s) for s in (
                (extracted_ids or {}).get("paper_title"),
                eee_metadata.get("benchmark_name"),
                _pretty,
            ) if s)
            if _overview_subject_conflicts(overview_text, _identity):
                logger.info("Overview husk-fill suppressed: EEE subject conflicts with identity (%r)",
                            _identity[:80])
            else:
                facts["benchmark_details.overview"] = overview_text

    if extracted_ids:
        if extracted_ids.get("paper_url"):
            facts["benchmark_details.paper_url"] = extracted_ids["paper_url"]
        if extracted_ids.get("hf_repo"):
            hf_repo = extracted_ids["hf_repo"]
            facts["benchmark_details.hf_url"] = f"https://huggingface.co/datasets/{hf_repo}"

    # collection_date is intentionally NOT seeded from a publication / HF-upload year -- those
    # are not the data-collection date (F6-2). It stays NS unless the LLM finds an explicit
    # collection period; _coerce_collection_date keeps that a year-or-NS downstream.

    # Route UnitXT structured catalog metadata (non-EEE path only; see workers.py).
    # Fill-only: only contributes metrics/format where EEE/HF have not already. The
    # catalog stores resolved artifacts under components[bucket][ref_id], ref_id like
    # "metrics.accuracy" / "formats.mcq" — strip the bucket prefix for the name.
    if unitxt_metadata:
        components = unitxt_metadata.get("components", {})
        if isinstance(components, dict):
            metric_refs = list(components.get("metrics", {}).keys())
            if metric_refs and "methodology.metrics" not in facts:
                facts["methodology.metrics"] = [r.split(".")[-1] for r in metric_refs]
            fmt_refs = list(components.get("formats", {}).keys())
            if fmt_refs and "data.format" not in facts:
                facts["data.format"] = fmt_refs[0].split(".")[-1]

    logger.info("Deterministic facts: %d fields extracted", len(facts))
    return facts


def _fill_paper_gaps(paper_items, paper_retriever, benchmark_name, verify_source,
                     identity_anchor=""):
    """Targeted RAG retrieval for paper fields that have no verified evidence yet.

    Same Evidence-Item format as the paper call. Quotes are verified against verify_source
    (the full paper text), doc=docling, so gap items share one reference with paper items.
    Returns (verified records, telemetry).
    """
    covered = {it["field"] for it in paper_items}
    targets = {path: query for path, query in evidence.gap_fields().items()
               if path not in covered}
    if not targets:
        return [], evidence.empty_call_telemetry(evidence.DOC_PAPER)

    logger.info("Gap-filling: targeted retrieval for %d fields: %s",
                len(targets), list(targets))

    gap_chunks: Dict[str, str] = {}
    for path, query in targets.items():
        try:
            chunks = paper_retriever.invoke(query)
            if chunks:
                gap_chunks[path] = "\n".join(c.page_content for c in chunks[:2])[:1500]
        except Exception as e:
            logger.debug("Gap-fill retrieval failed for %s: %s", path, e)

    if not gap_chunks:
        return [], evidence.empty_call_telemetry(evidence.DOC_PAPER)

    chunks_text = "\n\n".join(f"[For {path}]\n{text}" for path, text in gap_chunks.items())
    prompt = evidence.build_extraction_prompt(
        source_label="paper excerpts",
        fields=list(gap_chunks),
        benchmark_name=benchmark_name,
        identity_anchor=identity_anchor,
        source_text=chunks_text,
    )
    return _run_extraction(prompt, verify_source, evidence.DOC_PAPER, "Gap-fill")


# Hallucinated pipeline-org owners that leak into resource URLs -- the same furniture family as
# the Ergon-native / rollout-card prose jargon (the model invents these org names). Keyed on the
# literal token, never a benchmark name, so it stays corpus-general.
_FURNITURE_URL_RE = re.compile(r"\b(?:exgentic|ergon)\b", re.I)


def _is_nested_foreign_domain(url: str) -> bool:
    """True when a host that should own its path nests a FOREIGN domain as a path segment -- a
    malformed / injected resource like huggingface.co/datasets/mercor.com/... . Structural and
    conservative: only huggingface.co / github.com are checked, and a legit owner/name path (no
    TLD-bearing segment) passes."""
    m = re.match(r"^https?://([^/]+)(/.*)?$", url)
    if not m:
        return False
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in ("huggingface.co", "github.com"):
        return False
    segs = [s for s in (m.group(2) or "").split("/") if s]
    return any(re.fullmatch(r"[\w-]+\.(?:com|org|net|ai|io)", s, re.I) for s in segs)


def post_process_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize types and fix common schema issues in the generated card."""
    bd = card.get("benchmark_details", {})

    sb = bd.get("similar_benchmarks", [])
    if isinstance(sb, str):
        if any(neg in sb.lower() for neg in ["no ", "not ", "none", "no similar"]):
            bd["similar_benchmarks"] = []
        else:
            bd["similar_benchmarks"] = [sb]

    resources = bd.get("resources", [])
    if isinstance(resources, list):
        clean = []
        seen = set()
        for r in resources:
            if not isinstance(r, str):
                continue
            r = r.strip()
            if re.match(r"^https?://\S+$", r):
                url = r
            elif re.match(r"^[\w.-]+/[\w.-]+$", r):  # clean owner/name repo id, no spaces
                url = f"https://huggingface.co/datasets/{r}"
            else:
                m = re.search(r"https?://\S+", r)  # prose: extract the first real URL, else drop
                url = m.group(0) if m else None
            if url:
                url = url.replace("hugdingface.co", "huggingface.co")
                # Drop llm-stats furniture domains (model stats pages, not benchmark resources).
                if any(bad in url for bad in ("llm-stats.com/models", "api.llm-stats.")):
                    continue
                # Drop a malformed/injected URL nesting a foreign domain in a host that owns its
                # path (e.g. huggingface.co/datasets/mercor.com/...).
                if _is_nested_foreign_domain(url):
                    continue
                # Drop a hallucinated pipeline-org URL (Ergon/Exgentic furniture family).
                if _FURNITURE_URL_RE.search(url):
                    continue
                # Dedup paper mirrors: arxiv.org/abs/<id> and alphaxiv.org/abs/<id> are the same
                # paper -- key on the arxiv id so only the first mirror survives. Allow old-style
                # category/number ids (the "/") and strip a trailing version so vN/mirror variants
                # of the same paper still collapse.
                m_paper = re.search(r"(?:arxiv|alphaxiv)\.org/abs/([\w./-]+)", url)
                if m_paper:
                    paper_id = re.sub(r"v\d+$", "", m_paper.group(1).rstrip("/"))
                    key = f"paper:{paper_id}"
                else:
                    key = url
                if url.startswith("http") and key not in seen:
                    clean.append(url)
                    seen.add(key)
        bd["resources"] = clean

    languages = bd.get("languages", [])
    if isinstance(languages, list):
        bd["languages"] = [LANG_MAP.get(l, l) for l in languages]
    card["benchmark_details"] = bd

    purpose = card.get("purpose_and_intended_users", {})
    osu = purpose.get("out_of_scope_uses", [])
    if isinstance(osu, str):
        if any(neg in osu.lower() for neg in ["no ", "not ", "none"]):
            purpose["out_of_scope_uses"] = []
        else:
            purpose["out_of_scope_uses"] = [osu]
    card["purpose_and_intended_users"] = purpose

    if "flagged_fields" not in card:
        card["flagged_fields"] = {}
    if "missing_fields" not in card:
        card["missing_fields"] = []

    return card


def apply_deterministic_overrides(
    card: Dict[str, Any],
    deterministic_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Override LLM-generated fields with deterministic ground-truth values.

    Always overrides languages and license; fill-only for metrics and format.
    """
    overrides_applied = []

    _EMPTY = {"not specified", "not specified.", "no information found", ""}

    def _is_empty(val):
        if val is None:
            return True
        if isinstance(val, str) and val.strip().lower() in _EMPTY:
            return True
        if isinstance(val, list) and (
            not val or (len(val) == 1 and isinstance(val[0], str) and val[0].strip().lower() in _EMPTY)
        ):
            return True
        return False

    if "benchmark_details.languages" in deterministic_facts:
        card.setdefault("benchmark_details", {})["languages"] = \
            deterministic_facts["benchmark_details.languages"]
        overrides_applied.append("languages")

    # overview is fill-only: only an EEE benchmark description fills a Not-specified overview
    # (a husk), never clobbering a real Stage-B overview.
    if "benchmark_details.overview" in deterministic_facts:
        existing = card.get("benchmark_details", {}).get("overview")
        if _is_empty(existing):
            card.setdefault("benchmark_details", {})["overview"] = \
                deterministic_facts["benchmark_details.overview"]
            overrides_applied.append("overview")

    # data_type is always-override (matches card_utils _ALWAYS_OVERRIDE); keeps this site
    # byte-identical to the workers override for all six deterministic fields (S1/S10).
    if "benchmark_details.data_type" in deterministic_facts:
        card.setdefault("benchmark_details", {})["data_type"] = \
            deterministic_facts["benchmark_details.data_type"]
        overrides_applied.append("data_type")

    if "ethical_and_legal_considerations.data_licensing" in deterministic_facts:
        card.setdefault("ethical_and_legal_considerations", {})["data_licensing"] = \
            deterministic_facts["ethical_and_legal_considerations.data_licensing"]
        overrides_applied.append("license")

    if "methodology.metrics" in deterministic_facts:
        eee_metrics = deterministic_facts["methodology.metrics"]
        if eee_metrics and card.get("methodology"):
            existing = card["methodology"].get("metrics")
            if _is_empty(existing):
                card["methodology"]["metrics"] = eee_metrics
                overrides_applied.append("metrics")
            else:
                logger.debug("Skipping metrics override: LLM has %r", existing)

    if "data.size" in deterministic_facts:
        existing = card.get("data", {}).get("size")
        if _is_empty(existing):
            card.setdefault("data", {})["size"] = deterministic_facts["data.size"]
            overrides_applied.append("size")
        else:
            logger.debug("Skipping size override: LLM has %r", existing)

    if "data.format" in deterministic_facts:
        existing = card.get("data", {}).get("format")
        if _is_empty(existing):
            card.setdefault("data", {})["format"] = deterministic_facts["data.format"]
            overrides_applied.append("format")
        else:
            logger.debug("Skipping format override: LLM has %r", existing)

    if "data.collection_date" in deterministic_facts:
        existing = card.get("data", {}).get("collection_date")
        if _is_empty(existing):
            card.setdefault("data", {})["collection_date"] = deterministic_facts["data.collection_date"]
            overrides_applied.append("collection_date")

    if "data.size_breakdown" in deterministic_facts:
        existing = card.get("data", {}).get("size_breakdown")
        # fill-only: never clobber a real Stage-B breakdown dict (e.g. bigcode)
        if _is_empty(existing) or (isinstance(existing, dict) and not existing):
            card.setdefault("data", {})["size_breakdown"] = deterministic_facts["data.size_breakdown"]
            overrides_applied.append("size_breakdown")

    resources = card.get("benchmark_details", {}).get("resources", [])
    urls_added = []
    if "benchmark_details.paper_url" in deterministic_facts:
        url = deterministic_facts["benchmark_details.paper_url"]
        if url and url not in resources:
            resources.insert(0, url)
            urls_added.append("paper")
    if "benchmark_details.hf_url" in deterministic_facts:
        url = deterministic_facts["benchmark_details.hf_url"]
        if url and url not in resources:
            resources.append(url)
            urls_added.append("hf")
    if "benchmark_details.eee_source_urls" in deterministic_facts:
        for url in deterministic_facts["benchmark_details.eee_source_urls"]:
            if url and url not in resources:
                resources.append(url)
                urls_added.append("eee")
    card.setdefault("benchmark_details", {})["resources"] = resources
    if urls_added:
        overrides_applied.append(f"resources({','.join(urls_added)})")

    if overrides_applied:
        logger.info("Deterministic overrides applied: %s", ", ".join(overrides_applied))

    return card


# Logo fetched per HF org handle, cached for the 500-card batch.
_logo_cache: Dict[str, Optional[str]] = {}
# Org-confirmation overview cached per owner handle (None = not a confirmed org / a user).
_org_overview_cache: Dict[str, Optional[Dict[str, Any]]] = {}
_HF_OVERVIEW_URL = "https://huggingface.co/api/{kind}/{org}/overview"


def _hf_org_overview(owner: str) -> Optional[Dict[str, Any]]:
    """HF organizations/<owner> overview JSON, but only when <owner> is a confirmed org.

    The organizations endpoint 200s for a real org and 404s for a user handle, so a non-None
    return is positive proof the owner is an organization (not a person). Timeout-guarded,
    fail-soft (-> None), cached per owner."""
    if not owner or not isinstance(owner, str):
        return None
    if owner in _org_overview_cache:
        return _org_overview_cache[owner]
    result: Optional[Dict[str, Any]] = None
    try:
        resp = requests.get(_HF_OVERVIEW_URL.format(kind="organizations", org=owner), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                result = data
    except Exception as e:
        logger.warning("HF org lookup failed for '%s': %s", owner, e)
        result = None
    _org_overview_cache[owner] = result
    return result


def _is_identicon_avatar(raw_avatar: Any) -> bool:
    """True for an HF default identicon (huggingface.co/avatars/<hash>.svg), a placeholder,
    not a real logo. Matched on the RAW avatarUrl: a default identicon is a relative
    '/avatars/<hash>.svg' (or its absolute huggingface.co form), while a real uploaded avatar
    is an absolute cdn-avatars.huggingface.co URL whose path is not '/avatars/'."""
    return isinstance(raw_avatar, str) and "/avatars/" in raw_avatar


def _looks_like_person_list(value: Any) -> bool:
    """True only for a real person-name list. Guards against an HF org handle
    (a single token like 'bigcode') being mistaken for paper authors -- a person
    list either has multiple entries or an entry with a space ('Igor Ivanov')."""
    if not isinstance(value, list) or not value:
        return False
    names = [v for v in value if isinstance(v, str) and v.strip()]
    if not names:
        return False
    return len(names) > 1 or any(" " in n.strip() for n in names)


_BIBTEX_AUTHOR_RE = re.compile(r'author\s*=\s*(?:\{([^{}]*)\}|"([^"]*)")', re.IGNORECASE)


def _parse_bibtex_authors(readme_markdown: Any) -> Optional[List[str]]:
    """Extract a person list from the first BibTeX `author = { ... }` field in an HF readme.

    Handles `Last, First and Last, First` and `First Last and First Last`; splits on ` and `,
    trims a trailing `and others` / `et al.`. Returns None when no BibTeX author field is
    present; the caller still guards the result with _looks_like_person_list."""
    if not isinstance(readme_markdown, str) or "author" not in readme_markdown.lower():
        return None
    m = _BIBTEX_AUTHOR_RE.search(readme_markdown)
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or "").strip()
    if not raw:
        return None
    authors: List[str] = []
    for part in re.split(r"\s+and\s+", raw):
        part = part.strip().strip(",").strip()
        if not part or part.lower() in ("others", "et al", "et al."):
            continue
        if "," in part:  # "Last, First" -> "First Last"
            last, _, first = part.partition(",")
            part = f"{first.strip()} {last.strip()}".strip()
        if part:
            authors.append(part)
    return authors or None


def _fetch_org_logo(org: str) -> Optional[str]:
    """Resolve a real ORG logo URL for an HF owner handle. Org-only and general: accept the
    HF organizations/<org> overview avatar (never the users/ endpoint, which is a person's
    photo) and only when it is a real uploaded avatar, not a default identicon; else fall to
    github.com/<org>.png, but only when the handle is a confirmed org. A user handle returns
    None (no real org logo). Timeout-guarded, fail-soft (-> None), cached by org. Returns
    only a fetched real URL, never fabricated."""
    if not org or not isinstance(org, str):
        return None
    if org in _logo_cache:
        return _logo_cache[org]
    result: Optional[str] = None
    try:
        overview = _hf_org_overview(org)
        if overview is not None:  # confirmed organization (a user handle is None here)
            avatar = overview.get("avatarUrl")
            if isinstance(avatar, str) and avatar and not _is_identicon_avatar(avatar):
                result = avatar if avatar.startswith("http") else f"https://huggingface.co{avatar}"
            if result is None:
                gh = f"https://github.com/{org}.png"
                try:
                    if requests.head(gh, allow_redirects=True, timeout=10).status_code == 200:
                        result = gh
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Logo fetch failed for org '%s': %s", org, e)
        result = None
    _logo_cache[org] = result
    return result


def _alnum_key(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


# The first BibTeX entry head (`@type{key,`) and `title = { ... }` field, so the HF-README citation
# can be gated on benchmark identity before its authors are trusted. A README that mislabels its
# citation (mgsm's README carries the GSM8K bibtex cobbe2021gsm8k) must not leak the parent's authors.
_BIBTEX_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.IGNORECASE)
_BIBTEX_TITLE_RE = re.compile(r'title\s*=\s*(?:\{+([^{}]*)\}+|"([^"]*)")', re.IGNORECASE)


def _bibtex_identity(readme_markdown: Any) -> tuple:
    """(key, title) of the first BibTeX entry in an HF readme, each '' when absent. The citation key
    (e.g. 'cobbe2021gsm8k') and title are what tie a bibtex entry to a benchmark identity."""
    if not isinstance(readme_markdown, str) or not readme_markdown:
        return "", ""
    km = _BIBTEX_KEY_RE.search(readme_markdown)
    tm = _BIBTEX_TITLE_RE.search(readme_markdown)
    key = km.group(1).strip() if km else ""
    title = ((tm.group(1) or tm.group(2)) if tm else "") or ""
    return key, title.strip()


# Generic identity tokens that carry no benchmark-distinctive signal (a bibtex sharing only these is
# not corroborated as this benchmark's own citation). Plus the EEE function words / "benchmark"/
# "dataset" already in _IDENTITY_STOPWORDS, reused below.
_GENERIC_IDENTITY_TOKENS = {"benchmark", "benchmarks", "dataset", "datasets", "eval", "evals",
                            "evaluation", "task", "tasks", "test", "tests", "data", "suite",
                            "leaderboard", "corpus"}


def _distinctive_identity_tokens(eee_metadata: Optional[Dict[str, Any]],
                                 overview: Any) -> set:
    """Distinctive ACRONYM/SLUG tokens for a benchmark identity, used to gate an HF-README bibtex.

    The EEE benchmark_name/slug is the authoritative identity (the card's name may still be a paper
    title at display time). Split the slug on non-alnum (acemath-rewardbench -> acemath, rewardbench)
    AND keep the whole alnum-joined form (acemathrewardbench), add the overview's leading
    `Full Name (ACRONYM)` head acronym, drop generic/function words and pure-numeric tokens. Keyed on
    the SLUG/ACRONYM, never expanded name-words: 'math' appears in GSM8K's title, so an expanded-word
    match would FALSE-PASS the parent citation; the acronym 'mgsm' is absent from the GSM8K bibtex."""
    tokens: set = set()
    name = ""
    if isinstance(eee_metadata, dict):
        n = eee_metadata.get("benchmark_name")
        if isinstance(n, str):
            name = n
    for part in re.split(r"[^A-Za-z0-9]+", name):
        ak = _alnum_key(part)
        if ak:
            tokens.add(ak)
    whole = _alnum_key(name)
    if whole:
        tokens.add(whole)
    if isinstance(overview, str):
        m = _OVERVIEW_DEF_RE.match(overview)
        if m:
            ak = _alnum_key(m.group(2))
            if ak:
                tokens.add(ak)
    # Distinctive only: drop generic/function words and pure-numeric (and over-short) tokens.
    return {t for t in tokens
            if len(t) >= 3 and not t.isdigit()
            and t not in _GENERIC_IDENTITY_TOKENS and t not in _IDENTITY_STOPWORDS}


# A distinctive token must be at least this long to be trusted as a KEY SUBSTRING. A short acronym
# (3-4 chars: arc, nli, gqa, bbh) is a substring of common author-keyed words ('research',
# 'architecture', 'online', 'hierarchical') and would leak wrong authors via the key; those must
# corroborate via the whole-word TITLE instead. Long slugs (charxiv, gsm8k, acemath, rewardbench,
# activitynet) stay key-substring safe.
_KEY_SUBSTRING_MIN_LEN = 5


def _bibtex_matches_identity(bib_key: str, bib_title: str, identity_tokens: set) -> bool:
    """True when an HF-README bibtex plausibly belongs to THIS benchmark: at least one distinctive
    acronym/slug token corroborates it in the bibtex KEY or TITLE.

    Asymmetric on purpose, length-gated to close the wrong-author-leak class on BOTH sides:
    - TITLE (natural language): WHOLE-WORD match only, any length. A substring there collides ('arc'
      in 'architecture' would leak Neural Architecture Search authors onto an ARC card).
    - KEY (glued author+year+slug, 'wang2024charxiv'/'cobbe2021gsm8k'): SUBSTRING match, but only for
      LONG tokens (len >= _KEY_SUBSTRING_MIN_LEN). A short acronym (arc/nli) is a substring of common
      author-keyed words ('research'/'online') and leaks via the key, so short tokens must corroborate
      via the whole-word title (an ARC own-citation survives via its title 'ARC ...').
    The parent-citation case (mgsm identity vs GSM8K bibtex 'cobbe2021gsm8k', title 'Training Verifiers
    to Solve Math Word Problems') matches neither and is dropped. No distinctive token -> False."""
    if not identity_tokens:
        return False
    key_k = _alnum_key(bib_key)
    title_words = {_alnum_key(w) for w in re.findall(r"[A-Za-z0-9]+", bib_title or "")}
    title_words.discard("")
    return any((t in title_words) or (len(t) >= _KEY_SUBSTRING_MIN_LEN and t in key_k)
               for t in identity_tokens)


def _is_short_identity(token: str) -> bool:
    """A plausible short benchmark identity: a single token / acronym, or up to 3
    whitespace-separated words. Guards the subtitle strip so a long descriptive pre-colon
    phrase is not mistaken for a name and truncated."""
    if not isinstance(token, str):
        return False
    words = token.split()
    return 1 <= len(words) <= 3


def _name_is_paper_title(name: Any, paper_title: Any) -> bool:
    """True when the benchmark name is actually the resolved arXiv paper title (or its pre-
    subtitle head) rather than the benchmark identity -- detected by the title starting with
    the name (alnum-keyed). Lets the EEE benchmark identity win over a paper-title-shaped name."""
    if not isinstance(name, str) or not isinstance(paper_title, str):
        return False
    nk, pk = _alnum_key(name), _alnum_key(paper_title)
    if not nk or not pk:
        return False
    return pk.startswith(nk)


def _clean_caps_name(name: Any, candidates: tuple = ()) -> Any:
    """Normalize a raw paper-title benchmark name to its short identity.

    Drops the ': subtitle' (and the stray space before it) when the pre-colon token is a
    plausible short identity (single token / acronym / <= 3 words), so mixed-case titles
    like 'SuperGLUE: A Stickier Benchmark...' or 'API-Bank: A Comprehensive...' resolve to
    their identity while a genuinely long name is left intact. A long, mostly-uppercase
    string is also resolved to its pre-colon identity (legacy caps path; short acronyms
    like 'AA-LCR' are left alone). Either way, prefers a proper-cased identity (HF
    pretty_name / resolved paper title) whose pre-colon token matches. Structural, no
    benchmark names. Returns name unchanged when not triggered."""
    if not isinstance(name, str) or not name.strip():
        return name
    s = name.strip()
    pre = s.split(":", 1)[0].strip()
    # General path: strip a subtitle only when the pre-colon token is a short identity.
    strip_subtitle = (":" in s) and bool(pre) and pre != s and _is_short_identity(pre)
    # Legacy caps path: a long, mostly-uppercase string resolves to its pre-colon identity.
    letters = [c for c in s if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    caps_name = upper_ratio > 0.7 and (len(s) > 15 or " " in s)
    if not (strip_subtitle or caps_name):
        return name
    key = _alnum_key(pre)
    for cand in candidates:
        if not isinstance(cand, str) or not cand.strip():
            continue
        cand_pre = cand.split(":", 1)[0].strip()
        if cand_pre and _alnum_key(cand_pre) == key:
            return cand_pre
    return pre


# Evidence doc-label -> canonical provenance source (contract: source is a doc type,
# never a bare evidence id like 'E01' or a 'doc/status' compound).
_DOC_TO_SOURCE = {
    "docling": "paper", "abstract": "abstract", "html": "html",
    "hf_readme": "hf", "github_readme": "github",
}


def _normalize_provenance_sources(
    all_provenance: Dict[str, Any], evidence_items: Optional[List[Dict[str, Any]]]
) -> None:
    """Rewrite each field's provenance.source to the doc type of its cited evidence item.

    Stage-B echoes the cited evidence id (e.g. 'E01') or a 'doc/status' compound into
    source; the contract wants a clean doc type. Looks up evidence_ids in evidence_items,
    maps the doc via _DOC_TO_SOURCE. Leaves source in {deterministic, eee} (already clean)
    and display entries untouched. Mutates all_provenance in place."""
    id2doc = {
        it.get("evidence_id"): it.get("doc")
        for it in (evidence_items or []) if isinstance(it, dict)
    }
    for fields in all_provenance.values():
        if not isinstance(fields, dict):
            continue
        for entry in fields.values():
            if not isinstance(entry, dict):
                continue
            src = entry.get("source")
            if src in ("deterministic", "eee"):
                continue
            doc = None
            eids = entry.get("evidence_ids")
            if isinstance(eids, list) and eids:
                doc = id2doc.get(eids[0])
            if doc is None and isinstance(src, str) and src:
                # source may itself be an eid or a 'doc/status' compound -- recover the doc
                doc = id2doc.get(src) or src.split("/", 1)[0]
            if doc:
                entry["source"] = _DOC_TO_SOURCE.get(doc, doc)


# C3: coverage reconciliation backstops.

# A license token (SPDX-ish) for lifting a concise value out of a provenance quote.
_LICENSE_TOKEN_RE = re.compile(
    r"\bCC[ -]BY(?:[ -]SA|[ -]NC|[ -]ND)*[ -]?\d(?:\.\d)?"
    r"|\bCreative Commons[\w .-]*?\d(?:\.\d)?"
    r"|\bApache(?:\s+License)?,?\s*\d(?:\.\d)?"
    r"|\bMIT(?:\s+License)?\b"
    r"|\bBSD(?:[- ]\d)?\b|\bGPL(?:[- ]?v?\d)?\b|\bLGPL\b|\bAGPL\b|\bMPL\b|\bCC0\b"
    r"|\bODbL\b|\bODC[- ]?(?:BY|PDDL)?\b", re.I)
# A no-evidence marker: a provenance quote that itself states the field is unknown -> not liftable.
_NO_EVIDENCE_RE = re.compile(
    r"no evidence|not specified|not stated|not mentioned|not discussed|not provided|not reported"
    r"|\bno (?:[\w-]+\s+){0,4}(?:stated|found|provided|mentioned|reported|available)", re.I)


def _is_marker_evidence(ev: Any) -> bool:
    """True for empty / too-short / no-evidence-marker provenance text (not a usable value)."""
    s = ev.strip() if isinstance(ev, str) else ""
    return (not s) or len(s) < 6 or bool(_NO_EVIDENCE_RE.search(s))


def _concise_license(quote: str) -> str:
    """A concise license value from a provenance quote: prefer a short first sentence that names a
    license, else the distinct license tokens joined, else the first sentence."""
    if not isinstance(quote, str) or not quote.strip():
        return ""
    first = _first_sentences(quote, 1).strip()
    if len(first) <= 160 and _LICENSE_TOKEN_RE.search(first):
        return first
    hits: List[str] = []
    for m in _LICENSE_TOKEN_RE.finditer(quote):
        tok = re.sub(r"\s+", " ", m.group(0)).strip()
        if tok not in hits:
            hits.append(tok)
    return "; ".join(hits) if hits else first


# A strong code-benchmark signal -- deliberately stricter than _CODE_EXEC_RE (which matches
# incidental 'programmatic'/'execute') so a non-code benchmark never gets a code languages shape.
_STRONG_CODE_RE = re.compile(
    r"\bprogramming\b|\bcoding\b|code generation|code completion|source code"
    r"|software engineering|\bexercism\b|pass@\s*\d|\bunit test", re.I)


def _is_code_benchmark(card: Dict[str, Any], eee_metadata: Optional[Dict[str, Any]]) -> bool:
    """Strong code-benchmark signal: data_type already 'code', a pass@k metric, or a curated
    strong-code phrase in overview/methods/EEE description."""
    bd = card.get("benchmark_details") or {}
    if str(bd.get("data_type", "")).strip().lower() == "code":
        return True
    meth = card.get("methodology") or {}
    metrics = meth.get("metrics") if isinstance(meth.get("metrics"), list) else []
    if any(isinstance(m, str) and m.lower().startswith("pass@") for m in metrics):
        return True
    blob = " ".join([
        str(bd.get("overview", "")),
        " ".join(m for m in (meth.get("methods") or []) if isinstance(m, str)),
        " ".join(str((c or {}).get("evaluation_description", ""))
                 for c in ((eee_metadata or {}).get("metrics") or {}).values()
                 if isinstance(c, dict)),
    ])
    return bool(_STRONG_CODE_RE.search(blob))


def _languages_is_ns(value: Any) -> bool:
    """True when a languages value is missing / NS (a list of NS or an NS string)."""
    if value is None:
        return True
    if isinstance(value, list):
        return (not value) or all(_is_ns_str(v) for v in value)
    return _is_ns_str(value)


def apply_coverage_backstops(
    card: Dict[str, Any],
    provenance: Optional[Dict[str, Any]],
    eee_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """C3 reconciliation. Mutates card in place.
    Tier-1: lift a license that is NS in the card but present in provenance with a real source and
      substantive evidence. Allowlisted to data_licensing -- a widened all-field lift was tested
      across the d8 corpus and produced mostly type-mismatched re-lifts. Never touches
      overview/size/size_breakdown (C1/C2 own those).
    Tier-2: a code benchmark with NS languages gets ["not-applicable"] (contract S4.3 code shape;
      matches the HF-tag path so a no-HF variant is consistent with its sibling)."""
    # Tier-1: license lift.
    sec, field = "ethical_and_legal_considerations", "data_licensing"
    section = card.get(sec)
    if isinstance(section, dict) and _is_ns_str(section.get(field)):
        entry = ((provenance or {}).get(sec) or {}).get(field)
        if isinstance(entry, dict):
            src = str(entry.get("source", "")).strip().lower()
            ev = entry.get("evidence") or entry.get("quote")
            if src in ("paper", "hf", "html", "github", "eee") and not _is_marker_evidence(ev):
                value = _concise_license(ev if isinstance(ev, str) else "")
                if value:
                    section[field] = value
                    entry["status"] = "reconciled"
                    logger.info("Coverage backstop: lifted data_licensing from %s provenance", src)

    # Tier-2: code-benchmark languages.
    bd = card.get("benchmark_details")
    if isinstance(bd, dict) and _languages_is_ns(bd.get("languages")) and \
            _is_code_benchmark(card, eee_metadata):
        bd["languages"] = ["not-applicable"]
        logger.info("Coverage backstop: code benchmark -> languages not-applicable")


def apply_display_fields(
    card: Dict[str, Any],
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    eee_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fill display-only fields (authors, logo, org_url) from structured metadata.

    Display-only: shown to humans, excluded from every measurement (contract S8). Best
    effort and non-blocking. authors <- HF card_data person list, else the resolved paper's
    authors (extracted_ids), else an HF-README BibTeX whose citation is identity-consistent;
    org_url <- the dataset's HF org page; logo <- HF avatar / card image, else the org/user
    overview avatar or github.com/<org>.png (only real fetched URLs, never fabricated). When
    provenance is given, a derived sidecar entry is written for each field filled here
    (source = paper/hf/github, status = derived). eee_metadata supplies the authoritative
    benchmark identity used to gate the HF-README BibTeX authors."""
    bd = card.setdefault("benchmark_details", {})
    if not isinstance(bd, dict):
        return card

    def _record(field: str, value: Any, source: str, verified: bool = True) -> None:
        if provenance is None:
            return
        provenance.setdefault("benchmark_details", {})[field] = {
            "source": source,
            "evidence": value if isinstance(value, str) else ", ".join(map(str, value)),
            "quote_loc": {"doc": source, "char_start": None},
            "status": "derived",
            "verified": verified,
            "evidence_ids": [],
        }

    meta = _get_hf_meta(hf_metadata)
    card_data = meta.get("card_data", {}) if isinstance(meta, dict) else {}

    if not bd.get("authors"):
        authors, src = None, None
        # HF card_data: accept only a real person list (never the org handle in `author`).
        if isinstance(card_data, dict) and _looks_like_person_list(card_data.get("authors")):
            authors, src = card_data.get("authors"), "hf"
        if not authors:
            authors = (extracted_ids or {}).get("paper_authors")
            src = "paper"
        # HF readme BibTeX author list, behind HF card_data + resolver, before NS. Gated on benchmark
        # identity: a derivative dataset's README often carries its PARENT's bibtex (mgsm's README is
        # the GSM8K bibtex cobbe2021gsm8k), which would leak the wrong authors. Take the bibtex authors
        # only when the bibtex key/title shares a distinctive acronym/slug token with the EEE identity.
        if not authors:
            readme_md = meta.get("readme_markdown")
            bib = _parse_bibtex_authors(readme_md)
            if _looks_like_person_list(bib):
                bib_key, bib_title = _bibtex_identity(readme_md)
                id_tokens = _distinctive_identity_tokens(eee_metadata, bd.get("overview"))
                if _bibtex_matches_identity(bib_key, bib_title, id_tokens):
                    authors, src = bib, "hf"
        if isinstance(authors, str):
            authors = [authors]
        if isinstance(authors, list) and authors:
            cleaned = [str(a).strip() for a in authors if str(a).strip()]
            if cleaned:
                bd["authors"] = cleaned
                _record("authors", cleaned, src)

    org = None
    hf_repo = (extracted_ids or {}).get("hf_repo")
    if isinstance(hf_repo, str) and "/" in hf_repo:
        org = hf_repo.split("/", 1)[0].strip()
        # org_url only when the owner is a confirmed HF org; a user handle (404) -> NS, never
        # the personal HF profile.
        if org and not bd.get("org_url") and _hf_org_overview(org) is not None:
            bd["org_url"] = f"https://huggingface.co/{org}"
            _record("org_url", bd["org_url"], "hf")

    if not bd.get("logo"):
        for _k in ("avatarUrl", "avatar_url", "avatar", "logo"):
            val = meta.get(_k) if isinstance(meta, dict) else None
            if isinstance(val, str) and val.startswith("http") and not _is_identicon_avatar(val):
                bd["logo"] = val
                _record("logo", val, "hf")
                break
    # HF card image (dataset card YAML); only a real URL, never fabricated
    if not bd.get("logo") and isinstance(card_data, dict):
        for _k in ("thumbnail", "image", "logo"):
            val = card_data.get(_k)
            if isinstance(val, str) and val.startswith("http") and not _is_identicon_avatar(val):
                bd["logo"] = val
                _record("logo", val, "hf")
                break
    # Last resort: fetch the org avatar (HF overview) or github.com/<org>.png. Org-only, so a
    # user-owned dataset gets no logo. A fetched logo is never source-verified -> verified=False.
    if not bd.get("logo") and org:
        fetched = _fetch_org_logo(org)
        if fetched:
            bd["logo"] = fetched
            _record("logo", fetched, "github" if "github.com" in fetched else "hf", verified=False)

    card["benchmark_details"] = bd
    return card


def build_eee_prefill_block(det_facts: Dict[str, Any]) -> str:
    """Render the EEE evaluation-results and composite-suite blocks for Stage B.

    Reconstructs the [EEE] / composite [DETERMINISTIC] blocks that merge_extracted_facts
    used to produce, so EEE model scores and suite context survive once that merge
    function is removed ([A] owns the deletion). Source = eee. Returns '' when there is
    no EEE data. Stage B (owned by [B+V]) injects this into the methodology prefill.
    """
    blocks = []

    composite_ctx = det_facts.get("composite_context", "")
    if det_facts.get("benchmark_type") == "composite" and composite_ctx:
        blocks.append(f"[DETERMINISTIC] {composite_ctx}")

    eval_sum = det_facts.get("evaluation_summary")
    if isinstance(eval_sum, dict):
        lines = []
        metric = eval_sum.get("primary_metric", "score")
        for p in eval_sum.get("top_performers", [])[:5]:
            try:
                lines.append(f"- {p['model']}: {p['score']:.4f} ({metric})")
            except (KeyError, TypeError, ValueError):
                continue
        stats = eval_sum.get("score_statistics", {})
        if stats:
            lines.append(
                f"- Score statistics: mean={stats.get('mean')}, "
                f"std={stats.get('std_dev')}, range=[{stats.get('min')}, {stats.get('max')}]"
            )
        n = eval_sum.get("total_models_evaluated", 0)
        if n:
            lines.append(f"- Total models evaluated: {n}")
        if lines:
            blocks.append("[EEE] Evaluation results from Every Eval Ever:\n" + "\n".join(lines))

    return "\n\n".join(blocks)


def compute_card_confidence(
    has_paper: bool,
    has_hf_readme: bool,
    has_eee: bool,
    has_hf_basic: bool = False,
    has_html: bool = False,
) -> Dict[str, Any]:
    """Compute confidence level (high/medium/low) based on available sources."""
    if has_paper and (has_hf_readme or has_hf_basic or has_html):
        level = "high"
    elif has_paper or (has_hf_readme and has_eee) or (has_html and has_eee):
        level = "medium"
    else:
        level = "low"

    return {
        "confidence_level": level,
        "sources_available": {
            "paper": has_paper,
            "html": has_html,
            "hf_readme": has_hf_readme,
            "hf_basic_metadata": has_hf_basic,
            "eee": has_eee,
        },
    }


def _compact_hf_metadata(hf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trimmed subset of HF metadata relevant to composition."""
    meta = _get_hf_meta(hf_metadata)

    compact: Dict[str, Any] = {}
    for key in ("id", "tags", "license", "downloads", "likes"):
        if key in meta:
            compact[key] = meta[key]

    if "card_data" in meta and meta["card_data"]:
        compact["card_data"] = meta["card_data"]

    if "readme_markdown" in meta and meta["readme_markdown"]:
        compact["readme_excerpt"] = meta["readme_markdown"][:1500]

    return compact


def _compact_eee_metadata(eee_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trimmed subset of EEE metadata relevant to composition."""
    compact: Dict[str, Any] = {}

    if eee_metadata.get("benchmark_name"):
        compact["benchmark_name"] = eee_metadata["benchmark_name"]
    if eee_metadata.get("eval_library"):
        compact["eval_library"] = eee_metadata["eval_library"]
    if eee_metadata.get("source_urls"):
        compact["source_urls"] = eee_metadata["source_urls"][:5]

    metrics = eee_metadata.get("metrics", {})
    if metrics:
        # Hide the degenerate benchmark_score wrapper from the LLM context too, so it cannot
        # echo the proxy id into prose; keep it only if dropping it empties the set.
        real = {k: v for k, v in metrics.items()
                if not _is_degenerate_metric(k, v if isinstance(v, dict) else {})}
        metrics = real or metrics
        compact["metrics"] = {
            k: {
                "description": v.get("evaluation_description", ""),
                "lower_is_better": v.get("lower_is_better", False),
                "score_type": v.get("score_type", ""),
            }
            for k, v in list(metrics.items())[:10]
        }

    eval_summary = eee_metadata.get("evaluation_summary", {})
    if eval_summary:
        compact["evaluation_summary"] = {
            "total_models": eval_summary.get("total_models_evaluated", 0),
            "primary_metric": eval_summary.get("primary_metric", ""),
            "score_statistics": eval_summary.get("score_statistics", {}),
            "top_performers": eval_summary.get("top_performers", [])[:5],
        }

    return compact


class BenchmarkDetails(BaseModel):

    name: str = Field(
        ...,
        description="The official name of the benchmark as it appears in literature",
    )
    overview: str = Field(
        ...,
        description="2-3 sentence description of what the benchmark measures and its key characteristics",
    )
    data_type: str = Field(
        ...,
        description="The primary data modality (e.g., text, image, audio, multimodal, tabular)",
    )
    domains: List[str] = Field(
        ...,
        description="Specific application domains or subject areas (e.g., medical, legal, scientific, conversational AI)",
    )
    languages: List[str] = Field(
        ...,
        description="All languages supported in the dataset using full language names (e.g., 'English', 'Chinese', 'Spanish', 'Multilingual')",
    )
    similar_benchmarks: List[str] = Field(
        ...,
        description="Names of closely related or comparable benchmarks that measure similar capabilities",
    )
    resources: List[str] = Field(
        ...,
        description="URLs to official papers, datasets, leaderboards, and documentation",
    )
    authors: Optional[List[str]] = None
    logo: Optional[str] = None
    org_url: Optional[str] = None
    provenance: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )

    @field_validator("domains", "languages", "similar_benchmarks", "resources", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v):
        # LLM sometimes returns "Not specified" as a bare string for list fields
        return [v] if isinstance(v, str) else v


class PurposeAndIntendedUsers(BaseModel):

    goal: str = Field(
        ...,
        description="The primary objective and research question this benchmark addresses, including what capabilities or behaviors it aims to measure",
    )
    audience: List[str] = Field(
        ...,
        description="Who the benchmark is designed for -- who runs it and who uses its results (controlled vocabulary; see field spec)",
    )
    tasks: List[str] = Field(
        ...,
        description="Specific evaluation tasks or subtasks the benchmark covers (e.g., 'question answering', 'code generation', 'factual accuracy')",
    )
    limitations: str = Field(
        ...,
        description="Known limitations, biases, or constraints of the benchmark that users should be aware of",
    )
    out_of_scope_uses: List[str] = Field(
        ...,
        description="Explicit examples of inappropriate or unsupported use cases for this benchmark",
    )
    provenance: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )

    @field_validator("audience", "tasks", "out_of_scope_uses", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v):
        return [v] if isinstance(v, str) else v


class DataInfo(BaseModel):

    source: str = Field(
        ...,
        description="Detailed information about data origins, collection methods, and any preprocessing steps applied",
    )
    size: str = Field(
        ...,
        description="Dataset size. Prefer number of examples from paper (e.g., '817 questions'). "
        "If only disk size from HuggingFace is available, use that (e.g., '1.24 GB')",
    )
    format: str = Field(
        ...,
        description="The data format as described in the paper or README "
        "(e.g., 'JSON with question-answer pairs'). If only the HuggingFace hosting "
        "format is known, note it as such (e.g., 'parquet (HuggingFace hosting format)')",
    )
    annotation: str = Field(
        ...,
        description="Annotation methodology, quality control measures, inter-annotator agreement, and any human involvement in labeling",
    )
    size_breakdown: Union[Dict[str, int], str] = Field(
        default="Not specified",
        description="Per-subset or per-domain item counts as {subset_name: count}; 'Not specified' if not reported",
    )
    collection_date: str = Field(
        default="Not specified",
        description="Data collection or cutoff date/range (e.g. '2023-06' or '2022-2024'); distinct from the card creation date",
    )
    contamination_controls: str = Field(
        default="Not specified",
        description="Train-test leakage protections described in the source: canary string, held-out/private split, encryption",
    )
    provenance: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class Methodology(BaseModel):

    methods: List[str] = Field(
        ...,
        description="Evaluation approaches and techniques applied within the benchmark (e.g., 'zero-shot evaluation', 'few-shot prompting', 'fine-tuning')",
    )
    metrics: List[str] = Field(
        ...,
        description="Specific quantitative metrics used, named in readable form (e.g., 'accuracy', "
        "'F1-score', 'BLEU', 'exact match'). Do not emit a bare '<benchmark>.score' proxy; use "
        "'Not specified' when the source names no real metric",
    )
    calculation: str = Field(
        ...,
        description="Detailed explanation of how metrics are computed, including any normalization or aggregation methods",
    )
    interpretation: str = Field(
        ...,
        description="What scores mean in practice: the score scale and direction (whether higher or "
        "lower is better) and concrete reference points drawn from the reported baseline or human "
        "results; do not invent thresholds the source does not state",
    )
    baseline_results: str = Field(
        ...,
        description="Performance of established models or baselines, with specific numbers and context for comparison",
    )
    validation: str = Field(
        ...,
        description="Quality assurance measures, validation procedures, and steps taken to ensure reliable and reproducible evaluations",
    )
    human_baseline: str = Field(
        default="Not specified",
        description="Human or reference baseline result plus its conditions (time limit, tool/web access, expertise); distinct from baseline_results (model scores)",
    )
    judge_uses_llm: Union[bool, str] = Field(
        default="Not specified",
        description="Whether scoring uses an LLM-as-judge (true/false); 'Not specified' if not reported",
    )
    judge_num: Union[int, str] = Field(
        default="Not specified",
        description="Number of judge models used for LLM-as-judge scoring; 'Not specified' if not reported",
    )
    judge_models: List[str] = Field(
        default_factory=lambda: ["Not specified"],
        description="Names of the judge model(s) used for scoring",
    )
    judge_score_consolidation: str = Field(
        default="Not specified",
        description="How multiple judge scores are consolidated (e.g. majority vote, mean, single judge)",
    )
    validity_justification: str = Field(
        default="Not specified",
        description="What the paper states about construct validity or justification of its metric choice (extract-only; distinct from validation)",
    )
    provenance: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )

    @field_validator("methods", "metrics", "judge_models", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v):
        return [v] if isinstance(v, str) else v


class EthicalAndLegalConsiderations(BaseModel):

    privacy_and_anonymity: str = Field(
        ...,
        description="Data protection measures, anonymization or de-identification techniques, and "
        "handling of personally identifiable information; or an explicit statement that the data "
        "contains no personal information or is synthetic",
    )
    data_licensing: str = Field(
        ...,
        description="Specific license terms, usage restrictions, and redistribution permissions",
    )
    consent_procedures: str = Field(
        ...,
        description="How human participants or annotators consented and were compensated: informed "
        "consent, the crowdsourcing platform, participant rights, and withdrawal procedures",
    )
    compliance_with_regulations: str = Field(
        ...,
        description="Adherence to relevant regulations (GDPR, IRB approval, etc.) and ethical review processes",
    )
    provenance: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class BenchmarkCard(BaseModel):

    benchmark_details: BenchmarkDetails
    purpose_and_intended_users: PurposeAndIntendedUsers
    data: DataInfo
    methodology: Methodology
    ethical_and_legal_considerations: EthicalAndLegalConsiderations


SECTION_MODELS = {
    "benchmark_details": BenchmarkDetails,
    "purpose_and_intended_users": PurposeAndIntendedUsers,
    "data": DataInfo,
    "methodology": Methodology,
    "ethical_and_legal_considerations": EthicalAndLegalConsiderations,
}


_JUDGE_DETAIL_FIELDS = (
    "methodology.judge_num",
    "methodology.judge_models",
    "methodology.judge_score_consolidation",
)


def drop_inapplicable_judge_missing(card: Dict[str, Any], missing: Any) -> Any:
    """Remove judge-setup detail fields from a displayed missing_fields list when the
    benchmark does not use an LLM judge (judge_uses_llm is exactly False) -- they are
    not-applicable, not genuinely missing. Presentation-only; the frozen
    extract_missing_fields is unchanged. No-op when judge_uses_llm is True or unknown (NS)."""
    methodology = card.get("methodology")
    if not isinstance(methodology, dict) or methodology.get("judge_uses_llm") is not False:
        return missing
    if not isinstance(missing, list):
        return missing
    return [m for m in missing if m not in _JUDGE_DETAIL_FIELDS]


def enforce_card_shape(card: Dict[str, Any]) -> Dict[str, Any]:
    """Insert every section-model field that exclude_none dropped, so all cards share one
    key-set. List fields -> ["Not specified"], scalars/Unions -> "Not specified". Fills only
    absent keys (idempotent, never clobbers a value); skips provenance."""
    for section, model in SECTION_MODELS.items():
        sec = card.get(section)
        if not isinstance(sec, dict):
            continue
        for fname in model.model_fields:
            if fname == "provenance" or fname in sec:
                continue
            sec[fname] = _validator._canonical_ns(fname)
    return card


def extract_provenance(section_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split provenance metadata out of section data."""
    clean_data = dict(section_data)
    provenance = clean_data.pop("provenance", None) or {}
    return clean_data, provenance


def _build_group_model(name, section_names, section_models):
    """A transport-only wrapper composing the existing section models by reference."""
    fields = {sec: (section_models[sec], ...) for sec in section_names}
    return create_model(f"Group_{name}", **fields)


def _digest_block(evidence_digest, section_names):
    lines = []
    for sec in section_names:
        items = [it for key, its in evidence_digest.items()
                 if key == sec or key.startswith(sec + ".") for it in its]
        if not items:
            continue
        lines.append(f"## {sec}")
        for it in items:
            lines.append(
                f"- [{it.get('evidence_id', '?')}] "
                f"({it.get('field', sec)} | {it.get('doc', '?')}/{it.get('kind', 'stated')}) "
                f"{it.get('quote', '')}"
            )
    return "\n".join(lines) if lines else "(no evidence items for this group)"


def _established_block(det_facts, section_names):
    lines = []
    for key, val in (det_facts or {}).items():
        if isinstance(key, str) and "." in key and key.split(".", 1)[0] in section_names:
            lines.append(f"- {key} = {val}")
    if "methodology" in section_names and isinstance(det_facts, dict):
        # Seam 2: surface EEE eval results + composite context so EEE baselines survive
        # (build_eee_prefill_block replaces the line merge_extracted_facts used to emit).
        eee_block = build_eee_prefill_block(det_facts)
        if eee_block:
            lines.append(eee_block)
    return "\n".join(lines) if lines else "(nothing pre-filled for this group)"


_GLOSS_NOISE_RE = re.compile(r"\s*\([^()]*(?:controlled vocabulary|see field spec)[^()]*\)")


def _field_gloss(model_field) -> str:
    """A short role gloss from a field's own description; strips prompt-machinery
    parentheticals (vocab / field-spec asides) so the gloss carries field meaning."""
    desc = (getattr(model_field, "description", None) or "").strip()
    if not desc:
        return ""
    return " ".join(_GLOSS_NOISE_RE.sub("", desc).split())


def _field_spec_block(section_models, section_names, det_facts):
    lines = []
    for sec in section_names:
        lines.append(f"## {sec}")
        for fname, model_field in section_models[sec].model_fields.items():
            if fname == "provenance":
                continue
            path = f"{sec}.{fname}"
            if path in fs.DISPLAY_FIELDS:
                continue
            entry = f"- {fname} [{fs.field_class(path)}]"
            vocab = fs.enum_vocab(path)
            if path in fs.B_AUTHORED_ENUMS and vocab:
                entry += ": one or more of {" + ", ".join(sorted(vocab)) + "} or 'other:<text>'"
            elif path in fs.ESTABLISHED_FIELDS and path in det_facts:
                # only claim "already established" when a deterministic value is actually
                # shown above; otherwise B echoes the literal "already established" as the value
                entry += ": already established (copy the value shown above verbatim)"
            # A tuned FIELD_CAPS entry wins; otherwise fall back to the field's own
            # description as a generic role gloss.
            cap = fs.FIELD_CAPS.get(path)
            if cap:
                entry += f" -- {cap}"
            else:
                gloss = _field_gloss(model_field)
                if gloss:
                    entry += f" -- {gloss}"
            # Form guidance for any list-valued field: enumerate atomic items.
            if fname in fs.LIST_FIELDS:
                entry += (" -- enumerate atomic items, one discrete item per array element; "
                          "do NOT collapse into a summary sentence")
            lines.append(entry)
    return "\n".join(lines)


_STAGE_B_RULES = (
    "RULES:\n"
    "1. Use ONLY the evidence items below. No outside knowledge. No source text is shown to you on "
    "purpose; work from the quotes only.\n"
    "2. If a field has no supporting evidence and is not already established, output exactly \"Not specified\".\n"
    "3. Controlled (enum) fields: pick from the allowed vocabulary; if nothing fits use \"other:<short text>\".\n"
    "4. Prose (non-list) fields: 2-3 sentences, third person, your own synthesis of the cited "
    "quotes; no vocabulary.\n"
    "4b. List fields: enumerate atomic items, one discrete item per array element; do NOT collapse "
    "them into a single summary sentence. Each element is one item, not a sentence about the set.\n"
    "5. judge_* fields are decomposed from the single methodology.judge_setup evidence: set judge_uses_llm "
    "(true/false), judge_num (count), judge_models (list of BARE model identifiers only, e.g. GPT-4o or "
    "'self'; any judging-policy prose goes in judge_score_consolidation, never in judge_models), "
    "judge_score_consolidation (how scores combine). "
    "Set judge_uses_llm=true ONLY if the evidence describes an LLM scoring/grading MODEL OUTPUTS; evidence "
    "about how the dataset was built or annotated (e.g. human annotators, data construction) is NOT a judge.\n"
    "6. PROVENANCE (required for every filled, non-\"Not specified\", non-established field): add "
    "provenance[field] = {\"source\": \"<doc>\", \"evidence\": \"<the quote>\", \"evidence_ids\": [\"E..\"]}. "
    "Cite only evidence ids that appear in the evidence items; never invent ids.\n"
    "7. Never output an evidence id (e.g. E01) as a field value; evidence ids belong only in "
    "provenance.evidence_ids."
)


def _build_group_prompt(group_name, section_names, section_models, evidence_digest,
                        det_facts, composite_guidance):
    sec_label = ", ".join(section_names)
    system = (
        f"You are formatting verified evidence into the {sec_label} section(s) of a benchmark card. "
        f"You are a FORMATTER, not a researcher.\n"
        f"{composite_guidance}\n"
        f"{_STAGE_B_RULES}\n\n"
        f'Return ONE JSON object with a key per section; each section is an object of its fields plus a '
        f'"provenance" object.'
    )
    user = (
        f"=== FIELD SPECS ===\n{_field_spec_block(section_models, section_names, det_facts)}\n\n"
        f"=== ALREADY ESTABLISHED (copy, do not re-author) ===\n"
        f"{_established_block(det_facts, section_names)}\n\n"
        f"=== EVIDENCE ITEMS ===\n{_digest_block(evidence_digest, section_names)}\n\n"
        f"Produce JSON for: {sec_label}."
    )
    return system + "\n\n" + user


def _aggregate_validation(group_telemetry):
    groups = list(group_telemetry.values())
    n = len(groups) or 1
    n_other = sum(g.get("n_other", 0) for g in groups)
    n_enum = sum(g.get("n_enum_fields", 0) for g in groups)
    return {
        "first_pass_validity": sum(1 for g in groups if g.get("first_pass_valid")) / n,
        "iterations_to_valid_mean": sum(g.get("iterations_to_valid", 1) for g in groups) / n,
        "any_group_unconverged": any(not g.get("converged") for g in groups),
        "other_rate": (n_other / n_enum) if n_enum else 0.0,
        "ns_normalized": sum(g.get("ns_normalized", 0) for g in groups),
        "evidence_demoted": sum(g.get("evidence_demoted", 0) for g in groups),
        "established_mismatch": sum(g.get("established_mismatch", 0) for g in groups),
        "schema_invalid": sorted({p for g in groups for p in g.get("schema_invalid", [])}),
    }


@tool("compose_benchmark_card")
def compose_benchmark_card(
    unitxt_metadata: Optional[Dict[str, Any]] = None,
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    docling_output: Optional[Dict[str, Any]] = None,
    query: str = "",
    eee_metadata: Optional[Dict[str, Any]] = None,
    html_content: Optional[Dict[str, Any]] = None,
    github_readme: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose a benchmark card from collected metadata using source-isolated extraction."""

    logger.info("Composing benchmark card for: %s", query)

    has_paper = bool(docling_output and docling_output.get("success"))
    has_hf = bool(hf_metadata)
    has_hf_readme = bool(_get_hf_readme(hf_metadata))
    has_eee = bool(eee_metadata)
    has_html = bool(html_content and html_content.get("success"))
    has_github = bool(github_readme and github_readme.get("success"))
    has_abstract_fallback = not has_paper and bool(extracted_ids and extracted_ids.get("paper_abstract"))

    source_list = []
    if has_paper:
        source_list.append("Paper")
    elif has_abstract_fallback:
        source_list.append("Paper-abstract")
    if has_html:
        source_list.append("HTML")
    if has_github:
        source_list.append("GitHub-README")
    if has_hf_readme:
        source_list.append("HF-README")
    elif has_hf:
        source_list.append("HF-basic")
    if has_eee:
        source_list.append("EEE")
    logger.info("Available sources: %s", ", ".join(source_list) or "none")

    # Phase 1: Collect paper content (intro + RAG supplement)
    paper_retriever = None
    paper_vectorstore = None
    all_paper_content = ""

    if has_paper:
        paper_text = _normalize_docling_text(docling_output.get("filtered_text", ""))
        if paper_text:
            budget = Config.PAPER_EXTRACTION_BUDGET
            intro_chars = Config.PAPER_INTRO_CHARS

            intro_text = paper_text[:intro_chars]

            if len(paper_text) <= budget:
                all_paper_content = paper_text
                logger.info("Paper fits in budget (%d chars), using full text", len(paper_text))
            else:
                try:
                    if Config.DEFAULT_EMBEDDING_MODEL == "bge-large":
                        embeddings = HuggingFaceEmbeddings(
                            model_name="BAAI/bge-large-en-v1.5",
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                    elif Config.DEFAULT_EMBEDDING_MODEL == "e5-large":
                        embeddings = HuggingFaceEmbeddings(
                            model_name="intfloat/e5-large-v2",
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                    else:
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200,
                        separators=["\n\n", "\n", ". ", " "],
                    )
                    chunks = splitter.split_text(paper_text)
                    documents = [
                        Document(page_content=c, metadata={"chunk_idx": i})
                        for i, c in enumerate(chunks)
                    ]
                    # Unique per-benchmark collection: this inline paper-RAG path does
                    # not go through RAGRetriever, so without a name it would share
                    # chromadb's process-default "langchain" collection and a later
                    # benchmark could retrieve this one's paper chunks.
                    collection_name = sanitize_chroma_collection_name(query, suffix=uuid4().hex[:8])
                    paper_vectorstore = Chroma.from_documents(
                        documents, embeddings, collection_name=collection_name
                    )
                    paper_retriever = paper_vectorstore.as_retriever(search_kwargs={"k": 5})
                    logger.debug("Paper indexed: %d chunks", len(chunks))

                    seen = set()
                    rag_chunks = []
                    for queries in SECTION_QUERIES.values():
                        for sq in queries:
                            for chunk in paper_retriever.invoke(sq):
                                key = chunk.page_content[:200]
                                if key in seen or chunk.page_content[:100] in intro_text:
                                    continue
                                seen.add(key)
                                rag_chunks.append(chunk)

                    formatted = [f"[Abstract and Introduction]\n{intro_text}"]
                    used = len(intro_text)
                    for i, chunk in enumerate(rag_chunks, 1):
                        text = chunk.page_content
                        if used + len(text) > budget:
                            remaining = budget - used
                            if remaining > 100:
                                formatted.append(f"[Section {i}]\n{text[:remaining]}")
                            break
                        formatted.append(f"[Section {i}]\n{text}")
                        used += len(text)
                    all_paper_content = "\n\n".join(formatted)
                    logger.info(
                        "Paper: %d intro chars + %d RAG chunks, %d total chars for extraction",
                        len(intro_text), len(rag_chunks), used,
                    )
                except Exception as e:
                    logger.warning("Paper retrieval failed: %s", e)

            if not all_paper_content and paper_text:
                all_paper_content = paper_text[:budget]
                logger.info("Using raw paper text (truncated to %d chars)", budget)

    # Fallback: use paper abstract from resolver when Docling failed/skipped
    if not all_paper_content and has_abstract_fallback:
        abstract = extracted_ids["paper_abstract"]
        title = extracted_ids.get("paper_title", "")
        year = extracted_ids.get("paper_year", "")
        header = f"Paper: {title}" + (f" ({year})" if year else "")
        all_paper_content = f"{header}\n\nAbstract: {abstract}"
        logger.info("Using paper abstract as fallback (%d chars)", len(all_paper_content))

    # Extract paper title for name alias (helps with name mismatches like tau_bench_2 vs τ-bench)
    paper_title = ""
    if has_paper:
        paper_title = docling_output.get("metadata", {}).get("title", "")
    elif has_abstract_fallback and extracted_ids:
        paper_title = extracted_ids.get("paper_title", "")

    # Phase 2: Source-isolated verbatim-quote extraction (verified Evidence Items).
    # Phase 3 (gap-fill) reuses the inline paper-RAG collection, so it must outlive
    # the indexing block above; the finally frees it once gap-fill is done.
    try:
        identity_anchor = _get_benchmark_identity(query, hf_metadata, eee_metadata, paper_title=paper_title)
        if identity_anchor:
            logger.info("Benchmark identity: %s", identity_anchor)

        # Paper and gap-fill quotes are verified against the full paper text so they share one
        # docling reference; the model itself only ever reads all_paper_content.
        full_paper_text = ""
        if has_paper:
            full_paper_text = _normalize_docling_text(docling_output.get("filtered_text", ""))
        paper_source = full_paper_text or all_paper_content

        quote_telem = evidence.new_card_telemetry()

        paper_items = []
        if all_paper_content:
            # Label by actual source: abstract-only content must not claim docling.
            paper_doc = evidence.DOC_PAPER if has_paper else evidence.DOC_ABSTRACT
            paper_items, t = extract_facts_from_paper(
                all_paper_content, query, identity_anchor, paper_title=paper_title,
                verify_source=paper_source, doc=paper_doc,
            )
            evidence.merge_telemetry(quote_telem, t)

        hf_items = []
        if has_hf_readme:
            hf_items, t = extract_facts_from_hf_readme(hf_metadata, query, identity_anchor=identity_anchor)
            evidence.merge_telemetry(quote_telem, t)

        html_items = []
        if has_html:
            html_items, t = extract_facts_from_html(html_content, query, identity_anchor=identity_anchor)
            evidence.merge_telemetry(quote_telem, t)

        github_items = []
        if has_github:
            github_items, t = extract_facts_from_github_readme(github_readme, query, identity_anchor=identity_anchor)
            evidence.merge_telemetry(quote_telem, t)

        det_facts = extract_deterministic_facts(eee_metadata, hf_metadata, extracted_ids, unitxt_metadata)

        # Phase 3: Gap-fill paper fields that have no verified evidence yet
        gap_items = []
        if paper_items and paper_retriever:
            gap_items, t = _fill_paper_gaps(
                paper_items, paper_retriever, query, paper_source, identity_anchor=identity_anchor
            )
            evidence.merge_telemetry(quote_telem, t)
    finally:
        # Inline paper-RAG has no RAGRetriever.cleanup(); drop its collection so a
        # later benchmark in the same process can't retrieve these paper chunks.
        if paper_vectorstore is not None:
            try:
                paper_vectorstore.delete_collection()
            except Exception as e:
                logger.warning("Failed to delete paper RAG collection: %s", e)
            paper_vectorstore = None
            paper_retriever = None

    # Phase 4: Assign evidence ids and build the per-field digest (the only Stage-B input).
    # evidence_digest replaces merged_facts; the Stage-B section loop below reads
    # evidence_digest[field] instead of merged_facts.get(section) (handoff to [B+V]).
    evidence_items, evidence_digest = evidence.finalize_evidence(
        paper_items, gap_items, html_items, hf_items, github_items
    )
    logger.info(
        "Quote-verify: %d/%d quotes verified (%.0f%%); rejected=%s truncated=%d",
        quote_telem["verified"], quote_telem["emitted"],
        100.0 * evidence.verify_rate(quote_telem),
        quote_telem["reject_reasons"], quote_telem["truncated"],
    )

    # Phase 5: Compose sections in grouped Stage-B calls, validated and repaired per group.
    section_models = SECTION_MODELS
    fs.assert_groups_cover()

    # Seam 1: Stage A's verified per-field digest (built above in Phase 4) is the only
    # Stage-B input; it replaces the legacy section-text adapter.
    digest_ids = {it["evidence_id"] for items in evidence_digest.values() for it in items}
    det_filled_paths = {k for k in det_facts if isinstance(k, str) and "." in k}

    composite_guidance = ""
    if eee_metadata and eee_metadata.get("benchmark_type") == "composite":
        contains_list = ", ".join(eee_metadata.get("contains", []))
        composite_guidance = (
            "\nCOMPOSITE BENCHMARK CONTEXT: this benchmark is a composite suite containing: "
            f"{contains_list}. Describe what the suite measures overall and how the sub-benchmarks "
            "fit together; suite-level scoring and coverage belong here, per-sub-benchmark detail "
            "belongs in the individual cards.\n"
        )

    handler = get_llm_handler()
    generated_sections = {}
    all_provenance = {}
    all_flagged = {}
    group_telemetry = {}

    for group_name, group_sections in fs.ACTIVE_GROUPS:
        logger.info("Composing group: %s (%s)", group_name, ", ".join(group_sections))
        group_model = _build_group_model(group_name, group_sections, section_models)
        group_schema = group_model.model_json_schema()
        base_prompt = _build_group_prompt(
            group_name, group_sections, section_models, evidence_digest, det_facts, composite_guidance)

        def _generate(prompt, max_tokens=None, _schema=group_schema, _handler=handler):
            return _handler.generate_with_meta(prompt, response_format=_schema,
                                               max_completion_tokens=max_tokens)

        outcome = _validator.run_group_with_repair(
            group_name, group_sections, section_models, base_prompt, _generate,
            digest_ids, det_filled_paths=det_filled_paths,
        )

        for sec in group_sections:
            clean_section, section_provenance = extract_provenance(outcome.sections[sec])
            generated_sections[sec] = clean_section
            if section_provenance:
                all_provenance[sec] = section_provenance
        all_flagged.update(outcome.flagged)
        group_telemetry[group_name] = outcome.telemetry
        logger.info(
            "Group %s: first_pass=%s iterations=%s converged=%s",
            group_name, outcome.telemetry.get("first_pass_valid"),
            outcome.telemetry.get("iterations_to_valid"), outcome.telemetry.get("converged"),
        )

    # Phase 5b: Aboutness guard. Structural, model-independent: demote judge_* / data.source
    # when the evidence behind them is off-topic (construction/annotation routed to judge,
    # subset-construction routed to data.source). The §1 verifier checks presence, not topic.
    aboutness_telem = aboutness.apply_aboutness_guard(
        generated_sections, all_provenance, evidence_items, all_flagged)
    if aboutness_telem["demoted_fields"]:
        logger.info("Aboutness guard demoted: %s (judge_branch=%s)",
                    aboutness_telem["demoted_fields"], aboutness_telem["judge_branch"])

    # Phase 6: Assemble card

    try:
        final_card = BenchmarkCard(
            benchmark_details=BenchmarkDetails(**generated_sections["benchmark_details"]),
            purpose_and_intended_users=PurposeAndIntendedUsers(
                **generated_sections["purpose_and_intended_users"]
            ),
            data=DataInfo(**generated_sections["data"]),
            methodology=Methodology(**generated_sections["methodology"]),
            ethical_and_legal_considerations=EthicalAndLegalConsiderations(
                **generated_sections["ethical_and_legal_considerations"]
            ),
        )
    except Exception as e:
        logger.error("Failed to assemble final benchmark card: %s", e)
        raise

    benchmark_card_dict = final_card.model_dump(exclude_none=True)
    for section_key in benchmark_card_dict:
        if isinstance(benchmark_card_dict[section_key], dict):
            benchmark_card_dict[section_key].pop("provenance", None)

    # Phase 7: Deterministic overrides
    benchmark_card_dict = apply_deterministic_overrides(benchmark_card_dict, det_facts)

    # Phase 8: Post-processing
    benchmark_card_dict = post_process_card(benchmark_card_dict)

    # C2: drop a size / size_breakdown the card's own prose contradicts (runs before the
    # deterministic-provenance block so a dropped value emits no stale source=deterministic entry).
    _enforce_numeric_consistency(benchmark_card_dict, det_facts, all_provenance)

    # F5-2: scrub any raw EEE metric id the model leaked into prose (a degenerate wrapper id
    # becomes its human metric_name or is dropped; a real scorer's id becomes its canonical).
    _scrub_leaked_metric_ids(benchmark_card_dict, eee_metadata)

    # G2-1: drop a fabricated baseline leaderboard -- any baseline_results sentence whose reported
    # scores are not traceable to a source the composer was given (the verified evidence quotes + the
    # EEE/deterministic numeric content). Numbers are matched across scales (fraction/percent) so a
    # reformatted real baseline survives; structural, never a model-name list.
    _drop_ungrounded_baselines(benchmark_card_dict, evidence_items, det_facts)

    # M3: repair V4-Flash generation artifacts in prose (stray control-word suffix, field-final
    # truncation) and flag glued-word fragments. Keyed on the artifact token pattern, never a card.
    _apply_generation_artifact_scrub(benchmark_card_dict)

    # C4: promote a slug-proxy metric to the canonical metric the methodology prose names. Only when
    # EVERY structured entry is an `other:` proxy AND the prose names a controlled-vocab metric in a
    # definition context -- never invents, never reads a benchmark name. Leaves canonical metrics and
    # defensible composites untouched.
    _meth = benchmark_card_dict.get("methodology")
    if isinstance(_meth, dict):
        _metrics = _meth.get("metrics")
        if isinstance(_metrics, list) and _metrics and all(
            isinstance(_m, str) and _m.startswith("other:") for _m in _metrics
        ):
            _calc = _meth.get("calculation") if isinstance(_meth.get("calculation"), str) else ""
            _methods = [_m for _m in (_meth.get("methods") or []) if isinstance(_m, str)]
            _promoted = _metric_from_prose(_calc) or _metric_from_prose(*_methods)
            if _promoted:
                _meth["metrics"] = _promoted
                all_provenance.setdefault("methodology", {})["metrics"] = {
                    "source": "paper",
                    "evidence": (_calc or " ".join(_methods))[:300],
                    "quote": (_calc or " ".join(_methods))[:300],
                    "quote_loc": {"doc": "paper", "char_start": None},
                    "status": "promoted",
                    "verified": True,
                    "evidence_ids": [],
                }
                logger.info("Metric promoted from prose: %s", _promoted)

    # Provenance for deterministically filled fields (source=deterministic). The quote
    # is the structured value itself and is verified by construction (contract S2). Only
    # emitted where the card value actually equals the deterministic value, i.e. the
    # deterministic write won over the LLM (always-override fields, or fill-only hits).
    _det_doc = {
        "benchmark_details.languages": "hf",
        "benchmark_details.data_type": "hf",
        "data.format": "hf",
        "data.size": "hf",
        "ethical_and_legal_considerations.data_licensing": "hf",
        "methodology.metrics": "eee",
        "data.size_breakdown": "hf",
        "benchmark_details.overview": "eee",
    }
    for _dotted, _doc in _det_doc.items():
        _entry = _deterministic_field_provenance(
            _dotted, _doc, benchmark_card_dict, det_facts, eee_metadata
        )
        if _entry is None:
            continue
        _sec, _, _field = _dotted.partition(".")
        all_provenance.setdefault(_sec, {})[_field] = _entry

    # Display-only fields (authors, logo, org_url): best-effort, post-composition,
    # excluded from every measurement (contract S8). Never blocks composition.
    benchmark_card_dict = apply_display_fields(
        benchmark_card_dict, hf_metadata, extracted_ids, all_provenance, eee_metadata)

    # A3: normalize a raw ALL-CAPS paper-title name to its short identity (no-op otherwise).
    _bd = benchmark_card_dict.get("benchmark_details")
    if isinstance(_bd, dict) and _bd.get("name"):
        _bd["name"] = _clean_caps_name(
            _bd["name"],
            (det_facts.get("benchmark_details.pretty_name"),
             (extracted_ids or {}).get("paper_title")),
        )

    # A9: collection_date must be a year, never prose; a non-year fragment is dropped to NS.
    _data = benchmark_card_dict.get("data")
    if isinstance(_data, dict) and "collection_date" in _data:
        _data["collection_date"] = _coerce_collection_date(_data.get("collection_date"))

    # F2-8: never leave the name NS when EEE carries the benchmark identity. The single-EEE
    # path otherwise loses the name when paper resolution fails; fall back to the EEE
    # benchmark_name (a deterministic pipeline input, grounded downstream).
    if isinstance(_bd, dict) and isinstance(eee_metadata, dict):
        _eee_name = eee_metadata.get("benchmark_name")
        if isinstance(_eee_name, str) and _eee_name.strip():
            _cur = _bd.get("name")
            # Prefer the EEE benchmark identity over an NS name or a paper-title-derived name
            # (the LLM sometimes returns the arXiv title instead of the benchmark identity).
            if _is_ns_str(_cur) or _name_is_paper_title(
                _cur, (extracted_ids or {}).get("paper_title")
            ):
                _bd["name"] = _eee_name.strip()

    # F2-7: de-duplicate exact-duplicate strings within the audience list (display hygiene).
    _pu = benchmark_card_dict.get("purpose_and_intended_users")
    if isinstance(_pu, dict) and isinstance(_pu.get("audience"), list):
        _pu["audience"] = _dedup_preserve_order(_pu["audience"])

    # A7: attach EEE metric_name/unit as human labels in the provenance sidecar, keyed by the
    # canonical token (unchanged) and the raw id. The card's methodology.metrics is untouched.
    _labels = build_metric_labels(eee_metadata)
    if _labels:
        all_provenance.setdefault("methodology", {})["metrics_labels"] = _labels

    # A4: derive each field's provenance.source from its cited evidence item's doc type.
    _normalize_provenance_sources(all_provenance, evidence_items)

    # Re-add every model field exclude_none dropped, so all cards share one key-set.
    enforce_card_shape(benchmark_card_dict)

    # C3: reconciliation backstops (runs after enforce_card_shape so all fields exist as NS).
    # Tier-1 lifts a license held in provenance; Tier-2 gives a code benchmark the not-applicable
    # languages shape. Excludes overview/size/size_breakdown so it can never re-lift a C1/C2 drop.
    apply_coverage_backstops(benchmark_card_dict, all_provenance, eee_metadata)

    # Record schema_invalid residuals from the Stage-B validator loop.
    if all_flagged:
        benchmark_card_dict.setdefault("flagged_fields", {}).update(all_flagged)

    # Phase 9: Confidence
    confidence = compute_card_confidence(has_paper, has_hf_readme, has_eee, has_hf, has_html=has_html)

    return {
        "benchmark_card": benchmark_card_dict,
        "provenance": all_provenance if all_provenance else None,
        "composition_metadata": {
            "sources_used": {
                "unitxt": bool(unitxt_metadata),
                "huggingface": bool(hf_metadata),
                "extracted_ids": bool(extracted_ids),
                "docling": has_paper,
                "paper_abstract_fallback": has_abstract_fallback,
                "html": has_html,
                "eee": has_eee,
            },
            "query": query,
            "composition_timestamp": datetime.now().isoformat(),
            "generation_method": "stage_b_grouped_validated",
            "model_used": get_llm_handler().model_name,
            "confidence": confidence,
            "quote_verify": quote_telem,
            "validation": {
                "per_group": group_telemetry,
                "card": _aggregate_validation(group_telemetry),
                "aboutness": aboutness_telem,
            },
        },
    }
