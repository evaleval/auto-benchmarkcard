"""Compose structured benchmark cards from heterogeneous metadata via LLM synthesis.

Architecture: Extract (isolated) -> Merge -> Compose -> Override -> Post-process
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from auto_benchmarkcard.config import Config, get_llm_handler

logger = logging.getLogger(__name__)

PAPER_EXTRACTION_PROMPT = """Read this research paper about the benchmark "{benchmark_name}" and describe what it says about each field below.

{identity_anchor}
CRITICAL RULES:
1. Write in your own words based on what the paper says. Do NOT copy text verbatim — rephrase clearly.
2. Base your descriptions ONLY on the paper text provided. Do NOT use knowledge from your training data.
3. IMPORTANT: You are extracting facts about "{benchmark_name}" ONLY. Do NOT confuse it with any other benchmark mentioned or discussed in the paper. If the paper discusses multiple benchmarks or datasets, only describe what pertains to "{benchmark_name}" itself.
4. If the paper says nothing about a field, write: "- No information found"
5. For each field, write 1-3 clear sentences that capture the key information.
6. Use precise numbers when the paper provides them, but integrate them naturally into your description.
7. If a fact seems to be about a different topic than "{benchmark_name}", do NOT include it.

Describe these fields:

## benchmark_details
- name: Official benchmark name and any acronym expansion
- overview: What does it measure? How many tasks/datasets? What makes it distinctive?
- data_type: Primary data modality (text, image, audio, multimodal, tabular)
- domains: Research domains or subject areas covered
- similar_benchmarks: ONLY benchmarks the paper explicitly compares to or names as related. If none, write "- No information found"
- resources: URLs mentioned in the paper (homepage, leaderboard, GitHub)

## purpose_and_intended_users
- goal: What is the primary research objective?
- audience: Who is this benchmark designed for?
- tasks: What specific evaluation tasks or sub-tasks does it include?
- limitations: What limitations, biases, or constraints does the paper acknowledge?
- out_of_scope_uses: What is the benchmark explicitly NOT designed for?

## data
- source: Where does the data come from and how was it collected?
- size: How many examples, and what are the train/dev/test splits?
- format: How is the data structured?
- annotation: How was labeling done, who annotated, what quality control was used?

## methodology
- methods: How are models evaluated (zero-shot, few-shot, fine-tuning, etc.)?
- metrics: What metrics are used (list each by name)?
- calculation: How is the overall score computed from individual scores?
- interpretation: What score ranges indicate strong vs weak performance?
- baseline_results: What specific model results does the paper report? (include model names and scores)
- validation: What quality assurance or validation procedures were used?

## ethical_and_legal_considerations
- privacy_and_anonymity: How is PII handled? Is data anonymized?
- consent_procedures: How were crowdworkers or annotators compensated? What platform was used?
- compliance_with_regulations: Was there IRB approval, GDPR compliance, or ethical review?

PAPER TEXT:
{paper_content}"""

HF_README_EXTRACTION_PROMPT = """Read this HuggingFace dataset page about the benchmark "{benchmark_name}" and describe what it says about each field below.

{identity_anchor}
CRITICAL RULES:
1. Write in your own words based on what the page says. Do NOT copy text verbatim — rephrase clearly.
2. Base your descriptions ONLY on the text provided. Do NOT use knowledge from your training data.
3. You are extracting facts about "{benchmark_name}" ONLY. Do NOT confuse it with any other benchmark or dataset.
4. If information is not found, write: "- No information found"
5. For each field, write 1-3 clear sentences capturing the key information.

Describe these fields:

## benchmark_details
- overview: What does this benchmark measure? What makes it distinctive?
- domains: What research domains or subject areas does it cover?

## purpose_and_intended_users
- goal: What is the purpose of this benchmark?
- tasks: What evaluation tasks does it include?
- limitations: What limitations are mentioned?

## data
- source: Where does the data come from and how was it collected?
- size: How many examples or how large is the dataset?
- annotation: How was the annotation process conducted?

## methodology
- methods: How are models evaluated?
- metrics: What metrics are used?
- baseline_results: What model scores or baselines are reported?

## ethical_and_legal_considerations
- Any ethical considerations mentioned

HUGGINGFACE DATASET PAGE:
{hf_content}"""

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
) -> str:
    """Build a short identity anchor so the LLM doesn't confuse benchmarks."""
    signals = []

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


def extract_facts_from_paper(
    paper_content: str, benchmark_name: str, identity_anchor: str = ""
) -> str:
    """Extract benchmark facts from paper text in isolation (no other sources)."""
    if not paper_content or paper_content == "Not available":
        return ""

    prompt = PAPER_EXTRACTION_PROMPT.format(
        benchmark_name=benchmark_name,
        paper_content=paper_content,
        identity_anchor=identity_anchor,
    )

    try:
        from auto_benchmarkcard.config import get_llm_handler
        llm_handler = get_llm_handler()
        facts = llm_handler.generate(f"{_EXTRACTOR_SYSTEM}\n\n{prompt}")
        logger.info("Paper extraction: %d chars of facts", len(facts))
        return facts
    except Exception as e:
        logger.warning("Paper fact extraction failed: %s", e)
        return ""


def extract_facts_from_hf_readme(
    hf_metadata: Dict[str, Any],
    benchmark_name: str,
    hf_retriever: Any = None,
    identity_anchor: str = "",
) -> str:
    """Extract benchmark facts from HF README in isolation (capped at 15K chars)."""
    readme = _get_hf_readme(hf_metadata)
    if not readme or len(readme) < 100:
        return ""

    hf_content = readme[:15000]

    prompt = HF_README_EXTRACTION_PROMPT.format(
        benchmark_name=benchmark_name,
        hf_content=hf_content,
        identity_anchor=identity_anchor,
    )

    try:
        from auto_benchmarkcard.config import get_llm_handler
        llm_handler = get_llm_handler()
        facts = llm_handler.generate(f"{_EXTRACTOR_SYSTEM}\n\n{prompt}")
        logger.info("HF README extraction: %d chars of facts (from %d chars source)", len(facts), len(hf_content))
        return facts
    except Exception as e:
        logger.warning("HF README fact extraction failed: %s", e)
        return ""


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


def extract_deterministic_facts(
    eee_metadata: Optional[Dict[str, Any]] = None,
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract ground-truth facts from structured sources (no LLM involved)."""
    facts: Dict[str, Any] = {}

    if hf_metadata:
        meta = _get_hf_meta(hf_metadata)
        tags = meta.get("tags", [])
        if isinstance(tags, list):
            languages = []
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("language:"):
                    code = tag.split(":", 1)[1].strip()
                    languages.append(LANG_MAP.get(code, code))
            if languages:
                facts["benchmark_details.languages"] = languages

            for tag in tags:
                if isinstance(tag, str) and tag.startswith("license:"):
                    facts["ethical_and_legal_considerations.data_licensing"] = tag.split(":", 1)[1].strip()
                    break

            for tag in tags:
                if isinstance(tag, str) and tag.startswith("size_categories:"):
                    facts["data.size_category"] = tag.split(":", 1)[1].strip()
                    break

            for tag in tags:
                if isinstance(tag, str) and tag.startswith("format:"):
                    facts["data.format"] = tag.split(":", 1)[1].strip()
                    break

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

        if "ethical_and_legal_considerations.data_licensing" not in facts:
            license_val = meta.get("license")
            if not license_val:
                card_data = meta.get("card_data", {})
                if isinstance(card_data, dict):
                    license_val = card_data.get("license")
            if license_val:
                facts["ethical_and_legal_considerations.data_licensing"] = license_val

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

    if eee_metadata:
        metrics = eee_metadata.get("metrics", {})
        if metrics:
            facts["methodology.metrics"] = list(metrics.keys())
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

    if extracted_ids:
        if extracted_ids.get("paper_url"):
            facts["benchmark_details.paper_url"] = extracted_ids["paper_url"]
        if extracted_ids.get("hf_repo"):
            hf_repo = extracted_ids["hf_repo"]
            facts["benchmark_details.hf_url"] = f"https://huggingface.co/datasets/{hf_repo}"

    logger.info("Deterministic facts: %d fields extracted", len(facts))
    return facts


def check_cross_contamination(
    extracted_facts: str,
    source_text: str,
    benchmark_name: str,
    identity_anchor: str = "",
) -> str:
    """Flag extracted facts that likely came from a different benchmark.

    Layer 1: topic-mismatch (e.g. math terms for an instruction-following benchmark).
    Layer 2: proper-noun check (names absent from source text).
    Numbers are skipped to avoid false positives from normalization (8.5K vs 8500).
    """
    if not extracted_facts or not source_text:
        return extracted_facts

    source_lower = source_text.lower()

    topic_keywords: set = set()
    if identity_anchor:
        about_match = re.search(r'is about:\s*(.+?)\.?\s*Only', identity_anchor)
        if about_match:
            topics = about_match.group(1).split(",")
            topic_keywords = {t.strip().lower() for t in topics if t.strip()}

    _TOPIC_CLASH_PAIRS = {
        "instruction following": [
            "math word problem", "grade school math", "arithmetic",
            "mathematical reasoning", "solving math",
        ],
        "text generation": [],
        "evaluation": [],
        "mathematics": [
            "instruction following", "instruction-following",
        ],
        "grade school math": [
            "instruction following", "instruction-following",
        ],
        "question answering": [
            "math word problem", "grade school math",
        ],
    }

    off_topic_phrases: list = []
    for topic in topic_keywords:
        for key, phrases in _TOPIC_CLASH_PAIRS.items():
            if key in topic:
                off_topic_phrases.extend(phrases)

    result_lines = []
    flagged_count = 0

    for line in extracted_facts.split("\n"):
        stripped = line.strip()

        if (not stripped or stripped.startswith("#") or
                "no information found" in stripped.lower() or
                not stripped.startswith("-")):
            result_lines.append(line)
            continue

        line_lower = stripped.lower()
        topic_clash = False
        for phrase in off_topic_phrases:
            if phrase in line_lower and phrase not in source_lower:
                flagged_count += 1
                result_lines.append(
                    f"{line} [SUSPECT: topic '{phrase}' doesn't match benchmark domain]"
                )
                topic_clash = True
                break
        if topic_clash:
            continue

        cap_words = re.findall(r'\b[A-Z][A-Za-z0-9]{3,}(?:[-_][A-Za-z0-9]+)*\b', stripped)
        common = {
            "the", "this", "that", "not", "for", "and", "are", "was",
            "has", "its", "each", "what", "how", "who", "all", "use",
            "can", "may", "any", "set", "one", "two", "per", "new",
            "text", "data", "model", "task", "test", "used", "based",
            "from", "also", "with", "into", "they", "than", "more",
            "most", "only", "some", "such", "been", "were", "have",
            "does", "make", "made", "like", "using", "these", "those",
            "about", "other", "their", "there", "which", "would",
            "could", "should", "being", "after", "before", "paper",
            "level", "score", "high", "note", "type", "human",
            "found", "total", "each", "free", "full", "large",
            "small", "first", "second", "third", "final", "main",
            "true", "false", "null", "none", "english", "information",
        }
        cap_words = [w for w in cap_words if w.lower() not in common]

        ml_terms = {"accuracy", "precision", "recall", "bleu", "rouge",
                     "transformer", "language", "neural", "training",
                     "evaluation", "benchmark", "dataset", "annotation"}
        cap_words = [w for w in cap_words if (
            w.lower() not in ml_terms and
            w.lower() not in benchmark_name.lower()
        )]

        suspicious_words = []
        for w in cap_words:
            if w.lower() not in source_lower:
                suspicious_words.append(w)

        if len(suspicious_words) >= 2:
            flagged_count += 1
            result_lines.append(f"{line} [SUSPECT: terms '{', '.join(suspicious_words)}' not found in source]")
            continue
        elif len(suspicious_words) == 1:
            w = suspicious_words[0]
            # Single capitalized word pattern (e.g. "Upwork") — likely an org/platform name
            if len(w) >= 5 and w[0].isupper() and w[1:].islower():
                flagged_count += 1
                result_lines.append(f"{line} [SUSPECT: '{w}' not found in source]")
                continue

        result_lines.append(line)

    if flagged_count:
        logger.info("Contamination check: flagged %d potentially cross-contaminated facts for '%s'",
                     flagged_count, benchmark_name)
    return "\n".join(result_lines)


_GAP_QUERIES = {
    "domains": "domain area subject field topic application area research discipline",
    "limitations": "limitation weakness constraint bias shortcoming caveat",
    "annotation": "annotation annotator label crowdworker human judge agreement quality control",
    "source": "data collection source origin corpus created gathered compiled",
    "similar_benchmarks": "related work comparison benchmark dataset alternative prior",
    "audience": "intended user researcher practitioner developer community",
    "out_of_scope_uses": "not designed for out of scope limitation inappropriate use",
    "consent_procedures": "crowdworker consent compensation IRB ethical approval platform",
    "methods": "evaluation method zero-shot few-shot fine-tune prompt protocol",
    "baseline_results": "baseline result score performance accuracy model evaluation",
    "calculation": "overall score calculation aggregate average weighted macro",
}


def _fill_paper_gaps(
    paper_facts: str,
    paper_retriever: Any,
    benchmark_name: str,
    full_paper_text: str,
) -> str:
    """Re-retrieve paper chunks for fields that came back empty on first pass."""
    missing_fields = []
    current_section = ""
    for line in paper_facts.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
        elif stripped.startswith("- ") and "no information found" in stripped.lower():
            # Extract field name: "- domains: No information found" → "domains"
            field_match = re.match(r'-\s*(\w+):', stripped)
            if field_match:
                field_name = field_match.group(1)
                if field_name in _GAP_QUERIES:
                    missing_fields.append((current_section, field_name))

    if not missing_fields:
        return paper_facts

    logger.info("Gap-filling: targeted retrieval for %d missing fields: %s",
                len(missing_fields), [f[1] for f in missing_fields])

    gap_chunks: Dict[str, str] = {}
    for section, field in missing_fields:
        query = _GAP_QUERIES[field]
        try:
            chunks = paper_retriever.invoke(query)
            if chunks:
                text = "\n".join(c.page_content for c in chunks[:2])[:1500]
                gap_chunks[field] = text
        except Exception:
            pass

    if not gap_chunks:
        return paper_facts

    fields_text = "\n".join(
        f"- {field}: Extract this information from the text below."
        for _, field in missing_fields
        if field in gap_chunks
    )
    chunks_text = "\n\n".join(
        f"[For {field}]\n{text}" for field, text in gap_chunks.items()
    )

    gap_prompt = f"""Read the following paper excerpts about the benchmark "{benchmark_name}" and describe what they say about these specific fields.

RULES: Base your descriptions ONLY on the text provided. Rephrase in your own words. If the text does not contain the information, write: "- No information found"

Fields to describe:
{fields_text}

PAPER EXCERPTS:
{chunks_text}"""

    try:
        from auto_benchmarkcard.config import get_llm_handler
        llm_handler = get_llm_handler()
        gap_facts = llm_handler.generate(f"{_EXTRACTOR_SYSTEM}\n\n{gap_prompt}")

        filled_count = 0
        for _, field in missing_fields:
            pattern = rf'-\s*{field}:\s*(.+)'
            match = re.search(pattern, gap_facts, re.IGNORECASE)
            if match and "no information found" not in match.group(1).lower():
                new_value = match.group(0)
                old_pattern = rf'(-\s*{field}:\s*.*[Nn]o information found.*)'
                paper_facts = re.sub(old_pattern, new_value, paper_facts)
                filled_count += 1

        if filled_count:
            logger.info("Gap-filling: filled %d/%d missing fields", filled_count, len(missing_fields))
        else:
            logger.info("Gap-filling: no additional facts found in paper")

    except Exception as e:
        logger.warning("Gap-filling extraction failed: %s", e)

    return paper_facts


def merge_extracted_facts(
    paper_facts: str,
    hf_facts: str,
    deterministic_facts: Dict[str, Any],
    benchmark_name: str,
) -> Dict[str, str]:
    """Merge facts from all sources into per-section tagged strings.

    Priority: DETERMINISTIC > PAPER > HF_README.
    """
    merged: Dict[str, str] = {}

    for section in ALL_SECTIONS:
        parts = []

        if paper_facts:
            section_text = _extract_section_from_facts(paper_facts, section)
            if section_text:
                parts.append(f"[PAPER] Facts from research paper:\n{section_text}")

        if hf_facts:
            section_text = _extract_section_from_facts(hf_facts, section)
            if section_text:
                parts.append(f"[HF_README] Facts from HuggingFace dataset page:\n{section_text}")

        det_lines = []
        for key, value in deterministic_facts.items():
            if key.startswith(section + "."):
                field = key.split(".", 1)[1]
                if isinstance(value, list):
                    det_lines.append(f"- {field}: {', '.join(str(v) for v in value)}")
                elif isinstance(value, dict):
                    det_lines.append(f"- {field}: {json.dumps(value)}")
                else:
                    det_lines.append(f"- {field}: {value}")

        if det_lines:
            parts.append(
                "[DETERMINISTIC] Verified facts from structured metadata:\n"
                + "\n".join(det_lines)
            )

        if section == "methodology" and "evaluation_summary" in deterministic_facts:
            eval_sum = deterministic_facts["evaluation_summary"]
            eee_lines = []
            for p in eval_sum.get("top_performers", [])[:5]:
                metric = eval_sum.get("primary_metric", "score")
                eee_lines.append(f"- {p['model']}: {p['score']:.4f} ({metric})")
            stats = eval_sum.get("score_statistics", {})
            if stats:
                eee_lines.append(
                    f"- Score statistics: mean={stats.get('mean')}, "
                    f"std={stats.get('std_dev')}, "
                    f"range=[{stats.get('min')}, {stats.get('max')}]"
                )
            n = eval_sum.get("total_models_evaluated", 0)
            if n:
                eee_lines.append(f"- Total models evaluated: {n}")
            if eee_lines:
                parts.append(
                    "[EEE] Evaluation results from Every Eval Ever:\n"
                    + "\n".join(eee_lines)
                )

        merged[section] = "\n\n".join(parts) if parts else "No facts available for this section."

    return merged


def _extract_section_from_facts(facts_text: str, section_name: str) -> str:
    """Extract a named section from the LLM extraction output."""
    section_variants = [
        f"## {section_name}",
        f"## {section_name.replace('_', ' ')}",
        f"## {section_name.replace('_', ' ').title()}",
        f"**{section_name}**",
        f"**{section_name.replace('_', ' ').title()}**",
    ]

    lines = facts_text.split("\n")
    collecting = False
    collected = []

    for line in lines:
        stripped = line.strip().lower()

        if any(v.lower() in stripped for v in section_variants):
            collecting = True
            continue

        if collecting and (stripped.startswith("## ") or
                           (stripped.startswith("**") and stripped.endswith("**"))):
            is_other = any(
                s.replace("_", " ").lower() in stripped
                for s in ALL_SECTIONS if s != section_name
            )
            if is_other:
                break

        if collecting:
            collected.append(line)

    result = "\n".join(collected).strip()

    if result:
        clean = [l for l in result.split("\n")
                 if "[UNVERIFIED]" not in l and "[SUSPECT]" not in l]
        result = "\n".join(clean).strip()

    return result


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
            if "/" in r and not r.startswith("http") and not r.startswith("ftp"):
                r = f"https://huggingface.co/datasets/{r}"
            r = r.replace("hugdingface.co", "huggingface.co")
            if r.startswith("http") and r not in seen:
                clean.append(r)
                seen.add(r)
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

    if "data.format" in deterministic_facts:
        existing = card.get("data", {}).get("format")
        if _is_empty(existing):
            card.setdefault("data", {})["format"] = deterministic_facts["data.format"]
            overrides_applied.append("format")
        else:
            logger.debug("Skipping format override: LLM has %r", existing)

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


def compute_card_confidence(
    has_paper: bool,
    has_hf_readme: bool,
    has_eee: bool,
    has_hf_basic: bool = False,
) -> Dict[str, Any]:
    """Compute confidence level (high/medium/low) based on available sources."""
    if has_paper and (has_hf_readme or has_hf_basic):
        level = "high"
    elif has_paper or (has_hf_readme and has_eee):
        level = "medium"
    else:
        level = "low"

    return {
        "confidence_level": level,
        "sources_available": {
            "paper": has_paper,
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
        description="A comprehensive 2-3 sentence description explaining what the benchmark measures, its key characteristics, and its significance in the field",
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
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class PurposeAndIntendedUsers(BaseModel):

    goal: str = Field(
        ...,
        description="The primary objective and research question this benchmark addresses, including what capabilities or behaviors it aims to measure",
    )
    audience: List[str] = Field(
        ...,
        description="Target user groups (e.g., 'AI researchers', 'model developers', 'safety evaluators', 'industry practitioners')",
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
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


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
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
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
        description="Specific quantitative metrics used (e.g., 'accuracy', 'F1-score', 'BLEU', 'exact match')",
    )
    calculation: str = Field(
        ...,
        description="Detailed explanation of how metrics are computed, including any normalization or aggregation methods",
    )
    interpretation: str = Field(
        ...,
        description="Guidelines for interpreting scores, including score ranges, what constitutes good performance, and any caveats",
    )
    baseline_results: str = Field(
        ...,
        description="Performance of established models or baselines, with specific numbers and context for comparison",
    )
    validation: str = Field(
        ...,
        description="Quality assurance measures, validation procedures, and steps taken to ensure reliable and reproducible evaluations",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class EthicalAndLegalConsiderations(BaseModel):

    privacy_and_anonymity: str = Field(
        ...,
        description="Data protection measures, anonymization techniques, and handling of personally identifiable information",
    )
    data_licensing: str = Field(
        ...,
        description="Specific license terms, usage restrictions, and redistribution permissions",
    )
    consent_procedures: str = Field(
        ...,
        description="Details of informed consent processes, participant rights, and withdrawal procedures",
    )
    compliance_with_regulations: str = Field(
        ...,
        description="Adherence to relevant regulations (GDPR, IRB approval, etc.) and ethical review processes",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class BenchmarkCard(BaseModel):

    benchmark_details: BenchmarkDetails
    purpose_and_intended_users: PurposeAndIntendedUsers
    data: DataInfo
    methodology: Methodology
    ethical_and_legal_considerations: EthicalAndLegalConsiderations


def extract_provenance(section_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split provenance metadata out of section data."""
    clean_data = dict(section_data)
    provenance = clean_data.pop("provenance", None) or {}
    return clean_data, provenance


@tool("compose_benchmark_card")
def compose_benchmark_card(
    unitxt_metadata: Optional[Dict[str, Any]] = None,
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    docling_output: Optional[Dict[str, Any]] = None,
    query: str = "",
    eee_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose a benchmark card from collected metadata using source-isolated extraction."""

    logger.info("Composing benchmark card for: %s", query)

    has_paper = bool(docling_output and docling_output.get("success"))
    has_hf = bool(hf_metadata)
    has_hf_readme = bool(_get_hf_readme(hf_metadata))
    has_eee = bool(eee_metadata)

    source_list = []
    if has_paper:
        source_list.append("Paper")
    if has_hf_readme:
        source_list.append("HF-README")
    elif has_hf:
        source_list.append("HF-basic")
    if has_eee:
        source_list.append("EEE")
    logger.info("Available sources: %s", ", ".join(source_list) or "none")

    # Phase 1: Collect paper content (intro + RAG supplement)
    paper_retriever = None
    all_paper_content = ""

    if has_paper:
        paper_text = docling_output.get("filtered_text", "")
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
                    paper_vectorstore = Chroma.from_documents(documents, embeddings)
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

    # Phase 2: Source-isolated extraction
    identity_anchor = _get_benchmark_identity(query, hf_metadata, eee_metadata)
    if identity_anchor:
        logger.info("Benchmark identity: %s", identity_anchor)

    paper_facts = ""
    if all_paper_content:
        paper_facts = extract_facts_from_paper(all_paper_content, query, identity_anchor)

    hf_facts = ""
    if has_hf_readme:
        hf_facts = extract_facts_from_hf_readme(hf_metadata, query, identity_anchor=identity_anchor)

    det_facts = extract_deterministic_facts(eee_metadata, hf_metadata, extracted_ids)

    # Phase 3: Contamination check
    full_paper_text = ""
    if has_paper:
        full_paper_text = docling_output.get("filtered_text", "")

    if paper_facts and (full_paper_text or all_paper_content):
        paper_facts = check_cross_contamination(
            paper_facts, full_paper_text or all_paper_content, query, identity_anchor
        )

    if hf_facts and has_hf_readme:
        readme_text_for_check = _get_hf_readme(hf_metadata)
        if readme_text_for_check:
            hf_facts = check_cross_contamination(
                hf_facts, readme_text_for_check, query, identity_anchor
            )

    # Phase 3b: Gap-filling for missing fields
    if paper_facts and paper_retriever:
        paper_facts = _fill_paper_gaps(
            paper_facts, paper_retriever, query, full_paper_text or all_paper_content
        )

    # Phase 4: Merge facts
    merged_facts = merge_extracted_facts(paper_facts, hf_facts, det_facts, query)

    # Phase 5: Compose sections
    sections = [
        ("benchmark_details", BenchmarkDetails),
        ("purpose_and_intended_users", PurposeAndIntendedUsers),
        ("data", DataInfo),
        ("methodology", Methodology),
        ("ethical_and_legal_considerations", EthicalAndLegalConsiderations),
    ]

    gold_example_path = Path(__file__).parent / "gold_example.json"
    gold_example: Dict[str, Any] = {}
    try:
        gold_example = json.loads(gold_example_path.read_text())
    except Exception as e:
        logger.warning("Could not load gold example: %s", e)

    generated_sections = {}
    all_provenance = {}

    for section_name, section_class in sections:
        logger.info("Composing section: %s", section_name)

        section_facts = merged_facts.get(section_name, "No facts available.")

        gold_snippet = ""
        if gold_example and section_name in gold_example:
            gold_json = json.dumps(gold_example[section_name], indent=2)
            gold_json_escaped = gold_json.replace("{", "{{").replace("}", "}}")
            gold_snippet = (
                f"\n\nGOLD EXAMPLE (FORMAT reference — match style and detail level):\n"
                f"```json\n{gold_json_escaped}\n```"
            )

        section_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are documenting the AI benchmark "{{query}}". Generate the '{section_name}' section.

You are given DESCRIBED FACTS from multiple sources, each tagged with its origin:
- [PAPER]: From the research paper (most authoritative for methodology, baselines)
- [HF_README]: From the HuggingFace dataset page
- [DETERMINISTIC]: Verified facts from structured metadata (always correct)
- [EEE]: Evaluation results from Every Eval Ever

RULES:
1. Use ONLY the facts provided below. Do NOT add information from your own knowledge about this or any benchmark.
2. If facts from different sources conflict, prefer [DETERMINISTIC] > [PAPER] > [HF_README] > [EEE].
3. If a field has no facts from any source, write exactly "Not specified".
4. Write in third person. Be concise and clear. Match the gold example style.
5. Write in your own words — synthesize the facts into natural, readable descriptions. Do NOT copy raw text fragments.
6. Do NOT mention other benchmarks unless the [PAPER] facts explicitly name them.
7. For baseline_results: separate PAPER baselines (original results) from EEE results (evaluation suite results).
8. Do NOT invent numbers, dataset sizes, platform names, or methodology details that are not in the facts below.
9. SKIP any facts tagged [SUSPECT] — these may be from a different benchmark. Do not use them.
{gold_snippet}

PROVENANCE TRACKING (REQUIRED):
For every field you fill in (except "Not specified"), include a provenance entry:
{{{{
  "provenance": {{{{
    "field_name": {{{{
      "source": "paper|huggingface|eee|deterministic",
      "evidence": "the key fact that supports this value"
    }}}}
  }}}}
}}}}""",
            ),
            (
                "user",
                f"""Benchmark: {{query}}

SOURCE-TAGGED FACTS:
{{section_facts}}

Generate the {section_name} section.""",
            ),
        ])

        chain = section_prompt | get_llm_handler().with_structured_output(section_class)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                section_result = chain.invoke({
                    "query": query,
                    "section_facts": section_facts,
                })
                section_dict = section_result.model_dump()
                clean_section, section_provenance = extract_provenance(section_dict)
                generated_sections[section_name] = clean_section
                if section_provenance:
                    all_provenance[section_name] = section_provenance
                logger.info("Section %s completed", section_name)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Failed %s (attempt %d/%d): %s",
                        section_name, attempt + 1, max_retries, e,
                    )
                else:
                    logger.error(
                        "Failed %s after %d attempts: %s",
                        section_name, max_retries, e,
                    )
                    raise

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

    # Phase 9: Confidence
    confidence = compute_card_confidence(has_paper, has_hf_readme, has_eee, has_hf)

    return {
        "benchmark_card": benchmark_card_dict,
        "provenance": all_provenance if all_provenance else None,
        "composition_metadata": {
            "sources_used": {
                "unitxt": bool(unitxt_metadata),
                "huggingface": bool(hf_metadata),
                "extracted_ids": bool(extracted_ids),
                "docling": has_paper,
                "eee": has_eee,
            },
            "query": query,
            "composition_timestamp": datetime.now().isoformat(),
            "generation_method": "source_isolated_extraction",
            "model_used": get_llm_handler().model_name,
            "confidence": confidence,
        },
    }
