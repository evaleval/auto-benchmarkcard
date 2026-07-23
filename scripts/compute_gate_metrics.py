"""Composer gate v2 grounding metrics over a run output dir.

Classifies every measured field of every card in a batch run and reports the
reframed gate. Pure analysis of existing artifacts -- no model, no network.
The classifier, metrics, and thresholds are documented in this module.

Quote-verify-rate (gate metric 1) is telemetered upstream by verify_items and is
read elsewhere; this script computes gate metric 2, the structural hallucination
rate, plus the descriptive grounding breakdown.

Two field populations (per the gate design):
  full   = every non-display field in the five gold sections. The classifier runs
           over this set; it drives the grounding breakdown, per-source share and
           per-field table. Deterministic/EEE fields bucket as GROUNDED-DERIVED.
  judged = full minus JUDGE_SKIP (the frozen gold-judge skip set, which already
           contains the display fields and the six deterministic fields). The
           headline hallucination-rate and NS-rate are computed here. Derived
           fields are excluded from the hallucination denominator -- they are
           grounded by construction and would flatter the rate.

Usage:
  python scripts/compute_gate_metrics.py --run-dir <batch output dir>
  # -> <run-dir>/gate_metrics.json  (+ printed summary)
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # sibling scripts (judge_gold_set)

from auto_benchmarkcard.card_utils import (  # noqa: E402
    GOLD_SECTIONS, _ABSENCE_PLACEHOLDER_RE, _DISPLAY_ONLY_FIELDS,
    extract_card, is_not_specified,
)
from auto_benchmarkcard.tools.composer import aboutness, field_spec as fs  # noqa: E402
from judge_gold_set import JUDGE_SKIP  # noqa: E402

NS = "NS"
EXTRACTED = "GROUNDED-EXTRACTED"
DERIVED = "GROUNDED-DERIVED"
UNGROUNDED = "UNGROUNDED"
LABELS = (NS, EXTRACTED, DERIVED, UNGROUNDED)
_KEY = {NS: NS, EXTRACTED: "extracted", DERIVED: "derived", UNGROUNDED: "ungrounded"}

# Keys that can sit alongside real fields inside a section but are never measured.
_BOOKKEEPING = {"provenance", "flagged_fields", "missing_fields"}
# Only these provenance sources count as structured-derived (grounded in structured
# data, no text quote expected). Matches composer_gate_v2.md. Classified by source,
# NOT status: in the real artifacts deterministic fields carry status='stated'.
_DERIVED_SOURCES = {"deterministic", "eee"}

# A measured field whose Stage-B group failed to parse is collapsed to a "Not
# specified" placeholder on the card, but the failure is recorded in the top-level
# card["flagged_fields"] dict with this marker. That is data loss, NOT an honest
# abstention -- counting it as clean NS lets a parse failure read as a false GO.
_SCHEMA_INVALID_MARKER = "schema_invalid"
# Gate metric 2 bar. A structural hallucination-rate strictly above this is NO-GO.
HALLUCINATION_MAX = 0.02


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _value_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return ""


def is_ns_value(value):
    """Canonical 'Not specified' (incl. the list sentinel) or an absence placeholder
    that leaked through as the value. Both are honest abstentions, not claims."""
    if is_not_specified(value):
        return True
    text = _value_text(value)
    return bool(text) and bool(_ABSENCE_PLACEHOLDER_RE.search(text))


def _is_derived_source(source):
    primary = re.split(r"[/,]", str(source or "").strip().lower())[0].strip()
    return primary in _DERIVED_SOURCES


def classify(value, prov_entry):
    """Bucket one measured field. NS first (abstention), then a verified evidence
    quote (extracted), then a structured source (derived), else ungrounded."""
    if is_ns_value(value):
        return NS
    entry = prov_entry if isinstance(prov_entry, dict) else {}
    eids = entry.get("evidence_ids") or []
    if isinstance(eids, list) and eids:
        return EXTRACTED
    if _is_derived_source(entry.get("source")):
        return DERIVED
    return UNGROUNDED


def _prov_entry(prov, section, field):
    sec = prov.get(section) if isinstance(prov, dict) else None
    entry = sec.get(field) if isinstance(sec, dict) else None
    return entry if isinstance(entry, dict) else None


def classify_card(card, prov):
    """Per-field rows for every non-display gold-section field of one card."""
    rows = []
    for section in GOLD_SECTIONS:
        fields = card.get(section)
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            if key in _BOOKKEEPING:
                continue
            path = f"{section}.{key}"
            if path in _DISPLAY_ONLY_FIELDS:
                continue
            entry = _prov_entry(prov, section, key)
            rows.append({
                "path": path,
                "label": classify(value, entry),
                "source": entry.get("source") if entry else None,
                "judged": path not in JUDGE_SKIP,
            })
    return rows


def schema_invalid_fields(card):
    """Field paths in card['flagged_fields'] whose Stage-B group failed to parse.
    Such a field collapses to 'Not specified' on the card, so the classifier would
    otherwise score it as honest NS. It is data loss, not abstention; counting it
    keeps a parse failure from passing the gate as a clean card."""
    flagged = card.get("flagged_fields") if isinstance(card, dict) else None
    if not isinstance(flagged, dict):
        return []
    return sorted(path for path, reason in flagged.items()
                  if _SCHEMA_INVALID_MARKER in str(reason).lower())


def _counts(labels):
    c = Counter(labels)
    return {NS: c[NS], "extracted": c[EXTRACTED], "derived": c[DERIVED], "ungrounded": c[UNGROUNDED]}


def _rate(num, den):
    return round(num / den, 4) if den else None


def _metrics_from_rows(rows):
    """Full breakdown (all non-display fields) + judged-set gate metrics. Works for
    one card or for the rows pooled across cards (the aggregate, micro-averaged)."""
    judged_rows = [r for r in rows if r["judged"]]
    j = Counter(r["label"] for r in judged_rows)
    ns_j, ext_j, der_j, ung_j = j[NS], j[EXTRACTED], j[DERIVED], j[UNGROUNDED]
    hall_den = ext_j + ung_j  # derived excluded: grounded by construction
    share = Counter((r["source"] or "(no entry)").strip().lower() for r in rows)
    total = sum(share.values()) or 1
    return {
        "full_breakdown": _counts(r["label"] for r in rows),
        "n_fields_full": len(rows),
        "judged": {
            "counts": _counts(r["label"] for r in judged_rows),
            "n_fields": len(judged_rows),
            "non_ns": ext_j + der_j + ung_j,
        },
        "hallucination_rate": _rate(ung_j, hall_den),
        "hallucination_numerator": ung_j,
        "hallucination_denominator": hall_den,
        "ns_rate": _rate(ns_j, len(judged_rows)),
        "per_source_share": {
            k: {"count": v, "share": round(v / total, 4)}
            for k, v in sorted(share.items(), key=lambda kv: (-kv[1], kv[0]))
        },
    }


def gather_cards(run_dir):
    """(name, card, prov, hf, eee) for each per-card subdir under the batch run dir."""
    out = []
    pattern = os.path.join(run_dir, "*", "benchmarkcard", "benchmark_card_*.json")
    for card_path in sorted(glob.glob(pattern)):
        name = os.path.basename(card_path)[len("benchmark_card_"):-len(".json")]
        card_dir = os.path.dirname(os.path.dirname(card_path))
        prov_path = os.path.join(card_dir, "tool_output", "composer", f"provenance_{name}.json")
        hf_path = os.path.join(card_dir, "tool_output", "hf", f"{name}.json")
        eee_path = os.path.join(card_dir, "tool_output", "eee", f"{name}.json")
        out.append((name, extract_card(_load(card_path) or {}), _load(prov_path) or {},
                    _load(hf_path) or {}, _load(eee_path) or {}))
    return out


def _gather_paper_verification(run_dir):
    """Map card name -> paper-verification.json for each per-card subdir. Tolerates absence
    (HF-arxiv-tag short-circuit cards never write one)."""
    out = {}
    pattern = os.path.join(run_dir, "*", "tool_output", "paper_resolver", "paper-verification.json")
    for pv_path in sorted(glob.glob(pattern)):
        card_dir = os.path.dirname(os.path.dirname(os.path.dirname(pv_path)))
        siblings = glob.glob(os.path.join(card_dir, "benchmarkcard", "benchmark_card_*.json"))
        if not siblings:
            continue
        name = os.path.basename(siblings[0])[len("benchmark_card_"):-len(".json")]
        pv = _load(pv_path)
        if pv is not None:
            out[name] = pv
    return out


_VERIFY_PATTERNS = ("**/verify_items*.json", "**/quote_verify*.json", "**/*verify*telemetr*")


def _quote_verify(run_dir):
    for pat in _VERIFY_PATTERNS:
        hits = sorted(glob.glob(os.path.join(run_dir, pat), recursive=True))
        if hits:
            return {"source_file": os.path.relpath(hits[0], run_dir), "telemetry": _load(hits[0])}
    return {"source_file": None,
            "note": "no verify_items telemetry in this run. quote-verify-rate (gate metric 1) "
                    "is telemetered by verify_items (reject-reason log) and read elsewhere "
                    "(D6 pilot = 90.5%); not recomputed here -- the verifier is not reimplemented."}


def _verdict(schema_invalid_count, schema_invalid_by_card, hallucination_rate):
    """GO/NO-GO over the two bars this script owns. NO-GO if any field was lost to a
    parse failure (data loss masquerading as 'Not specified') or the structural
    hallucination-rate exceeds the bar. Gate metric 1 (quote-verify-rate >= 90%) is
    telemetered upstream by verify_items and is not asserted here."""
    hall_exceeds = hallucination_rate is not None and hallucination_rate > HALLUCINATION_MAX
    reasons = []
    if schema_invalid_count:
        n_cards = len(schema_invalid_by_card)
        reasons.append(
            f"{schema_invalid_count} schema_invalid field(s) across {n_cards} card(s): "
            "unparseable Stage-B output collapsed to 'Not specified' (data loss, not abstention)")
    if hall_exceeds:
        reasons.append(
            f"hallucination-rate {_pct(hallucination_rate)} exceeds the {_pct(HALLUCINATION_MAX)} bar")
    return {
        "result": "NO-GO" if reasons else "GO",
        "reasons": reasons,
        "checks": {
            "schema_invalid": {
                "pass": schema_invalid_count == 0,
                "count": schema_invalid_count,
                "by_card": schema_invalid_by_card,
            },
            "hallucination_rate": {
                "pass": not hall_exceeds,
                "rate": hallucination_rate,
                "threshold": HALLUCINATION_MAX,
            },
        },
        "note": "Gate metric 1 (quote-verify-rate >= 90%) is telemetered upstream by "
                "verify_items and read elsewhere; it is not asserted by this script.",
    }


def _docling_degraded(run_dir):
    """Cards that resolved a paper_url but built abstract-only (docling got nothing).

    Reads the per-card docling_telemetry sidecar written by run_composer. Purely
    descriptive triage telemetry — does not feed the GO/NO-GO verdict.
    """
    out = []
    pattern = os.path.join(run_dir, "*", "tool_output", "composer", "docling_telemetry_*.json")
    for p in sorted(glob.glob(pattern)):
        tel = _load(p) or {}
        if tel.get("degraded_to_abstract_only"):
            name = os.path.basename(p)[len("docling_telemetry_"):-len(".json")]
            out.append({
                "card": name,
                "reason": tel.get("reason"),
                "http_status": tel.get("http_status"),
            })
    return {"count": len(out), "cards": out}


# Gate v3: a descriptive content layer over the two grounding bars above. These checks
# are GENERAL linters for the full corpus -- they key on field-class, enum vocab, URL
# shape and telemetry, never on a benchmark name or an expected value. They do NOT feed
# the GO/NO-GO verdict; they annotate a clean verdict with content defects the grounding
# bars are blind to (a verbatim-grounded value can still be the wrong thing to say).

# Enumerable list fields whose elements should be atomic items, not prose or a summary.
# (methods/out_of_scope_uses/resources/authors/judge_models are prose- or URL-valued by
# design and are linted elsewhere or not at all.)
_ENUM_LIST_FIELDS = {
    "benchmark_details.domains", "benchmark_details.similar_benchmarks",
    "benchmark_details.languages", "purpose_and_intended_users.tasks",
    "purpose_and_intended_users.audience",
}
# A list element that summarises the whole collection instead of naming one member
# ("7 domains covering 139 libraries"): a leading count+noun or an aggregate connector.
_SUMMARY_ELEMENT_RE = re.compile(
    r"^\s*[\d,]+\s+\w+|\b(?:covering|spanning|comprising|consisting of)\b", re.IGNORECASE)
_ATOMIC_MAX_WORDS = 14  # an element longer than ~1 short sentence is not an atomic item
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+[A-Z]")  # a mid-string sentence break

# A short machine-usable name vs a paper title: ALL-CAPS-and-long, or simply many-word.
_NAME_MAX_LEN = 40
_NAME_MAX_WORDS = 8
_NAME_CAPS_RATIO = 0.6

# Canonical metric shapes: pass@k, other:<id>, or a known metric token.
_PASS_AT_K_RE = re.compile(r"^pass\s*[@_ ]?\s*\d+$", re.IGNORECASE)
_KNOWN_METRICS = {
    "accuracy", "f1", "bleu", "rouge", "exact match", "exact_match", "win-rate",
    "win_rate", "precision", "recall", "ndcg", "ndcg@k", "mrr", "auc", "perplexity",
    "bertscore", "meteor", "em",
}

# Methodology fields whose evidence describes how scoring works (judge-aboutness inputs).
_SCORING_PROV_FIELDS = (
    "methods", "metrics", "calculation",
    "judge_uses_llm", "judge_models", "judge_num", "judge_setup",
)


def _real_list(v):
    return isinstance(v, list) and bool(v) and not is_not_specified(v)


def _real_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _prov_evidence_texts(prov, section, fields):
    """The verified-quote text behind a set of fields, read from the provenance sidecar."""
    out = []
    sec = prov.get(section) if isinstance(prov, dict) else None
    if not isinstance(sec, dict):
        return out
    for f in fields:
        e = sec.get(f)
        if isinstance(e, dict) and isinstance(e.get("evidence"), str) and e["evidence"].strip():
            out.append(e["evidence"])
    return out


def _aboutness_flags(name, card, prov, hf=None, eee=None):
    """Fields that are off-topic / mis-about despite being verbatim-grounded. Reuses the
    aboutness.py predicates read-only; every trigger keys on field-class, never on a name."""
    flags = []
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}
    meth = card.get("methodology") if isinstance(card.get("methodology"), dict) else {}
    pu = card.get("purpose_and_intended_users") if isinstance(card.get("purpose_and_intended_users"), dict) else {}

    # judge mis-aboutness: an asserted LLM-judge cluster with no evidence that an LLM
    # SCORES model outputs (the bigcode/biolp class). Read-only mirror of the in-pipeline
    # aboutness._guard_judge_cluster: trigger = asserted + no output-scoring evidence.
    asserts = (meth.get("judge_uses_llm") is True or _real_list(meth.get("judge_models"))
               or _real_int(meth.get("judge_num")))
    if asserts:
        quotes = _prov_evidence_texts(prov, "methodology", _SCORING_PROV_FIELDS)
        quotes += [v for v in (meth.get("judge_models") or []) if isinstance(v, str)]
        if not any(aboutness.describes_output_scoring(q) for q in quotes):
            exec_seen = any(aboutness.describes_exec_scoring(q) for q in quotes)
            flags.append({"card": name, "path": "methodology.judge_uses_llm",
                          "kind": "judge_mis_aboutness",
                          "detail": "judge cluster asserted but no output-scoring evidence"
                                    + (" (scoring looks execution-based)" if exec_seen else "")})

    # judge DENIAL mis-aboutness: an explicit judge_uses_llm=False that the source contradicts
    # by NAMING a judge / equality-checker model (the aa-lcr V4-Flash misread). Read-only mirror
    # of the EEE injection's detector over the SAME EEE + HF text -- fires independently of
    # whether the injection ran. Descriptive only; never feeds the verdict.
    if meth.get("judge_uses_llm") is False:
        model = aboutness.extract_judge_model(aboutness.collect_judge_scan_text(eee, hf))
        if model:
            flags.append({"card": name, "path": "methodology.judge_uses_llm",
                          "kind": "judge_denial_mis_aboutness",
                          "detail": f"judge_uses_llm=False but source names a judge model: {model!r}"})

    # modality/data_type mismatch: data_type carrying metric/methodology text instead of a
    # modality. aboutness.is_metric_only_data_type is the exact predicate Stage-A uses.
    dt = bd.get("data_type")
    if isinstance(dt, str) and not is_not_specified(dt) and aboutness.is_metric_only_data_type(dt):
        flags.append({"card": name, "path": "benchmark_details.data_type",
                      "kind": "data_type_not_modality",
                      "detail": f"reads as metric/method text, not a modality: {dt!r}"})

    # related-benchmark leakage (conservative): a named similar_benchmark appears verbatim in
    # a core about-this-benchmark field while this benchmark's own name does not. Keyed on the
    # card's own similar_benchmarks list, so still corpus-general.
    sims = [s for s in (bd.get("similar_benchmarks") or [])
            if isinstance(s, str) and not is_not_specified(s) and len(s) > 3]
    own = str(bd.get("name") or "").strip()
    if sims:
        own_re = re.compile(r"\b" + re.escape(own) + r"\b", re.IGNORECASE) if own else None
        sim_res = [(s, re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)) for s in sims]
        for path, value in (("benchmark_details.overview", bd.get("overview")),
                            ("purpose_and_intended_users.goal", pu.get("goal")),
                            ("purpose_and_intended_users.tasks", pu.get("tasks"))):
            text = _value_text(value)
            if not text:
                continue
            for s, sre in sim_res:
                if sre.search(text) and not (own_re and own_re.search(text)):
                    flags.append({"card": name, "path": path, "kind": "related_benchmark_leakage",
                                  "detail": f"mentions related benchmark {s!r} but not this benchmark"})
                    break
    return flags


def _is_canonical_metric(m):
    s = m.strip()
    if _PASS_AT_K_RE.match(s):
        return True
    if s.lower().startswith("other:") and len(s) > len("other:"):
        payload = s[len("other:"):]
        # the degenerate llm-stats proxy other:<slug>.score is not canonical; genuine other:<token> stays
        return not (payload.lower().endswith(".score") and len(payload) > len(".score"))
    return s.lower() in _KNOWN_METRICS


def _is_bare_url(u):
    """A single bare http(s) URL. Garbled/concatenated/prose values fail; a legitimate
    aggregator URL (e.g. llm-stats.com) is well-formed and passes by construction."""
    s = str(u).strip()
    if not s or any(ch.isspace() for ch in s) or s.lower().count("http") != 1:
        return False
    p = urlparse(s)
    return p.scheme in ("http", "https") and bool(p.netloc)


def _is_clean_data_type(dt):
    """A single modality token, a comma-list of them, or a clean other:<single-token>."""
    for part in str(dt).split(","):
        tok = part.strip().lower()
        if not tok:
            return False
        if tok.startswith("other:"):
            payload = tok[len("other:"):].strip()
            if not payload or any(ch.isspace() for ch in payload):
                return False
        elif tok not in fs.DATA_TYPE_VOCAB:
            return False
    return True


def _is_title_like_name(nm):
    s = nm.strip()
    letters = [c for c in s if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    allcaps_long = len(s) > _NAME_MAX_LEN and caps_ratio > _NAME_CAPS_RATIO
    return allcaps_long or len(s.split()) > _NAME_MAX_WORDS


def _is_atomic_element(el):
    s = el.strip()
    if len(s.split()) > _ATOMIC_MAX_WORDS:
        return False
    if _SENTENCE_BOUNDARY_RE.search(s):
        return False
    return not _SUMMARY_ELEMENT_RE.search(s)


def _enum_list_values(card):
    out = []
    for path in sorted(_ENUM_LIST_FIELDS):
        section, field = path.split(".", 1)
        sec = card.get(section)
        if isinstance(sec, dict) and isinstance(sec.get(field), list):
            out.append((path, sec[field]))
    return out


def _format_lint(name, card):
    """Machine-readability lint of structured fields. Structure-only, corpus-general."""
    flags = []
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}
    meth = card.get("methodology") if isinstance(card.get("methodology"), dict) else {}

    for m in (meth.get("metrics") or []):
        if isinstance(m, str) and not is_not_specified(m) and not _is_canonical_metric(m):
            flags.append({"card": name, "path": "methodology.metrics", "kind": "metric_noncanonical",
                          "detail": f"want pass@k / other:<id> / known token, got {m!r}"})

    for r in (bd.get("resources") or []):
        if isinstance(r, str) and not is_not_specified(r) and not _is_bare_url(r):
            flags.append({"card": name, "path": "benchmark_details.resources", "kind": "resource_not_url",
                          "detail": f"not a single bare URL: {r!r}"})

    dt = bd.get("data_type")
    if isinstance(dt, str) and not is_not_specified(dt) and not _is_clean_data_type(dt):
        flags.append({"card": name, "path": "benchmark_details.data_type", "kind": "data_type_not_atomic",
                      "detail": f"not a single clean modality token: {dt!r}"})

    nm = bd.get("name")
    if isinstance(nm, str) and not is_not_specified(nm) and _is_title_like_name(nm):
        flags.append({"card": name, "path": "benchmark_details.name", "kind": "name_title_like",
                      "detail": f"reads as a title, not a short identity: {nm!r}"})

    for path, value in _enum_list_values(card):
        for el in value:
            if isinstance(el, str) and not is_not_specified(el) and not _is_atomic_element(el):
                flags.append({"card": name, "path": path, "kind": "list_not_atomic",
                              "detail": f"summary/sentence, not an atomic item: {el!r}"})

    return flags


# Generic tokens that carry no dataset identity; dropped from both sides before the
# token-overlap test so a shared "benchmark"/"data" cannot mask a real mismatch.
_ID_STOPWORDS = {
    "benchmark", "bench", "dataset", "data", "eval", "evaluation", "test", "tests",
    "hf", "huggingface", "the", "for", "and", "of", "v1", "v2", "v3", "v4",
}


def _id_tokens(*texts):
    """Identity tokens for a name / repo id: the separator-split tokens (minus generic
    stopwords) plus each input's fully-squashed form, so a concatenated name matches a
    separated one (superglue <-> super_glue) without a bare substring match (arc != archives)."""
    toks = set()
    for t in texts:
        if not isinstance(t, str):
            continue
        low = t.lower()
        for tok in re.split(r"[^a-z0-9]+", low):
            if len(tok) >= 2 and tok not in _ID_STOPWORDS:
                toks.add(tok)
        squashed = re.sub(r"[^a-z0-9]+", "", low)
        if len(squashed) >= 3:
            toks.add(squashed)
    return toks


def _subject_vs_slug_flags(name, card):
    """The card's declared subject reads as a paper title yet shares no identity token with the folder
    slug -- a wrong paper's title became the card identity (the bfcl shape). Title-like is required so a
    short correct identity that merely uses different tokens than its slug (mmlu, an acronym; tau^2-Bench)
    is not flagged. The slug (name param) is the ground-truth identity and is never fed into the subject
    side, so a wrong title cannot self-match. Descriptive only."""
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}
    bd_name = bd.get("name")
    if not isinstance(bd_name, str) or is_not_specified(bd_name) or not _is_title_like_name(bd_name):
        return []
    slug_tokens = _id_tokens(name)
    subject_tokens = _id_tokens(bd_name)
    if not slug_tokens or not subject_tokens or not slug_tokens.isdisjoint(subject_tokens):
        return []
    return [{"card": name, "path": "benchmark_details.name", "kind": "subject_vs_slug",
             "detail": f"name reads as a title of a different subject than slug {name!r}: {bd_name!r}"}]


def _dataset_identity_flags(name, card, hf):
    """Resolved HF dataset whose repo id shares no name token with the card. Surfaces a
    mis-resolved dataset (e.g. arc -> banned-historical-archives) whose metadata the grounding
    bars would otherwise count as grounded. Descriptive only; never flags a missing resolution."""
    repo_id = hf.get("id") if isinstance(hf, dict) else None
    if not isinstance(repo_id, str) or not repo_id.strip():
        return []
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}
    bd_name = bd.get("name")
    card_name = bd_name if isinstance(bd_name, str) and not is_not_specified(bd_name) else ""
    card_tokens = _id_tokens(name, card_name)
    repo_tokens = _id_tokens(repo_id, *repo_id.split("/"))  # split org/name so each squashes alone
    if not card_tokens or not repo_tokens or not card_tokens.isdisjoint(repo_tokens):
        return []
    return [{"card": name, "path": "benchmark_details.name", "kind": "dataset_identity_mismatch",
             "detail": f"resolved HF dataset {repo_id!r} shares no name token with card {name!r}"}]


def _none_override_flags(name, pv):
    """A confident LLM 'none' verdict that nonetheless shipped a resolved_url -- the Tier-1
    regression signal (today's aa-index). Descriptive; goes empty once confident-none is final."""
    if not isinstance(pv, dict):
        return []
    lv = pv.get("llm_verification") or {}
    if (lv.get("match_index") == "none" and not lv.get("error")
            and (lv.get("confidence") or 0) >= 0.7 and pv.get("resolved_url")):
        return [{"card": name, "path": "paper_resolver.resolved_url", "kind": "none_override",
                 "detail": f"confident none (conf={lv.get('confidence')}) but resolved "
                           f"{pv.get('resolved_url')!r}"}]
    return []


_DEGRADED_SOURCE_STATUSES = {"errored", "rate_limited", "auth_error"}


def _paper_search_degraded_flags(name, pv):
    """A paper-search source that did not run cleanly (errored / rate_limited / auth_error) --
    so a None/weak resolution can be read as a degraded search rather than a true no-paper.
    Descriptive only; goes empty when every source returns ok or 0-results."""
    if not isinstance(pv, dict):
        return []
    flags = []
    for st in pv.get("sources_tried", []):
        status = st.get("status")
        if status in _DEGRADED_SOURCE_STATUSES:
            flags.append({"card": name, "path": f"sources_tried.{st.get('source')}",
                          "kind": "paper_search_degraded",
                          "detail": f"{st.get('source')} reported {status} ({st.get('results', 0)} results)"})
    return flags


def _paper_binding_flags(name, pv):
    """Resolve-time paper-binding verifier outcomes per card, from the paper sidecar. A
    pre-set/extracted/curated paper_url that was verified leaves a `binding` record; emit a descriptive
    flag for a rejected binding (the bypass was caught -> dropped to fall-through/thin) and an
    unverified-degraded keep (a hesitant/absent LLM verdict kept a pre-set binding). Also reads the
    top-level `prior_binding` (a pre-set that was rejected then fell through to a search/curated keep),
    so a dropped bypass is surfaced even when the final binding succeeded. Mirrors _hf_match_flags /
    _paper_search_degraded_flags: descriptive only, does NOT feed the GO/NO-GO verdict."""
    if not isinstance(pv, dict):
        return []
    out = []

    def _emit(b):
        if not isinstance(b, dict):
            return
        verdict = b.get("verdict")
        if verdict == "rejected":
            out.append({"card": name, "path": "paper_url", "kind": "paper_binding_rejected",
                        "detail": f"binding {b.get('paper_url')!r} ({b.get('binding_source')}) rejected "
                                  f"({b.get('reason')})"})
        elif verdict == "unverified_degraded":
            out.append({"card": name, "path": "paper_url", "kind": "paper_binding_unverified",
                        "detail": f"binding {b.get('paper_url')!r} ({b.get('binding_source')}) kept "
                                  f"unverified ({b.get('reason')})"})

    _emit(pv.get("binding"))
    _emit(pv.get("prior_binding"))   # a rejected pre-set carried through a fall-through keep
    return out


def _hf_sidecar_name(p):
    """Derive the card name for an hf_verifier sidecar path from its benchmarkcard sibling."""
    card_dir = os.path.dirname(os.path.dirname(os.path.dirname(p)))
    siblings = glob.glob(os.path.join(card_dir, "benchmarkcard", "benchmark_card_*.json"))
    return (os.path.basename(siblings[0])[len("benchmark_card_"):-len(".json")]
            if siblings else os.path.basename(card_dir))


def _hf_match_flags(run_dir):
    """Resolve-time HF-match verifier outcomes per card, from the hf_verifier sidecar. Emits a
    descriptive flag for a rejected match (verdict 'rejected' -> binding cleared to honest-thin), an
    unverified-degraded keep (verdict 'unverified_degraded' -> kept under a weak/absent LLM verdict),
    and a verifier_error (verdict 'verifier_error' -> the verifier crashed and the binding was kept
    by the fail-safe; LOUD so a mass crash is not silent). Mirrors _paper_search_degraded_flags /
    _docling_degraded: descriptive only, does NOT feed the GO/NO-GO verdict."""
    out = []
    pattern = os.path.join(run_dir, "*", "tool_output", "hf_verifier", "hf-verification.json")
    for p in sorted(glob.glob(pattern)):
        rec = _load(p) or {}
        verdict = rec.get("verdict")
        name = _hf_sidecar_name(p)
        if verdict == "rejected":
            out.append({"card": name, "path": "hf_repo", "kind": "hf_match_rejected",
                        "detail": f"repo {rec.get('repo_id')!r} rejected ({rec.get('reason')})"})
        elif verdict == "unverified_degraded":
            out.append({"card": name, "path": "hf_repo", "kind": "hf_match_unverified",
                        "detail": f"repo {rec.get('repo_id')!r} kept unverified ({rec.get('reason')})"})
        elif verdict == "verifier_error":
            out.append({"card": name, "path": "hf_repo", "kind": "hf_match_error",
                        "detail": f"repo {rec.get('repo_id')!r} kept on verifier crash ({rec.get('reason')})"})
    return out


def _hf_verification_summary(run_dir):
    """Per-verdict count breakdown over the hf_verifier sidecars (confirmed / rejected /
    unverified_degraded / skipped_curated / verifier_error / other). The denominator (total) exposes
    a mass verifier crash: 'hf_match_verification: 0 flags' can never be confused with '0 cards
    verified' because the summary shows how many cards the verifier actually scored. Descriptive
    only; does NOT feed the GO/NO-GO verdict."""
    counts = Counter()
    pattern = os.path.join(run_dir, "*", "tool_output", "hf_verifier", "hf-verification.json")
    for p in sorted(glob.glob(pattern)):
        rec = _load(p) or {}
        counts[rec.get("verdict") or "unknown"] += 1
    return {"total": sum(counts.values()), "by_verdict": dict(counts)}


def _shell_detector_flags(name, card, pv):
    """Near-empty shell: no paper resolved AND benchmark_details name + overview both collapsed
    to NS (the parse-husk class). General predicate, not keyed on any benchmark name."""
    if isinstance(pv, dict) and pv.get("resolved_url"):
        return []
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}

    def _ns(v):
        return v is None or is_not_specified(v)

    if _ns(bd.get("name")) and _ns(bd.get("overview")):
        return [{"card": name, "path": "benchmark_details", "kind": "shell_detector",
                 "detail": "no paper resolved and benchmark_details name + overview both NS"}]
    return []


def _repo_paper_mismatch_flags(name, card, hf):
    """Gate twin of Tier-2b on the resolved HF sidecar: the repo basename is a derivative (bears
    the card's name token plus extra distinctive tokens -> the bixbench case _dataset_identity
    MISSES) or an aggregate (shares no name token -> the bold case). Descriptive only."""
    repo_id = hf.get("id") if isinstance(hf, dict) else None
    if not isinstance(repo_id, str) or not repo_id.strip():
        return []
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}
    bd_name = bd.get("name")
    card_name = bd_name if isinstance(bd_name, str) and not is_not_specified(bd_name) else ""
    card_tokens = _id_tokens(name, card_name)
    basename = repo_id.split("/")[-1]
    base_tokens = _id_tokens(basename)
    if not card_tokens or not base_tokens:
        return []
    squashed = re.sub(r"[^a-z0-9]+", "", basename.lower())
    if card_tokens.isdisjoint(base_tokens):
        kind = "aggregate_repo"
        detail = f"resolved HF repo {repo_id!r} shares no name token with card {name!r}"
    else:
        extra = {t for t in base_tokens
                 if t not in card_tokens and t != squashed and len(t) >= 3}
        if not extra:
            return []
        kind = "derivative_repo"
        detail = f"resolved HF repo {repo_id!r} bears {name!r} plus extra tokens {sorted(extra)}"
    return [{"card": name, "path": "benchmark_details.name",
             "kind": f"repo_paper_mismatch_{kind}", "detail": detail}]


def _source_coverage(run_dir):
    """Per-card paper coverage from the docling telemetry sidecar: full_text | abstract_only
    | no_paper. Descriptive context so NS/flag counts are read in light of what source the
    composer actually had (e.g. biolp is abstract-only because the paper host blocked the
    batch), not as failure. Feeds the C1 relabel signal."""
    by_card, counts = {}, Counter()
    pattern = os.path.join(run_dir, "*", "tool_output", "composer", "docling_telemetry_*.json")
    for p in sorted(glob.glob(pattern)):
        name = os.path.basename(p)[len("docling_telemetry_"):-len(".json")]
        tel = _load(p) or {}
        if tel.get("full_text"):
            label = "full_text"
        elif tel.get("degraded_to_abstract_only") or tel.get("paper_url_resolved"):
            label = "abstract_only"
        else:
            label = "no_paper"
        by_card[name] = {"label": label, "reason": tel.get("reason"),
                         "http_status": tel.get("http_status")}
        counts[label] += 1
    return {"counts": dict(counts), "by_card": dict(sorted(by_card.items()))}


def _flag_block(flags):
    return {"count": len(flags), "flags": flags,
            "by_card": dict(Counter(f["card"] for f in flags))}


# Magnitude buckets emitted by the HF size-category tooling, read as an (lo, hi) range.
_SIZE_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
# Breakdown rows whose key names an aggregate (a subtotal / grand total) double-count the
# leaf rows; dropped before summing so a faithful "splits + totals" breakdown is not read as
# exceeding its own size (the arc shape: Challenge_Total / Easy_Total / Total). Split on
# non-alnum because \btotal\b does not match inside "Challenge_Total".
_AGG_BREAKDOWN_TOKENS = {"total", "totals", "sum", "overall", "combined", "all",
                         "aggregate", "subtotal"}


def _parse_size_bucket(size):
    """data.size as an (lo, hi) example range, or None when unparseable / NS. Handles the HF
    magnitude buckets ('Less than 1K examples', '1K to 10K examples') and a leading explicit
    integer for free-text sizes ('23,679 ... prompts' -> exact)."""
    if not isinstance(size, str) or is_not_specified(size):
        return None
    t = size.strip().lower().replace(",", "")

    def _num(tok):
        m = re.match(r"^([\d.]+)\s*([kmb]?)$", tok.replace(" ", ""))
        return float(m.group(1)) * _SIZE_MULT.get(m.group(2), 1) if m else None

    m = re.search(r"less than\s*([\d.]+\s*[kmb]?)", t)
    if m:
        hi = _num(m.group(1))
        return (0.0, hi) if hi else None
    m = re.search(r"([\d.]+\s*[kmb]?)\s*(?:to|-|–)\s*([\d.]+\s*[kmb]?)", t)
    if m:
        lo, hi = _num(m.group(1)), _num(m.group(2))
        if lo and hi:
            return (lo, hi)
    m = re.match(r"^([\d]+)\b", t)
    if m:
        v = float(m.group(1))
        return (v, v)
    return None


def _breakdown_leaf_sum(bd):
    """(sum, n_leaves) over the numeric leaf rows of a size_breakdown dict, excluding aggregate
    rows whose key names a subtotal / total (tokenised on non-alnum)."""
    if not isinstance(bd, dict):
        return 0, 0
    leaves = []
    for k, v in bd.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if set(re.split(r"[^a-z0-9]+", str(k).lower())) & _AGG_BREAKDOWN_TOKENS:
            continue
        leaves.append(v)
    return sum(leaves), len(leaves)


def _ints_in_text(text):
    return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text or "")]


def _prose_magnitude(text):
    """A dataset magnitude implied by an 'M <unit>s ... average of N ... per <unit>' phrase,
    returned as M*N, or None. Conservative: needs both the 'per <unit>' rate and a matching
    '<M> <unit>(s)' count, so a lone large integer never triggers it."""
    if not text:
        return None
    best = None
    for m in re.finditer(r"average of\s+([\d,]+)\s+[\w\s-]*?\bper\s+([a-z]+)", text, re.IGNORECASE):
        n = int(m.group(1).replace(",", ""))
        unit = re.escape(m.group(2).lower())
        cm = re.search(r"([\d,]+)\s+(?:[a-z]+\s+){0,2}" + unit + r"(?:es|s)?\b", text, re.IGNORECASE)
        if cm:
            best = max(best or 0, int(cm.group(1).replace(",", "")) * n)
    return best


def _numeric_contradiction_flags(name, card):
    """data.size / data.size_breakdown contradicting the magnitude the card's own prose implies.
    Three card-internal, corpus-general signals (any fires): a breakdown that is percentages
    (sums to 100 while the prose states a larger total); a breakdown summing above the stated
    size upper bound; an overview whose 'M units * N per unit' magnitude exceeds the size bucket.
    Descriptive only; keyed on shape and arithmetic, never on a benchmark name."""
    flags = []
    data = card.get("data") if isinstance(card.get("data"), dict) else {}
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}
    overview = _value_text(bd.get("overview"))
    rng = _parse_size_bucket(data.get("size"))
    breakdown = data.get("size_breakdown")

    if isinstance(breakdown, dict):
        leaf_sum, n_leaves = _breakdown_leaf_sum(breakdown)
        prose_totals = sorted({i for i in _ints_in_text(overview) if i > 100})
        if n_leaves >= 2 and abs(leaf_sum - 100) < 1e-9 and prose_totals:
            flags.append({"card": name, "path": "data.size_breakdown",
                          "kind": "numeric_internal_contradiction",
                          "detail": f"size_breakdown sums to 100 (reads as percentages) but the overview "
                                    f"states larger totals {prose_totals}"})
        elif rng and n_leaves >= 1 and leaf_sum > rng[1]:
            flags.append({"card": name, "path": "data.size_breakdown",
                          "kind": "numeric_internal_contradiction",
                          "detail": f"size_breakdown sums to {leaf_sum:.0f}, above the stated size upper "
                                    f"bound {rng[1]:.0f}"})

    if rng:
        prod = _prose_magnitude(overview)
        if prod and prod > rng[1]:
            flags.append({"card": name, "path": "data.size",
                          "kind": "numeric_internal_contradiction",
                          "detail": f"overview implies ~{prod} examples, above the stated size upper "
                                    f"bound {rng[1]:.0f}"})
    return flags


# General (non-benchmark) value vocabularies for the two value-typed fields below: a field left
# 'Not specified' whose own evidence still names a value of its type was dropped, not honestly
# abstained. Licence identifiers and language names, never benchmark names.
_LICENSE_RE = re.compile(
    r"\b(cc[\s-]?by(?:[\s-]?[a-z]{2})*(?:[\s-]?\d(?:\.\d)?)?|cc0|creative commons|mit license|"
    r"apache(?:\s+license)?[\s,]*2(?:\.0)?|[al]?gpl(?:[\s-]?v?\d)?|bsd(?:[\s-]\d)?|"
    r"open data commons|odc[\s-]?by|mpl|mozilla public)\b", re.IGNORECASE)
# Unambiguous, single-token language names only (no 'go' / 'r' / 'c' which collide with prose).
_LANGUAGE_NAMES = (
    "english", "chinese", "mandarin", "french", "german", "spanish", "japanese", "korean",
    "russian", "arabic", "portuguese", "italian", "hindi", "dutch", "vietnamese", "turkish",
    "bengali", "indonesian", "python", "java", "javascript", "typescript", "rust", "ruby",
    "php", "scala", "kotlin", "swift", "perl", "haskell", "fortran", "matlab",
)
_LANGUAGE_RE = re.compile(r"\b(" + "|".join(_LANGUAGE_NAMES) + r")\b", re.IGNORECASE)


def _is_usable_evidence(evidence):
    """The text behind a provenance entry carries a real value, not an abstention placeholder."""
    return (isinstance(evidence, str) and bool(evidence.strip())
            and not is_ns_value(evidence)
            and "no evidence provided" not in evidence.lower())


def _available_but_dropped_flags(name, card, prov):
    """A value-typed field shipped as 'Not specified' while its own evidence still names a value
    of that type -- the composer had it and dropped it. Scoped to fields with a recognisable
    closed-form value (a licence id; a language name) so the signal stays low-FP; a generic
    'has any evidence' rule fires on tangential quotes (filenames, repo lists). Descriptive only;
    the vocabularies are general, never keyed on a benchmark name. A field set to any real value
    -- including 'not-applicable' -- is filled, not NS, so it never fires."""
    flags = []
    el = card.get("ethical_and_legal_considerations")
    el = el if isinstance(el, dict) else {}
    bd = card.get("benchmark_details") if isinstance(card.get("benchmark_details"), dict) else {}

    if is_not_specified(el.get("data_licensing")):
        entry = _prov_entry(prov, "ethical_and_legal_considerations", "data_licensing")
        ev = entry.get("evidence") if entry else None
        if _is_usable_evidence(ev):
            m = _LICENSE_RE.search(ev)
            if m:
                flags.append({"card": name, "path": "ethical_and_legal_considerations.data_licensing",
                              "kind": "available_but_dropped",
                              "detail": f"data_licensing NS but evidence names a licence: {m.group(0)!r}"})

    if is_not_specified(bd.get("languages")):
        entry = _prov_entry(prov, "benchmark_details", "languages")
        ev = entry.get("evidence") if entry else None
        src = "languages"
        if not (_is_usable_evidence(ev) and _LANGUAGE_RE.search(ev)):
            # cross-field fallback: a code benchmark's language lives in its data_type evidence
            entry = _prov_entry(prov, "benchmark_details", "data_type")
            ev = entry.get("evidence") if entry else None
            src = "data_type"
        if _is_usable_evidence(ev):
            m = _LANGUAGE_RE.search(ev)
            if m:
                flags.append({"card": name, "path": "benchmark_details.languages",
                              "kind": "available_but_dropped",
                              "detail": f"languages NS but {src} evidence names a language: {m.group(0)!r}"})
    return flags


def compute(run_dir):
    cards = gather_cards(run_dir)
    pv_by_card = _gather_paper_verification(run_dir)
    per_card, all_rows, flagged = {}, [], []
    schema_invalid_by_card = []
    aboutness_all, format_all, dataset_all = [], [], []
    none_override_all, shell_all, repo_mismatch_all = [], [], []
    subject_vs_slug_all = []
    numeric_contra_all, avail_dropped_all = [], []
    paper_search_degraded_all = []
    paper_binding_all = []
    per_field = defaultdict(lambda: {NS: 0, "extracted": 0, "derived": 0, "ungrounded": 0, "judged": None})
    for name, card, prov, hf, eee in cards:
        rows = classify_card(card, prov)
        all_rows.extend(rows)
        aboutness_all.extend(_aboutness_flags(name, card, prov, hf, eee))
        format_all.extend(_format_lint(name, card))
        dataset_all.extend(_dataset_identity_flags(name, card, hf))
        subject_vs_slug_all.extend(_subject_vs_slug_flags(name, card))
        pv = pv_by_card.get(name)
        none_override_all.extend(_none_override_flags(name, pv))
        paper_search_degraded_all.extend(_paper_search_degraded_flags(name, pv))
        paper_binding_all.extend(_paper_binding_flags(name, pv))
        shell_all.extend(_shell_detector_flags(name, card, pv))
        repo_mismatch_all.extend(_repo_paper_mismatch_flags(name, card, hf))
        numeric_contra_all.extend(_numeric_contradiction_flags(name, card))
        avail_dropped_all.extend(_available_but_dropped_flags(name, card, prov))
        m = _metrics_from_rows(rows)
        si_fields = schema_invalid_fields(card)
        m["schema_invalid"] = {"count": len(si_fields), "fields": si_fields}
        per_card[name] = m
        if si_fields:
            schema_invalid_by_card.append({"card": name, "count": len(si_fields), "fields": si_fields})
        for r in rows:
            pf = per_field[r["path"]]
            pf[_KEY[r["label"]]] += 1
            pf["judged"] = r["judged"]
            if r["judged"] and r["label"] == UNGROUNDED:
                flagged.append({"card": name, "path": r["path"], "source": r["source"]})
    aggregate = _metrics_from_rows(all_rows)
    aggregate["per_field"] = dict(sorted(per_field.items()))
    aggregate["hallucination_fields"] = flagged  # the judged ungrounded fields (numerator)
    schema_invalid_total = sum(d["count"] for d in schema_invalid_by_card)
    aggregate["schema_invalid"] = {"count": schema_invalid_total, "by_card": schema_invalid_by_card}
    return {
        "run_dir": os.path.abspath(run_dir),
        "n_cards": len(cards),
        "cards": [c[0] for c in cards],
        "classifier": {"labels": list(LABELS), "derived_sources": sorted(_DERIVED_SOURCES)},
        "quote_verify_rate": _quote_verify(run_dir),
        "docling_degraded": _docling_degraded(run_dir),
        "verdict": _verdict(schema_invalid_total, schema_invalid_by_card, aggregate["hallucination_rate"]),
        "aggregate": aggregate,
        "per_card": per_card,
        "aboutness": _flag_block(aboutness_all),
        "format_lint": _flag_block(format_all),
        "dataset_identity": _flag_block(dataset_all),
        "subject_vs_slug": _flag_block(subject_vs_slug_all),
        "none_override": _flag_block(none_override_all),
        "paper_search_degraded": _flag_block(paper_search_degraded_all),
        "paper_binding_verification": _flag_block(paper_binding_all),
        "hf_match_verification": _flag_block(_hf_match_flags(run_dir)),
        "hf_verification_summary": _hf_verification_summary(run_dir),
        "shell_detector": _flag_block(shell_all),
        "repo_paper_mismatch": _flag_block(repo_mismatch_all),
        "source_coverage": _source_coverage(run_dir),
        "numeric_contradiction": _flag_block(numeric_contra_all),
        "available_but_dropped": _flag_block(avail_dropped_all),
    }


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_summary(metrics):
    print(f"Run: {metrics['run_dir']}")
    print(f"Cards: {metrics['n_cards']}  ({', '.join(metrics['cards'])})")
    print("\nfull-set counts | judged-set headline (hallu% = ungrounded / non-NS-non-derived judged)"
          " | parse = schema_invalid fields shipped as 'Not specified'")
    hdr = (f"{'card':16s} {'NS':>3s} {'extr':>4s} {'driv':>4s} {'ungr':>4s} | "
           f"{'judg':>4s} {'hallu%':>7s} {'NS%':>6s} {'parse':>5s}")
    print(hdr)
    print("-" * len(hdr))

    def _row(name, m):
        fb = m["full_breakdown"]
        si = m.get("schema_invalid", {}).get("count", 0)
        print(f"{name:16s} {fb[NS]:>3d} {fb['extracted']:>4d} {fb['derived']:>4d} {fb['ungrounded']:>4d} | "
              f"{m['judged']['n_fields']:>4d} {_pct(m['hallucination_rate']):>7s} {_pct(m['ns_rate']):>6s} {si:>5d}")

    for name, m in metrics["per_card"].items():
        _row(name, m)
    print("-" * len(hdr))
    a = metrics["aggregate"]
    _row("AGGREGATE", a)

    print(f"\nHeadline (judged set, {a['judged']['n_fields']} fields over the run):")
    print(f"  hallucination-rate = {a['hallucination_numerator']}/{a['hallucination_denominator']} "
          f"= {_pct(a['hallucination_rate'])}   [gate: <= 2%]")
    print(f"  NS-rate            = {_pct(a['ns_rate'])}   (descriptive context, not a bar)")
    if a["hallucination_fields"]:
        print("  judged ungrounded (numerator):")
        for f in a["hallucination_fields"]:
            print(f"    {f['card']:16s} {f['path']:42s} source={f['source']!r}")
    print(f"\nFull-set grounding breakdown (all non-display fields): {a['full_breakdown']}")
    print("Per-source share (full set):")
    for src, d in a["per_source_share"].items():
        print(f"  {src:22s} {d['count']:>4d}  {_pct(d['share'])}")
    qv = metrics["quote_verify_rate"]
    if qv.get("source_file"):
        print(f"\nquote-verify telemetry: {qv['source_file']}")
    else:
        print(f"\nquote-verify-rate: {qv['note']}")
    dd = metrics.get("docling_degraded", {})
    print(f"\ndocling-degraded (paper_url resolved but built abstract-only): {dd.get('count', 0)} card(s)")
    for c in dd.get("cards", []):
        print(f"  {c['card']:16s} reason={c.get('reason')} status={c.get('http_status')}")
    _print_gate(metrics)
    _print_gate_v3(metrics)


def _print_gate(metrics):
    """Prominent GO/NO-GO block. Lists the offending card+fields when a parse failure
    has silently shipped a card with data collapsed to 'Not specified'."""
    v = metrics["verdict"]
    si = v["checks"]["schema_invalid"]
    hr = v["checks"]["hallucination_rate"]
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"GATE: {v['result']}")
    print(bar)
    print(f"  schema_invalid fields (parse failures = data loss, not abstention): "
          f"{si['count']}  [{'PASS' if si['pass'] else 'FAIL'}]")
    print(f"  hallucination-rate: {_pct(hr['rate'])}  (bar <= {_pct(hr['threshold'])})  "
          f"[{'PASS' if hr['pass'] else 'FAIL'}]")
    if si["by_card"]:
        print("\n  offending cards -- fields shipped as 'Not specified' after a parse failure:")
        for entry in si["by_card"]:
            print(f"    {entry['card']} ({entry['count']}):")
            for path in entry["fields"]:
                print(f"      {path}")
    if v["reasons"]:
        print("\n  NO-GO:")
        for reason in v["reasons"]:
            print(f"    - {reason}")
    print(f"\n  {v['note']}")
    print(bar)


def _print_gate_v3(metrics):
    """Descriptive triage layer (Gate v3). Does NOT affect GO/NO-GO -- it annotates a clean
    verdict with content defects the two grounding bars cannot see. Lists offending field
    paths and notes which checks are candidates to promote to hard bars on the ship model."""
    ab = metrics.get("aboutness", {})
    fl = metrics.get("format_lint", {})
    di = metrics.get("dataset_identity", {})
    sc = metrics.get("source_coverage", {})
    bar = "-" * 64
    print(f"\n{bar}")
    print("Gate v3 -- descriptive triage (does NOT affect GO/NO-GO)")
    print(bar)
    print(f"  aboutness flags  : {ab.get('count', 0)}")
    for f in ab.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    print(f"  format_lint flags: {fl.get('count', 0)}")
    for f in fl.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    print(f"  dataset_identity : {di.get('count', 0)}")
    for f in di.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    svs = metrics.get("subject_vs_slug", {})
    print(f"  subject_vs_slug  : {svs.get('count', 0)}")
    for f in svs.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    no = metrics.get("none_override", {})
    print(f"  none_override    : {no.get('count', 0)}")
    for f in no.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    psd = metrics.get("paper_search_degraded", {})
    print(f"  paper_search_degraded: {psd.get('count', 0)}")
    for f in psd.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    pbv = metrics.get("paper_binding_verification", {})
    print(f"  paper_binding_verification: {pbv.get('count', 0)}")
    for f in pbv.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    hfm = metrics.get("hf_match_verification", {})
    hfs = metrics.get("hf_verification_summary", {})
    print(f"  hf_match_verification: {hfm.get('count', 0)} flags "
          f"({hfs.get('total', 0)} cards verified: {hfs.get('by_verdict', {})})")
    for f in hfm.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    sd = metrics.get("shell_detector", {})
    print(f"  shell_detector   : {sd.get('count', 0)}")
    for f in sd.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    rpm = metrics.get("repo_paper_mismatch", {})
    print(f"  repo_paper_mismatch: {rpm.get('count', 0)}")
    for f in rpm.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    print(f"  source_coverage  : {sc.get('counts', {})}")
    for card, d in sc.get("by_card", {}).items():
        print(f"    {card:16s} {d['label']:14s} reason={d.get('reason')}")
    nc = metrics.get("numeric_contradiction", {})
    print(f"  numeric_contradiction: {nc.get('count', 0)}")
    for f in nc.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    ad = metrics.get("available_but_dropped", {})
    print(f"  available_but_dropped: {ad.get('count', 0)}")
    for f in ad.get("flags", []):
        print(f"    {f['card']:16s} {f['path']:34s} {f['kind']}: {f['detail']}")
    print("\n  Promotion candidates (descriptive now; hard-bar once confirmed on the ship model):")
    print("    judge_mis_aboutness, judge_denial_mis_aboutness, metric_noncanonical, data_type_not_atomic,")
    print("    data_type_not_modality, dataset_identity_mismatch, numeric_internal_contradiction,")
    print("    available_but_dropped")
    print("  Stay descriptive: list_not_atomic, name_title_like, related_benchmark_leakage,")
    print("    paper_search_degraded, source_coverage (context for NS/flag counts, not a defect).")
    print(bar)


def main():
    ap = argparse.ArgumentParser(description="Composer gate v2 grounding metrics over a run output dir.")
    ap.add_argument("--run-dir", required=True, help="batch output dir (parent of <name>_<ts> dirs)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        sys.exit(f"not a directory: {args.run_dir}")
    metrics = compute(args.run_dir)
    if metrics["n_cards"] == 0:
        sys.exit(f"no cards under {args.run_dir} (expected */benchmarkcard/benchmark_card_*.json)")
    out_path = os.path.join(args.run_dir, "gate_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print_summary(metrics)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
