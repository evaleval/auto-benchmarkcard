"""Score the synthetic wrong-referent probes against the production referent layers.

Input  eval/probes/referent_probes.jsonl (scripts/generate_referent_probes.py)
Output eval/probes/probe_scores.json + a human summary on stdout

--splice switches to the VERBATIM-span splice probes
(eval/probes/splice_probes.jsonl, scripts/generate_splice_probes.py) and scores them
through the real path: evidence.verify_items with the frame and doc index, then
aboutness.apply_assignment_gates. It reports detection AND false-positive rate together,
plus a pre-fix/post-fix comparison of evidence._resolve_referent (--pre-fix scores only
the old behaviour, via a monkeypatched local copy; src/ is never modified). See the
SPLICE MODE section at the bottom of this file.

Every probe carries two spans: original_text (a genuine, correctly bound span of the
cached source) and perturbed_text (the same span with ONE same-type referent token
swapped, so the claim is bound to the wrong referent). Both spans are pushed through
the SAME pipeline, so the instrument reports a true-positive rate on perturbed AND a
false-positive rate on original. Without both numbers a "detector" that rejects
everything would score 100%.

Pipeline, reconstructed from production (no reimplementation of the checks):

  Stage A source    composer_tool._normalize_docling_text(docling.filtered_text) for
                    docling probes, _get_hf_readme(hf)[:15000] for hf_readme probes.
                    docstructure.build_index() over that same string, so section /
                    table offsets compose with the verified char_start.
  frame             frame.build_frame(slug, eee_metadata, hf_json, doc_index,
                    paper_text=paper_source, llm_handler=None). The LLM sibling pass
                    is deliberately NOT run (see LIMITATIONS), so the frame here is
                    the deterministic subset of the production frame.
  L1 quote-verify   evidence.verify_items() -> reject_reasons.no_match when the span
                    is not a verbatim substring of the normalized source.
  L2 referent       evidence._resolve_referent(claimed="", quote, frame). claimed is
                    left empty on purpose: that is the only branch where the
                    deterministic resolver may contradict the model, because a claimed
                    id that exists in the frame is returned unchecked. A span resolved
                    to a non-target node is the signal the sibling gate acts on.
  L3 gates          aboutness.apply_assignment_gates() on the verified record, run once
                    per field in GATE_FIELD_PANEL (every headline + numeric gate field).
                    A probe counts as caught by L3 if ANY panel field produces a
                    drop/refile on the perturbed span that it does not produce on the
                    original span. That is the most generous reading of L3, i.e. an
                    upper bound on its discriminative power.
  L4 EAV            eav.run_eav() is model-based. It sits behind --with-eav and is NOT
                    exercised: the default path makes zero network calls.

LIMITATIONS baked into this instrument (also emitted into the JSON):
  * frame.build_frame runs with llm_handler=None, so sibling nodes that the production
    LLM sibling pass would add are missing. The frame layer measured here is a LOWER
    bound on the production frame.
  * composer_tool._flag_version_conflicts is a CARD-level, cross-field check (it needs
    a version token majority over >= 2 filled fields). A single span cannot exercise
    it, so it is reported as not-exercisable rather than faked.
  * The probes perturb the TEXT. A real wrong-referent splice keeps the text verbatim
    and moves only the binding. See the honest-read section printed at the end.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from auto_benchmarkcard.tools.composer import aboutness  # noqa: E402
from auto_benchmarkcard.tools.composer import docstructure  # noqa: E402
from auto_benchmarkcard.tools.composer import evidence  # noqa: E402
from auto_benchmarkcard.tools.composer import frame as frame_mod  # noqa: E402
from auto_benchmarkcard.tools.composer.composer_tool import (  # noqa: E402
    _get_hf_readme,
    _normalize_docling_text,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PATH_BASE = os.path.dirname(REPO)

HF_CONTEXT_CAP = 15000  # composer_tool.extract_facts_from_hf_readme

# Every field the assignment gates can act on: the headline set (sibling gate) plus the
# numeric set (claim_role / table_anchor / subset refile).
GATE_FIELD_PANEL = sorted(
    set(aboutness._GATE_HEADLINE_FIELDS) | set(aboutness._GATE_NUMERIC_FIELDS)
)

PROBE_TYPES = ("subset_swap", "version_swap", "split_swap", "number_swap")

PROBE_INPUTS = []  # raw probe dicts, filled in main(); used by probe_quality_audit


# ----------------------------------------------------------------- source loading


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _run_dir_for(doc_rel):
    """<run_dir> from '<run_dir>/tool_output/(docling|hf)/<slug>.json'."""
    abs_doc = os.path.join(PATH_BASE, doc_rel)
    return os.path.dirname(os.path.dirname(os.path.dirname(abs_doc))), abs_doc


class CardContext:
    """Everything Stage A needs for one (card, doc) pair, built once and cached."""

    def __init__(self, slug, doc_rel):
        self.slug = slug
        self.doc_rel = doc_rel
        self.run_dir, self.abs_doc = _run_dir_for(doc_rel)
        self.notes = []

        tool_output = os.path.join(self.run_dir, "tool_output")
        self.hf = _load_json(os.path.join(tool_output, "hf", slug + ".json"))
        self.eee = _load_json(os.path.join(tool_output, "eee", slug + ".json"))
        docling = _load_json(os.path.join(tool_output, "docling", slug + ".json"))

        # paper_source: exactly composer_tool's Stage-A string.
        raw_filtered = (docling or {}).get("filtered_text", "") or ""
        self.paper_source = _normalize_docling_text(raw_filtered)
        # the probe generator used `filtered_text or text`; keep the generator's string
        # so a mismatch can be attributed instead of scored as a detection.
        self.generator_text = raw_filtered or ((docling or {}).get("text", "") or "")

        self.doc_index = None
        if self.paper_source:
            try:
                self.doc_index = docstructure.build_index(self.paper_source)
            except Exception as exc:  # pragma: no cover - defensive, same as production
                self.notes.append("doc_index failed: %s" % exc)

        if "/hf/" in doc_rel:
            self.doc_token = evidence.DOC_HF
            readme = _get_hf_readme(self.hf)
            self.verify_source = readme[:HF_CONTEXT_CAP]
            self.generator_source = readme
            self.verify_index = None  # production passes no doc_index for HF items
            self.source_kind = "hf_readme"
        else:
            self.doc_token = evidence.DOC_PAPER
            self.verify_source = self.paper_source
            self.generator_source = self.generator_text
            self.verify_index = self.doc_index
            self.source_kind = "docling"

        try:
            self.frame = frame_mod.build_frame(
                slug, self.eee, self.hf, doc_index=self.doc_index,
                paper_text=self.paper_source, llm_handler=None,
            )
        except Exception as exc:  # pragma: no cover
            self.notes.append("frame failed: %s" % exc)
            self.frame = None

        self.frame_counts = (
            (self.frame or {}).get("telemetry", {}).get("builder_counts", {})
        )
        self.target_id = (self.frame or {}).get("target_id", "target")


# ----------------------------------------------------------------- one variant run


def _run_variant(ctx, text, field, claim_role="primary_result"):
    """Push one span through verify -> referent -> gates for ONE field.

    Returns a dict with the layer outcomes. shown_text is left empty: the probe window
    is a slice of the source, not of the assembled model context, so the outside_shown
    check is not the axis under test.
    """
    item = {"field": field, "quote": text, "kind": "stated",
            "claim_role": claim_role, "referent": "", "note": ""}
    records, telem = evidence.verify_items(
        [item], ctx.verify_source, ctx.doc_token,
        shown_text="", doc_index=ctx.verify_index, frame=ctx.frame,
    )
    out = {
        "verified": bool(records),
        "reject_reason": "",
        "referent": "",
        "referent_how": "",
        "region": "",
        "table_id": "",
        "char_start": None,
        "gate_actions": [],
        "gate_survived": bool(records),
    }
    if not records:
        reasons = [k for k, v in telem["reject_reasons"].items() if v]
        out["reject_reason"] = reasons[0] if reasons else "unknown"
        return out

    rec = records[0]
    out["referent"] = rec.get("referent", "")
    out["region"] = rec.get("region", "")
    out["table_id"] = rec.get("table_id", "")
    out["char_start"] = rec.get("char_start")
    how = (telem.get("referent_resolution") or {})
    out["referent_how"] = next((k for k, v in how.items() if v), "")

    gate_telem = {}
    kept = aboutness.apply_assignment_gates([copy.deepcopy(rec)], ctx.frame, gate_telem)
    out["gate_actions"] = [
        {"gate": a["gate"], "action": a["action"]} for a in gate_telem.get("actions", [])
    ]
    out["gate_survived"] = bool(kept)
    return out


def _bypass_variant(ctx, text, anchor_start, field, claim_role="primary_result"):
    """L1-BYPASS counterfactual: what the referent + gate layers would do to this span
    IF Stage-A verbatim verification did not stop it first.

    This is the case that matters operationally: a real wrong-referent splice keeps the
    quote verbatim and moves only the binding, so L1 never fires and L2/L3 are the only
    line of defence. The record is built the way verify_items would have built it, with
    section/region/table taken from the doc index at the ORIGINAL span's offset (a real
    splice sits at a real location in the document).
    """
    quote = evidence.normalize_ws(text)
    referent, how = evidence._resolve_referent("", quote, ctx.frame) if ctx.frame else ("", "")
    region = table_id = ""
    section_path = ""
    if ctx.verify_index is not None and anchor_start is not None:
        sec = ctx.verify_index.locate(anchor_start)
        if sec is not None:
            section_path = " > ".join(sec.path)
            region = sec.region
        tab = ctx.verify_index.table_at(anchor_start)
        if tab is not None:
            table_id = tab.table_id
    rec = {"field": field, "quote": quote, "char_start": anchor_start, "doc": ctx.doc_token,
           "kind": "stated", "note": "", "referent": referent, "claim_role": claim_role,
           "section_path": section_path, "region": region, "table_id": table_id}
    gate_telem = {}
    kept = aboutness.apply_assignment_gates([copy.deepcopy(rec)], ctx.frame, gate_telem)
    return {
        "referent": referent, "referent_how": how, "region": region, "table_id": table_id,
        "survived": bool(kept),
        "actions": [{"gate": a["gate"], "action": a["action"]}
                    for a in gate_telem.get("actions", [])],
    }


def _bypass_analysis(ctx, probe):
    """Run the L1-bypass counterfactual over the whole gate field panel."""
    anchor = evidence.verify_quote(evidence.normalize_ws(probe["original_text"]),
                                   evidence.normalize_ws(ctx.verify_source or ""))
    out = {"anchor_char_start": anchor}
    for variant, text in (("original", probe["original_text"]),
                          ("perturbed", probe["perturbed_text"])):
        runs = {f: _bypass_variant(ctx, text, anchor, f) for f in GATE_FIELD_PANEL}
        first = runs[GATE_FIELD_PANEL[0]]
        out[variant] = {
            "referent": first["referent"],
            "referent_how": first["referent_how"],
            "region": first["region"],
            "table_id": first["table_id"],
            "gate_drop_fields": sorted(f for f, r in runs.items() if not r["survived"]),
            "gate_signature": sorted(
                "%s|%s|%s" % (f, a["gate"], a["action"])
                for f, r in runs.items() for a in r["actions"]
                if a["action"] in ("drop", "drop_duplicate", "refile")),
        }
    o, p = out["original"], out["perturbed"]
    caught = "none"
    if p["referent"] != o["referent"]:
        caught = "frame_referent_mismatch"
    elif set(p["gate_signature"]) - set(o["gate_signature"]):
        caught = "deterministic_gate"
    out["caught_by"] = caught
    out["detected"] = caught != "none"
    return out


def _gate_signature(runs):
    """The set of drop/refile actions over the whole field panel."""
    sig = set()
    for field, r in runs.items():
        for a in r["gate_actions"]:
            if a["action"] in ("drop", "drop_duplicate", "refile"):
                sig.add((field, a["gate"], a["action"]))
    return sig


def score_probe(ctx, probe):
    """Full layered scoring of one probe. Returns the per-probe result record."""
    res = {
        "card": probe["card"],
        "probe_type": probe["probe_type"],
        "doc_kind": ctx.source_kind,
        "swapped_from": probe["swapped_from"],
        "swapped_to": probe["swapped_to"],
        "gen_char_start": probe["char_start"],
        "notes": [],
    }

    per_variant = {}
    for variant, text in (("original", probe["original_text"]),
                          ("perturbed", probe["perturbed_text"])):
        runs = {f: _run_variant(ctx, text, f) for f in GATE_FIELD_PANEL}
        any_run = runs[GATE_FIELD_PANEL[0]]
        per_variant[variant] = {
            "verified": any_run["verified"],
            "reject_reason": any_run["reject_reason"],
            "referent": any_run["referent"],
            "referent_how": any_run["referent_how"],
            "region": any_run["region"],
            "table_id": any_run["table_id"],
            "char_start": any_run["char_start"],
            "gate_drop_fields": sorted(
                f for f, r in runs.items() if r["verified"] and not r["gate_survived"]
            ),
            "gate_signature": sorted("|".join(x) for x in _gate_signature(runs)),
        }

    o, p = per_variant["original"], per_variant["perturbed"]

    # --- source-fidelity attribution: did the ORIGINAL fail for a reason that has
    # nothing to do with the detector (production truncation / generator-vs-production
    # source mismatch)?
    fidelity = ""
    if not o["verified"]:
        norm_o = evidence.normalize_ws(probe["original_text"])
        in_gen = norm_o in evidence.normalize_ws(ctx.generator_source or "")
        if ctx.source_kind == "hf_readme" and probe["char_start"] > HF_CONTEXT_CAP:
            fidelity = "outside_production_hf_15k_cap"
        elif in_gen:
            fidelity = "generator_source_differs_from_production_source"
        else:
            fidelity = "original_not_in_any_source"
    res["original_failure_cause"] = fidelity

    # Relaxed-source rescoring: for spans the production HF 15K cap excludes, check both
    # variants against the UNTRUNCATED cached source, so the probe is not simply lost.
    res["relaxed_source"] = None
    if fidelity == "outside_production_hf_15k_cap":
        full = evidence.normalize_ws(ctx.generator_source or "")
        res["relaxed_source"] = {
            "original_verifies": evidence.normalize_ws(probe["original_text"]) in full,
            "perturbed_verifies": evidence.normalize_ws(probe["perturbed_text"]) in full,
        }

    # --- layer attribution, first layer that fires on the perturbed span and NOT on
    # the original span (a layer that fires on both is not discriminating).
    layer = "none"
    detail = ""
    if not p["verified"] and o["verified"]:
        layer = "quote_verify_no_match"
        detail = p["reject_reason"]
    elif p["verified"] and o["verified"]:
        # A referent that is non-target on BOTH spans is not discriminating, so only a
        # drift between the two spans counts as a catch.
        if p["referent"] != o["referent"]:
            layer = "frame_referent_mismatch"
            detail = "%s -> %s (%s)" % (o["referent"], p["referent"], p["referent_how"])
        else:
            added = set(p["gate_signature"]) - set(o["gate_signature"])
            if added:
                layer = "deterministic_gate"
                detail = ";".join(sorted(added))
    elif not p["verified"] and not o["verified"]:
        layer = "unscorable"
        detail = fidelity or "both spans unverifiable"
    res["layer"] = layer
    res["layer_detail"] = detail
    res["detected"] = layer in ("quote_verify_no_match", "frame_referent_mismatch",
                                "deterministic_gate")
    res["original_rejected"] = not o["verified"]

    # --- diagnostics that are NOT available in production (they need the original as
    # a reference). Reported separately so they can never inflate the headline rate.
    if ctx.frame is not None:
        m_o = frame_mod.match_mention(ctx.frame, probe["original_text"])
        m_p = frame_mod.match_mention(ctx.frame, probe["perturbed_text"])
        res["oracle_frame_mentions_original"] = m_o
        res["oracle_frame_mentions_perturbed"] = m_p
        res["oracle_mention_drift"] = sorted(set(m_o) ^ set(m_p))
        res["swapped_from_in_frame"] = bool(
            frame_mod.match_mention(ctx.frame, " %s " % probe["swapped_from"]))
        res["swapped_to_in_frame"] = bool(
            frame_mod.match_mention(ctx.frame, " %s " % probe["swapped_to"]))
    # L1-bypass counterfactual (see _bypass_analysis): the only measurement here
    # that speaks to the referent layer's own power.
    res["l1_bypass"] = _bypass_analysis(ctx, probe)
    res["variants"] = per_variant
    return res


# ----------------------------------------------------------------- reporting


# --------------------------------------------------------------- probe-quality audit

# Broader than frame._MODEL_WORD_RE (which only looks 12 chars back and requires the
# token to abut the family word): here any model-family word within 40 chars before the
# swapped token marks the probe as a MODEL version rather than a benchmark version.
_MODEL_FAMILY_RE = __import__("re").compile(
    r"\b(gpt|llama|claude|gemini|gemma|qwen|mistral|mixtral|deepseek|palm|bert|phi|grok"
    r"|vicuna|falcon|opus|sonnet|haiku|sea-lion|sealion|aya|commandr|command-r|nemo"
    r"|yi|glm|internlm|olmo)\b", __import__("re").IGNORECASE)
_CITATION_RE = __import__("re").compile(r"https?://|arxiv|et al\.|\(\d{4}\)|huggingface\.co",
                                        __import__("re").IGNORECASE)


def _num_value(tok):
    try:
        return float(tok.rstrip("%").replace(",", ""))
    except ValueError:
        return None


def probe_quality_audit(results, probes):
    """Measure the probe set's own softness, independent of any detector.

    A wrong-referent probe is only informative if the perturbed claim stays PLAUSIBLE.
    Two known soft spots: number swaps that land on an implausible magnitude, and
    version swaps that move a MODEL's version (or a citation/URL token) rather than the
    benchmark's, which is not a card referent at all.
    """
    ratios, implausible, long_decimals = [], 0, 0
    model_versions, citation_versions = 0, 0
    subset_self_contradiction = 0
    for probe, res in zip(probes, results):
        t = probe["probe_type"]
        if t == "number_swap":
            a, b = _num_value(probe["swapped_from"]), _num_value(probe["swapped_to"])
            if a and b and min(a, b) > 0:
                r = max(a, b) / min(a, b)
                ratios.append(round(r, 3))
                if r >= 10:
                    implausible += 1
            frac = probe["swapped_to"].split(".")[-1] if "." in probe["swapped_to"] else ""
            if len(frac) > 6:
                long_decimals += 1
        elif t == "version_swap":
            text = probe["original_text"]
            i = text.find(probe["swapped_from"])
            before = text[max(0, i - 40):i] if i >= 0 else ""
            chunk = text[max(0, i - 60):i + 60] if i >= 0 else ""
            if _MODEL_FAMILY_RE.search(before):
                model_versions += 1
            if _CITATION_RE.search(chunk):
                citation_versions += 1
        elif t == "subset_swap":
            window = probe["perturbed_text"].lower()
            if probe["swapped_from"].lower() in window:
                subset_self_contradiction += 1
    ratios.sort()
    n = len(ratios)
    return {
        "number_swap": {
            "n_with_ratio": n,
            "median_magnitude_ratio": ratios[n // 2] if n else None,
            "min_ratio": ratios[0] if n else None,
            "max_ratio": ratios[-1] if n else None,
            "ratio_ge_10_implausible": implausible,
            "swapped_in_value_has_over_6_decimals": long_decimals,
        },
        "version_swap": {
            "model_family_version_not_benchmark_version": model_versions,
            "inside_citation_or_url_context": citation_versions,
        },
        "subset_swap": {
            "perturbed_window_still_names_the_swapped_out_subset": subset_self_contradiction,
        },
    }


def _rate(num, den):
    return round(num / den, 4) if den else None


def summarize(results, contexts, args):
    total = len(results)
    scorable = [r for r in results if r["layer"] != "unscorable"]
    detected = [r for r in scorable if r["detected"]]
    fp = [r for r in results if r["original_rejected"]]
    fp_clean = [r for r in fp if r["original_failure_cause"] == "original_not_in_any_source"]

    by_type = {}
    for t in PROBE_TYPES:
        rows = [r for r in results if r["probe_type"] == t]
        sc = [r for r in rows if r["layer"] != "unscorable"]
        by_type[t] = {
            "n": len(rows),
            "scorable": len(sc),
            "detected": sum(1 for r in sc if r["detected"]),
            "detection_rate_scorable": _rate(sum(1 for r in sc if r["detected"]), len(sc)),
            "originals_rejected": sum(1 for r in rows if r["original_rejected"]),
            "layers": dict(Counter(r["layer"] for r in rows)),
        }

    by_layer = dict(Counter(r["layer"] for r in results))
    by_layer_type = defaultdict(Counter)
    for r in results:
        by_layer_type[r["probe_type"]][r["layer"]] += 1

    # frame reach: could the referent layer even see the swapped tokens?
    frame_reach = {
        t: {
            "swapped_from_is_frame_node": sum(
                1 for r in results if r["probe_type"] == t and r.get("swapped_from_in_frame")),
            "swapped_to_is_frame_node": sum(
                1 for r in results if r["probe_type"] == t and r.get("swapped_to_in_frame")),
            "n": sum(1 for r in results if r["probe_type"] == t),
        }
        for t in PROBE_TYPES
    }

    # L1-bypass counterfactual aggregates
    bypass_by_type = {}
    for t in PROBE_TYPES:
        rows = [r for r in results if r["probe_type"] == t]
        bypass_by_type[t] = {
            "n": len(rows),
            "detected": sum(1 for r in rows if r["l1_bypass"]["detected"]),
            "rate": _rate(sum(1 for r in rows if r["l1_bypass"]["detected"]), len(rows)),
            "caught_by": dict(Counter(r["l1_bypass"]["caught_by"] for r in rows)),
        }
    bypass = {
        "detected": sum(1 for r in results if r["l1_bypass"]["detected"]),
        "rate": _rate(sum(1 for r in results if r["l1_bypass"]["detected"]), total),
        "caught_by": dict(Counter(r["l1_bypass"]["caught_by"] for r in results)),
        "false_positive_note": (
            "no FPR is definable here: the bypass detector is comparative (it needs the "
            "original span as a reference), so it cannot be run on a single production "
            "item. It is a ceiling on L2+L3 discriminative power, not a deployable check."),
        "by_swap_type": bypass_by_type,
    }

    relaxed = [r for r in results if r.get("relaxed_source")]
    relaxed_summary = {
        "n": len(relaxed),
        "original_verifies_untruncated": sum(
            1 for r in relaxed if r["relaxed_source"]["original_verifies"]),
        "perturbed_verifies_untruncated": sum(
            1 for r in relaxed if r["relaxed_source"]["perturbed_verifies"]),
    }

    return {
        "probes": total,
        "probe_quality_audit": probe_quality_audit(results, PROBE_INPUTS),
        "l1_bypass_counterfactual": bypass,
        "relaxed_source_rescoring": relaxed_summary,
        "scorable": len(scorable),
        "unscorable": total - len(scorable),
        "detected": len(detected),
        "detection_rate_scorable": _rate(len(detected), len(scorable)),
        "detection_rate_all_probes": _rate(len(detected), total),
        "false_positives_originals_rejected": len(fp),
        "false_positive_rate_raw": _rate(len(fp), total),
        "false_positive_rate_detector_attributable": _rate(len(fp_clean), total),
        "original_failure_causes": dict(
            Counter(r["original_failure_cause"] for r in results if r["original_rejected"])),
        "by_swap_type": by_type,
        "by_layer": by_layer,
        "by_layer_and_type": {k: dict(v) for k, v in by_layer_type.items()},
        "frame_reach": frame_reach,
        "gate_field_panel": GATE_FIELD_PANEL,
        "eav_run": bool(args.with_eav),
        "cards": {
            slug: {
                "doc": ctx.doc_rel,
                "source_kind": ctx.source_kind,
                "verify_source_chars": len(ctx.verify_source or ""),
                "frame_nodes": len((ctx.frame or {}).get("nodes", [])),
                "frame_builder_counts": ctx.frame_counts,
                "doc_index_sections": len((ctx.doc_index.sections if ctx.doc_index else [])),
                "notes": ctx.notes,
            }
            for slug, ctx in contexts.items()
        },
        "limitations": [
            "frame.build_frame ran with llm_handler=None: the production LLM sibling "
            "pass is absent, so the frame layer measured here is a LOWER bound.",
            "composer_tool._flag_version_conflicts is a card-level cross-field check "
            "and cannot be exercised by a single span; it is not scored.",
            "eav.run_eav (the model-based attribution audit) was NOT run; --with-eav "
            "gates it and was not used.",
            "verify_items ran with shown_text='' (the outside_shown check is skipped). "
            "Production paper extraction passes shown_text=all_paper_content, a "
            "budget/RAG-selected slice of the paper that depends on the embedding stack "
            "and the query; it is not reproducible offline from the probe alone. Some "
            "probe spans would therefore have been rejected as outside_shown in "
            "production regardless of the perturbation.",
            "The probes perturb the source TEXT, so Stage-A verbatim verification is "
            "on the critical path. They do NOT measure detection of a verbatim quote "
            "whose referent binding alone is wrong, which is the production splice "
            "failure mode the referent layer exists for.",
        ],
    }


def print_summary(s, results):
    P = print
    P("=" * 78)
    P("REFERENT-SWAP PROBE SCORES  (deterministic, zero LLM calls)")
    P("=" * 78)
    P("probes                        %d" % s["probes"])
    P("scorable                      %d   (unscorable: %d)" % (s["scorable"], s["unscorable"]))
    P("DETECTED (true positives)     %d / %d  = %s" % (
        s["detected"], s["scorable"], s["detection_rate_scorable"]))
    P("detection rate over ALL       %s" % s["detection_rate_all_probes"])
    P("")
    P("CONTROL on original_text (genuine, correctly bound spans):")
    P("  originals rejected          %d / %d  = %s  (raw FPR)" % (
        s["false_positives_originals_rejected"], s["probes"], s["false_positive_rate_raw"]))
    P("  detector-attributable FPR   %s" % s["false_positive_rate_detector_attributable"])
    for cause, n in sorted(s["original_failure_causes"].items()):
        P("    %-52s %d" % (cause, n))
    P("")
    P("BY SWAP TYPE")
    P("  %-14s %4s %4s %4s %8s %8s" % ("type", "n", "scor", "det", "rate", "orig_rej"))
    for t in PROBE_TYPES:
        b = s["by_swap_type"][t]
        P("  %-14s %4d %4d %4d %8s %8d" % (
            t, b["n"], b["scorable"], b["detected"],
            b["detection_rate_scorable"], b["originals_rejected"]))
    P("")
    P("PER-LAYER ATTRIBUTION (first layer firing on perturbed but NOT on original)")
    for layer, n in sorted(s["by_layer"].items(), key=lambda kv: -kv[1]):
        P("  %-28s %d" % (layer, n))
    P("")
    P("  layer x swap type")
    for t in PROBE_TYPES:
        row = s["by_layer_and_type"].get(t, {})
        P("    %-14s %s" % (t, ", ".join("%s=%d" % kv for kv in sorted(row.items()))))
    P("")
    P("FRAME REACH (is the swapped token even a node in the deterministic frame?)")
    for t in PROBE_TYPES:
        f = s["frame_reach"][t]
        P("  %-14s from_in_frame %d/%d   to_in_frame %d/%d" % (
            t, f["swapped_from_is_frame_node"], f["n"],
            f["swapped_to_is_frame_node"], f["n"]))
    P("")
    b = s["l1_bypass_counterfactual"]
    P("L1-BYPASS COUNTERFACTUAL (what L2 frame/referent + L3 gates catch on their own,")
    P("i.e. the realistic splice where the quote stays verbatim and only the binding moves)")
    P("  detected                    %d / %d  = %s" % (b["detected"], s["probes"], b["rate"]))
    P("  caught_by                   %s" % ", ".join(
        "%s=%d" % kv for kv in sorted(b["caught_by"].items())))
    for t in PROBE_TYPES:
        bt = b["by_swap_type"][t]
        P("    %-14s %d/%d  (%s)" % (t, bt["detected"], bt["n"], bt["rate"]))
    P("  NOTE: %s" % b["false_positive_note"])
    P("")
    rx = s["relaxed_source_rescoring"]
    if rx["n"]:
        P("RELAXED-SOURCE RESCORING of the %d spans past the production HF 15K cap:" % rx["n"])
        P("  original verifies against the untruncated readme   %d/%d" % (
            rx["original_verifies_untruncated"], rx["n"]))
        P("  perturbed verifies against the untruncated readme  %d/%d" % (
            rx["perturbed_verifies_untruncated"], rx["n"]))
        P("  -> with the cap relaxed these would be %d more quote-verify catches, 0 more FPs."
          % (rx["original_verifies_untruncated"] - rx["perturbed_verifies_untruncated"]))
    P("")
    P("CARDS / SOURCES")
    for slug, c in sorted(s["cards"].items()):
        P("  %-28s %-10s src=%6d chars  frame_nodes=%2d  sections=%3d" % (
            slug, c["source_kind"], c["verify_source_chars"],
            c["frame_nodes"], c["doc_index_sections"]))
    P("")
    q = s["probe_quality_audit"]
    P("PROBE-QUALITY AUDIT (softness of the probe set itself, detector-independent)")
    nq = q["number_swap"]
    P("  number_swap magnitude ratio  median %s  range %s..%s" % (
        nq["median_magnitude_ratio"], nq["min_ratio"], nq["max_ratio"]))
    P("    implausible (ratio >= 10x)                 %d / %d" % (
        nq["ratio_ge_10_implausible"], nq["n_with_ratio"]))
    P("    swapped-in value with > 6 decimals         %d" % (
        nq["swapped_in_value_has_over_6_decimals"]))
    vq = q["version_swap"]
    P("  version_swap moves a MODEL family version    %d" % (
        vq["model_family_version_not_benchmark_version"]))
    P("  version_swap sits in a citation / URL chunk  %d" % (
        vq["inside_citation_or_url_context"]))
    P("  subset_swap window still names the old name  %d" % (
        q["subset_swap"]["perturbed_window_still_names_the_swapped_out_subset"]))
    P("")
    P("LIMITATIONS")
    for lim in s["limitations"]:
        P("  - %s" % lim)
    if s["false_positive_rate_raw"] is None:
        P("")
        P("WARNING: the FPR could not be computed; without it this instrument is worthless.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", default=None,
                    help="probe file; defaults to referent_probes.jsonl, or "
                         "splice_probes.jsonl under --splice")
    ap.add_argument("--out", default=None,
                    help="output json; defaults to probe_scores.json, or "
                         "splice_scores.json under --splice")
    ap.add_argument("--splice", action="store_true",
                    help="score the VERBATIM-span splice probes "
                         "(scripts/generate_splice_probes.py) through the real "
                         "verify_items + assignment-gate path instead of the swap probes")
    ap.add_argument("--pre-fix", dest="pre_fix_only", action="store_true",
                    help="--splice only: score ONLY against the pre-a9b5177 resolver "
                         "(monkeypatched local copy; src/ is not modified)")
    ap.add_argument("--with-eav", action="store_true",
                    help="ALSO run the model-based EAV audit (eav.run_eav). This makes "
                         "paid LLM calls; the default path is offline and deterministic.")
    args = ap.parse_args()

    if args.splice:
        args.probes = args.probes or os.path.join(REPO, "eval", "probes",
                                                  "splice_probes.jsonl")
        args.out = args.out or os.path.join(REPO, "eval", "probes", "splice_scores.json")
        if args.with_eav:
            raise SystemExit("--with-eav is not wired for splice mode; this path is offline.")
        run_splice_mode(args)
        return

    args.probes = args.probes or os.path.join(REPO, "eval", "probes",
                                              "referent_probes.jsonl")
    args.out = args.out or os.path.join(REPO, "eval", "probes", "probe_scores.json")

    probes = [json.loads(line) for line in open(args.probes) if line.strip()]
    PROBE_INPUTS.extend(probes)

    contexts = {}
    results = []
    for probe in probes:
        key = probe["card"] + "::" + probe["doc"]
        if key not in contexts:
            contexts[key] = CardContext(probe["card"], probe["doc"])
        results.append(score_probe(contexts[key], probe))

    if args.with_eav:
        # Designed, deliberately not exercised. run_eav needs an llm_handler and would
        # bill per batch; wire it here only when a paid run is explicitly wanted.
        from auto_benchmarkcard.llm_handler import get_llm_handler  # noqa: F401
        raise SystemExit(
            "--with-eav is a design hook only: the EAV layer is model-based and this "
            "run is required to stay offline. Wire get_llm_handler() into eav.run_eav "
            "here when a paid run is authorized."
        )

    ctx_by_slug = {}
    for key, ctx in contexts.items():
        ctx_by_slug[key.split("::")[0] + " (" + ctx.source_kind + ")"] = ctx

    summary = summarize(results, ctx_by_slug, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "probes": results}, f, indent=2)
    print_summary(summary, results)
    print("")
    print("wrote %s" % args.out)



# =====================================================================================
# SPLICE MODE  (eval/probes/splice_probes.jsonl, scripts/generate_splice_probes.py)
# =====================================================================================
#
# The swap probes above edit the source text, so Stage A stops them and the referent
# layer never gets to speak. The splice probes are VERBATIM spans: Stage A is silent by
# construction and the referent layer is the only thing that can adjudicate them. This
# mode runs them through the REAL path (evidence.verify_items with the frame and doc
# index, then aboutness.apply_assignment_gates) and reports, per probe, which layer
# decided and how.
#
# WHAT MAY BE READ AS A MEASUREMENT, AND WHAT MAY NOT
# ---------------------------------------------------------------------------------
# An audit of the first version of this instrument showed that its headline was
# analytically forced: the probe generator selected positives with frame.match_mention,
# which is the same predicate evidence._resolve_referent decides with, so "detection
# 1.0" and "FPR 0.0" were both re-derivable in closed form without ever calling the
# resolver. That is a spec-conformance test printed in the shape of a measurement.
#
# The probe file now carries a "channel" per probe, and this scorer keeps the three
# apart and labels each with what it can support:
#
#   mention  circular by construction. Reported as CONFORMANCE, never as detection.
#            Its value is 1.0 for positives and 0.0 for negatives before the run.
#            It is still worth running: it is the regression test that the override
#            rule and its comparison exemption are wired the way the spec says.
#   section  independent. Selection reads only the enclosing section heading; the
#            resolver only ever sees the quote. Both misses and false positives are
#            possible, and both occur. THIS is the measurement.
#   claim    branch coverage. claimed_referent is varied away from "target", which is
#            the only branch a9b5177 touched. claim_wrong is a class BOTH resolvers
#            fail by design.
#
# Every probe carries expected_referent; scoring is one comparison (resolved ==
# expected) and the channel decides how the resulting number may be quoted.

_SPLICE_NODE_TYPES = ("sibling", "subset", "version", "split", "metric")
_MENTION_NEGATIVES = ("target_only", "comparison", "collision", "bare")
_MENTION_BLINDSPOT = ("multi_nontarget",)
_CLAIM_KINDS = ("claim_correct", "claim_name", "claim_unknown", "claim_wrong")

ADJUDICATION_PATH = os.path.join(REPO, "eval", "probes", "splice_adjudication.jsonl")


def _pre_fix_resolve_referent(claimed, quote, frame):
    """PRE-FIX copy of evidence._resolve_referent (the version before commit a9b5177).

    Verbatim reconstruction from `git show a9b5177`: an exact node id always wins with
    no corroboration, so a claimed "target" (always a valid node id) returned target and
    the quote scan below was unreachable. Installed by monkeypatch under --pre-fix; src/
    is never modified and no old commit is checked out.
    """
    from auto_benchmarkcard.tools.composer import frame as frame_mod
    ids = {n["id"] for n in frame.get("nodes", [])}
    claimed = (claimed or "").strip()
    if claimed in ids:
        return claimed, "claimed_id"
    if claimed:
        hits = frame_mod.match_mention(frame, claimed)
        if len(hits) == 1:
            return hits[0], "claimed_name"
    quote_hits = frame_mod.match_mention(frame, quote)
    target_id = frame.get("target_id", "target")
    non_target = [h for h in quote_hits if h != target_id]
    if len(non_target) == 1 and target_id not in quote_hits:
        return non_target[0], "quote_mention"
    return target_id, "default_target"


def load_adjudication():
    """Hand labels for a sample of probes: {(card, span): {label, note, ...}}.

    label is one of target / non_target / undecidable and answers the only question the
    machine labels beg: is this span, read in its paper, actually a claim about the
    target? The file is written by a human reader (see the report), NOT by
    frame.match_mention, which is the point."""
    rows = {}
    if not os.path.exists(ADJUDICATION_PATH):
        return rows
    for line in open(ADJUDICATION_PATH):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[(r["card"], r["span"])] = r
    return rows


def score_splice_probe(ctx, probe):
    """One splice probe through the production path. Returns the per-probe record."""
    target_id = ctx.target_id
    item = {
        "field": probe["field"],
        "quote": probe["span"],
        "kind": "stated",
        "claim_role": probe.get("claim_role", "primary_result"),
        "referent": probe.get("claimed_referent", "target"),
        "note": "",
    }
    res = {
        "card": probe["card"],
        "channel": probe.get("channel", "mention"),
        "class_label": probe.get("class_label", ""),
        "kind": probe["kind"],
        "node_type": probe.get("node_type", ""),
        "sibling_relation": probe.get("sibling_relation", ""),
        "referential_check": probe.get("referential_check", "n/a"),
        "referential_reason": probe.get("referential_reason", ""),
        "field": probe["field"],
        "region": probe.get("region", ""),
        "section_heading": probe.get("section_heading", ""),
        "span": probe["span"],
        "true_referent": probe["true_referent"],
        "expected_referent": probe.get("expected_referent", probe["true_referent"]),
        "claimed_referent": probe.get("claimed_referent", "target"),
        "target_id": target_id,
        "stage_a_ok": False,
        "stage_a_reject": "",
        "resolved_referent": "",
        "resolved_how": "",
        "gate_actions": [],
        "gate_survived": None,
        "adjudicated_by": "",
    }
    if ctx.frame is None:
        # A card whose frame failed resolves every referent to "" and would read as a
        # move off target, i.e. as a detection. That is a broken probe, not a result.
        res["stage_a_reject"] = "frame_build_failed"
        res["adjudicated_by"] = "BROKEN_PROBE_no_frame"
        return res

    records, telem = evidence.verify_items(
        [item], ctx.verify_source, ctx.doc_token,
        shown_text="", doc_index=ctx.verify_index, frame=ctx.frame,
    )
    res["stage_a_ok"] = bool(records)
    if not records:
        # An unverifiable span is a BROKEN PROBE, not a detection: the generator
        # guarantees verbatim spans, so this can only mean source drift.
        reasons = [k for k, v in telem["reject_reasons"].items() if v]
        res["stage_a_reject"] = reasons[0] if reasons else "unknown"
        res["adjudicated_by"] = "BROKEN_PROBE_stage_a_reject"
        return res

    rec = records[0]
    how = (telem.get("referent_resolution") or {})
    res["resolved_referent"] = rec.get("referent", "")
    res["resolved_how"] = next((k for k, v in how.items() if v), "")

    gate_telem = {}
    kept = aboutness.apply_assignment_gates([copy.deepcopy(rec)], ctx.frame, gate_telem)
    res["gate_actions"] = ["%s:%s" % (a["gate"], a["action"])
                           for a in gate_telem.get("actions", [])]
    res["gate_survived"] = bool(kept)
    res["gate_effective"] = [a for a in res["gate_actions"]
                             if a.split(":")[1] in ("drop", "drop_duplicate", "refile")]
    # The region gate is referent-INDEPENDENT (it keys off docstructure alone), so it
    # fires identically pre- and post-fix. Only the "sibling" gate family acts on the
    # resolved referent, so it alone measures what the referent layer CHANGES about the
    # card.
    res["gate_referent_driven"] = [a for a in res["gate_effective"]
                                   if a.startswith("sibling:")]

    off_target = res["resolved_referent"] != target_id
    res["off_target"] = off_target
    res["correct"] = res["resolved_referent"] == res["expected_referent"]
    if off_target:
        res["adjudicated_by"] = res["resolved_how"] or "referent"
    elif res["gate_effective"]:
        res["adjudicated_by"] = "deterministic_gate:" + res["gate_effective"][0]
    else:
        res["adjudicated_by"] = "none"
    return res


def _bucket(rows):
    return {"n": len(rows), "distinct_spans": len({r["span"] for r in rows}),
            "cards": len({r["card"] for r in rows})}


def summarize_splice(results, label, adjudication):
    """Per-channel summary. No channel's number is reported without its label."""
    live = [r for r in results if r["stage_a_ok"]]
    broken = [r for r in results if not r["stage_a_ok"]]

    def sel(cls):
        return [r for r in live if r["class_label"] == cls]

    # ------------------------------------------------------------------ mention
    mpos = sel("m_positive")
    m_moved = [r for r in mpos if r["off_target"]]
    by_type = {}
    for t in _SPLICE_NODE_TYPES:
        rows = [r for r in mpos if r["node_type"] == t]
        d = [r for r in rows if r["off_target"]]
        by_type[t] = dict(_bucket(rows), moved_off_target=len(d),
                          rate=_rate(len(d), len(rows)),
                          contained=sum(1 for r in rows if r.get("gate_referent_driven")),
                          adjudicated_by=dict(Counter(r["adjudicated_by"] for r in rows)))
    mneg = {}
    for k in _MENTION_NEGATIVES + _MENTION_BLINDSPOT:
        rows = sel(k)
        f = [r for r in rows if r["off_target"]]
        mneg[k] = dict(_bucket(rows), moved_off_target=len(f),
                       rate=_rate(len(f), len(rows)),
                       moved_to=dict(Counter(r["resolved_referent"] for r in f)))
    clean = [r for r in live if r["class_label"] in _MENTION_NEGATIVES]
    clean_fp = [r for r in clean if r["off_target"]]
    strict = [r for r in mpos if r["referential_check"] == "ok"]
    weak = [r for r in mpos if r["referential_check"] != "ok"]

    sib = [r for r in mpos if r["node_type"] == "sibling"]
    sib_rel = dict(Counter(r["sibling_relation"] for r in sib))

    mention = {
        "READ_THIS_FIRST": (
            "CIRCULAR BY CONSTRUCTION. The generator selects these probes with "
            "frame.match_mention, the predicate evidence._resolve_referent decides "
            "with, so every number below is derivable in closed form from the probe "
            "labels without running the resolver: positives move off target, clean "
            "negatives do not. Quote this as a CONFORMANCE test (the override rule and "
            "its comparison exemption are wired as specified), never as a detection "
            "rate or a false-positive rate. The measured numbers are in the 'section' "
            "channel."),
        "positives_conformance": dict(
            _bucket(mpos), moved_off_target=len(m_moved),
            conformance_rate=_rate(len(m_moved), len(mpos)),
            by_node_type=by_type,
            adjudicated_by=dict(Counter(r["adjudicated_by"] for r in mpos)),
            sibling_relation=sib_rel,
            sibling_relation_note=(
                "'contains' siblings are COMPONENTS of the target (EEE 'contains': "
                "HELM's scenarios, a suite's member benchmarks); 'appears_in' siblings "
                "are parent suites. aboutness's sibling gate drops both from headline "
                "fields, but only the second is a rival entity. A quote about a "
                "component being dropped is a different, milder event than a rival "
                "benchmark's claim being caught."),
        ),
        "positives_strict": dict(
            _bucket(strict),
            moved_off_target=sum(1 for r in strict if r["off_target"]),
            rate=_rate(sum(1 for r in strict if r["off_target"]), len(strict)),
            definition=("positives whose span passes the frame's OWN node-creation "
                        "context guard, re-applied to the span. Machine-adjudicated, "
                        "not ground truth: see the adjudication block."),
        ),
        "positives_weak_or_unguarded": dict(
            _bucket(weak),
            moved_off_target=sum(1 for r in weak if r["off_target"]),
            rate=_rate(sum(1 for r in weak if r["off_target"]), len(weak)),
            reasons=dict(Counter(r["referential_reason"] for r in weak)),
            warning=("These spans only CONTAIN a node's name in a context production's "
                     "own node builder would have rejected. Every one that moves is the "
                     "resolver OVER-BINDING. Read as a false-positive channel."),
        ),
        "negatives": dict(
            _bucket(clean), moved_off_target=len(clean_fp),
            rate=_rate(len(clean_fp), len(clean)),
            by_kind={k: mneg[k] for k in _MENTION_NEGATIVES},
            note=("0 by construction, same circularity. 'comparison' now means the span "
                  "names the target AND one distinctively-named sibling/subset (a real "
                  "two-entity sentence); the common-word collisions that used to be "
                  "counted as comparisons are split out as 'collision'."),
        ),
        "negatives_blindspot": {
            "by_kind": {k: mneg[k] for k in _MENTION_BLINDSPOT},
            "note": ("multi_nontarget: 2+ non-target nodes and no target, so the "
                     "'sole non-target' rule leaves it on target. Not decidable from a "
                     "mention list, so a blind spot, not a control."),
        },
    }

    # ------------------------------------------------------------------ section
    spos, sneg = sel("s_positive"), sel("s_negative")
    s_caught = [r for r in spos if r["off_target"]]
    s_exact = [r for r in spos if r["correct"]]
    s_fp = [r for r in sneg if r["off_target"]]
    section = {
        "READ_THIS_FIRST": (
            "INDEPENDENT SELECTION. A span is chosen by the section heading that "
            "encloses it (docstructure.DocIndex.locate(char_start).path); the span's "
            "own text is never read during selection and the resolver never sees the "
            "heading. Both numbers below can come out anywhere in [0,1]. The section "
            "signal is a HYPOTHESIS about aboutness, not certainty: a sentence under a "
            "heading naming benchmark X is usually but not always about X, which is "
            "what the hand adjudication in this file bounds."),
        "positives": dict(
            _bucket(spos), moved_off_target=len(s_caught),
            detection_rate=_rate(len(s_caught), len(spos)),
            resolved_to_the_section_node=len(s_exact),
            exact_rate=_rate(len(s_exact), len(spos)),
            contained=sum(1 for r in spos if r.get("gate_referent_driven")),
            adjudicated_by=dict(Counter(r["adjudicated_by"] for r in spos)),
            note=("A miss here means the section heading names a non-target node and "
                  "the sentence inside it does not, so the quote-only resolver leaves "
                  "the claim on the target. That is the recall the mention channel "
                  "cannot see."),
        ),
        "negatives": dict(
            _bucket(sneg), moved_off_target=len(s_fp),
            false_positive_rate=_rate(len(s_fp), len(sneg)),
            moved_to=dict(Counter(r["resolved_referent"] for r in s_fp)),
            moved_to_node_type=dict(Counter(r["resolved_referent"].split(":")[0]
                                            for r in s_fp)),
            contained=sum(1 for r in sneg if r.get("gate_referent_driven")),
            note=("MEASURED false positives: the enclosing heading names the target and "
                  "no other node, yet the resolver moved the claim off the target "
                  "because the sentence happens to contain some node's name."),
            examples=[{"card": r["card"], "heading": r["section_heading"],
                       "moved_to": r["resolved_referent"],
                       "span": r["span"][:150]} for r in s_fp[:12]],
        ),
    }

    # ------------------------------------------------------------------ claim
    claim = {"READ_THIS_FIRST": (
        "BRANCH COVERAGE. Every probe in the other channels claims 'target', which is "
        "exactly and only the branch a9b5177 changed, so their pre/post delta is forced "
        "and carries no information about the rest of the resolver. Here "
        "claimed_referent is varied. claim_wrong is the other production splice shape: "
        "a target sentence attributed to another node. Both resolvers obey it "
        "unchecked, by design (evidence.py: 'a claimed sibling/subset is already the "
        "specific answer'), so it is a class this layer provably does not catch.")}
    for k in _CLAIM_KINDS:
        rows = sel(k)
        ok = [r for r in rows if r["correct"]]
        claim[k] = dict(_bucket(rows), correct=len(ok),
                        rate=_rate(len(ok), len(rows)),
                        resolved_how=dict(Counter(r["resolved_how"] for r in rows)),
                        resolved_to=dict(Counter(r["resolved_referent"].split(":")[0]
                                                 for r in rows)))

    # ------------------------------------------------------------------ containment
    cont_rows = [r for r in live if r.get("gate_referent_driven")]
    cells = Counter((r["node_type"] or "-", r["field"], a)
                    for r in cont_rows for a in r["gate_referent_driven"])
    containment = {
        "WHAT_THIS_IS": (
            "NOT a rate. Only two cells of aboutness.apply_assignment_gates consume a "
            "resolved referent: a sibling referent drops a quote from a headline field, "
            "and a subset referent re-files a data.size quote to data.size_breakdown. "
            "version / split / metric referents have NO deterministic consumer, so a "
            "detected item keeps flowing with a non-target referent stamped on it and "
            "only the model-based EAV audit and the Stage-B digest ever see it. A "
            "containment 'rate' over a probe set is therefore a property of the probe "
            "mix (node types chosen by the generator's quota, fields chosen by its "
            "stand-in router) rather than of the system, and is not reported here."),
        "gate_table_coverage": "2 of the 5 node types the frame can produce have a "
                               "deterministic consumer; the subset consumer fires only "
                               "on data.size",
        "events": len(cont_rows),
        "by_node_type_field_action": {" | ".join(k): v for k, v in sorted(cells.items())},
        "all_gate_actions": dict(Counter(a for r in live for a in r["gate_actions"])),
    }

    # ------------------------------------------------------------------ adjudication
    adj = {"WHAT_THIS_IS": (
        "Hand labels for a sample of probes, read against the paper with surrounding "
        "context by a single annotator (the author of this instrument, LLM-assisted). "
        "Not independent of the instrument and not a substitute for a second rater; it "
        "is here because every other label in this file is machine-derived and the one "
        "question that decides whether a 'detection' is a detection at all -- is this "
        "span about the target? -- cannot be answered by the matcher under test.")}
    if adjudication:
        labelled = [r for r in live if (r["card"], r["span"]) in adjudication]
        by_class = defaultdict(lambda: Counter())
        cm = Counter()
        harmful = []
        for r in labelled:
            a = adjudication[(r["card"], r["span"])]
            lab = a["label"]
            by_class[r["class_label"]][lab] += 1
            if lab == "undecidable":
                continue
            moved = r["off_target"]
            cm[("about_target" if lab == "target" else "about_non_target",
                "moved" if moved else "stayed")] += 1
            if lab == "target" and r.get("gate_referent_driven"):
                harmful.append({"card": r["card"], "field": r["field"],
                                "action": r["gate_referent_driven"],
                                "moved_to": r["resolved_referent"],
                                "span": r["span"][:140]})
        tp = cm[("about_non_target", "moved")]
        fn = cm[("about_non_target", "stayed")]
        fp = cm[("about_target", "moved")]
        tn = cm[("about_target", "stayed")]
        # A move counts as a STRICT hit only when the annotator also agreed the node the
        # resolver picked is the entity the span is about. Several spans are not about
        # the target and still get bound to the wrong node (a split word collision), and
        # crediting those as detections would repeat the mistake this instrument exists
        # to avoid.
        strict_tp = sum(
            1 for r in labelled
            if adjudication[(r["card"], r["span"])]["label"] == "non_target"
            and r["off_target"]
            and adjudication[(r["card"], r["span"])].get("resolved_node_ok") == "yes")
        # section positives: every mover was adjudicated, the non-movers were sampled,
        # so recall has to be re-weighted onto the full class.
        spos_all = [r for r in live if r["class_label"] == "s_positive"]
        spos_moved = [r for r in spos_all if r["off_target"]]
        spos_stayed = [r for r in spos_all if not r["off_target"]]
        lab_stayed = [adjudication[(r["card"], r["span"])] for r in spos_stayed
                      if (r["card"], r["span"]) in adjudication]
        stayed_nt = sum(1 for a in lab_stayed if a["label"] == "non_target")
        stayed_dec = sum(1 for a in lab_stayed if a["label"] != "undecidable")
        moved_nt = sum(1 for r in spos_moved
                       if (r["card"], r["span"]) in adjudication
                       and adjudication[(r["card"], r["span"])]["label"] == "non_target")
        est_fn = (stayed_nt / len(lab_stayed) * len(spos_stayed)) if lab_stayed else 0.0
        adj.update({
            "n_labelled": len(labelled),
            "labels": dict(Counter(adjudication[(r["card"], r["span"])]["label"]
                                   for r in labelled)),
            "labels_by_class": {k: dict(v) for k, v in sorted(by_class.items())},
            "confusion_matrix_on_decidable": {
                "true_positive_moved_and_about_non_target": tp,
                "false_negative_stayed_but_about_non_target": fn,
                "false_positive_moved_but_about_target": fp,
                "true_negative_stayed_and_about_target": tn,
            },
            "precision": _rate(tp, tp + fp),
            "recall_unweighted_do_not_quote": _rate(tp, tp + fn),
            "strict_true_positives_right_node_too": strict_tp,
            "strict_precision": _rate(strict_tp, tp + fp),
            "section_positive_recall_reweighted": {
                "note": ("Every s_positive that MOVED was adjudicated; the ones that "
                         "stayed were sampled, so the raw recall above over-counts. This "
                         "row re-weights the sampled non-movers onto the whole class."),
                "moved_and_about_non_target": moved_nt,
                "sampled_non_movers": len(lab_stayed),
                "of_them_about_non_target": stayed_nt,
                "non_movers_total": len(spos_stayed),
                "estimated_missed_off_target_claims": round(est_fn, 1),
                "estimated_recall": _rate(moved_nt, moved_nt + est_fn)
                if (moved_nt + est_fn) else "n/a",
                "sampled_non_movers_decidable": stayed_dec,
            },
            "by_stratum": {},
            "harmful_containment": harmful,
            "harmful_containment_note": (
                "gate actions that dropped or re-filed a span a human read as a claim "
                "ABOUT THE TARGET. These are cards being damaged by the referent layer, "
                "and they are counted inside the raw containment event total."),
            "section_channel_only_confusion": {},
            "mention_strict_bucket_check": {},
            "sampling_design": (
                "Strata: every section-channel false positive (all movers among the "
                "s_negatives), every section-channel positive that moved, a random "
                "sample of section positives that stayed, a random sample of "
                "section negatives that stayed, every referent-driven containment "
                "event, and a random sample of the mention channel's 'strict' "
                "positives. Rates computed over this file are stratum-weighted where "
                "the row says so and raw counts otherwise."),
        })
        strata = defaultdict(Counter)
        for r in labelled:
            a = adjudication[(r["card"], r["span"])]
            strata[a.get("stratum", "?")][a["label"]] += 1
        adj["by_stratum"] = {k: dict(v) for k, v in sorted(strata.items())}

        sec_rows = [r for r in labelled
                    if r["class_label"] in ("s_positive", "s_negative")]
        scm = Counter()
        for r in sec_rows:
            a = adjudication[(r["card"], r["span"])]
            if a["label"] == "undecidable":
                continue
            scm[(a["label"], "moved" if r["off_target"] else "stayed")] += 1
        adj["section_channel_only_confusion"] = {
            "true_positive": scm[("non_target", "moved")],
            "false_negative": scm[("non_target", "stayed")],
            "false_positive": scm[("target", "moved")],
            "true_negative": scm[("target", "stayed")],
            "precision": _rate(scm[("non_target", "moved")],
                               scm[("non_target", "moved")] + scm[("target", "moved")]),
            "note": ("The section channel alone, which is the only channel whose "
                     "selection is independent of the resolver. Movers were sampled "
                     "exhaustively and non-movers by random sample, so precision is "
                     "unbiased but the raw recall is not: use the re-weighted row."),
        }
        mstrict = [r for r in labelled if r["class_label"] == "m_positive"
                   and r["referential_check"] == "ok"]
        adj["mention_strict_bucket_check"] = {
            "n_labelled": len(mstrict),
            "really_about_the_target": sum(
                1 for r in mstrict
                if adjudication[(r["card"], r["span"])]["label"] == "target"),
            "really_about_a_non_target_and_bound_to_the_right_node": sum(
                1 for r in mstrict
                if adjudication[(r["card"], r["span"])]["label"] == "non_target"
                and adjudication[(r["card"], r["span"])].get("resolved_node_ok") == "yes"),
            "note": ("The 'strict' bucket is machine-adjudicated by re-applying the "
                     "frame's own context guards to the span. This row is the check on "
                     "that label: a strict positive that a reader calls a claim about "
                     "the TARGET is a false positive the machine label calls a "
                     "detection."),
        }
    else:
        adj["n_labelled"] = 0

    headline = (
        "SECTION channel (independent): detection %s on %d section-bound positives, "
        "FPR %s on %d section-bound negatives. MENTION channel is conformance-only." % (
            section["positives"]["detection_rate"], section["positives"]["n"],
            section["negatives"]["false_positive_rate"], section["negatives"]["n"]))

    return {
        "label": label,
        "broken_probes": len(broken),
        "broken_probe_reasons": dict(Counter(r["stage_a_reject"] for r in broken)),
        "headline": headline,
        "mention_channel_conformance_only": mention,
        "section_channel_measured": section,
        "claim_channel_branch_coverage": claim,
        "containment": containment,
        "hand_adjudication": adj,
    }


def print_splice_summary(s):
    P = print
    P("-" * 78)
    P("%s" % s["label"].upper())
    P("-" * 78)
    if s["broken_probes"]:
        P("  !! %d BROKEN PROBE(S): %s" % (s["broken_probes"], s["broken_probe_reasons"]))
    sec = s["section_channel_measured"]
    men = s["mention_channel_conformance_only"]
    cl = s["claim_channel_branch_coverage"]

    P("")
    P("SECTION CHANNEL  (independent selection -- this is the measurement)")
    p, n = sec["positives"], sec["negatives"]
    P("  positives  n=%3d (%d cards)  moved off target=%3d  DETECTION RATE=%s" % (
        p["n"], p["cards"], p["moved_off_target"], p["detection_rate"]))
    P("             resolved to the section-bound node itself: %d (%s)" % (
        p["resolved_to_the_section_node"], p["exact_rate"]))
    P("  negatives  n=%3d (%d cards)  moved off target=%3d  FALSE-POSITIVE RATE=%s" % (
        n["n"], n["cards"], n["moved_off_target"], n["false_positive_rate"]))
    if n["moved_to_node_type"]:
        P("             over-bound to: %s" % ", ".join(
            "%s=%d" % kv for kv in sorted(n["moved_to_node_type"].items())))
    for ex in n["examples"][:6]:
        P("               - %-16s [%s] -> %s | %s" % (
            ex["card"], ex["heading"][:34], ex["moved_to"], ex["span"][:70]))

    P("")
    P("MENTION CHANNEL  (CONFORMANCE ONLY, circular by construction)")
    mp = men["positives_conformance"]
    P("  positives  n=%3d (%d distinct spans, %d cards)  moved off target=%3d  rate=%s"
      % (mp["n"], mp["distinct_spans"], mp["cards"], mp["moved_off_target"],
         mp["conformance_rate"]))
    P("  %-10s %4s %5s %8s %6s  %s" % ("node type", "n", "moved", "rate", "contd",
                                       "distinct spans / cards"))
    for t in _SPLICE_NODE_TYPES:
        r = mp["by_node_type"][t]
        P("  %-10s %4d %5d %8s %6d  %d / %d" % (
            t, r["n"], r["moved_off_target"], r["rate"], r["contained"],
            r["distinct_spans"], r["cards"]))
    if mp["sibling_relation"]:
        P("  sibling nodes by EEE relation: %s" % ", ".join(
            "%s=%d" % kv for kv in sorted(mp["sibling_relation"].items())))
    st, wk = men["positives_strict"], men["positives_weak_or_unguarded"]
    P("  strict (passes the frame's own context guard)  n=%3d moved=%3d (%s)"
      % (st["n"], st["moved_off_target"], st["rate"]))
    P("  weak/unguarded (name match only)               n=%3d moved=%3d (%s)"
      % (wk["n"], wk["moved_off_target"], wk["rate"]))
    for reason, cnt in sorted(wk["reasons"].items(), key=lambda kv: -kv[1]):
        P("     %-52s %d" % (reason, cnt))
    mn = men["negatives"]
    P("  negatives  n=%3d  moved off target=%d (%s)" % (
        mn["n"], mn["moved_off_target"], mn["rate"]))
    for k in _MENTION_NEGATIVES:
        r = mn["by_kind"][k]
        P("     %-16s n=%3d  moved=%d (%s)" % (k, r["n"], r["moved_off_target"],
                                               r["rate"]))
    for k in _MENTION_BLINDSPOT:
        r = men["negatives_blindspot"]["by_kind"][k]
        P("     %-16s n=%3d  moved=%d (%s)   [blind spot, not in any rate]" % (
            k, r["n"], r["moved_off_target"], r["rate"]))

    P("")
    P("CLAIM CHANNEL  (branch coverage of _resolve_referent)")
    P("  %-16s %4s %8s %8s  %s" % ("kind", "n", "correct", "rate", "resolved via"))
    for k in _CLAIM_KINDS:
        r = cl[k]
        P("  %-16s %4d %8d %8s  %s" % (
            k, r["n"], r["correct"], r["rate"],
            ", ".join("%s=%d" % kv for kv in sorted(r["resolved_how"].items()))))

    c = s["containment"]
    P("")
    P("CONTAINMENT (what the referent layer CHANGES about a card) -- %d events" % c["events"])
    for cell, n in sorted(c["by_node_type_field_action"].items()):
        P("  %-58s %d" % (cell, n))
    P("  gate actions over all probes: %s" % ", ".join(
        "%s=%d" % kv for kv in sorted(c["all_gate_actions"].items())))

    a = s["hand_adjudication"]
    if a.get("n_labelled"):
        P("")
        P("HAND ADJUDICATION  (%d spans labelled by a human reader)" % a["n_labelled"])
        cm = a["confusion_matrix_on_decidable"]
        P("                          moved off target   stayed on target")
        P("  really about non-target %12d %18d" % (
            cm["true_positive_moved_and_about_non_target"],
            cm["false_negative_stayed_but_about_non_target"]))
        P("  really about the target %12d %18d" % (
            cm["false_positive_moved_but_about_target"],
            cm["true_negative_stayed_and_about_target"]))
        P("  precision=%s (strict, right node too: %s)" % (
            a["precision"], a["strict_precision"]))
        rw = a["section_positive_recall_reweighted"]
        P("  section-positive recall, re-weighted: %s  (%d moved and really off-target;"
          " %d of %d sampled non-movers were off-target claims -> ~%s missed of %d)" % (
              rw["estimated_recall"], rw["moved_and_about_non_target"],
              rw["of_them_about_non_target"], rw["sampled_non_movers"],
              rw["estimated_missed_off_target_claims"], rw["non_movers_total"]))
        P("  label mix: %s" % ", ".join("%s=%d" % kv for kv in sorted(a["labels"].items())))
        P("  NOTE: the matrix above pools strata (section channel, containment events, a")
        P("  sample of mention 'strict' positives), so it is a diagnostic, not a rate.")
        sc = a["section_channel_only_confusion"]
        if sc:
            P("  SECTION CHANNEL ONLY  tp=%d fn=%d fp=%d tn=%d  precision=%s" % (
                sc["true_positive"], sc["false_negative"], sc["false_positive"],
                sc["true_negative"], sc["precision"]))
        mb = a["mention_strict_bucket_check"]
        if mb.get("n_labelled"):
            P("  MENTION 'strict' bucket check: %d labelled, %d are really about the "
              "TARGET" % (mb["n_labelled"], mb["really_about_the_target"]))
            P("     (%d are about a non-target AND bound to the right node)"
              % mb["really_about_a_non_target_and_bound_to_the_right_node"])
        P("  HARMFUL containment (a target claim dropped/refiled): %d" % len(
            a["harmful_containment"]))
        for h in a["harmful_containment"][:8]:
            P("     - %-16s %-28s %s | %s" % (h["card"], h["field"], h["action"],
                                              h["span"][:60]))
    P("")


SPLICE_LIMITATIONS = [
    "CIRCULARITY, the finding that decides what this instrument is worth: the mention "
    "channel selects positives with frame.match_mention, the predicate the resolver "
    "decides with, so its numbers are analytically forced and are reported as "
    "CONFORMANCE, never as detection or false-positive rates. The measured numbers come "
    "from the section channel, whose selection signal (the enclosing heading) the "
    "resolver never sees.",
    "The section channel's ground truth is a HYPOTHESIS, not certainty: a sentence "
    "under a heading that names benchmark X is usually about X, but not always. The "
    "hand-adjudicated sample is what bounds that error, and it is one annotator.",
    "The hand adjudication is by the author of the instrument, LLM-assisted, with no "
    "second rater and no inter-rater agreement. It is evidence, not ground truth.",
    "Section-bound positives are concentrated in few papers (a paper only has sections "
    "named after other benchmarks if it is a suite/meta paper), so that class has a "
    "small effective sample even when its row count is not small. Read the distinct-"
    "span and card counts printed next to every n.",
    "The pre/post delta is confined to claimed_referent == 'target', which is the only "
    "branch a9b5177 changed. The share of production traffic that claims 'target' is "
    "NOT measured here (composition_metadata persists the resolution counts but not the "
    "claimed value per record), so the delta cannot be scaled to production as it "
    "stands.",
    "claim_wrong shows a class both resolvers fail: a claimed non-target id is obeyed "
    "unchecked by design. If the extractor ever mis-attributes in that direction, "
    "nothing downstream of this layer contests it.",
    "frame.build_frame ran with llm_handler=None, so the production LLM sibling pass is "
    "absent. Sibling nodes come only from EEE contains/appears_in, and most of those "
    "are 'contains' (components of the target), not rival benchmarks.",
    "The probes are SYNTHETIC splices: a real span of the paper, relabelled. No model "
    "produced them, so they say nothing about how often the extractor actually splices.",
    "verify_items ran with shown_text='' (the outside_shown check is skipped). "
    "Production passes a RAG/budget-selected slice of the paper, which is not "
    "reproducible offline; some spans would have been rejected there regardless.",
    "Paper (docling) source only. HF-readme items take a different path (15K cap, no "
    "doc_index, no region gates) and are not exercised.",
    "The field on each probe comes from a stand-in router in the generator, not from "
    "the LLM extractor that assigns fields in production. Any number whose denominator "
    "depends on the field mix (containment above all) is therefore not a production "
    "rate, and containment is reported as an event count and a gate-table coverage "
    "statement instead.",
    "Cached docling sources: a large share of the cards come from archive run trees "
    "that may predate the current docling version, so the Stage-A string is not "
    "guaranteed to be byte-identical to what a fresh run would produce. The probe file "
    "records the lane per probe.",
    "The wrong-paper screen removes cards whose cached paper is a different document "
    "entirely (six such cards were in the previous probe set). It cannot catch a "
    "HOMONYM: a card whose slug matches an unrelated paper's own name still passes.",
    "eav.run_eav (the model-based attribution audit) was NOT run: this path makes zero "
    "LLM calls.",
]


def run_splice_mode(args):
    probes = [json.loads(line) for line in open(args.probes) if line.strip()]
    adjudication = load_adjudication()
    contexts = {}

    def ctx_for(probe):
        key = probe["card"] + "::" + probe["doc"]
        if key not in contexts:
            contexts[key] = CardContext(probe["card"], probe["doc"])
        return contexts[key]

    def score_all():
        return [score_splice_probe(ctx_for(p), p) for p in probes]

    def dist(rows):
        return {"n": len(rows), "distinct_spans": len({r["span"] for r in rows}),
                "cards": len({r["card"] for r in rows})}

    out = {
        "probe_set": {
            "file": os.path.relpath(args.probes, REPO),
            "n": len(probes),
            "cards": len({p["card"] for p in probes}),
            "by_channel": {ch: dist([p for p in probes if p.get("channel") == ch])
                           for ch in ("mention", "section", "claim")},
            "by_class": {c: dist([p for p in probes if p.get("class_label") == c])
                         for c in sorted({p.get("class_label", "") for p in probes})},
            "fields": dict(Counter(p["field"] for p in probes)),
            "source_lane": dict(Counter(p.get("doc_lane", "") for p in probes)),
            "claimed_referent_kinds": dict(Counter(
                "target" if p["claimed_referent"] == "target" else "non_target_or_other"
                for p in probes)),
        },
        "adjudication_file": os.path.relpath(ADJUDICATION_PATH, REPO)
        if adjudication else None,
        "limitations": SPLICE_LIMITATIONS,
    }

    print("=" * 78)
    print("SPLICE PROBE SCORES  (verbatim spans; real verify_items + assignment gates)")
    print("=" * 78)
    print("probes %d over %d cards" % (out["probe_set"]["n"], out["probe_set"]["cards"]))
    for ch, d in out["probe_set"]["by_channel"].items():
        print("  %-9s n=%-4d distinct spans=%-4d cards=%d" % (
            ch, d["n"], d["distinct_spans"], d["cards"]))
    print("  hand-adjudicated labels available: %d" % len(adjudication))
    print("")

    if not args.pre_fix_only:
        post = score_all()
        out["post_fix"] = summarize_splice(post, "current resolver (post-fix, a9b5177)",
                                           adjudication)
        out["probes_post_fix"] = post
        print_splice_summary(out["post_fix"])

    # --- pre-fix comparison: monkeypatch a local copy of the old resolver.
    saved = evidence._resolve_referent
    try:
        evidence._resolve_referent = _pre_fix_resolve_referent
        pre = score_all()
    finally:
        evidence._resolve_referent = saved
    out["pre_fix"] = summarize_splice(pre, "pre-fix resolver (before a9b5177)",
                                      adjudication)
    if args.pre_fix_only:
        out["probes_pre_fix"] = pre
    print_splice_summary(out["pre_fix"])

    if "post_fix" in out:
        a, b = out["pre_fix"], out["post_fix"]

        def g(s, *path):
            cur = s
            for k in path:
                cur = cur[k]
            return cur

        rows = [
            ("SECTION detection (measured)", "section_channel_measured", "positives",
             "detection_rate"),
            ("SECTION FPR (measured)", "section_channel_measured", "negatives",
             "false_positive_rate"),
            ("mention conformance (forced)", "mention_channel_conformance_only",
             "positives_conformance", "conformance_rate"),
            ("mention negatives (forced)", "mention_channel_conformance_only",
             "negatives", "rate"),
            ("claim_correct correct", "claim_channel_branch_coverage",
             "claim_correct", "rate"),
            ("claim_wrong correct", "claim_channel_branch_coverage",
             "claim_wrong", "rate"),
            ("claim_name correct", "claim_channel_branch_coverage",
             "claim_name", "rate"),
            ("claim_unknown correct", "claim_channel_branch_coverage",
             "claim_unknown", "rate"),
        ]
        out["delta"] = {name: [g(a, *path), g(b, *path)] for name, *path in
                        [(r[0], *r[1:]) for r in rows]}
        out["delta_note"] = (
            "Every probe outside the claim channel claims referent 'target', which is "
            "exactly and only the branch a9b5177 changed. The claim-channel rows are "
            "identical pre and post by construction: they exercise branches the commit "
            "did not touch. Read the delta as a wiring regression test on one branch, "
            "not as a system-level improvement estimate.")
        print("=" * 78)
        print("BEFORE / AFTER")
        print("=" * 78)
        print("  %-34s %-12s %-12s" % ("", "pre-fix", "post-fix"))
        for name, *path in rows:
            print("  %-34s %-12s %-12s" % (name, g(a, *path), g(b, *path)))
        print("")
        print("  %s" % out["delta_note"])
        pre_moved = a["mention_channel_conformance_only"][
            "positives_conformance"]["moved_off_target"]
        if pre_moved:
            print("  WARNING: the pre-fix resolver moved %d mention positive(s) off "
                  "target. The diagnosis predicts 0" % pre_moved)
            print("  by construction (a claimed 'target' is always a valid node id and "
                  "was returned unchecked).")
        else:
            print("  pre-fix movement is 0 on every probe claiming 'target', as the "
                  "diagnosis predicts: the old")
            print("  resolver returned the claimed id unchecked and its quote scan was "
                  "dead code.")
        print("")

    print("LIMITATIONS")
    for lim in SPLICE_LIMITATIONS:
        print("  - %s" % lim)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
