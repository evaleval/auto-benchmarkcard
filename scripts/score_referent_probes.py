"""Score the synthetic wrong-referent probes against the production referent layers.

Input  eval/probes/referent_probes.jsonl (scripts/generate_referent_probes.py)
Output eval/probes/probe_scores.json + a human summary on stdout

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
    ap.add_argument("--probes", default=os.path.join(REPO, "eval", "probes",
                                                     "referent_probes.jsonl"))
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "probes",
                                                  "probe_scores.json"))
    ap.add_argument("--with-eav", action="store_true",
                    help="ALSO run the model-based EAV audit (eav.run_eav). This makes "
                         "paid LLM calls; the default path is offline and deterministic.")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
