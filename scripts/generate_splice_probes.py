"""Build VERBATIM-span splice probes for the referent layer.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
generate_referent_probes.py builds every positive by EDITING the source text. Stage A
(evidence.verify_quote) is an exact substring test, so every edited span dies there:
the probe never reaches the referent layer at all.

A production splice has the opposite shape. The span stays BYTE-IDENTICAL to the source
and only the BINDING moves: a sentence that is really about a sibling, a subset, a
version, a split, or a metric gets filed under the target benchmark. Stage A is silent
on it by construction, so the referent layer is the only thing that can adjudicate it.
Every span emitted here is a slice of
evidence.normalize_ws(_normalize_docling_text(docling.filtered_text)), so it verifies
verbatim by construction (asserted, see SELF-CHECK below).

THE CIRCULARITY PROBLEM, AND WHAT WE DO ABOUT IT
--------------------------------------------------------------------------------
The first version of this generator selected positives with frame.match_mention, which
is the SAME predicate evidence._resolve_referent uses to decide. An audit re-derived
100% of that instrument's headline in closed form from the probe labels alone, without
ever calling the resolver: detection 1.0 and FPR 0.0 were properties of the selection
rule, not measurements of the resolver. That criticism is correct and it is not fixable
by adding caveats to a tautology.

So this generator now emits THREE selection channels, and the scorer reports them apart:

  channel "mention"  selection predicate == resolver predicate. Kept, but relabelled:
                     it is a SPEC-CONFORMANCE check ("does the resolver act on what is
                     in front of it"), whose value is 1.0 by construction. It is never
                     reported as a detection rate.

  channel "section"  INDEPENDENT selection. A span is chosen by the SECTION HEADING that
                     encloses it (docstructure.DocIndex.locate(char_start).path). The
                     span's own text is never consulted during selection, and the
                     resolver never sees the heading, so it can fail in both directions.
                     s_positive: the heading names exactly one non-target node (a
                     sibling or a distinctively-named subset) and never the target, so a
                     sentence inside it is a candidate wrong-referent claim.
                     s_negative: the heading names the target and no non-target node, so
                     a sentence inside it should stay on the target. Any s_negative the
                     resolver moves off target is a MEASURED false positive.

  channel "claim"    branch coverage of _resolve_referent. Every probe in the other two
                     channels claims "target", which is exactly and only the branch
                     commit a9b5177 changed, so the pre/post delta on them is forced.
                     This channel varies claimed_referent instead: a correct non-target
                     id, a WRONG non-target id (the other production splice shape, which
                     both resolvers obey unchecked -- a class the resolver provably
                     fails), a node NAME, and an unknown id.

Every probe carries expected_referent, so the scorer computes one thing (resolved ==
expected) and the channel decides how the number may be read.

SELF-CHECK: every emitted span is re-verified with evidence.verify_quote against
normalize_ws(source). A span that does not verify is a broken probe, not a detection,
and the script exits non-zero without writing.

Deterministic and offline: frame.build_frame runs with llm_handler=None, no network,
no model calls.

Usage:
  python scripts/generate_splice_probes.py
  python scripts/generate_splice_probes.py --cards blink BBH
  python scripts/generate_splice_probes.py --out /tmp/probes.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from auto_benchmarkcard.tools.composer import docstructure  # noqa: E402
from auto_benchmarkcard.tools.composer import evidence  # noqa: E402
from auto_benchmarkcard.tools.composer import frame as frame_mod  # noqa: E402
from auto_benchmarkcard.tools.composer.composer_tool import (  # noqa: E402
    _normalize_docling_text,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PATH_BASE = os.path.dirname(REPO)

# Checked in next to the probes so the set is reconstructible from the repo alone
# (an earlier default pointed at a per-session scratchpad file).
DEFAULT_INDEX = os.path.join(REPO, "eval", "probes", "docling_index.json")

# Fallback discovery roots, used only when no index file exists.
INTEGRATION = os.path.join(REPO, ".claude", "worktrees", "integration")
SOURCE_ROOTS = [
    os.path.join(REPO, "output", "*", "output"),
    os.path.join(REPO, "output", "current", "*", "output"),
    os.path.join(INTEGRATION, "batch_output", "*", "output"),
    os.path.join(INTEGRATION, "output", "*", "output"),
]

# A long paid regen writes here while this script runs. Nothing may be read from it.
FORBIDDEN_PATH_MARK = os.path.join("output", "phase3_gate_regen20d")

# Sentence boundaries on the whitespace-normalized string (all runs are single spaces).
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

MIN_SPAN_CHARS = 60
MAX_SPAN_CHARS = evidence.QUOTE_CAP  # 300; over-cap quotes get truncated by verify_items
MIN_SPAN_WORDS = 8

NODE_TYPES = ("sibling", "subset", "version", "split", "metric")

MENTION_NEGATIVE_KINDS = ("target_only", "comparison", "collision", "bare",
                          "multi_nontarget")
# The buckets that can be read as a false-positive control at all. "comparison" is the
# exemption a9b5177 preserves; "collision" is the common-word case that used to be
# mislabelled as comparison; multi_nontarget is a known blind spot, reported apart.
MENTION_CLEAN_NEGATIVES = ("target_only", "comparison", "collision", "bare")

CLAIM_KINDS = ("claim_correct", "claim_wrong", "claim_name", "claim_unknown")

# Field routing. Headline fields only (the ones aboutness's gates act on).
FIELD_SIZE = "data.size"
FIELD_BASELINE = "methodology.baseline_results"
FIELD_OVERVIEW = "benchmark_details.overview"

_SIZE_STATEMENT_RE = re.compile(
    r"\b\d[\d,.]*\s*(?:k|m|thousand|million)?\s+"
    r"(examples?|instances?|questions?|samples?|problems?|items?|prompts?|pairs?|"
    r"tasks?|documents?|images?|dialogues?|annotations?|videos?|clips?)\b",
    re.IGNORECASE)
_RESULT_CUE_RE = re.compile(
    r"\b(accuracy|f1|bleu|rouge|score[sd]?|performance|precision|recall|achiev\w*|"
    r"outperform\w*|pass@\d+|win rate|error rate|success rate)\b|%", re.IGNORECASE)
_COMPARISON_CUE_RE = re.compile(
    r"\b(compar\w+|versus|vs\.?|than|whereas|while|unlike|similar to|in contrast|"
    r"outperform\w*|relative to|both|as well as|other benchmarks?)\b", re.IGNORECASE)


# --------------------------------------------------------------------- span hygiene
# Junk-span filters. NONE of them consults the frame, so they cannot bias any channel
# toward or away from the resolver: they only remove strings that are not prose.
_FUNCTION_WORDS = frozenset(
    "the of and to in a is are for we that with on as by this it from at be an our "
    "was were which can not or their its these those has have had than then but each "
    "all more most such when where how they them he she his her you".split())
_INTERNAL_CAPS_RE = re.compile(r"[a-z][A-Z]")
_TOC_DOTS_RE = re.compile(r"\.\s?\.\s?\.\s?\.")
_AFFIL_RE = re.compile(
    r"\b(University|Universit\w+|Institute|Laborator\w+|Corresponding author|"
    r"Google DeepMind|Microsoft Research|equal contribution)\b")


def span_is_prose(s):
    """(ok, reason). Frame-independent junk filter."""
    toks = s.split(" ")
    if len(toks) < MIN_SPAN_WORDS:
        return False, "too_few_words"
    if s.lstrip().startswith("#"):
        return False, "heading_line"
    if "<!-- image -->" in s:
        return False, "image_marker"
    if s.count("|") >= 2:
        return False, "table_or_toc_row"
    if _TOC_DOTS_RE.search(s):
        return False, "toc_dots"
    if "@" in s or (_AFFIL_RE.search(s) and s.count(",") >= 4):
        return False, "front_matter_block"
    if sum(c.isalpha() for c in s) < 0.6 * len(s):
        return False, "low_alpha_ratio"
    fw = sum(1 for t in toks if t.strip(".,;:()[[]\"'").lower() in _FUNCTION_WORDS)
    if fw < 3:
        return False, "not_prose_few_function_words"
    weird = sum(1 for t in toks if _INTERNAL_CAPS_RE.search(t))
    if weird > 0.15 * len(toks):
        return False, "ocr_garbage"
    return True, ""


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _discover_from_roots():
    """Fallback discovery: {slug: run_dir} from the run-dir globs, newest first."""
    import glob
    card_re = re.compile(r"^(.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")
    found = {}
    for root in SOURCE_ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "*")), reverse=True):
            m = card_re.match(os.path.basename(d))
            if not m:
                continue
            slug = m.group(1)
            if slug in found:
                continue
            if os.path.exists(os.path.join(d, "tool_output", "docling", slug + ".json")):
                found[slug] = d
    return found


def load_index(index_path):
    """{slug: run_dir}. Relative paths resolve against PATH_BASE so the checked-in
    index works on any machine. Anything under the live regen dir is refused."""
    idx = _load_json(index_path) if index_path and os.path.exists(index_path) else None
    if not idx:
        return _discover_from_roots()
    out = {}
    for slug, v in sorted(idx.items()):
        path = v.get("path") if isinstance(v, dict) else v
        if not path:
            continue
        if not os.path.isabs(path):
            path = os.path.join(PATH_BASE, path)
        if FORBIDDEN_PATH_MARK in path:
            continue
        out[slug] = path
    return out


# Generic English nouns. A frame node whose NAME is one of these is matched by
# frame.match_mention wherever the plain word occurs, so a "mention" of it is weak
# evidence of a reference. Used to LABEL probes and to keep junk nodes out of the
# independent section channel; never to filter production.
_COMMON_WORDS = frozenset({
    "train", "training", "test", "dev", "validation", "accuracy", "score", "scores",
    "all", "default", "data", "image", "images", "text", "music", "math", "code",
    "base", "main", "full", "mini", "core", "hard", "easy", "medium", "large",
    "small", "long", "short", "public", "private", "open", "closed", "general",
    # junk subset/sibling nodes observed in real frames (frame._subset_nodes has no
    # distinctiveness guard, so common nouns become bindable node names)
    "document", "documents", "response", "responses", "generation", "standard",
    "examples", "example", "setting", "settings", "problems", "problem",
    "questions", "question", "answer", "answers", "vision", "audio", "video",
    "reasoning", "knowledge", "language", "english", "single", "multi", "human",
    "level", "levels", "task", "tasks", "subset", "subsets", "split", "splits",
    "input", "output", "prompt", "prompts", "context", "summary", "label", "labels",
})

# a capitalized token (a product/library name) adjacent to a version token, on EITHER
# side ("Sphinx v4.0.0" and "v4.0.0 of Sphinx" are both third-party version tokens)
_PROPER_BEFORE_VERSION_RE = re.compile(r"\b([A-Z][A-Za-z0-9_+.-]{2,})\s*[-( ]*$")
_PROPER_AFTER_VERSION_RE = re.compile(r"^\s*(?:of|in|from|for)?\s*\b([A-Z][A-Za-z0-9_+.-]{2,})")


def _norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _flat(s):
    return re.sub(r"[^a-z0-9.]+", "", str(s).lower())


class CardSources:
    """Deterministic Stage-A source + frame + doc index for one card, no LLM, no network."""

    def __init__(self, slug, run_dir):
        self.slug = slug
        self.run_dir = run_dir
        self.ok = False
        self.reason = ""
        self.norm_source = ""
        self.target_match_how = ""
        if FORBIDDEN_PATH_MARK in run_dir:
            self.reason = "refused: live regen directory"
            return
        tool_output = os.path.join(run_dir, "tool_output")
        self.doc_path = os.path.join(tool_output, "docling", slug + ".json")
        docling = _load_json(self.doc_path)
        if not docling or not docling.get("success"):
            self.reason = "no successful docling"
            return
        raw = docling.get("filtered_text") or ""
        # exactly composer_tool's Stage-A string for paper items
        self.paper_source = _normalize_docling_text(raw)
        if len(self.paper_source) < 2000:
            self.reason = "paper text under 2000 chars"
            return
        self.eee = _load_json(os.path.join(tool_output, "eee", slug + ".json"))
        self.hf = _load_json(os.path.join(tool_output, "hf", slug + ".json"))
        try:
            self.doc_index = docstructure.build_index(self.paper_source)
        except Exception as exc:
            self.doc_index = None
            self.reason = "doc_index failed: %s" % exc
        try:
            self.frame = frame_mod.build_frame(
                slug, self.eee, self.hf, doc_index=self.doc_index,
                paper_text=self.paper_source, llm_handler=None,
            )
        except Exception as exc:
            self.reason = "frame failed: %s" % exc
            return
        self.target_id = self.frame.get("target_id", "target")
        self.nodes = {n["id"]: n for n in self.frame.get("nodes", [])}
        self.node_type = {i: n["type"] for i, n in self.nodes.items()}
        self.node_name = {i: n["name"] for i, n in self.nodes.items()}
        self.norm_source = evidence.normalize_ws(self.paper_source)
        if self.doc_index is None:
            self.reason = self.reason or "no doc index"
            return
        ok, why = self.screen_paper()
        if not ok:
            self.reason = "wrong-paper screen: " + why
            return
        self.ok = True

    # ----------------------------------------------------------- wrong-paper screen
    def spellings(self, node_id):
        n = self.nodes.get(node_id) or {}
        return [s for s in [n.get("name", ""), *(n.get("aliases") or [])] if s]

    def screen_paper(self):
        """The cached docling file must actually be THIS benchmark's paper.

        Six cards in the previous probe set were built on an unrelated paper (card
        'ace' on a quantum-optics paper, 'mathvision' on the Kimi k1.5 report, 'bbq' on
        an adversarial-training paper, ...); no span from such a card can be
        target-about, so every label on it is noise.

        Screen: some target spelling, the slug, or a slug part of >= 4 chars must occur
        in the paper HEAD (as a token, or as a punctuation-free substring for names of
        >= 6 chars, which is what catches 'LiveCodeBench Pro' for slug
        'livecodebenchpro'), or at least 5 times in the body.

        A pass that rests only on a short slug PART is recorded as weak: it cannot tell
        a homonym apart ('vibe-simulation' matches the VIBE human-pose paper on 'vibe').
        """
        head = self.paper_source[:6000]
        flat_head = _flat(head)
        best = 0
        cands = [(s, "spelling") for s in self.spellings(self.target_id)]
        cands.append((self.slug, "slug"))
        cands += [(p, "slug_part") for p in re.split(r"[-_ ]", self.slug) if len(p) >= 4]
        for cand, origin in cands:
            cand = (cand or "").strip()
            if len(cand) < 4:
                continue
            f = _flat(cand)
            if len(f) >= 6 and f in flat_head:
                self.target_match_how = "flat_head:%s:%s" % (origin, cand)
                return True, ""
            if frame_mod._token_present(head, cand):
                self.target_match_how = "token_head:%s:%s" % (origin, cand)
                return True, ""
            best = max(best, sum(1 for _ in frame_mod._iter_token_hits(
                self.paper_source, cand)))
        if best >= 5:
            self.target_match_how = "body_hits:%d" % best
            return True, ""
        return False, "target name absent from paper head (%d body hits)" % best

    # ------------------------------------------------------------- mention helpers
    def alias_hits(self, node_id, text):
        """(start, end) token hits of any spelling of node_id inside text."""
        out = []
        for sp in self.spellings(node_id):
            if len(sp.strip()) < frame_mod.MIN_ALIAS_CHARS:
                continue
            out.extend(frame_mod._iter_token_hits(text, sp))
        return out

    def target_named_outside(self, span, winner_id):
        """True if the span names the target somewhere the winning node's own name does
        not cover. frame.match_mention resolves overlaps longest-alias-first, so
        'the HealthBench score' reports ONLY metric:healthbench-score and hides that the
        target was named; 'WMDP-Chem' legitimately covers its own 'WMDP'."""
        win = self.alias_hits(winner_id, span)
        for st, en in self.alias_hits(self.target_id, span):
            if not any(ws <= st and en <= we for ws, we in win):
                return True
        return False

    def target_derived(self, node_id):
        """A metric/version/setting node whose NAME contains the target's name is the
        target under another label ('HealthBench score'), not a rival entity."""
        if self.node_type.get(node_id) not in ("metric", "version", "setting"):
            return False
        tg = [_flat(s) for s in self.spellings(self.target_id) if len(s) >= 4]
        for sp in self.spellings(node_id):
            f = _flat(sp)
            if any(t and t in f for t in tg):
                return True
            if len(f) >= 3 and any(f in t for t in tg):
                return True
        return False

    def sibling_relation(self, node_id):
        """'contains' (the node is a COMPONENT of the target) vs 'appears_in' (the node
        is a parent suite the target belongs to). frame._eee_sibling_nodes puts both
        under type 'sibling', and aboutness's sibling gate drops both from headline
        fields. Only the second is a rival entity: a quote about a component of the
        target is not a splice in the same sense. Read off the node description, which
        is the only place the relation survives."""
        if self.node_type.get(node_id) != "sibling":
            return ""
        d = (self.nodes.get(node_id) or {}).get("description", "")
        if "'contains'" in d:
            return "contains"
        if "'appears_in'" in d:
            return "appears_in"
        return "llm_or_unknown"

    def distinctive(self, node_id):
        """Node name specific enough that a section heading naming it is a real signal."""
        name = self.node_name.get(node_id, "")
        if len(name) < 5 or _norm_name(name) in _COMMON_WORDS:
            return False
        if all(w in _COMMON_WORDS for w in _norm_name(name).split()):
            return False
        return True


def referential_check(src, span, node_id, node_type):
    """How strong is the evidence that this span REFERS to the node, rather than merely
    containing a word that happens to be the node's name?

    Re-applies the frame's OWN node-creation guards to the span. Labels probes; never
    filters production. Returns (verdict, reason), verdict in {ok, weak, unguarded}."""
    name = src.node_name.get(node_id, "")
    low = name.lower()
    if node_type in ("sibling", "subset"):
        if _norm_name(name) in _COMMON_WORDS or len(name) < 5:
            return "weak", "short_or_common_name"
        if all(w in _COMMON_WORDS for w in _norm_name(name).split()):
            return "weak", "generic_multiword_name"
        return "ok", ""
    if node_type == "split":
        for rx in (frame_mod._SPLIT_CONTEXT_RE, frame_mod._SPLIT_YAML_RE):
            for m in rx.finditer(span):
                if frame_mod._SPLIT_CANON.get(m.group(1).lower(), m.group(1).lower()) == low:
                    return "ok", ""
        return "weak", "split_word_without_split_context"
    if node_type == "version":
        if not frame_mod._RELEASE_CUE_RE.search(span):
            return "weak", "version_token_without_release_cue"
        for rx in (frame_mod._VERSION_RE, frame_mod._VERSION_WORD_RE):
            for m in rx.finditer(span):
                if "v" + m.group(1) != low and m.group(1) != low.lstrip("v"):
                    continue
                before = span[max(0, m.start() - 40):m.start()]
                after = span[m.end():m.end() + 40]
                if (_PROPER_BEFORE_VERSION_RE.search(before)
                        or _PROPER_AFTER_VERSION_RE.search(after)):
                    return "weak", "third_party_version_suspect"
        return "ok", ""
    if node_type == "metric":
        # production has NO context guard for metric mentions at all
        return "unguarded", "metric_name_match_has_no_production_context_guard"
    return "unguarded", "unknown_node_type"


def pick_field(span):
    """Deterministic headline-field routing from the span's own content.

    NOTE (honesty): production assigns the field with the LLM extractor. This router is
    a stand-in, so any rate whose denominator depends on the field mix (containment, in
    particular) is a property of this function as much as of the system. data.size is
    only used for spans that actually state a count of something."""
    has_digit = any(c.isdigit() for c in span)
    if has_digit and _RESULT_CUE_RE.search(span):
        return FIELD_BASELINE
    if _SIZE_STATEMENT_RE.search(span):
        return FIELD_SIZE
    return FIELD_OVERVIEW


def iter_spans(norm_source):
    """Verbatim prose spans of the normalized source (frame-independent filters only)."""
    for sent in _SENT_RE.split(norm_source):
        s = sent.strip()
        if not (MIN_SPAN_CHARS <= len(s) <= MAX_SPAN_CHARS):
            continue
        ok, _ = span_is_prose(s)
        if ok:
            yield s


def classify_mention(src, span):
    """Channel 'mention': the selection predicate IS frame.match_mention.

    (class_label, true_referent, hits, node_type) or None to skip."""
    hits = frame_mod.match_mention(src.frame, span)
    non_target = [h for h in hits if h != src.target_id]
    if not hits:
        return ("bare", src.target_id, hits, "")
    if src.target_id in hits:
        if not non_target:
            return ("target_only", src.target_id, hits, "")
        if len(non_target) == 1:
            nid = non_target[0]
            ntype = src.node_type.get(nid, "")
            # A real comparison sentence names two ENTITIES. "we test models on X" is a
            # word collision with split:test, not a comparison, and counting it as one
            # inflated the bucket that is supposed to prove the exemption holds: 30 of
            # 38 "comparison" negatives in the previous set were split-word collisions.
            if ntype in ("sibling", "subset") and src.distinctive(nid):
                return ("comparison", src.target_id, hits, ntype)
            return ("collision", src.target_id, hits, ntype)
        return None  # target + several non-targets: not a clean control
    if len(non_target) == 1:
        nid = non_target[0]
        ntype = src.node_type.get(nid, "")
        if not ntype:
            return None
        if src.target_derived(nid):
            return None  # 'the HealthBench score': the node IS the target, relabelled
        if src.target_named_outside(span, nid):
            return None  # the span does name the target; match_mention hid it
        return ("m_positive", nid, hits, ntype)
    return ("multi_nontarget", src.target_id, hits, "")


def section_binding(src, char_start):
    """(node_id or '', heading, 'positive'|'negative'|'') from the ENCLOSING SECTION
    HEADING only. The span's text is never read here: this is the independent channel."""
    sec = src.doc_index.locate(char_start) if src.doc_index is not None else None
    if sec is None:
        return "", "", ""
    heading = " ".join(sec.path)
    hits = frame_mod.match_mention(src.frame, heading)
    non_target = [h for h in hits if h != src.target_id]
    if src.target_id in hits and not non_target:
        if sec.region in ("related_work", "ethics"):
            return "", heading, ""
        return src.target_id, heading, "negative"
    if len(non_target) == 1 and src.target_id not in hits:
        nid = non_target[0]
        if src.node_type.get(nid) not in ("sibling", "subset"):
            return "", heading, ""   # a heading matching split:train is a word collision
        if not src.distinctive(nid):
            return "", heading, ""
        if src.target_derived(nid):
            return "", heading, ""
        return nid, heading, "positive"
    return "", heading, ""


def build_probe(src, span, channel, class_label, true_ref, expected_ref, claimed,
                hits, node_type, heading, selection_signal):
    """One probe record; char_start is what production's verify_quote would return."""
    char_start = evidence.verify_quote(span, src.norm_source)
    region = section_path = table_id = ""
    if src.doc_index is not None and char_start is not None:
        sec = src.doc_index.locate(char_start)
        if sec is not None:
            section_path = " > ".join(sec.path)
            region = sec.region
        tab = src.doc_index.table_at(char_start)
        if tab is not None:
            table_id = tab.table_id
    if channel == "mention" and class_label == "m_positive":
        rc = referential_check(src, span, true_ref, node_type)
    else:
        rc = ("n/a", "")
    doc_rel = os.path.relpath(src.doc_path, PATH_BASE)
    return {
        "card": src.slug,
        "doc": doc_rel,
        "doc_lane": ("archive" if doc_rel.startswith("archive") else "current"),
        "channel": channel,
        "class_label": class_label,
        "kind": "positive" if class_label in ("m_positive", "s_positive") else "negative",
        "selection_signal": selection_signal,
        "span": span,
        "char_start": char_start,
        "true_referent": true_ref,
        "expected_referent": expected_ref,
        "claimed_referent": claimed,
        "matched_node_ids": hits,
        "node_type": node_type,
        "referent_name": src.node_name.get(true_ref, ""),
        "sibling_relation": src.sibling_relation(true_ref) if node_type == "sibling" else "",
        "comparison_cue": bool(_COMPARISON_CUE_RE.search(span)),
        "target_match_how": src.target_match_how,
        "section_heading": heading,
        "referential_check": rc[0],
        "referential_reason": rc[1],
        "field": pick_field(span),
        "claim_role": "primary_result",
        "region": region,
        "section_path": section_path,
        "table_id": table_id,
        "source_kind": "docling",
    }


def collect(src, per_card_cap, section_cap):
    """Candidate probes for one card, capped per bucket so one card cannot dominate.

    Returns (mention_pools, section_pools, claim_seed_pools)."""
    men = defaultdict(list)   # key -> probes   (m_positive:<ntype> | negative kind)
    sec = defaultdict(list)   # 's_positive' | 's_negative'
    claim_seed = defaultdict(list)
    seen = set()
    for span in iter_spans(src.norm_source):
        if span in seen:
            continue
        seen.add(span)
        cs = evidence.verify_quote(span, src.norm_source)
        if cs is None:
            continue

        # ---- channel "section": selection reads ONLY the enclosing heading
        node, heading, role = section_binding(src, cs)
        if role == "positive":
            b = sec["s_positive"]
            if len(b) < section_cap:
                b.append(build_probe(
                    src, span, "section", "s_positive", node, node, "target", [],
                    src.node_type.get(node, ""), heading, "section_heading"))
        elif role == "negative":
            b = sec["s_negative"]
            if len(b) < section_cap:
                b.append(build_probe(
                    src, span, "section", "s_negative", src.target_id, src.target_id,
                    "target", [], "", heading, "section_heading"))

        # ---- channel "mention": selection predicate == resolver predicate
        c = classify_mention(src, span)
        if c is None:
            continue
        label, true_ref, hits, ntype = c
        if label == "m_positive":
            key = "m_positive:" + ntype
            expected = true_ref
        else:
            key = label
            expected = src.target_id
        b = men[key]
        if len(b) < per_card_cap:
            b.append(build_probe(
                src, span, "mention", label if label == "m_positive" else label,
                true_ref, expected, "target", hits, ntype, heading,
                "quote_match_mention"))

        # ---- channel "claim": seeds. A span whose quote names exactly one non-target
        # node seeds the correct/name/unknown claims; a target-only span seeds the
        # WRONG claim (a target fact attributed to another node).
        if label == "m_positive" and len(claim_seed["nontarget"]) < per_card_cap:
            claim_seed["nontarget"].append((span, true_ref, ntype, heading, hits))
        if label == "target_only" and len(claim_seed["target"]) < per_card_cap:
            others = sorted(i for i in src.nodes
                            if i != src.target_id and i not in hits
                            and src.node_type.get(i) in ("sibling", "subset"))
            if others:
                claim_seed["target"].append((span, others[0], src.node_type[others[0]],
                                             heading, hits))
    return men, sec, claim_seed


def round_robin(pools, quota, rng):
    """Take up to `quota` items, cycling over pools (a pool = one card) so the output
    is spread over cards instead of exhausting the first big paper.

    CONSUMES the caller's pools. The previous version copied them (`list(p)`), so a
    caller that called it twice got the same candidates back: 30 of 150 positives in
    the first shipped probe set were duplicate spans and the sibling class was 6
    distinct sentences presented as 11."""
    live = [p for p in pools if p]
    for p in live:
        rng.shuffle(p)
    out = []
    while live and len(out) < quota:
        nxt = []
        for p in live:
            if len(out) >= quota:
                nxt.append(p)
                continue
            out.append(p.pop())
            if p:
                nxt.append(p)
        live = nxt
    return out


def allocate(pools_by_key, keys, budget, rng):
    """Equal quota per key, redistributing what a key cannot supply. Records the true
    candidate supply BEFORE anything is consumed."""
    supply = {k: sum(len(p) for p in pools_by_key.get(k, [])) for k in keys}
    out = []
    remaining = [k for k in keys if pools_by_key.get(k)]
    while remaining and budget > 0:
        share = max(1, budget // len(remaining))
        took = False
        for k in list(remaining):
            got = round_robin(pools_by_key[k], min(share, budget), rng)
            if got:
                out.extend(got)
                budget -= len(got)
                took = True
            if not any(pools_by_key[k]):
                remaining.remove(k)
            if budget <= 0:
                break
        if not took:
            break
    return out, supply


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--cards", nargs="*", default=None)
    ap.add_argument("--max-cards", type=int, default=400)
    ap.add_argument("--per-card-cap", type=int, default=2)
    ap.add_argument("--section-per-card-cap", type=int, default=6,
                    help="section-bound spans are scarce and concentrated in few "
                         "papers, so this cap is looser than --per-card-cap")
    ap.add_argument("--mention-positives", type=int, default=120)
    ap.add_argument("--mention-negatives", type=int, default=160)
    ap.add_argument("--section-positives", type=int, default=60)
    ap.add_argument("--section-negatives", type=int, default=80)
    ap.add_argument("--claim-probes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "probes",
                                                  "splice_probes.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    index = load_index(args.index)
    slugs = args.cards if args.cards else sorted(index)
    if not args.cards:
        rng.shuffle(slugs)

    used, skipped = [], []
    men_pools = defaultdict(list)
    sec_pools = defaultdict(list)
    claim_pools = defaultdict(list)
    for slug in slugs:
        if len(used) >= args.max_cards:
            break
        run_dir = index.get(slug)
        if not run_dir:
            skipped.append((slug, "not in index"))
            continue
        src = CardSources(slug, run_dir)
        if not src.ok:
            skipped.append((slug, src.reason))
            continue
        men, sec, seeds = collect(src, args.per_card_cap, args.section_per_card_cap)
        if not men and not sec:
            skipped.append((slug, "no candidate spans"))
            continue
        for k, probes in men.items():
            men_pools[k].append(probes)
        for k, probes in sec.items():
            sec_pools[k].append(probes)
        for k, rows in seeds.items():
            claim_pools[k].append([(src, r) for r in rows])
        used.append((slug, os.path.relpath(run_dir, PATH_BASE),
                     len(src.frame.get("nodes", [])),
                     sum(len(v) for v in men.values()), sum(len(v) for v in sec.values())))

    pos_keys = ["m_positive:" + t for t in NODE_TYPES]
    sel_men_pos, sup_men_pos = allocate(men_pools, pos_keys, args.mention_positives, rng)
    sel_men_neg, sup_men_neg = allocate(men_pools, list(MENTION_NEGATIVE_KINDS),
                                        args.mention_negatives, rng)
    sel_sec_pos, sup_sec_pos = allocate(sec_pools, ["s_positive"],
                                        args.section_positives, rng)
    sel_sec_neg, sup_sec_neg = allocate(sec_pools, ["s_negative"],
                                        args.section_negatives, rng)

    # ---- channel "claim": branch coverage of _resolve_referent
    per_kind = max(1, args.claim_probes // len(CLAIM_KINDS))
    seeds_nt, sup_nt = allocate(claim_pools, ["nontarget"], per_kind * 3, rng)
    seeds_tg, sup_tg = allocate(claim_pools, ["target"], per_kind, rng)
    claim_probes = []
    for i, (src, (span, nid, ntype, heading, hits)) in enumerate(seeds_nt):
        kind = ("claim_correct", "claim_name", "claim_unknown")[i % 3]
        if kind == "claim_correct":
            claimed, expected = nid, nid
        elif kind == "claim_name":
            claimed, expected = src.node_name.get(nid, ""), nid
        else:
            claimed, expected = "sibling:__not-a-node__", nid
        claim_probes.append(build_probe(
            src, span, "claim", kind, nid, expected, claimed, hits, ntype, heading,
            "claimed_referent_variant"))
    for src, (span, nid, ntype, heading, hits) in seeds_tg:
        # the span names the TARGET only; the claim attributes it to another node.
        # A correct resolver keeps it on the target. Both resolvers obey the claim.
        claim_probes.append(build_probe(
            src, span, "claim", "claim_wrong", src.target_id, src.target_id, nid,
            hits, ntype, heading, "claimed_referent_variant"))

    probes = sel_men_pos + sel_men_neg + sel_sec_pos + sel_sec_neg + claim_probes

    # ---- global dedup: one (card, span, class_label, claimed) row at most, and never
    # the same (card, span) twice inside a channel.
    seen_rows, seen_chan, seen_span = set(), set(), set()
    deduped = []
    for p in probes:
        k1 = (p["card"], p["span"], p["class_label"], p["claimed_referent"])
        k2 = (p["card"], p["span"], p["channel"])
        # the same sentence can reach the set through two card slugs built on the SAME
        # paper (swe-bench / swe-bench-verified), which would double-count one piece of
        # evidence, so a span text is used once per class.
        k3 = (p["span"], p["class_label"], p["claimed_referent"])
        if k1 in seen_rows or k2 in seen_chan or k3 in seen_span:
            continue
        seen_rows.add(k1)
        seen_chan.add(k2)
        seen_span.add(k3)
        deduped.append(p)
    n_dropped_dupes = len(probes) - len(deduped)
    probes = deduped
    probes.sort(key=lambda p: (p["card"], p["channel"], p["class_label"],
                               p["char_start"] or 0, p["claimed_referent"]))

    # ------------------------------------------------------------ SELF-CHECK
    cache = {}
    broken = []
    for p in probes:
        key = (p["card"], p["doc"])
        if key not in cache:
            run_dir = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.join(PATH_BASE, p["doc"]))))
            s = CardSources(p["card"], run_dir)
            cache[key] = s.norm_source
        norm_src = cache[key]
        idx = evidence.verify_quote(evidence.normalize_ws(p["span"]), norm_src)
        if idx is None:
            broken.append((p["card"], p["span"][:80]))
        elif idx != p["char_start"]:
            broken.append((p["card"], "char_start drift %s != %s" % (idx, p["char_start"])))
        if len(evidence.normalize_ws(p["span"])) > evidence.QUOTE_CAP:
            broken.append((p["card"], "over QUOTE_CAP, would be truncated"))
    if broken:
        print("SELF-CHECK FAILED: %d probe(s) do not verify verbatim." % len(broken),
              file=sys.stderr)
        for card, why in broken[:20]:
            print("  %-24s %s" % (card, why), file=sys.stderr)
        raise SystemExit(2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ honest distribution
    def dist(rows):
        return "%d rows / %d distinct spans / %d cards" % (
            len(rows), len({r["span"] for r in rows}), len({r["card"] for r in rows}))

    supply = {}
    supply.update(sup_men_pos)
    supply.update(sup_men_neg)
    supply.update(sup_sec_pos)
    supply.update(sup_sec_neg)
    P = print
    P("=" * 78)
    P("SPLICE PROBES  (verbatim spans; Stage A silent by construction)")
    P("=" * 78)
    P("wrote %d probes to %s" % (len(probes), args.out))
    P("  dropped as duplicates during selection: %d" % n_dropped_dupes)
    P("  SELF-CHECK: %d/%d spans verify verbatim via evidence.verify_quote" % (
        len(probes), len(probes)))
    P("")
    P("CHANNELS  (the scorer reports these apart; read the caveat on each)")
    for ch, note in (("mention", "selection predicate == resolver predicate -> SPEC "
                                 "CONFORMANCE ONLY, not detection"),
                     ("section", "selection from the enclosing heading only -> the "
                                 "resolver can fail either way"),
                     ("claim", "claimed_referent varied -> branch coverage of "
                               "_resolve_referent")):
        rows = [p for p in probes if p["channel"] == ch]
        P("  %-9s %-58s %s" % (ch, note, dist(rows)))
    P("")
    P("MENTION CHANNEL: positives by node type (candidate supply measured BEFORE")
    P("selection, so it is a real pool size)")
    for t in NODE_TYPES:
        rows = [p for p in probes if p["class_label"] == "m_positive"
                and p["node_type"] == t]
        P("  %-10s %s   (pool: %d)" % (t, dist(rows), supply.get("m_positive:" + t, 0)))
    P("MENTION CHANNEL: negatives")
    for k in MENTION_NEGATIVE_KINDS:
        rows = [p for p in probes if p["class_label"] == k]
        P("  %-16s %s   (pool: %d)" % (k, dist(rows), supply.get(k, 0)))
    P("")
    P("SECTION CHANNEL (independent selection)")
    for k in ("s_positive", "s_negative"):
        rows = [p for p in probes if p["class_label"] == k]
        P("  %-16s %s   (pool: %d)" % (k, dist(rows), supply.get(k, 0)))
    P("CLAIM CHANNEL (branch coverage)")
    for k in CLAIM_KINDS:
        rows = [p for p in probes if p["class_label"] == k]
        P("  %-16s %s" % (k, dist(rows)))
    P("")
    P("REFERENTIAL-PLAUSIBILITY AUDIT of the mention positives (the frame's own")
    P("context guards, re-applied to the span; labels probes, never filters production)")
    P("  %-10s %5s %5s %10s  top reason" % ("node type", "ok", "weak", "unguarded"))
    mpos = [p for p in probes if p["class_label"] == "m_positive"]
    for t in NODE_TYPES:
        rows = [p for p in mpos if p["node_type"] == t]
        if not rows:
            continue
        v = Counter(r["referential_check"] for r in rows)
        reasons = Counter(r["referential_reason"] for r in rows if r["referential_reason"])
        top = reasons.most_common(1)[0] if reasons else ("", 0)
        P("  %-10s %5d %5d %10d  %s%s" % (
            t, v.get("ok", 0), v.get("weak", 0), v.get("unguarded", 0),
            top[0], (" x%d" % top[1]) if top[1] else ""))
    allv = Counter(p["referential_check"] for p in mpos)
    P("  TOTAL      %5d %5d %10d" % (
        allv.get("ok", 0), allv.get("weak", 0), allv.get("unguarded", 0)))
    P("")
    P("FIELDS")
    for fld, n in sorted(Counter(p["field"] for p in probes).items()):
        P("  %-34s %d" % (fld, n))
    P("REGIONS")
    for reg, n in sorted(Counter(p["region"] or "(none)" for p in probes).items(),
                         key=lambda kv: -kv[1]):
        P("  %-22s %d" % (reg, n))
    P("SOURCE LANE (cached docling provenance; archive trees may predate the current")
    P("docling version, so the Stage-A string is not guaranteed to be today's)")
    for lane, n in sorted(Counter(p["doc_lane"] for p in probes).items()):
        P("  %-22s %d" % (lane, n))
    P("")
    P("CARDS USED (%d)   nodes / mention cand / section cand / selected" % len(used))
    for slug, d, nodes, nmen, nsec in sorted(used):
        n_here = sum(1 for p in probes if p["card"] == slug)
        if not n_here:
            continue
        P("  %-26s nodes=%3d men=%2d sec=%2d selected=%2d  %s" % (
            slug, nodes, nmen, nsec, n_here, d))
    P("")
    P("CARDS SKIPPED: %d" % len(skipped))
    for why, n in Counter(w.split(":")[0] for _, w in skipped).most_common():
        P("  %-40s %d" % (why, n))
    wrong = [(s, w) for s, w in skipped if w.startswith("wrong-paper")]
    P("  wrong-paper screen rejected %d card(s):" % len(wrong))
    for s, w in wrong[:20]:
        P("    %-24s %s" % (s, w))
    P("")
    P("HONEST READ")
    P("  * The MENTION channel is circular by construction: its positives are selected")
    P("    with frame.match_mention, the same predicate evidence._resolve_referent")
    P("    decides with. Its detection number is 1.0 before the resolver runs. It is a")
    P("    spec-conformance check, and the scorer prints it as one.")
    P("  * The SECTION channel is the measurement. Selection reads only the enclosing")
    P("    heading; the resolver only ever sees the quote. It can miss (heading names a")
    P("    non-target node, quote does not) and it can over-bind (heading names the")
    P("    target, quote contains some other node's name).")
    P("  * The CLAIM channel is the only place a probe can fail for both resolvers:")
    P("    claim_wrong attributes a target sentence to another node and both the pre-fix")
    P("    and the post-fix resolver obey it unchecked.")
    P("  * frame.build_frame ran with llm_handler=None, so the production LLM sibling")
    P("    pass is absent. Sibling nodes here are the EEE-declared ones only.")


if __name__ == "__main__":
    main()
