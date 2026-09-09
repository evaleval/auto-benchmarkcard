"""Synthesize wrong-referent (splice) probes from cached per-card sources.

FactCC/CLIFF-style entity swaps: take genuine source text (docling paper
text, HF readme), swap one same-type referent token for another plausible
one, and record the perturbation. Zero human labels; every probe is a
guaranteed wrong-referent positive for stress-testing an assignment
verifier.

Probe types:
  subset_swap   a subset/config name of the card for another subset name
                of the same card (names from size_breakdown keys or HF
                card_data configs)
  version_swap  a version token (v1, v2.0, "version 1.0") for a sibling
                version, preferring another version token from the same doc
  split_swap    a dataset split word (train/validation/dev/test) for a
                different split word
  number_swap   a numeric literal in a results-like sentence for a
                different numeric literal that also occurs elsewhere in
                the same document, so both numbers are genuine and only
                the binding is wrong

Sources are searched in this checkout (output/*/output, the phase dirs behind
output/current, and batch_output/*/output), then in any additional read-only
roots configured with BENCHMARKCARD_SOURCE_ROOTS. Cards without cached text are skipped.

Usage:
  python scripts/generate_referent_probes.py --cards acebench ai2d
  python scripts/generate_referent_probes.py --limit 10   # auto-discover
"""

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import extract_card  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH_BASE = os.path.dirname(REPO)

# Local cache roots first, followed by any explicitly configured archive roots.
SOURCE_ROOTS = [
    os.path.join(REPO, "output", "*", "output"),
    os.path.join(REPO, "output", "current", "*", "output"),
    os.path.join(REPO, "batch_output", "*", "output"),
]
# Each path/glob must point to a directory containing timestamped card run dirs.
# Separate multiple roots with os.pathsep (":" on Unix, ";" on Windows).
SOURCE_ROOTS.extend(
    os.path.expanduser(path)
    for path in os.environ.get("BENCHMARKCARD_SOURCE_ROOTS", "").split(os.pathsep)
    if path
)

CARD_DIR_RE = re.compile(r"^(.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")

SPLIT_WORDS = ["train", "validation", "test", "dev"]
SPLIT_CANON = {"training": "train"}

# split word followed by a set/split noun, or a "split: train" style binding
SPLIT_CONTEXT_RE = re.compile(
    r"\b(train|training|validation|dev|test)\b"
    r"(?=\s+(?:set|sets|split|splits|data|examples|instances|partition)\b)",
    re.IGNORECASE,
)
SPLIT_YAML_RE = re.compile(r"\bsplits?:\s*(train|training|validation|dev|test)\b", re.IGNORECASE)

VERSION_RE = re.compile(r"(?<![\w.])[vV](\d+(?:\.\d+)*)(?![\w.])")
VERSION_WORD_RE = re.compile(r"\b[vV]ersion\s+(\d+(?:\.\d+)*)(?![\w.])")

NUMBER_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(%?)(?![\w.,%])")
RESULTS_KEYWORDS = [
    "accuracy", "score", "f1", "pass@", "bleu", "rouge", "achiev", "performance",
    "precision", "recall", "win rate", "error rate", "solve", "correct", "%",
]

WINDOW = 150


def find_card_dirs(name):
    """All cached run dirs for a card, local roots before configured archives."""
    dirs = []
    for root in SOURCE_ROOTS:
        hits = glob.glob(os.path.join(root, name + "_20*"))
        # newest run first within each root
        dirs.extend(sorted(hits, reverse=True))
    return dirs


def discover_cards():
    names = set()
    for root in SOURCE_ROOTS:
        for d in glob.glob(os.path.join(root, "*")):
            m = CARD_DIR_RE.match(os.path.basename(d))
            if m:
                names.add(m.group(1))
    return sorted(names)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_sources(card_dir, name):
    """Return {doc_label: (source_path, text)} for the cached docs we trust."""
    docs = {}
    docling = load_json(os.path.join(card_dir, "tool_output", "docling", name + ".json"))
    if docling and docling.get("success"):
        text = docling.get("filtered_text") or docling.get("text") or ""
        if len(text) > 200:
            docs["docling"] = (os.path.join(card_dir, "tool_output", "docling", name + ".json"), text)
    hf = load_json(os.path.join(card_dir, "tool_output", "hf", name + ".json"))
    if hf:
        readme = hf.get("readme_markdown") or ""
        if len(readme) > 200:
            docs["hf_readme"] = (os.path.join(card_dir, "tool_output", "hf", name + ".json"), readme)
    return docs, hf


def subset_names(card_dir, name, hf):
    """Subset/config names from the card's own content and cached HF configs."""
    names = []
    card = load_json(os.path.join(card_dir, "benchmarkcard", "benchmark_card_" + name + ".json"))
    if card:
        card = extract_card(card)
        sb = card.get("data", {}).get("size_breakdown")
        if isinstance(sb, dict):
            names.extend(sb.keys())
        for key in ("subtasks", "variants", "subsets"):
            val = card.get("benchmark_details", {}).get(key)
            if isinstance(val, list):
                names.extend(v for v in val if isinstance(v, str))
    if hf and isinstance(hf.get("card_data"), dict):
        for cfg in hf["card_data"].get("configs") or []:
            if isinstance(cfg, dict) and cfg.get("config_name"):
                names.append(cfg["config_name"])
    out = []
    for n in names:
        n = n.strip()
        if len(n) >= 3 and not n.isdigit() and n.lower() not in SPLIT_WORDS and n not in out:
            out.append(n)
    return out


def has_boundary(text, start, end):
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def match_case(new, old):
    if old[:1].isupper() and new[:1].islower():
        return new[:1].upper() + new[1:]
    return new


def make_probe(card, ptype, doc_label, doc_path, text, start, old, new):
    a = max(0, start - WINDOW)
    b = min(len(text), start + len(old) + WINDOW)
    return {
        "card": card,
        "probe_type": ptype,
        "doc": os.path.relpath(doc_path, PATH_BASE),
        "original_text": text[a:b],
        "perturbed_text": text[a:start] + new + text[start + len(old):b],
        "swapped_from": old,
        "swapped_to": new,
        "char_start": start,
    }


def gen_subset_swaps(card, doc_label, doc_path, text, names, rng, cap):
    probes = []
    if len(names) < 2:
        return probes
    for old in names:
        if len(probes) >= cap:
            break
        # take an occurrence where some sibling is absent from the local window,
        # otherwise the perturbed sentence lists the same subset twice and the
        # probe is trivial; skip the name if every occurrence is an enumeration
        candidates = [n for n in names if n != old]
        start, absent = text.find(old), []
        tries = 0
        while start != -1 and tries < 20:
            tries += 1
            if has_boundary(text, start, start + len(old)):
                window = text[max(0, start - WINDOW):start + len(old) + WINDOW].lower()
                absent = [n for n in candidates if n.lower() not in window]
                if absent:
                    break
            start = text.find(old, start + 1)
        if start == -1 or not absent:
            continue
        new = rng.choice(absent)
        probes.append(make_probe(card, "subset_swap", doc_label, doc_path, text, start, old, new))
    return probes


def bump_major(token):
    """v1 -> v2, v1.0 -> v2.0, 2.0 -> 3.0; keeps the token's format."""
    prefix = token[0] if token[0] in "vV" else ""
    parts = token[len(prefix):].split(".")
    parts[0] = str(int(parts[0]) + 1)
    return prefix + ".".join(parts)


def in_url_chunk(text, start, end):
    """True if the token sits in a whitespace-delimited chunk with a slash,
    i.e. a URL, DOI, or path, where version tokens are not benchmark versions."""
    a = start
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    b = end
    while b < len(text) and not text[b].isspace():
        b += 1
    return "/" in text[a:b]


def gen_version_swaps(card, doc_label, doc_path, text, rng, cap):
    matches = list(VERSION_RE.finditer(text)) + list(VERSION_WORD_RE.finditer(text))
    matches = [m for m in matches if not in_url_chunk(text, m.start(), m.end())]
    if not matches:
        return []
    matches.sort(key=lambda m: m.start())
    tokens = []
    for m in matches:
        tok = m.group(0) if m.re is VERSION_RE else m.group(1)
        if tok not in tokens:
            tokens.append(tok)
    probes = []
    seen_starts = set()
    for m in matches:
        if len(probes) >= cap:
            break
        if m.re is VERSION_RE:
            old, start = m.group(0), m.start()
        else:
            old, start = m.group(1), m.start(1)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        # prefer a genuine sibling token from the same doc, same v-prefix style
        siblings = [t for t in tokens if t != old and t.lstrip("vV") != old.lstrip("vV")
                    and (t[0] in "vV") == (old[0] in "vV")]
        new = rng.choice(siblings) if siblings else bump_major(old)
        probes.append(make_probe(card, "version_swap", doc_label, doc_path, text, start, old, new))
    return probes


def gen_split_swaps(card, doc_label, doc_path, text, rng, cap):
    probes = []
    matches = list(SPLIT_CONTEXT_RE.finditer(text))
    matches += [m for m in SPLIT_YAML_RE.finditer(text)]
    matches.sort(key=lambda m: m.start(m.lastindex or 0))
    seen_starts = set()
    for m in matches:
        if len(probes) >= cap:
            break
        grp = m.lastindex or 0
        old, start = m.group(grp), m.start(grp)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        canon = SPLIT_CANON.get(old.lower(), old.lower())
        new = match_case(rng.choice([s for s in SPLIT_WORDS if s != canon]), old)
        probes.append(make_probe(card, "split_swap", doc_label, doc_path, text, start, old, new))
    return probes


def num_shape(m):
    if m.group(2):
        return "percent"
    if "." in m.group(1):
        return "decimal"
    return "int"


def looks_like_year(m):
    if num_shape(m) != "int":
        return False
    try:
        return 1900 <= int(m.group(1)) <= 2035
    except ValueError:
        return False


def gen_number_swaps(card, doc_label, doc_path, text, rng, cap):
    matches = [m for m in NUMBER_RE.finditer(text) if not looks_like_year(m)]
    by_shape = {}
    for m in matches:
        by_shape.setdefault(num_shape(m), []).append(m.group(0))
    probes = []
    low = text.lower()
    for m in matches:
        if len(probes) >= cap:
            break
        shape = num_shape(m)
        if shape == "int" and len(m.group(1)) < 2:
            continue
        ctx = low[max(0, m.start() - 120):m.end() + 120]
        if not any(k in ctx for k in RESULTS_KEYWORDS):
            continue
        pool = sorted({v for v in by_shape.get(shape, []) if v != m.group(0)})
        if not pool:
            continue
        new = rng.choice(pool)
        probes.append(make_probe(card, "number_swap", doc_label, doc_path, text, m.start(), m.group(0), new))
    return probes


def gen_card_probes(name, seed, cap):
    rng = random.Random("%d:%s" % (seed, name))
    for card_dir in find_card_dirs(name):
        docs, hf = load_sources(card_dir, name)
        if not docs:
            continue
        names = subset_names(card_dir, name, hf)
        probes = []
        for doc_label in sorted(docs):
            doc_path, text = docs[doc_label]
            probes += gen_subset_swaps(name, doc_label, doc_path, text, names, rng, cap)
            probes += gen_version_swaps(name, doc_label, doc_path, text, rng, cap)
            probes += gen_split_swaps(name, doc_label, doc_path, text, rng, cap)
            probes += gen_number_swaps(name, doc_label, doc_path, text, rng, cap)
        return probes, card_dir
    return [], None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", nargs="*", default=None, help="card names; default: auto-discover")
    ap.add_argument("--limit", type=int, default=10, help="max cards in auto-discover mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-doc", type=int, default=2, help="cap per probe type per source doc")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "probes", "referent_probes.jsonl"))
    args = ap.parse_args()

    cards = args.cards if args.cards else discover_cards()
    all_probes = []
    used, skipped = [], []
    for name in cards:
        if not args.cards and len(used) >= args.limit:
            break
        probes, card_dir = gen_card_probes(name, args.seed, args.max_per_doc)
        if card_dir is None:
            skipped.append(name)
            continue
        used.append((name, os.path.relpath(card_dir, PATH_BASE)))
        all_probes.extend(probes)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in all_probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = {}
    for p in all_probes:
        counts[p["probe_type"]] = counts.get(p["probe_type"], 0) + 1
    print("wrote %d probes to %s" % (len(all_probes), args.out))
    for t in ("subset_swap", "version_swap", "split_swap", "number_swap"):
        print("  %-12s %d" % (t, counts.get(t, 0)))
    print("cards used (%d):" % len(used))
    for name, d in used:
        n = sum(1 for p in all_probes if p["card"] == name)
        print("  %-30s %2d probes  %s" % (name, n, d))
    if skipped:
        print("skipped (no cached text): %s" % ", ".join(skipped))


if __name__ == "__main__":
    main()
