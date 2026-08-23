"""Referent frame: the typed list of things evidence for this card could be about.

Phase 2 foundation. The frame enumerates, per card, the candidate referents that
wrong-section-splice errors confuse: the target benchmark itself, sibling
benchmarks named in the same paper or suite, subsets/configs, versions, splits,
evaluation settings, and metrics with their role. Downstream, every evidence
item names the frame node it is about, and assignment checks compare that claim
against the node type the field expects.

Node ids are stable within a card ("target", "sibling:fibble", "subset:hard",
"version:2.0", "split:test", "metric:accuracy"). Deterministic sources fill
first (EEE metadata, HF configs, size_breakdown keys, version regexes over the
paper head); one optional fail-soft LLM pass enriches sibling benchmarks named
in the paper. The frame is telemetry-visible and saved as a composer sidecar.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NODE_TYPES = (
    "target", "sibling", "subset", "version", "split", "setting", "metric",
)
METRIC_ROLES = ("primary", "secondary", "meta_eval", "unknown")

# paper head window for version/split harvesting and the LLM sibling prompt
HEAD_CHARS = 4000
MAX_LLM_SIBLINGS = 12
MIN_ALIAS_CHARS = 3  # single very-short aliases are ignored by match_mention

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# version tokens; same shapes and URL exclusion as scripts/generate_referent_probes.py
_VERSION_RE = re.compile(r"(?<![\w.])[vV](\d+(?:\.\d+)*)(?![\w.])")
_VERSION_WORD_RE = re.compile(r"\b[vV]ersion\s+(\d+(?:\.\d+)*)(?![\w.])")
_RELEASE_CUE_RE = re.compile(
    r"\b(release[sd]?|version|update[sd]?|changelog|revision|revised|successor"
    r"|introduc\w+|extend\w+|supersede\w*)\b", re.IGNORECASE)
# a version token right after a model family word is a model version, not a benchmark version
_MODEL_WORD_RE = re.compile(
    r"\b(gpt|llama|claude|gemini|gemma|qwen|mistral|mixtral|deepseek|palm|bert"
    r"|phi|grok|vicuna|falcon|opus|sonnet|haiku)\b[-. ]*$", re.IGNORECASE)

_SPLIT_WORDS = ("train", "validation", "test", "dev")
_SPLIT_CANON = {"training": "train"}
# split word bound to a set/split noun, so bare verbs ("we test models") never count
_SPLIT_CONTEXT_RE = re.compile(
    r"\b(train|training|validation|dev|test)\b"
    r"(?=\s+(?:set|sets|split|splits|data|examples|instances|partition)\b)",
    re.IGNORECASE,
)
_SPLIT_YAML_RE = re.compile(r"\bsplits?:\s*(train|training|validation|dev|test)\b", re.IGNORECASE)
_SPLIT_NAME_RE = re.compile(r"-\s*name:\s*(train|training|validation|dev|test)\b", re.IGNORECASE)

_CONFIG_NAME_RE = re.compile(r"^\s*-?\s*config_name:\s*[\"']?([^\"'\n]+?)[\"']?\s*$", re.MULTILINE)
_PRETTY_NAME_RE = re.compile(r"^\s*pretty_name:\s*[\"']?([^\"'\n]+?)[\"']?\s*$", re.MULTILINE)


def _norm(name: str) -> str:
    """Normalized form used for dedup across builders."""
    return _NON_ALNUM_RE.sub(" ", str(name).lower()).strip()


def _slug(name: str) -> str:
    return _NON_ALNUM_RE.sub("-", str(name).lower()).strip("-") or "unnamed"


def _in_url_chunk(text: str, start: int, end: int, token: str = "") -> bool:
    """True if the token sits in a whitespace-delimited chunk that looks like a
    URL, DOI, or path. A token that itself contains a slash (CNN/DailyMail) is
    only excluded when the chunk carries a scheme."""
    a = start
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    b = end
    while b < len(text) and not text[b].isspace():
        b += 1
    chunk = text[a:b]
    if "/" in token:
        return "://" in chunk
    return "/" in chunk


def _has_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _iter_token_hits(text: str, token: str):
    """Whole-token, case-insensitive, URL-excluded (start, end) occurrences."""
    low = text.lower()
    tok = token.lower()
    start = low.find(tok)
    while start != -1:
        end = start + len(tok)
        if _has_boundary(text, start, end) and not _in_url_chunk(text, start, end, token):
            yield start, end
        start = low.find(tok, start + 1)


def _token_present(text: str, token: str) -> bool:
    return next(_iter_token_hits(text, token), None) is not None


def _add_node(nodes: List[Dict[str, Any]], index: Dict[str, Dict[str, Any]],
              node: Dict[str, Any]) -> bool:
    """Append node unless a node with the same normalized name (or alias) exists;
    on a duplicate, merge the new spellings into the existing node's aliases.
    A normalized form belongs to exactly one node: the first to claim it."""
    key = _norm(node["name"])
    if not key:
        return False
    existing = index.get(key)
    if existing is None:
        kept = []
        for alias in node.get("aliases", []):
            k = _norm(alias)
            if not k:
                continue
            owner = index.get(k)
            if owner is not None and owner is not node:
                continue
            low = alias.lower()
            if low == node["name"].lower() or any(low == a.lower() for a in kept):
                continue
            index[k] = node
            kept.append(alias)
        node["aliases"] = kept
        nodes.append(node)
        index[key] = node
        return True
    for spelling in [node["name"], *node.get("aliases", [])]:
        k = _norm(spelling)
        if not k:
            continue
        owner = index.get(k)
        if owner is not None and owner is not existing:
            continue
        index[k] = existing
        known = [existing["name"], *existing["aliases"]]
        if all(spelling.lower() != x.lower() for x in known):
            existing["aliases"].append(spelling)
    return False


def _target_node(benchmark_name: str, eee_metadata: Optional[Dict[str, Any]],
                 hf_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    aliases: List[str] = []

    def take(value):
        if isinstance(value, str) and value.strip():
            aliases.append(value.strip())

    if isinstance(eee_metadata, dict):
        take(eee_metadata.get("benchmark_name"))
        # not present in current EEE dumps, kept defensively
        for key in ("full_name", "pretty_name", "paper_title", "title"):
            take(eee_metadata.get(key))
    if isinstance(hf_json, dict):
        repo_id = hf_json.get("id")
        if isinstance(repo_id, str) and repo_id.strip():
            take(repo_id.rsplit("/", 1)[-1])
        card_data = hf_json.get("card_data")
        if isinstance(card_data, dict):
            take(card_data.get("pretty_name"))
        # card_data is {} in cached HF dumps; the pretty_name lives in the README frontmatter
        fm = _readme_frontmatter(hf_json.get("readme_markdown"))
        m = _PRETTY_NAME_RE.search(fm)
        if m:
            take(m.group(1))

    seen = {benchmark_name.lower()}
    unique = []
    for alias in aliases:
        k = alias.lower()
        if k not in seen:
            seen.add(k)
            unique.append(alias)
    return {
        "id": "target",
        "type": "target",
        "name": benchmark_name,
        "aliases": unique,
        "description": "The target benchmark this card documents.",
    }


def _eee_sibling_nodes(eee_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    if not isinstance(eee_metadata, dict):
        return nodes
    for key, desc in (("contains", "Suite member listed in EEE metadata 'contains'."),
                      ("appears_in", "Parent suite listed in EEE metadata 'appears_in'.")):
        for name in eee_metadata.get(key) or []:
            if isinstance(name, str) and name.strip():
                name = name.strip()
                nodes.append({
                    "id": "sibling:" + _slug(name),
                    "type": "sibling",
                    "name": name,
                    "aliases": [],
                    "description": desc,
                })
    return nodes


def _readme_frontmatter(readme) -> str:
    if not isinstance(readme, str) or not readme.startswith("---"):
        return ""
    end = readme.find("\n---", 3)
    if end < 0:
        return ""
    return readme[3:end]


def _subset_nodes(eee_metadata: Optional[Dict[str, Any]],
                  hf_json: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    named: List[Tuple[str, str]] = []
    if isinstance(hf_json, dict):
        card_data = hf_json.get("card_data")
        if isinstance(card_data, dict):
            for cfg in card_data.get("configs") or []:
                if isinstance(cfg, dict) and isinstance(cfg.get("config_name"), str):
                    named.append((cfg["config_name"], "HF dataset config."))
                elif isinstance(cfg, str):
                    named.append((cfg, "HF dataset config."))
        # cached HF dumps serialize card_data as {}; read configs from the README frontmatter
        fm = _readme_frontmatter(hf_json.get("readme_markdown"))
        for m in _CONFIG_NAME_RE.finditer(fm):
            named.append((m.group(1), "HF dataset config."))
    if isinstance(eee_metadata, dict):
        breakdown = eee_metadata.get("size_breakdown")
        if isinstance(breakdown, dict):
            for key in breakdown:
                if isinstance(key, str):
                    named.append((key, "Subset named in EEE size_breakdown."))

    nodes = []
    for name, desc in named:
        name = name.strip()
        low = name.lower()
        # split words and generic placeholder configs are not subset referents
        if len(name) < 3 or name.isdigit() or low in _SPLIT_WORDS or low in ("default", "all"):
            continue
        nodes.append({
            "id": "subset:" + _slug(name),
            "type": "subset",
            "name": name,
            "aliases": [],
            "description": desc,
        })
    return nodes


def _metric_nodes(eee_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    if not isinstance(eee_metadata, dict):
        return nodes
    metrics = eee_metadata.get("metrics")
    if not isinstance(metrics, dict):
        return nodes
    summary = eee_metadata.get("evaluation_summary")
    primary_id = summary.get("primary_metric") if isinstance(summary, dict) else None
    for rid, cfg in metrics.items():
        cfg = cfg if isinstance(cfg, dict) else {}
        name = cfg.get("metric_name")
        if not isinstance(name, str) or not name.strip():
            name = rid.rsplit(".", 1)[-1]
        name = name.strip()
        details = cfg.get("additional_details")
        marked = isinstance(details, dict) and details.get("alphaxiv_is_primary") in (True, "true", "True")
        role = "primary" if rid == primary_id or marked else "unknown"
        # suite cards key some metrics by suite-MEMBER name (helm-lite: "MMLU" -> "EM");
        # only id-like keys become aliases, a benchmark-shaped key names a sibling instead
        aliases = []
        if _norm(rid) != _norm(name) and ("." in rid or "_" in rid or rid == rid.lower()):
            aliases = [rid]
        nodes.append({
            "id": "metric:" + _slug(name),
            "type": "metric",
            "name": name,
            "aliases": aliases,
            "role": role,
            "description": "Metric tracked in EEE metadata.",
        })
    return nodes


def _version_nodes(paper_text: str, target_aliases: List[str]) -> List[Dict[str, Any]]:
    """Standalone version tokens in the paper head, near the benchmark name or a
    release context; URL/DOI/path chunks and model-name versions are excluded."""
    head = paper_text[:HEAD_CHARS]
    if not head:
        return []
    anchors = [a for a in target_aliases if len(a) >= MIN_ALIAS_CHARS]
    found: Dict[str, Dict[str, Any]] = {}
    for rx, worded in ((_VERSION_RE, False), (_VERSION_WORD_RE, True)):
        for m in rx.finditer(head):
            if _in_url_chunk(head, m.start(), m.end()):
                continue
            if not worded and _MODEL_WORD_RE.search(head, max(0, m.start() - 12), m.start()):
                continue
            window = head[max(0, m.start() - 200):m.end() + 200]
            # the word form carries its own release context ("version 3")
            if not worded and not _RELEASE_CUE_RE.search(window) \
                    and not any(_token_present(window, a) for a in anchors):
                continue
            number = m.group(1)
            node = found.get(number)
            if node is None:
                found[number] = node = {
                    "id": "version:" + number,
                    "type": "version",
                    "name": "v" + number,
                    "aliases": [],
                    "description": "Version token found in the paper head.",
                }
            raw = m.group(0)
            known = [node["name"], *node["aliases"]]
            if all(raw.lower() != x.lower() for x in known):
                node["aliases"].append(raw)
    return list(found.values())


def _split_nodes(paper_text: str, hf_json: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: Dict[str, set] = {}

    def record(raw: str):
        canon = _SPLIT_CANON.get(raw.lower(), raw.lower())
        words.setdefault(canon, set()).add(raw.lower())

    head = paper_text[:HEAD_CHARS]
    for m in _SPLIT_CONTEXT_RE.finditer(head):
        record(m.group(1))
    if isinstance(hf_json, dict):
        fm = _readme_frontmatter(hf_json.get("readme_markdown"))
        for m in _SPLIT_YAML_RE.finditer(fm):
            record(m.group(1))
        # dataset_info splits blocks list entries as "- name: train"
        for block in re.finditer(r"splits:", fm):
            for m in _SPLIT_NAME_RE.finditer(fm, block.end(), block.end() + 800):
                record(m.group(1))

    nodes = []
    for canon in sorted(words):
        aliases = sorted(w for w in words[canon] if w != canon)
        nodes.append({
            "id": "split:" + canon,
            "type": "split",
            "name": canon,
            "aliases": aliases,
            "description": "Dataset split mentioned in the paper head or HF metadata.",
        })
    return nodes


def _extract_json_list(raw: str) -> Optional[str]:
    """First balanced [...] block in raw (string-aware), or None."""
    if not raw:
        return None
    start = raw.find("[")
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
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def _llm_sibling_names(llm_handler, target_name: str, target_aliases: List[str],
                       paper_text: str, doc_index) -> List[str]:
    """One LLM call over the paper head (+ section titles); returns the raw name
    list. Callers validate against the paper text; this never checks presence."""
    titles = []
    for section in (getattr(doc_index, "sections", None) or [])[:40]:
        title = getattr(section, "title", "")
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    known = ", ".join([target_name, *target_aliases][:6])
    parts = [
        f'You are indexing a research paper about the benchmark "{target_name}" '
        f"(also known as: {known}).",
        "List the names of OTHER benchmarks or datasets mentioned in the material "
        f'below, i.e. ones that are NOT "{target_name}" itself.',
        'Answer with ONLY a JSON array of name strings, e.g. ["MMLU", "HellaSwag"]. '
        "Answer [] if none are mentioned.",
    ]
    if titles:
        parts.append("Section titles:\n" + "\n".join("- " + t for t in titles))
    parts.append("Paper text (beginning):\n" + paper_text[:HEAD_CHARS])
    raw = llm_handler.generate("\n\n".join(parts))
    if not isinstance(raw, str):
        return []

    parsed = None
    for candidate in (raw.strip(), _extract_json_list(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            break
        except (ValueError, TypeError):
            continue
    if not isinstance(parsed, list):
        return []
    names = []
    for item in parsed:
        if isinstance(item, str) and 3 <= len(item.strip()) <= 80:
            names.append(item.strip())
    return names


def build_frame(
    benchmark_name: str,
    eee_metadata: Optional[Dict[str, Any]],
    hf_json: Optional[Dict[str, Any]],
    doc_index=None,
    paper_text: str = "",
    llm_handler=None,
) -> Dict[str, Any]:
    """Return {"target_id": "target", "nodes": [...], "telemetry": {...}}.

    Each node: {"id", "type", "name", "aliases": [...], "description": one line,
    and for metrics "role"}. Deterministic builders run always; the LLM sibling
    pass runs only when llm_handler and paper_text are given and never raises
    (fail-soft to the deterministic frame). Dedup by normalized name across
    builders; the target node always exists and carries the resolved benchmark
    name plus HF/EEE aliases.
    """
    nodes: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    merged = 0

    target = _target_node(benchmark_name, eee_metadata, hf_json)
    _add_node(nodes, index, target)
    counts["target"] = 1

    builders = [
        ("siblings_eee", _eee_sibling_nodes(eee_metadata)),
        ("subsets", _subset_nodes(eee_metadata, hf_json)),
        ("metrics", _metric_nodes(eee_metadata)),
        ("versions", _version_nodes(paper_text, [target["name"], *target["aliases"]])),
        ("splits", _split_nodes(paper_text, hf_json)),
    ]
    for label, batch in builders:
        added = 0
        for node in batch:
            if _add_node(nodes, index, node):
                added += 1
            else:
                merged += 1
        counts[label] = added

    telemetry: Dict[str, Any] = {"llm_pass": "skipped"}
    counts["siblings_llm"] = 0
    if llm_handler is not None and paper_text:
        try:
            names = _llm_sibling_names(llm_handler, target["name"], target["aliases"],
                                       paper_text, doc_index)
            kept = 0
            for name in names:
                if kept >= MAX_LLM_SIBLINGS:
                    break
                # the LLM is never trusted: the name must occur literally in the paper
                if not _token_present(paper_text, name):
                    continue
                node = {
                    "id": "sibling:" + _slug(name),
                    "type": "sibling",
                    "name": name,
                    "aliases": [],
                    "description": "Benchmark or dataset named in the paper (LLM sibling pass).",
                }
                if _add_node(nodes, index, node):
                    kept += 1
                else:
                    merged += 1
            counts["siblings_llm"] = kept
            telemetry["llm_pass"] = "ok"
            telemetry["llm_returned"] = len(names)
        except Exception as exc:
            logger.warning("frame LLM sibling pass failed: %s", exc)
            telemetry["llm_pass"] = "failed"
            telemetry["llm_pass_error"] = str(exc)[:200]

    telemetry["builder_counts"] = counts
    telemetry["merged_duplicates"] = merged
    telemetry["total_nodes"] = len(nodes)
    return {"target_id": "target", "nodes": nodes, "telemetry": telemetry}


def node_lookup(frame: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """id -> node mapping for O(1) checks."""
    return {node["id"]: node for node in frame.get("nodes", [])}


def match_mention(frame: Dict[str, Any], text: str) -> List[str]:
    """Frame node ids whose name or alias occurs in text (whole-token,
    case-insensitive, longest-alias-first). Deterministic; used by the
    sibling-name veto and subset cross-check gates."""
    if not text:
        return []
    entries = []
    for node in frame.get("nodes", []):
        for alias in [node.get("name", ""), *(node.get("aliases") or [])]:
            alias = (alias or "").strip()
            if len(alias) >= MIN_ALIAS_CHARS:
                entries.append((alias, node["id"]))
    # longest alias first so a name nested inside a longer match never double-counts
    entries.sort(key=lambda e: (-len(e[0]), e[0].lower()))
    claimed: List[Tuple[int, int]] = []
    hits: List[Tuple[int, str]] = []
    for alias, node_id in entries:
        for start, end in _iter_token_hits(text, alias):
            if any(start < ce and cs < end for cs, ce in claimed):
                continue
            claimed.append((start, end))
            hits.append((start, node_id))
    seen = set()
    ordered = []
    for _, node_id in sorted(hits):
        if node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)
    return ordered
