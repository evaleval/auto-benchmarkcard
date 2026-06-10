"""Source-scoped retrieval and source assembly for the validation stage.

Ship-config building blocks measured on the gold set (validation week): BM25
re-retrieval of each atom's contexts from the card's OWN source bundle
(docling + html + HF readme), and assembly of the primary source text used for
neutral-atom escalation. The data-level functions take the in-memory tool
outputs the production pipeline already holds in state; the *_from_run_dir /
run_dir variants load the same JSON artifacts from a run directory for offline
use (scripts/revalidate_gold_set.py).

scripts/judge_gold_set.py keeps its own assemble_source variant on purpose:
the judge is the measurement instrument (different rag-fill semantics, EEE
block) and must not be coupled to the ship module.
"""

import glob
import json
import math
import os
import re
from collections import Counter

from auto_benchmarkcard.tools.rag.format_converter import normalize_context_for_nli
from auto_benchmarkcard.tools.rag.indexer import MetadataIndexer

SOURCE_CAP = 280_000  # chars; judge parity (scripts/judge_gold_set.py)

_TOKEN_RE = re.compile(r"\d+\.\d+|\w+")
_NUM_LIT_RE = re.compile(r"\d+(?:[.,]\d+)+%?|\d{2,}")
_NUM_BOOST = 2.0


def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _doc_text(obj):
    if not obj:
        return ""
    if isinstance(obj, dict):
        return obj.get("text") or obj.get("content") or obj.get("markdown") or ""
    return str(obj)


def _readme(hf_json):
    readme = (hf_json or {}).get("readme_markdown") or (hf_json or {}).get("readme") or ""
    # whitespace-only readme must not count as a present source
    return readme if readme.strip() else ""


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def bm25_index(texts):
    docs = [_tokenize(t) for t in texts]
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    return {
        "tfs": [Counter(d) for d in docs],
        "lens": [len(d) for d in docs],
        "avgdl": (sum(len(d) for d in docs) / n) if n else 0.0,
        "idf": {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()},
    }


def rank_chunks(index, texts, query, top_k):
    """Deterministic BM25 (k1=1.5, b=0.75) with an exact-substring boost for
    number literals -- embeddings are digit-blind, scores like 0.712 are the
    highest-signal anchors benchmark claims have."""
    k1, b = 1.5, 0.75
    q_terms = set(_tokenize(query))
    lits = set(_NUM_LIT_RE.findall(query))
    scores = []
    for i, tf in enumerate(index["tfs"]):
        s = 0.0
        dl = index["lens"][i] or 1
        for t in q_terms:
            f = tf.get(t)
            if f:
                s += index["idf"].get(t, 0.0) * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * dl / (index["avgdl"] or 1))
                )
        for lit in lits:
            if lit in texts[i]:
                s += _NUM_BOOST
        scores.append((s, i))
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scores[:top_k]]


def assemble_source_text(docling_data, html_data, hf_json, rag_context_texts=None, cap=SOURCE_CAP):
    """Primary source text: docling + html + HF readme; optional retrieved-context
    fill up to the cap. Pass rag_context_texts=None for the no-rag-fill ship config."""
    doc = _doc_text(docling_data)
    html = _doc_text(html_data)
    readme = _readme(hf_json)

    parts = []
    if doc:
        parts.append("[PAPER / DOCUMENT]\n" + doc)
    if html:
        parts.append("[WEBPAGE]\n" + html)
    if readme:
        parts.append("[HF README]\n" + readme)
    primary = "\n\n".join(parts)
    if len(primary) >= cap or not rag_context_texts:
        return primary[:cap]
    primary += "\n\n[RETRIEVED CONTEXT]\n" + "\n\n".join(rag_context_texts)
    return primary[:cap]


def bundle_documents(benchmark_name, docling_data, html_data, hf_json):
    """Chunked langchain Documents from the card's own source bundle (docling +
    html + HF readme only -- readme-only hf payload keeps judge parity, no
    basic_info/builder docs)."""
    readme = _readme(hf_json)
    return MetadataIndexer().create_documents(
        unitxt_data=None,
        hf_data={"readme_markdown": readme} if readme else None,
        benchmark_name=benchmark_name,
        docling_data=docling_data,
        html_data=html_data,
    )


def source_scoped_rag(rag, benchmark_name, docling_data, html_data, hf_json, top_k=4):
    """Re-retrieve each atom's contexts from the card's OWN source bundle
    instead of reusing the generation-time RAG picks. Atoms stay untouched
    (ids, text, original, label, field); only their contexts are replaced.
    Context entries are duplicated per atom with unique ids (src_<atom>_<j>,
    mirroring the c_a{n}_{k} convention) because Context.set_atom would
    otherwise overwrite the atom pointer for shared ids."""
    docs = [d for d in bundle_documents(benchmark_name, docling_data, html_data, hf_json)
            if (d.page_content or "").strip()]
    texts = [d.page_content for d in docs]
    if not texts:
        return rag, {"source_scoped": False, "fallback": "empty_bundle"}
    idx = bm25_index(texts)
    new_atoms, contexts = [], []
    for atom in rag.get("atoms", []):
        field_words = (atom.get("field") or "").replace(".", " ").replace("_", " ")
        query = f"{atom.get('text', '')} {benchmark_name} {field_words}"
        cids = []
        for j, i in enumerate(rank_chunks(idx, texts, query, top_k)):
            cid = f"src_{atom['id']}_{j}"
            cids.append(cid)
            contexts.append({
                "id": cid, "title": benchmark_name,
                "text": normalize_context_for_nli(texts[i]),
                "source": docs[i].metadata.get("source", "?"),
            })
        new_atoms.append({**atom, "contexts": cids})
    info = {
        "source_scoped": True, "n_chunks": len(texts), "top_k": top_k,
        "chunks_by_source": dict(Counter(d.metadata.get("source", "?") for d in docs)),
    }
    return {**rag, "atoms": new_atoms, "contexts": contexts}, info


def _read_rag_context_texts(rag_dir):
    texts = []
    rp = _first(os.path.join(rag_dir, "*.jsonl"))
    if rp:
        try:
            for line in open(rp):
                o = json.loads(line)
                for c in o.get("contexts") or []:
                    texts.append(c.get("text", "") if isinstance(c, dict) else str(c))
        except Exception:  # noqa: BLE001
            pass
    return texts


def assemble_source(run_dir, include_rag_fill=True):
    """Full source text for neutral escalation, loaded from a run directory;
    mirrors judge_gold_set.assemble_source (primary docling+html+readme, RAG
    contexts fill the remaining budget) so 'neutral after escalation' is judged
    against the same text the judge saw. include_rag_fill=False drops the
    retrieved-context fill (no-rag-fill ship config / contamination ablation)."""
    to = os.path.join(run_dir, "tool_output")
    docling = _load_json(_first(os.path.join(to, "docling", "*.json")))
    html = _load_json(_first(os.path.join(to, "html", "*.json")))
    hf = _load_json(_first(os.path.join(to, "hf", "*.json"))) or {}
    primary = assemble_source_text(docling, html, hf)
    if len(primary) >= SOURCE_CAP or not include_rag_fill:
        return primary
    return assemble_source_text(
        docling, html, hf,
        rag_context_texts=_read_rag_context_texts(os.path.join(to, "rag")),
    )


def source_scoped_rag_from_run_dir(rag, run_dir, benchmark_name, hf_json, top_k=4):
    """source_scoped_rag with docling/html loaded from a run directory."""
    to = os.path.join(run_dir, "tool_output")
    docling = _load_json(_first(os.path.join(to, "docling", "*.json")))
    html = _load_json(_first(os.path.join(to, "html", "*.json")))
    return source_scoped_rag(rag, benchmark_name, docling, html, hf_json, top_k=top_k)
