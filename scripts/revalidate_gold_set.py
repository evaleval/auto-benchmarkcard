#!/usr/bin/env python3
"""Re-run ONLY FactReasoner on the frozen gold-set cards with a chosen config.

FactReasoner is a terminal annotator: it sets flags, it does not change card
content. So to measure a new FactReasoner config we re-score the persisted RAG
results (tool_output/rag/formatted_rag_results_<name>.jsonl) and re-apply the
pipeline's flagging, writing cards whose content is identical to the frozen v31
cards but whose flagged_fields reflect the new FactReasoner run. No composer/RAG
runs, so no HF credits are consumed and there is no composer-model confound.

Runs on RITS only. The active FactReasoner backend is asserted to be RITS before
any model call (hard-wired backend + .env + runtime assert).

Usage:
  python scripts/revalidate_gold_set.py --label fr1                 # baseline (revise off)
  python scripts/revalidate_gold_set.py --label fr1_revise --revise # decontextualization on
"""
import argparse
import glob
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# RITS-only. Config reads LLM_ENGINE_TYPE at import, and load_dotenv must not
# override our choice with the .env default (hf).
os.environ["LLM_ENGINE_TYPE"] = "rits"
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"), override=False)
os.environ["LLM_ENGINE_TYPE"] = "rits"

for _n in ("LiteLLM", "httpx", "mellea", "asyncio", "openai"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

from auto_benchmarkcard.card_utils import (  # noqa: E402
    apply_deterministic_overrides,
    backfill_from_provenance,
    extract_card,
    extract_hf_tags,
    normalize_not_specified,
)
from auto_benchmarkcard.config import Config  # noqa: E402
from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import (  # noqa: E402
    _RITS_ROUTES,
    _create_factreasoner_backend,
    evaluate_factuality,
    evaluate_factuality_two_tier,
    flag_benchmark_card_fields,
    load_formatted_rag_results,
)
from auto_benchmarkcard.tools.rag.format_converter import (  # noqa: E402
    normalize_context_for_nli,
    save_formatted_results,
)
from auto_benchmarkcard.tools.rag.indexer import MetadataIndexer  # noqa: E402
from auto_benchmarkcard.validation_policy import ANALYTICAL_FIELDS, IDENTITY_FIELDS  # noqa: E402


def assert_rits() -> None:
    """Refuse to run unless FactReasoner is actually wired to RITS."""
    if Config.LLM_ENGINE_TYPE.lower() != "rits":
        sys.exit(f"REFUSING: LLM_ENGINE_TYPE={Config.LLM_ENGINE_TYPE!r} != 'rits'")
    backend = _create_factreasoner_backend(Config.FACTREASONER_MODEL)
    if backend is None:
        sys.exit("REFUSING: FactReasoner backend is None (RITS_API_KEY/RITS_API_URL missing?)")
    base = str(getattr(backend, "_base_url", ""))
    if "rits" not in base:
        sys.exit(f"REFUSING: backend base_url is not RITS: {base!r}")
    if Config.FACTREASONER_MODEL.lower() not in _RITS_ROUTES:
        sys.exit(
            f"REFUSING: FACTREASONER_MODEL {Config.FACTREASONER_MODEL!r} has no explicit "
            f"RITS route (would derive a possibly-invalid served model). Add it to _RITS_ROUTES."
        )
    print(f"[RITS OK] backend={type(backend).__name__} model={Config.FACTREASONER_MODEL} base={base}")


def _locate(entry):
    name = entry["name"]
    run_dir = os.path.dirname(os.path.dirname(entry["card_path"]))
    rag = os.path.join(run_dir, "tool_output", "rag", f"formatted_rag_results_{name}.jsonl")
    prov = os.path.join(run_dir, "tool_output", "composer", f"provenance_{name}.json")
    hf = os.path.join(run_dir, "tool_output", "hf", f"{name}.json")
    eee = os.path.join(run_dir, "tool_output", "eee", f"{name}.json")
    return name, run_dir, rag, prov, hf, eee, entry["card_path"]


SOURCE_CAP = 280_000  # chars; judge parity (scripts/judge_gold_set.py)


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


def _assemble_source(run_dir, include_rag_fill=True):
    """Full source text for neutral escalation; mirrors judge_gold_set.assemble_source
    (primary docling+html+readme, RAG contexts fill the remaining budget) so
    'neutral after escalation' is judged against the same text the judge saw.
    --no-rag-fill drops the retrieved-context fill (contamination ablation)."""
    to = os.path.join(run_dir, "tool_output")
    doc = _doc_text(_load_json(_first(os.path.join(to, "docling", "*.json"))))
    html = _doc_text(_load_json(_first(os.path.join(to, "html", "*.json"))))
    hf = _load_json(_first(os.path.join(to, "hf", "*.json"))) or {}
    readme = hf.get("readme_markdown") or hf.get("readme") or ""

    parts = []
    if doc:
        parts.append("[PAPER / DOCUMENT]\n" + doc)
    if html:
        parts.append("[WEBPAGE]\n" + html)
    if readme:
        parts.append("[HF README]\n" + readme)
    primary = "\n\n".join(parts)
    if len(primary) >= SOURCE_CAP or not include_rag_fill:
        return primary[:SOURCE_CAP]
    rag_texts = []
    rp = _first(os.path.join(to, "rag", "*.jsonl"))
    if rp:
        try:
            for line in open(rp):
                o = json.loads(line)
                for c in o.get("contexts") or []:
                    rag_texts.append(c.get("text", "") if isinstance(c, dict) else str(c))
        except Exception:  # noqa: BLE001
            pass
    if rag_texts:
        primary += "\n\n[RETRIEVED CONTEXT]\n" + "\n\n".join(rag_texts)
    return primary[:SOURCE_CAP]


def _bundle_documents(run_dir, name, hf_json):
    """Chunked langchain Documents from the card's own source bundle (docling +
    html + HF readme only -- readme-only hf payload keeps judge parity, no
    basic_info/builder docs)."""
    to = os.path.join(run_dir, "tool_output")
    docling = _load_json(_first(os.path.join(to, "docling", "*.json")))
    html = _load_json(_first(os.path.join(to, "html", "*.json")))
    readme = (hf_json or {}).get("readme_markdown") or (hf_json or {}).get("readme") or ""
    return MetadataIndexer().create_documents(
        unitxt_data=None,
        hf_data={"readme_markdown": readme} if readme else None,
        benchmark_name=name,
        docling_data=docling,
        html_data=html,
    )


_TOKEN_RE = re.compile(r"\d+\.\d+|\w+")
_NUM_LIT_RE = re.compile(r"\d+(?:[.,]\d+)+%?|\d{2,}")
_NUM_BOOST = 2.0


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _bm25_index(texts):
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


def _rank_chunks(index, texts, query, top_k):
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


def _source_scoped_rag(rag, run_dir, name, hf_json, top_k=4):
    """Re-retrieve each frozen atom's contexts from the card's OWN source bundle
    instead of reusing the generation-time RAG picks. Atoms stay frozen (ids,
    text, original, label, field untouched); only their contexts are replaced.
    Context entries are duplicated per atom with unique ids (src_<atom>_<j>,
    mirroring the c_a{n}_{k} convention) because Context.set_atom would
    otherwise overwrite the atom pointer for shared ids."""
    docs = [d for d in _bundle_documents(run_dir, name, hf_json)
            if (d.page_content or "").strip()]
    texts = [d.page_content for d in docs]
    if not texts:
        return rag, {"source_scoped": False, "fallback": "empty_bundle"}
    idx = _bm25_index(texts)
    new_atoms, contexts = [], []
    for atom in rag.get("atoms", []):
        field_words = (atom.get("field") or "").replace(".", " ").replace("_", " ")
        query = f"{atom.get('text', '')} {name} {field_words}"
        cids = []
        for j, i in enumerate(_rank_chunks(idx, texts, query, top_k)):
            cid = f"src_{atom['id']}_{j}"
            cids.append(cid)
            contexts.append({
                "id": cid, "title": name,
                "text": normalize_context_for_nli(texts[i]),
                "source": docs[i].metadata.get("source", "?"),
            })
        new_atoms.append({**atom, "contexts": cids})
    info = {
        "source_scoped": True, "n_chunks": len(texts), "top_k": top_k,
        "chunks_by_source": dict(Counter(d.metadata.get("source", "?") for d in docs)),
    }
    return {**rag, "atoms": new_atoms, "contexts": contexts}, info


def _apply_pipeline_flagging(card, field_analysis, provenance, retrieved_contexts, hf_json,
                             eee_json=None, artifact_verify=True, high_signal=False,
                             legacy_policy=False, backfill_contexts=None):
    """Reproduce workers.run_factreasoner flag steps (workers.py:795-840) so the
    flagged_fields match what the production pipeline would set.

    Toggles allow byte-identical re-application of older configs: legacy_policy
    re-adds methodology.interpretation to the analytical suppression set,
    artifact_verify=False restores blind structured-source trust, and
    high_signal=False keeps the original content-word grounding.
    backfill_contexts lets a source-scoped run keep backfilling card VALUES from
    the frozen RAG contexts (card content must not drift from the frozen v31
    cards) while the unflag decisions see the new contexts.
    """
    # The frozen v31 cards carry their original flagged_fields; without stripping
    # them, a card whose fresh run produces zero flags silently inherits the stale
    # v31 set (flag_benchmark_card_fields only overwrites when non-empty).
    card = {k: v for k, v in card.items() if k != "flagged_fields"}
    flagged = flag_benchmark_card_fields(
        benchmark_card=card,
        field_analysis=field_analysis,
        threshold=Config.DEFAULT_FACTUALITY_THRESHOLD,
        provenance=provenance,
        retrieved_contexts=retrieved_contexts,
        structured_artifacts={"eee": eee_json, "hf": hf_json} if artifact_verify else None,
        high_signal_grounding=high_signal,
    )
    analytical = ANALYTICAL_FIELDS | ({"methodology.interpretation"} if legacy_policy else frozenset())
    ff = flagged.get("flagged_fields")
    if isinstance(ff, dict):
        field_details = field_analysis.get("field_details", {})
        drop = [
            f
            for f in ff
            if f in analytical
            or (f in IDENTITY_FIELDS and field_details.get(f, {}).get("all_neutral") is True)
        ]
        for f in drop:
            del ff[f]
    if provenance:
        flagged = backfill_from_provenance(
            flagged, provenance,
            backfill_contexts if backfill_contexts is not None else retrieved_contexts,
        )
    flagged = normalize_not_specified(flagged)
    # Deterministic HF overrides clear stale flags on overridden fields (workers.py:828-840).
    hf_overrides = extract_hf_tags(hf_json) if hf_json else {}
    if hf_overrides:
        flagged = apply_deterministic_overrides(flagged, hf_overrides)
        ffl = flagged.get("flagged_fields")
        if isinstance(ffl, dict):
            for dotted_key in hf_overrides:
                for k in (dotted_key.split(".")[-1], dotted_key):
                    ffl.pop(k, None)
    return flagged


def _process(entry, revise, source_scoped=False, escalate=False, window_chars=12000,
             rag_fill=True, high_signal=False):
    name, run_dir, rag_p, prov_p, hf_p, eee_p, card_p = _locate(entry)
    rag = load_formatted_rag_results(rag_p)
    card = extract_card(json.load(open(card_p)))
    provenance = json.load(open(prov_p)) if os.path.exists(prov_p) else None
    hf_json = json.load(open(hf_p)) if os.path.exists(hf_p) else None
    eee_json = json.load(open(eee_p)) if os.path.exists(eee_p) else None
    # Card-value backfill must keep seeing the FROZEN generation-time contexts:
    # with the broader source-scoped pool it would fill Not-specified values and
    # drift card content away from the frozen v31 cards (invalidating the judge
    # labels). Only the unflag decisions get the new contexts.
    frozen_contexts = [c.get("text", "") for c in rag.get("contexts", [])]
    ret_info = None
    if source_scoped:
        rag, ret_info = _source_scoped_rag(rag, run_dir, name, hf_json)
    retrieved_contexts = [c.get("text", "") for c in rag.get("contexts", [])]
    if escalate:
        source = _assemble_source(run_dir, include_rag_fill=rag_fill)
        fr = evaluate_factuality_two_tier(
            rag, source,
            model=Config.FACTREASONER_MODEL,
            cache_dir=Config.FACTREASONER_CACHE_DIR,
            merlin_path=str(Config.MERLIN_BIN),
            revise_atoms=revise,
            window_chars=window_chars,
        )
    else:
        fr = evaluate_factuality(
            rag,
            model=Config.FACTREASONER_MODEL,
            cache_dir=Config.FACTREASONER_CACHE_DIR,
            merlin_path=str(Config.MERLIN_BIN),
            revise_atoms=revise,
        )
    flagged_card = _apply_pipeline_flagging(
        card, fr.get("field_analysis", {}), provenance, retrieved_contexts, hf_json,
        eee_json=eee_json, high_signal=high_signal,
        backfill_contexts=frozen_contexts,
    )
    return name, flagged_card, fr, rag, ret_info


def main():
    ap = argparse.ArgumentParser(description="Re-run FactReasoner on the frozen gold set (RITS only).")
    ap.add_argument("--label", required=True, help="config label, e.g. fr1 or fr1_revise")
    ap.add_argument("--revise", action="store_true", help="enable atom decontextualization")
    ap.add_argument("--source-scoped", action="store_true",
                    help="retrieve per-atom contexts from the card's own source bundle "
                         "(deterministic BM25) instead of the frozen RAG contexts")
    ap.add_argument("--escalate-neutral", action="store_true",
                    help="second NLI pass for all-neutral atoms against full-source windows")
    ap.add_argument("--window-chars", type=int, default=12000, help="escalation window size")
    ap.add_argument("--no-rag-fill", action="store_true",
                    help="escalation source = primary bundle only (no retrieved-context fill)")
    ap.add_argument("--high-signal", action="store_true",
                    help="number/name veto in the provenance grounding check "
                         "(sound only with --source-scoped contexts)")
    ap.add_argument("--manifest", default="gold_set/manifest.json")
    ap.add_argument("--out-root", default="gold_set/revalidate", help="where flagged cards are written")
    ap.add_argument("--workers", type=int, default=3, help="parallel cards (each is independent)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cards (testing)")
    args = ap.parse_args()

    assert_rits()
    print(f"[config] label={args.label} revise_atoms={args.revise} "
          f"source_scoped={args.source_scoped} escalate={args.escalate_neutral} "
          f"window={args.window_chars} rag_fill={not args.no_rag_fill} "
          f"high_signal={args.high_signal}")

    man = json.load(open(os.path.join(REPO, args.manifest)))
    entries = man["cards"] if isinstance(man, dict) and "cards" in man else man
    if args.limit:
        entries = entries[: args.limit]

    out_dir = os.path.join(REPO, args.out_root, args.label)
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    def work(entry):
        t = time.time()
        name, flagged_card, fr, rag, ret_info = _process(
            entry, args.revise, source_scoped=args.source_scoped,
            escalate=args.escalate_neutral, window_chars=args.window_chars,
            rag_fill=not args.no_rag_fill, high_signal=args.high_signal,
        )
        cdir = os.path.join(out_dir, name, "benchmarkcard")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, f"benchmark_card_{name}.json"), "w") as f:
            json.dump({"benchmark_card": flagged_card}, f, indent=2)
        # Also persist factuality_results so eval_gold_set's validation-coverage check
        # (looks for tool_output/factreasoner/*.json with num_atoms>0) sees this run.
        frdir = os.path.join(out_dir, name, "tool_output", "factreasoner")
        os.makedirs(frdir, exist_ok=True)
        with open(os.path.join(frdir, f"factuality_results_{name}.json"), "w") as f:
            json.dump(fr, f, indent=2)
        if ret_info and ret_info.get("source_scoped"):
            # Persist the rewritten rag for audit/offline re-use.
            rdir = os.path.join(out_dir, name, "tool_output", "rag")
            os.makedirs(rdir, exist_ok=True)
            save_formatted_results(rag, os.path.join(rdir, f"formatted_rag_results_{name}.jsonl"))
        esc = fr.get("escalation") or {}
        return name, {
            "stratum": entry.get("stratum"),
            "flagged_fields": flagged_card.get("flagged_fields", {}),
            "n_flagged": len(flagged_card.get("flagged_fields", {})),
            "num_atoms": fr["results"]["num_atoms"],
            "retrieval": ret_info,
            "escalation": {
                "escalated": esc.get("escalated_atoms", 0),
                "resolved": len(esc.get("resolved", {})),
                "still_neutral": len(esc.get("still_neutral", [])),
                "num_windows": esc.get("num_windows", 0),
                "skipped": esc.get("skipped"), "failed": esc.get("failed"),
            } if args.escalate_neutral else None,
            "secs": round(time.time() - t, 1),
        }

    started = time.time()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, e): e for e in entries}
            for fut in as_completed(futs):
                e = futs[fut]
                try:
                    name, info = fut.result()
                    results[name] = info
                    print(f"  {name:22} flags={info['n_flagged']:>2} atoms={info['num_atoms']:>3} {info['secs']}s")
                except Exception as ex:  # noqa: BLE001
                    print(f"  {e['name']:22} ERR {type(ex).__name__}: {str(ex)[:140]}")
    else:
        for e in entries:
            try:
                name, info = work(e)
                results[name] = info
                print(f"  {name:22} flags={info['n_flagged']:>2} atoms={info['num_atoms']:>3} {info['secs']}s")
            except Exception as ex:  # noqa: BLE001
                print(f"  {e['name']:22} ERR {type(ex).__name__}: {str(ex)[:140]}")

    summary = {
        "label": args.label,
        "revise_atoms": args.revise,
        "source_scoped": args.source_scoped,
        "escalate_neutral": args.escalate_neutral,
        "window_chars": args.window_chars,
        "rag_fill": not args.no_rag_fill,
        "high_signal": args.high_signal,
        "engine": "rits",
        "model": Config.FACTREASONER_MODEL,
        "n_cards": len(results),
        "total_flags": sum(v["n_flagged"] for v in results.values()),
        "elapsed_s": round(time.time() - started, 1),
        "per_card": results,
    }
    sp = os.path.join(REPO, "gold_set", f"revalidate_{args.label}.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {sp}: {len(results)} cards, {summary['total_flags']} total flags, {summary['elapsed_s']}s")
    print(f"flagged cards under {out_dir} (use as eval_gold_set --cards-root {os.path.relpath(out_dir, REPO)})")


if __name__ == "__main__":
    main()
