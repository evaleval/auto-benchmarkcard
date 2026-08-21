"""Offline check for the A1 fix: provenance-based un-flagging must be grounded in
the retrieved source, not merely asserted by the composer.

Runs against saved pipeline outputs (factuality_results + formatted_rag_results +
provenance) under output/. Makes NO inference calls.

Reports, per card, how many fields the OLD logic would have un-flagged (because the
composer asserted source+evidence) but the NEW logic correctly keeps flagged
(because that evidence is absent from the retrieved chunks) -- i.e. hallucinations
the old self-consistency check was hiding.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import backfill_from_provenance  # noqa: E402
from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import (  # noqa: E402
    _determine_flag_reason,
    _evidence_grounded_in_contexts,
    _is_structured_source,
    flag_benchmark_card_fields,
)

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")
THRESHOLD = 0.8


def _load_jsonl_first(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def find_trios(limit=8):
    trios = []
    for fact in glob.glob(os.path.join(OUTPUT_ROOT, "**", "factuality_results_*.json"), recursive=True):
        tool_dir = os.path.dirname(os.path.dirname(fact))  # .../tool_output/factreasoner -> tool_output
        rag = glob.glob(os.path.join(tool_dir, "rag", "formatted_rag_results_*.jsonl"))
        prov = glob.glob(os.path.join(tool_dir, "composer", "provenance_*.json"))
        if rag and prov:
            trios.append((fact, rag[0], prov[0]))
        if len(trios) >= limit:
            break
    return trios


def unit_checks():
    contexts = [
        "The dataset contains 817 expert-written questions across physics and biology.",
        "Models are scored with exact match accuracy.",
    ]
    pos = _evidence_grounded_in_contexts("817 expert-written questions in physics", contexts)
    neg = _evidence_grounded_in_contexts("the corpus has 5000000 synthetic dialogues", contexts)
    empty = _evidence_grounded_in_contexts("anything at all here", [])
    print(f"  grounded(real evidence)      = {pos}  (expect True)")
    print(f"  grounded(fabricated evidence)= {neg}  (expect False)")
    print(f"  grounded(no contexts)        = {empty}  (expect False)")
    return pos and (not neg) and (not empty)


def backfill_unit_checks():
    contexts = ["The benchmark uses exact match accuracy over 1319 test problems."]
    g = backfill_from_provenance(
        {"data": {"size": "Not specified"}},
        {"data": {"size": {"source": "paper", "evidence": "1319 test problems in the dataset"}}},
        contexts,
    )["data"]["size"] != "Not specified"
    ug = backfill_from_provenance(
        {"data": {"size": "Not specified"}},
        {"data": {"size": {"source": "paper", "evidence": "9999999 synthetic dialogues never mentioned"}}},
        contexts,
    )["data"]["size"] == "Not specified"
    st = backfill_from_provenance(
        {"benchmark_details": {"languages": ["Not specified"]}},
        {"benchmark_details": {"languages": {"source": "deterministic", "evidence": "languages: English"}}},
        contexts,
    )["benchmark_details"]["languages"] != ["Not specified"]
    print(f"  backfill(grounded prose)   filled     = {g}  (expect True)")
    print(f"  backfill(ungrounded prose) kept-empty = {ug}  (expect True)")
    print(f"  backfill(structured fact)  filled     = {st}  (expect True)")
    return g and ug and st


def analyze_trio(fact_path, rag_path, prov_path):
    fact = json.load(open(fact_path))
    prov = json.load(open(prov_path))
    rag = _load_jsonl_first(rag_path)

    field_analysis = fact.get("field_analysis", {})
    field_details = field_analysis.get("field_details", {})
    contexts = [c.get("text", "") for c in rag.get("contexts", [])]

    old_unflag = 0   # old logic would lift the flag (source+evidence asserted)
    kept_by_fix = 0  # new logic keeps it flagged (evidence not in source)
    examples = []

    for field_name, stats in field_details.items():
        if field_name.endswith(".name"):
            continue
        should_flag, _, _ = _determine_flag_reason(stats, THRESHOLD)
        if not should_flag:
            continue
        parts = field_name.split(".")
        fp = prov.get(parts[0], {}).get(parts[1] if len(parts) > 1 else "", {})
        if not (fp.get("source") and fp.get("evidence")):
            continue
        old_unflag += 1
        new_unflag = _is_structured_source(fp.get("source")) or _evidence_grounded_in_contexts(
            fp["evidence"], contexts
        )
        if not new_unflag:
            kept_by_fix += 1
            if len(examples) < 2:
                examples.append((field_name, f"[{fp.get('source')}] {str(fp['evidence'])[:70]}"))

    # sanity: the real function must run end to end with contexts
    flag_benchmark_card_fields({}, field_analysis, THRESHOLD, prov, contexts)

    return len(field_details), len(contexts), old_unflag, kept_by_fix, examples


def main():
    print("== Unit checks on _evidence_grounded_in_contexts ==")
    units_ok = unit_checks()

    print("\n== Unit checks on backfill_from_provenance (A1b) ==")
    backfill_ok = backfill_unit_checks()
    units_ok = units_ok and backfill_ok

    trios = find_trios()
    print(f"\n== Real-data check ({len(trios)} cards) ==")
    if not trios:
        print("  no saved trios found under output/ -- skipping real-data check")

    tot_old, tot_fixed = 0, 0
    for fact, rag, prov in trios:
        name = os.path.basename(fact).replace("factuality_results_", "").replace(".json", "")
        try:
            nfields, nctx, old_unflag, kept, examples = analyze_trio(fact, rag, prov)
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            continue
        tot_old += old_unflag
        tot_fixed += kept
        msg = f"  {name:32s} fields={nfields:3d} ctx={nctx:3d} old-unflag={old_unflag:2d} kept-by-fix={kept:2d}"
        print(msg)
        for fn, ev in examples:
            print(f"        keeps flagged: {fn}  <- evidence not in source: \"{ev}\"")

    print(f"\nTotal across cards: old logic would un-flag {tot_old} fields; "
          f"fix keeps {tot_fixed} of them flagged (ungrounded provenance).")
    print(f"\nUNIT CHECKS: {'PASS' if units_ok else 'FAIL'}")
    sys.exit(0 if units_ok else 1)


if __name__ == "__main__":
    main()
