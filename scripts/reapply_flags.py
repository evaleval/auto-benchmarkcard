#!/usr/bin/env python3
"""Re-apply ONLY the flagging logic on persisted FactReasoner results.

Zero inference: atoms, marginals and contexts are frozen from an existing
revalidate label (e.g. fr1_ph). This lets pure flag-post-processing changes
(policy, artifact verification, high-signal grounding) be measured offline,
isolated from NLI run-to-run variance.

Usage:
  python scripts/reapply_flags.py --source-label fr1_ph_clean --label selftest --self-test
  python scripts/reapply_flags.py --source-label fr1_ph_clean --label fr1A
  python scripts/reapply_flags.py --source-label fr1_ph_clean --label fr1B --artifact-verify --legacy-policy
  python scripts/reapply_flags.py --source-label fr1_ph --label fr1ABC --artifact-verify --high-signal

--self-test forces the legacy toggles and byte-compares every produced card
against the source label's card; any diff means the flagging code has drifted
since the source run was produced (exit 1).
"""
import argparse
import filecmp
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)

# Importing the revalidate harness pins LLM_ENGINE_TYPE=rits and loads the
# fact_reasoner stack, but nothing here makes a model call (assert_rits is not
# invoked) -- this script is offline by construction.
from revalidate_gold_set import _apply_pipeline_flagging, _locate  # noqa: E402

from auto_benchmarkcard.card_utils import extract_card  # noqa: E402
from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import load_formatted_rag_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Offline flag re-application on persisted field_analysis.")
    ap.add_argument("--source-label", required=True, help="existing gold_set/revalidate/<label> run to reuse")
    ap.add_argument("--label", required=True, help="output label")
    ap.add_argument("--manifest", default="gold_set/manifest.json")
    ap.add_argument("--legacy-policy", action="store_true",
                    help="re-add methodology.interpretation to the analytical suppression set")
    ap.add_argument("--artifact-verify", action="store_true",
                    help="verify structured-source claims against eee/hf artifacts instead of blind trust")
    ap.add_argument("--high-signal", action="store_true",
                    help="number/name veto in the provenance grounding check")
    ap.add_argument("--self-test", action="store_true",
                    help="legacy toggles + byte-compare against the source label (expects identity)")
    args = ap.parse_args()

    legacy_policy, artifact_verify, high_signal = args.legacy_policy, args.artifact_verify, args.high_signal
    if args.self_test:
        if artifact_verify or high_signal:
            sys.exit("--self-test reproduces the legacy config; drop --artifact-verify/--high-signal")
        legacy_policy, artifact_verify, high_signal = True, False, False

    man = json.load(open(args.manifest))
    entries = man["cards"] if isinstance(man, dict) and "cards" in man else man

    src_root = os.path.join("gold_set", "revalidate", args.source_label)
    out_root = os.path.join("gold_set", "revalidate", args.label)
    results, diffs = {}, []

    for entry in entries:
        name, run_dir, rag_p, prov_p, hf_p, eee_p, card_p = _locate(entry)
        fr_path = os.path.join(src_root, name, "tool_output", "factreasoner", f"factuality_results_{name}.json")
        if not os.path.exists(fr_path):
            print(f"  {name:22} SKIP no persisted factuality results in {src_root}")
            continue
        fr = json.load(open(fr_path))
        card = extract_card(json.load(open(card_p)))
        provenance = json.load(open(prov_p)) if os.path.exists(prov_p) else None
        hf_json = json.load(open(hf_p)) if os.path.exists(hf_p) else None
        eee_json = json.load(open(eee_p)) if os.path.exists(eee_p) else None
        rag = load_formatted_rag_results(rag_p)
        retrieved_contexts = [c.get("text", "") for c in rag.get("contexts", [])]

        flagged = _apply_pipeline_flagging(
            card, fr.get("field_analysis", {}), provenance, retrieved_contexts, hf_json,
            eee_json=eee_json, artifact_verify=artifact_verify,
            high_signal=high_signal, legacy_policy=legacy_policy,
        )

        cdir = os.path.join(out_root, name, "benchmarkcard")
        os.makedirs(cdir, exist_ok=True)
        out_card = os.path.join(cdir, f"benchmark_card_{name}.json")
        with open(out_card, "w") as f:
            json.dump({"benchmark_card": flagged}, f, indent=2)
        frdir = os.path.join(out_root, name, "tool_output", "factreasoner")
        os.makedirs(frdir, exist_ok=True)
        shutil.copyfile(fr_path, os.path.join(frdir, f"factuality_results_{name}.json"))

        flagged_fields = flagged.get("flagged_fields", {})
        results[name] = {"flagged_fields": flagged_fields, "n_flagged": len(flagged_fields)}
        line = f"  {name:22} flags={len(flagged_fields):>2}"
        if args.self_test:
            src_card = os.path.join(src_root, name, "benchmarkcard", f"benchmark_card_{name}.json")
            same = os.path.exists(src_card) and filecmp.cmp(out_card, src_card, shallow=False)
            line += "  identical" if same else "  DIFF"
            if not same:
                diffs.append(name)
        print(line)

    summary = {
        "label": args.label, "source_label": args.source_label,
        "legacy_policy": legacy_policy, "artifact_verify": artifact_verify,
        "high_signal": high_signal, "self_test": args.self_test,
        "n_cards": len(results),
        "total_flags": sum(v["n_flagged"] for v in results.values()),
        "per_card": results,
    }
    sp = os.path.join("gold_set", f"reapply_{args.label}.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {sp}: {len(results)} cards, {summary['total_flags']} total flags")
    if args.self_test:
        if diffs:
            sys.exit(f"SELF-TEST FAILED: {len(diffs)} cards differ from {args.source_label}: {diffs}")
        print(f"SELF-TEST OK: all {len(results)} cards byte-identical to {args.source_label}")


if __name__ == "__main__":
    main()
