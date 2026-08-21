"""Stage sampled cards for the frozen faithfulness judge.

judge_gold_set.prepare() derives run_dir = dirname(dirname(card_path)) and reads
sources from <run_dir>/tool_output. For refreshed cards the shipped corpus card
differs from the card inside the cached run dir, so staging marries the two:

  <out>/rundirs/<slug>/benchmarkcard/benchmark_card_<slug>.json   byte-copy of the
                                                                  frozen corpus card
  <out>/rundirs/<slug>/tool_output -> <cached run_dir>/tool_output  symlink

and writes <out>/manifest.json in the gold_set manifest shape so the frozen
prepare/ingest run over it unchanged (via s_judge.py's GOLD_DIR repoint).

Usage:
  python scripts/s_stage.py --sample eval/s150/sample.json --out-dir eval/s150
  python scripts/s_stage.py --from-gold-manifest --out-dir eval/dryrun/gold
"""

import argparse
import json
import os
import shutil
import sys

from check_frozen import check

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

SOURCE_SUBDIRS = ["docling", "html", "hf", "eee", "rag"]


def _load(p):
    with open(p) as f:
        return json.load(f)


def _rel(p):
    return os.path.relpath(os.path.abspath(p), os.path.abspath(REPO))


def stage_card(name, corpus_card_abs, run_dir_abs, out_dir_abs):
    """Copy the frozen card and link tool_output. Returns the staged card path (abs)."""
    card_dir = os.path.join(out_dir_abs, "rundirs", name, "benchmarkcard")
    os.makedirs(card_dir, exist_ok=True)
    staged_card = os.path.join(card_dir, f"benchmark_card_{name}.json")
    shutil.copyfile(corpus_card_abs, staged_card)

    link = os.path.join(out_dir_abs, "rundirs", name, "tool_output")
    target = os.path.join(run_dir_abs, "tool_output")
    if not os.path.isdir(target):
        sys.exit(f"{name}: no tool_output at {target}")
    if os.path.islink(link) or os.path.exists(link):
        if os.path.islink(link):
            os.unlink(link)
        else:
            sys.exit(f"{name}: {link} exists and is not a symlink")
    os.symlink(target, link)
    return staged_card


def audit_sources(run_dir_abs):
    to = os.path.join(run_dir_abs, "tool_output")
    out = {}
    for sub in SOURCE_SUBDIRS:
        d = os.path.join(to, sub)
        if not os.path.isdir(d):
            out[sub] = {"files": 0, "bytes": 0}
            continue
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if os.path.isfile(os.path.join(d, f))]
        out[sub] = {"files": len(files), "bytes": sum(os.path.getsize(f) for f in files)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Stage sampled cards for the frozen judge.")
    ap.add_argument("--sample", default=None, help="sample.json from s_sample.py")
    ap.add_argument("--from-gold-manifest", action="store_true",
                    help="stage the 22 gold cards from gold_set/manifest.json (dry run)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="stage only the first N cards")
    args = ap.parse_args()
    check()

    if bool(args.sample) == args.from_gold_manifest:
        sys.exit("pass exactly one of --sample / --from-gold-manifest")

    out_dir_abs = os.path.abspath(os.path.join(REPO, args.out_dir))
    os.makedirs(out_dir_abs, exist_ok=True)

    if args.from_gold_manifest:
        gold = _load(os.path.join(REPO, "gold_set", "manifest.json"))
        entries = [{"name": e["name"], "stratum": e["stratum"],
                    "corpus_card": e["card_path"],
                    "source_run_dir": os.path.dirname(os.path.dirname(e["card_path"]))}
                   for e in gold["cards"]]
    else:
        sample = _load(os.path.join(REPO, args.sample))
        entries = sample["cards"]

    if args.limit:
        entries = entries[:args.limit]

    report, manifest_cards = {}, []
    warnings = 0
    for e in entries:
        name = e["name"]
        corpus_card_abs = os.path.join(REPO, e["corpus_card"])
        run_dir_abs = os.path.join(REPO, e["source_run_dir"])
        staged = stage_card(name, corpus_card_abs, run_dir_abs, out_dir_abs)
        audit = audit_sources(run_dir_abs)
        primary_empty = all(audit[s]["files"] == 0 for s in ("docling", "html", "hf"))
        if primary_empty:
            warnings += 1
            print(f"warning: {name}: docling+html+hf all empty (judge sees only EEE/RAG)")
        report[name] = {"run_dir": e["source_run_dir"], "sources": audit,
                        "primary_empty": primary_empty}
        manifest_cards.append({"name": name, "stratum": e["stratum"],
                               "card_path": _rel(staged)})

    manifest = {"n": len(manifest_cards),
                "strata": sorted({c["stratum"] for c in manifest_cards}),
                "cards": manifest_cards}
    with open(os.path.join(out_dir_abs, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir_abs, "staging_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"staged {len(manifest_cards)} cards -> {args.out_dir}/rundirs "
          f"({warnings} primary-empty warnings)")
    print(f"manifest -> {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
