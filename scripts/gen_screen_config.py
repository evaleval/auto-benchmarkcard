"""Generate the args config for the s_screen_review.js workflow from a sample file.

The workflow script has no filesystem access of its own; this config is passed
verbatim as the Workflow args and also persisted next to the screen outputs for
the record.

Usage:
  python scripts/gen_screen_config.py --sample eval/s150/sample.json \
      --stage-root eval/s150 --outdir eval/s150/screen [--limit N] [--batch 25]
"""

import argparse
import json
import os

from check_frozen import check

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

MODEL_ALIAS = "fable"          # Workflow agent opts alias
MODEL_ID = "claude-fable-5"    # recorded in every verdict + screen_meta


def main():
    ap = argparse.ArgumentParser(description="Screen workflow config generator.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--stage-root", required=True, help="dir holding rundirs/ (from s_stage.py)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()
    check()

    with open(os.path.join(REPO, args.sample)) as f:
        sample = json.load(f)
    cards = sample["cards"][:args.limit] if args.limit else sample["cards"]

    outdir_abs = os.path.abspath(os.path.join(REPO, args.outdir))
    cfg = {
        "cards": [{"slug": c["name"],
                   "staged_dir": os.path.abspath(
                       os.path.join(REPO, args.stage_root, "rundirs", c["name"]))}
                  for c in cards],
        "outdir": outdir_abs,
        "model_alias": MODEL_ALIAS,
        "model_id": MODEL_ID,
        "batch": args.batch,
        "sample_file": args.sample,
    }
    os.makedirs(outdir_abs, exist_ok=True)
    os.makedirs(os.path.join(outdir_abs, "verdicts"), exist_ok=True)
    dest = os.path.join(outdir_abs, "screen_config.json")
    with open(dest, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"config -> {os.path.relpath(dest, REPO)} ({len(cfg['cards'])} cards)")
    print("pass this file's JSON verbatim as the Workflow args for s_screen_review.js")


if __name__ == "__main__":
    main()
