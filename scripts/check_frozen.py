"""Frozen-instrument md5 guard for the eval harness.

The eval design freezes four files (the faithfulness judge and its imports).
Every eval entry point calls check() first; a FROZEN mismatch is a hard error,
a WATCH mismatch is a loud warning (eval_gold_set.py may gain additive
functions; a screen-script change invalidates screen_meta.script_md5).

Usage:
  python scripts/check_frozen.py
  python scripts/check_frozen.py --write-baseline # record the current md5s
"""

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASELINE = os.path.join(REPO, "eval", "corpus", "frozen_code_hashes_md5.json")

FROZEN = [
    "scripts/judge_gold_set.py",
    "src/auto_benchmarkcard/card_utils.py",
    "src/auto_benchmarkcard/validation_policy.py",
    "src/auto_benchmarkcard/tools/rag/atomizer.py",
]
WATCH = [
    "scripts/eval_gold_set.py",
    "s_screen_review.js",
]


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def current():
    out = {"frozen": {}, "watch": {}}
    for rel in FROZEN:
        out["frozen"][rel] = file_md5(os.path.join(REPO, rel))
    for rel in WATCH:
        p = os.path.join(REPO, rel)
        out["watch"][rel] = file_md5(p) if os.path.exists(p) else None
    return out


def check(quiet=True):
    """Verify frozen files against the committed baseline. Exits on violation."""
    if not os.path.exists(BASELINE):
        sys.exit(f"check_frozen: no baseline at {os.path.relpath(BASELINE, REPO)} "
                 "(run scripts/check_frozen.py --write-baseline once)")
    with open(BASELINE) as f:
        base = json.load(f)
    now = current()
    bad = [rel for rel, md5 in base["frozen"].items() if now["frozen"].get(rel) != md5]
    if bad:
        for rel in bad:
            print(f"check_frozen: FROZEN file modified: {rel} "
                  f"(baseline {base['frozen'][rel]}, now {now['frozen'].get(rel)})", file=sys.stderr)
        sys.exit("check_frozen: frozen instrument changed - refusing to run")
    for rel, md5 in base.get("watch", {}).items():
        if now["watch"].get(rel) != md5:
            print(f"check_frozen: WARNING watch file changed: {rel} "
                  f"(baseline {md5}, now {now['watch'].get(rel)})", file=sys.stderr)
    if not quiet:
        print(f"check_frozen: {len(base['frozen'])} frozen ok, {len(base.get('watch', {}))} watched")


def main():
    ap = argparse.ArgumentParser(description="Eval-harness frozen-file md5 guard.")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    if args.write_baseline:
        if os.path.exists(BASELINE):
            check()  # never re-baseline over a frozen-file violation
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as f:
            json.dump(current(), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"baseline -> {os.path.relpath(BASELINE, REPO)}")
        return
    check(quiet=False)


if __name__ == "__main__":
    main()
