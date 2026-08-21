"""Wrapper around the frozen faithfulness judge (scripts/judge_gold_set.py).

The frozen module resolves every path through its module-level GOLD_DIR constant
at call time. This wrapper loads the module, repoints GOLD_DIR at an eval
directory (which s_stage.py made gold_set-shaped), and calls the frozen
prepare()/ingest() verbatim. judge_gold_set.py itself is never modified
(check_frozen guards that).

Subcommands:
  prepare      --eval-dir eval/s150 --label s150
               frozen prepare over <eval-dir>/manifest.json
               -> /tmp/gold/judge_inputs_<label>/ + <eval-dir>/judge/inputs_md5.json
  ingest       --eval-dir eval/s150 --label s150 --results <results.json>
               [--judge-model claude-fable-5] [--prompt-version v2-eee]
               frozen ingest -> <eval-dir>/judge_<label>.json
  replay-v31b  --out /tmp/replay_v31b.json
               re-emit gold_set/judge_v31b.json per-card verdicts as a
               {name: verdict} results file (zero-cost ingest validation)
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

from check_frozen import check

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JUDGE_MODULE = os.path.join(REPO, "scripts", "judge_gold_set.py")

DEFAULT_JUDGE_MODEL = "claude-fable-5"  # gold_set/judge_v31b.json judge_info pin


def load_frozen(eval_dir=None):
    spec = importlib.util.spec_from_file_location("judge_gold_set_frozen", JUDGE_MODULE)
    jgs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jgs)
    if eval_dir is not None:
        jgs.GOLD_DIR = os.path.abspath(os.path.join(REPO, eval_dir))
    return jgs


def write_inputs_md5(eval_dir, label):
    inputs_dir = f"/tmp/gold/judge_inputs_{label}"
    out = {}
    for fn in sorted(os.listdir(inputs_dir)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(inputs_dir, fn)
        with open(p, "rb") as f:
            data = f.read()
        out[fn[:-5]] = {"md5": hashlib.md5(data).hexdigest(), "bytes": len(data)}
    dest_dir = os.path.join(REPO, eval_dir, "judge")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "inputs_md5.json")
    with open(dest, "w") as f:
        json.dump({"label": label, "inputs_dir": inputs_dir, "cards": out}, f,
                  indent=2, sort_keys=True)
        f.write("\n")
    print(f"inputs md5 -> {os.path.relpath(dest, REPO)} ({len(out)} cards)")


def replay_v31b(out_path):
    src = os.path.join(REPO, "gold_set", "judge_v31b.json")
    with open(src) as f:
        judge = json.load(f)
    results = {name: {"name": name,
                      "field_verdicts": c["field_verdicts"],
                      "risk_verdicts": c["risk_verdicts"]}
               for name, c in judge["per_card"].items()}
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"replayed {len(results)} v31b verdicts -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Eval-dir wrapper for the frozen judge module.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--label", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--eval-dir", required=True)
    i.add_argument("--label", required=True)
    i.add_argument("--results", required=True)
    i.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    i.add_argument("--prompt-version", default=None,
                   help="default: the frozen module's JUDGE_PROMPT_VERSION")
    r = sub.add_parser("replay-v31b")
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    check()

    if args.cmd == "replay-v31b":
        replay_v31b(args.out)
        return

    if not os.path.exists(os.path.join(REPO, args.eval_dir, "manifest.json")):
        sys.exit(f"no manifest.json in {args.eval_dir} (run s_stage.py first)")
    jgs = load_frozen(args.eval_dir)
    if args.cmd == "prepare":
        jgs.prepare(args.label)
        write_inputs_md5(args.eval_dir, args.label)
    elif args.cmd == "ingest":
        pv = args.prompt_version or jgs.JUDGE_PROMPT_VERSION
        jgs.ingest(args.label, args.results, args.judge_model, pv)


if __name__ == "__main__":
    main()
