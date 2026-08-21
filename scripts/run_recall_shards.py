#!/usr/bin/env python3
"""Run the recall refresh over many slugs in parallel shards.

refresh_from_cache.py processes its slugs sequentially in one process and writes to
one refresh_<timestamp> dir. To parallelize the recall compose pass we launch N shard
processes over disjoint slug groups, staggered so each lands in its own run dir
(REFRESH_OUT/refresh_<ts> is keyed to the second and is not overridable).

Usage:
  python scripts/run_recall_shards.py <slugfile> --shards 8 --compose --phases abc
  python scripts/run_recall_shards.py <slugfile> --shards 5            # dry (no compose), mechanics test

<slugfile> has one slug per line. The script prints each shard's run dir + compose-error
count and writes a manifest (shard_runs.json) listing the run dirs for the assembly step.
It does NOT assemble or commit anything.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_slugs(path):
    with open(path) as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def split_round_robin(items, n):
    groups = [[] for _ in range(n)]
    for i, it in enumerate(items):
        groups[i % n].append(it)
    return [g for g in groups if g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugfile")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--compose", action="store_true", help="real compose (needs VPN/RITS; spends)")
    ap.add_argument("--phases", default="abc")
    ap.add_argument("--stagger", type=float, default=1.0, help="seconds between shard launches")
    ap.add_argument("--run-root", default=os.environ.get("REFRESH_RUN_ROOT", ""),
                    help="REFRESH_RUN_ROOT for the shards (cached tool_output)")
    ap.add_argument("--tag", default="recall", help="label for this shard batch's output tree")
    args = ap.parse_args()

    slugs = read_slugs(args.slugfile)
    groups = split_round_robin(slugs, args.shards)
    base_out = os.path.join(REPO, "output", "recall_shards", args.tag)
    log_dir = os.path.join(base_out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    if args.run_root:
        env["REFRESH_RUN_ROOT"] = args.run_root

    print(f"{len(slugs)} slugs -> {len(groups)} shards; compose={'REAL (spends)' if args.compose else 'skipped (dry)'}")
    procs = []
    for i, group in enumerate(groups):
        cmd = [sys.executable, "scripts/refresh_from_cache.py", *group, "--phases", args.phases]
        if args.compose:
            cmd.append("--compose")
        shard_env = dict(env)
        shard_env["REFRESH_OUT_DIR"] = os.path.join(base_out, f"shard_{i:02d}")
        logpath = os.path.join(log_dir, f"shard_{i:02d}.log")
        logf = open(logpath, "w")
        p = subprocess.Popen(cmd, cwd=REPO, env=shard_env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((i, p, logf, logpath, len(group)))
        print(f"  launched shard {i:02d}: {len(group)} slugs -> {logpath}")
        if i < len(groups) - 1:
            time.sleep(args.stagger)

    run_dirs = []
    for i, p, logf, logpath, n in procs:
        rc = p.wait()
        logf.close()
        with open(logpath) as f:
            out = f.read()
        m = re.search(r"run dir:\s*(\S+)", out)
        rundir = m.group(1) if m else None
        errm = re.search(r"compose failed", out)
        pending = "PASS-2 PENDING-ON-VPN" in out
        run_dirs.append({"shard": i, "rc": rc, "n": n, "run_dir": rundir,
                         "had_compose_error": bool(errm), "all_failed": pending})
        print(f"  shard {i:02d}: rc={rc} n={n} run_dir={rundir} "
              f"compose_err={bool(errm)} all_failed={pending}")

    manifest = os.path.join(log_dir, "shard_runs.json")
    with open(manifest, "w") as f:
        json.dump({"slugfile": args.slugfile, "compose": args.compose,
                   "phases": args.phases, "shards": run_dirs}, f, indent=2)
    print(f"\nmanifest: {manifest}")
    any_failed = any(s["all_failed"] for s in run_dirs)
    if any_failed:
        print("WARNING: at least one shard had all composes fail (VPN/RITS?). Inspect logs.")
        sys.exit(2)


if __name__ == "__main__":
    main()
