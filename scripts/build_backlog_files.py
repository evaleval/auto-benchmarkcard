#!/usr/bin/env python3
"""Build per-phase backlog files from the triage CSV and reaudit JSON.

Inputs:
  output/audit/llm_stats_triage.csv
  output/audit/reaudit_2026-05-06.json

Outputs (in repo root):
  phase2a_smoke9.txt        webhook-failed entries (must-generate w/ status)
  phase2b_must.txt          remaining must-generate (no webhook attempt)
  phase2c_high.txt          llm-stats card HIGH (72)
  phase2d_medium.txt        llm-stats card MEDIUM (39)
  phase2e_low.txt           llm-stats card LOW (148)
  phase2f_subtasks.txt      group-under entries (subtasks)
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRIAGE_CSV = ROOT / "output/audit/llm_stats_triage.csv"
AUDIT_JSON = ROOT / "output/audit/reaudit_2026-05-06.json"


def write_backlog(name: str, keys: list[str], note: str = "") -> None:
    path = ROOT / name
    with open(path, "w") as f:
        if note:
            f.write(f"# {note}\n")
        for k in keys:
            f.write(f"{k}\n")
    print(f"  {name}: {len(keys)} entries")


def main() -> None:
    triage = list(csv.DictReader(open(TRIAGE_CSV)))
    audit = json.load(open(AUDIT_JSON))

    # phase2a + phase2b: split must_generate by webhook attempt
    must_gen = audit.get("must_generate", [])
    smoke9 = [m["v13_key"] for m in must_gen if m.get("webhook_status") == "generation_failed"]
    rest_must = [m["v13_key"] for m in must_gen if m.get("webhook_status") != "generation_failed"]

    # phases 2c/2d/2e: llm-stats card by confidence
    by_conf = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for r in triage:
        if r["recommendation"] == "card" and r["confidence"] in by_conf:
            by_conf[r["confidence"]].append(r["v13_key"])

    # phase 2f: subtasks (group-under-X)
    subtasks = [r["v13_key"] for r in triage if r["recommendation"].startswith("group-under")]

    print("Building backlog files in", ROOT)
    write_backlog("phase2a_smoke9.txt", smoke9, "9 webhook-failed entries (re-run after pipeline fixes)")
    write_backlog("phase2b_must.txt", rest_must, "Non-llm-stats must-generate, never attempted")
    write_backlog("phase2c_high.txt", by_conf["HIGH"], "llm-stats card HIGH (72)")
    write_backlog("phase2d_medium.txt", by_conf["MEDIUM"], "llm-stats card MEDIUM (39)")
    write_backlog("phase2e_low.txt", by_conf["LOW"], "llm-stats card LOW (148)")
    write_backlog("phase2f_subtasks.txt", subtasks, "Subtasks (group-under-X) — same pipeline, separate-file pattern")

    total = len(smoke9) + len(rest_must) + sum(len(v) for v in by_conf.values()) + len(subtasks)
    print(f"\nTotal cards to generate: {total}")


if __name__ == "__main__":
    main()
