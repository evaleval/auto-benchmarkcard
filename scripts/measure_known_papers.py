"""A3 redundancy measurement: does the automatic resolver find the SAME paper as the
hardcoded KNOWN_PAPERS table?

For a representative sample of table entries, this bypasses the table and runs the
normal resolver (OpenAlex/S2 search + LLM verify), then compares the resolved arXiv id
to the table's. High agreement => the table is mostly redundant and can be removed /
shrunk to a few overrides. Low agreement => the resolver is too weak; build the agentic
source-finder before removing the table.

SAFETY: makes NO inference calls unless you pass --run. Each sampled benchmark costs
~1-2 LLM calls (metadata pre-query + 70B verify) + free scholarly-API lookups. Pick the
engine via LLM_ENGINE_TYPE in .env (hf for a small sample, rits for the full table).

Usage:
    python scripts/measure_known_papers.py            # dry run: prints the plan, 0 calls
    python scripts/measure_known_papers.py --run      # executes the comparison
    python scripts/measure_known_papers.py --run --n 20
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ARXIV_RE = re.compile(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)


def arxiv_id(url):
    if not url:
        return None
    m = ARXIV_RE.search(url)
    return m.group(1) if m else url.strip().rstrip("/").lower()


def pick_sample(keys, n):
    """Deterministic spread across the (category-grouped) table order."""
    if len(keys) <= n:
        return keys
    step = max(1, len(keys) // n)
    return keys[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the resolver (makes inference calls)")
    ap.add_argument("--n", type=int, default=20, help="sample size")
    args = ap.parse_args()

    from auto_benchmarkcard.tools.eee import paper_resolver as pr
    from auto_benchmarkcard.config import Config

    keys = list(pr.KNOWN_PAPERS.keys())
    sample = pick_sample(keys, args.n)

    print(f"KNOWN_PAPERS entries: {len(keys)}")
    print(f"Sample ({len(sample)}): {', '.join(sample)}")
    print(f"Engine: LLM_ENGINE_TYPE={Config.LLM_ENGINE_TYPE}")
    print(f"Estimated cost if run: ~{len(sample)}-{2 * len(sample)} LLM calls + free scholarly-API lookups\n")

    if not args.run:
        print("DRY RUN — no inference made. Re-run with --run to execute.")
        print("Table values for the sample:")
        for k in sample:
            print(f"  {k:28s} -> {pr.KNOWN_PAPERS[k]}")
        return

    table = dict(pr.KNOWN_PAPERS)
    agree, disagree, missed = 0, [], []
    try:
        pr.KNOWN_PAPERS = {}  # bypass the override so we measure the resolver alone
        for k in sample:
            expected = arxiv_id(table[k])
            try:
                res = pr.resolve_paper(suite_name=k)
                got_url = res.get("url") if isinstance(res, dict) else res
            except Exception as e:
                got_url = None
                print(f"  {k:28s} ERROR {e}")
            got = arxiv_id(got_url)
            if got is None:
                missed.append(k)
                status = "MISS"
            elif got == expected:
                agree += 1
                status = "ok"
            else:
                disagree.append((k, expected, got))
                status = "DIFF"
            print(f"  {k:28s} [{status:4s}] table={expected}  resolver={got}")
    finally:
        pr.KNOWN_PAPERS = table  # always restore

    n = len(sample)
    print(f"\nAgreement: {agree}/{n} ({100 * agree // n if n else 0}%)  "
          f"| different: {len(disagree)}  | resolver found nothing: {len(missed)}")
    if disagree:
        print("Disagreements (table vs resolver):")
        for k, exp, got in disagree:
            print(f"  {k}: {exp} != {got}")
    if missed:
        print(f"Resolver found nothing for: {', '.join(missed)}")
    print("\nReading: high agreement -> table mostly redundant, safe to remove/shrink. "
          "Low agreement / many misses -> resolver too weak, keep table until source-finder lands.")


if __name__ == "__main__":
    main()
