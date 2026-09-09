"""Offline test for strip_page_furniture (cross-contamination fix).

Feeds captured llm-stats scrapes (produced by the unpatched scraper, so they still contain the
leaderboard table + related-benchmarks carousel) into the strip and checks that other-benchmark
furniture is dropped while each benchmark's genuine hero summary survives. No network, no LLM.

Run: PYTHONPATH=src python scripts/test_furniture_strip.py
Set BENCHMARKCARD_CAPTURED_RUNS to the directory containing d6_pilot if needed.
"""

import json
import os
from pathlib import Path

from auto_benchmarkcard.tools.html.html_tool import strip_page_furniture

# Captured scrapes are optional local data, outside the public source tree.
FIXTURE_BASE = Path(os.environ.get(
    "BENCHMARKCARD_CAPTURED_RUNS", Path(__file__).resolve().parents[1] / "output"
)).expanduser() / "d6_pilot" / "output"
FIXTURES = {
    "aa-lcr": "aa-lcr_2026-06-19_13-56/tool_output/html/aa-lcr.json",
    "bigcodebench": "bigcodebench_2026-06-19_13-43/tool_output/html/bigcodebench.json",
    "biolp-bench": "biolp-bench_2026-06-19_14-03/tool_output/html/biolp-bench.json",
}

# Per fixture: strings that MUST be gone after the strip, and strings that MUST survive.
DROPPED = {
    "aa-lcr": ["448 multiple-choice questions", "biology, physics, and chemistry",
               "extends MMLU", "Progress Over Time"],
    "bigcodebench": ["448 multiple-choice questions", "Progress Over Time"],
    "biolp-bench": ["448 multiple-choice questions", "SuperGPQA", "Progress Over Time"],
}
KEPT = {
    "aa-lcr": ["evaluates large language models on long context and reasoning tasks",
               "leads the AA-LCR leaderboard with a score"],
    "bigcodebench": ["139 libraries and 7 domains"],
    "biolp-bench": ["biological laboratory protocols"],  # genuine biology stays (positional cut)
}


def _load(name: str):
    path = FIXTURE_BASE / FIXTURES[name]
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    failures = []
    ran = 0

    for name in FIXTURES:
        data = _load(name)
        if data is None:
            print(f"SKIP {name}: fixture not found")
            continue
        ran += 1
        clean, removed = strip_page_furniture(data["text"], data["url"])

        if removed <= 0:
            failures.append(f"{name}: expected furniture removed, got removed={removed}")
        for bad in DROPPED[name]:
            if bad in clean:
                failures.append(f"{name}: contaminant survived: {bad!r}")
        for good in KEPT[name]:
            if good not in clean:
                failures.append(f"{name}: genuine content lost: {good!r}")
        print(f"OK   {name}: removed {removed} chars; {len(clean)} kept")

    # Non-llm-stats pages must be an exact no-op, even with furniture-looking lines.
    noop_text = "Foo bar\nProgress Over Time\n| a | b |\nMy Leaderboard"
    out, removed = strip_page_furniture(noop_text, "https://huggingface.co/datasets/x")
    if out != noop_text or removed != 0:
        failures.append(f"non-llm-stats not a no-op: removed={removed}")

    # No marker present on an llm-stats page -> unchanged (never over-strip).
    no_marker = "AA-LCR\nA long-context retrieval benchmark with real documents."
    out, removed = strip_page_furniture(no_marker, "https://llm-stats.com/benchmarks/aa-lcr")
    if out != no_marker or removed != 0:
        failures.append(f"no-marker llm-stats page should be unchanged: removed={removed}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if ran == 0:
        print("\nNO FIXTURES FOUND — cannot validate against captured scrapes")
        return 1
    print(f"\nALL FURNITURE-STRIP ASSERTIONS PASSED ({ran} fixtures + 2 edge cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
