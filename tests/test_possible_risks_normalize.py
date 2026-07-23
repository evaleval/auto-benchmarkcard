"""possible_risks description normalization. The detector path wraps description in a list,
the deterministic single-judge rule writes a plain string; integrate_risks_into_benchmark_card
must collapse every item's description to a string so the shape is consistent. Network-free.
"""

from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import (
    integrate_risks_into_benchmark_card,
)


def test_list_and_string_descriptions_both_become_strings():
    card = {"benchmark_details": {}}
    risks = [
        {"category": "C1", "description": ["a risk paragraph"], "url": None},
        {"category": "C2", "description": "already a string", "url": "http://x"},
    ]
    out = integrate_risks_into_benchmark_card(card, risks)
    descs = [r["description"] for r in out["possible_risks"]]
    assert descs == ["a risk paragraph", "already a string"]
    assert all(isinstance(d, str) for d in descs)
