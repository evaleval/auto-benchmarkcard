"""Referent frame: the typed list of things evidence for this card could be about.

Phase 2 foundation. The frame enumerates, per card, the candidate referents that
wrong-section-splice errors confuse: the target benchmark itself, sibling
benchmarks named in the same paper or suite, subsets/configs, versions, splits,
evaluation settings, and metrics with their role. Downstream, every evidence
item names the frame node it is about, and assignment checks compare that claim
against the node type the field expects.

Node ids are stable within a card ("target", "sibling:fibble", "subset:hard",
"version:2.0", "split:test", "metric:accuracy"). Deterministic sources fill
first (EEE metadata, HF configs, size_breakdown keys, version regexes over the
paper head); one optional fail-soft LLM pass enriches sibling benchmarks named
in the paper. The frame is telemetry-visible and saved as a composer sidecar.
"""

from typing import Any, Dict, List, Optional

NODE_TYPES = (
    "target", "sibling", "subset", "version", "split", "setting", "metric",
)
METRIC_ROLES = ("primary", "secondary", "meta_eval", "unknown")


def build_frame(
    benchmark_name: str,
    eee_metadata: Optional[Dict[str, Any]],
    hf_json: Optional[Dict[str, Any]],
    doc_index=None,
    paper_text: str = "",
    llm_handler=None,
) -> Dict[str, Any]:
    """Return {"target_id": "target", "nodes": [...], "telemetry": {...}}.

    Each node: {"id", "type", "name", "aliases": [...], "description": one line,
    and for metrics "role"}. Deterministic builders run always; the LLM sibling
    pass runs only when llm_handler and paper_text are given and never raises
    (fail-soft to the deterministic frame). Dedup by normalized name across
    builders; the target node always exists and carries the resolved benchmark
    name plus HF/EEE aliases.
    """
    raise NotImplementedError


def node_lookup(frame: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """id -> node mapping for O(1) checks."""
    raise NotImplementedError


def match_mention(frame: Dict[str, Any], text: str) -> List[str]:
    """Frame node ids whose name or alias occurs in text (whole-token,
    case-insensitive, longest-alias-first). Deterministic; used by the
    sibling-name veto and subset cross-check gates."""
    raise NotImplementedError
