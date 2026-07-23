"""Before/after per-field rates across the run, old card vs merged card.

These are DESCRIPTIVE heuristics for the human-review delta table, not gates: the
phases set their own targets. On a null refresh old == merged, so every delta is
exactly zero, which is itself a check on the merge. Each metric is a per-card
classifier; the rate is the mean over all cards and the delta is new minus old.
"""

from typing import Any, Callable, Dict, List, Tuple

from auto_benchmarkcard.card_utils import is_not_specified

_INTERP_ENUM = {"higher_is_better", "lower_is_better"}
_STORAGE_TOKENS = {
    "json", "jsonl", "csv", "tsv", "parquet", "txt", "text", "xml", "yaml", "yml",
    "zip", "arrow", "hdf5", "npz", "pickle", "pkl",
}

# Recall fields whose Not-specified rate Phase B aims to lower. Dotted paths into
# the inner card. Reported one NS-rate metric per field.
RECALL_FIELDS = [
    "methodology.calculation",
    "methodology.interpretation",
    "methodology.validation",
    "methodology.human_baseline",
    "data.collection_date",
    "data.contamination_controls",
    "ethical_and_legal_considerations.privacy_and_anonymity",
    "ethical_and_legal_considerations.consent_procedures",
    "ethical_and_legal_considerations.compliance_with_regulations",
    "purpose_and_intended_users.out_of_scope_uses",
    "purpose_and_intended_users.limitations",
]


def _inner(card: Dict[str, Any]) -> Dict[str, Any]:
    return card.get("benchmark_card", card)


def _get(card: Dict[str, Any], path: str) -> Any:
    node: Any = card
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _interp_prose(inner: Dict[str, Any]) -> bool:
    v = _get(inner, "methodology.interpretation")
    return bool(v) and not is_not_specified(v) and str(v).strip().lower() not in _INTERP_ENUM


def _metrics_real(inner: Dict[str, Any]) -> bool:
    v = _get(inner, "methodology.metrics")
    if not isinstance(v, list) or is_not_specified(v):
        return False
    for m in v:
        if not isinstance(m, str):
            continue
        ml = m.strip().lower()
        if ml and not ml.startswith("other:") and ml != "accuracy":
            return True
    return False


def _size_precise(inner: Dict[str, Any]) -> bool:
    v = _get(inner, "data.size")
    return isinstance(v, str) and not is_not_specified(v) and any(ch.isdigit() for ch in v)


def _format_structure(inner: Dict[str, Any]) -> bool:
    v = _get(inner, "data.format")
    if not isinstance(v, str) or is_not_specified(v):
        return False
    return v.strip().lower() not in _STORAGE_TOKENS


def _ns_rate_classifier(path: str) -> Callable[[Dict[str, Any]], bool]:
    def cls(inner: Dict[str, Any]) -> bool:
        v = _get(inner, path)
        return v is None or is_not_specified(v)
    return cls


def _risks_specific(inner: Dict[str, Any]) -> bool:
    name = _get(inner, "benchmark_details.name")
    risks = _get(inner, "possible_risks")
    if not isinstance(risks, list) or not isinstance(name, str) or is_not_specified(name):
        return False
    token = name.strip().lower()
    if len(token) < 3:
        return False
    for r in risks:
        if isinstance(r, dict) and token in str(r.get("description", "")).lower():
            return True
    return False


def _classifiers() -> List[Tuple[str, Callable[[Dict[str, Any]], bool]]]:
    metrics: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
        ("interpretation_prose_rate", _interp_prose),
        ("metrics_real_rate", _metrics_real),
        ("size_precise_rate", _size_precise),
        ("format_structure_rate", _format_structure),
        ("possible_risks_specific_rate", _risks_specific),
    ]
    for path in RECALL_FIELDS:
        metrics.append((f"ns_rate::{path}", _ns_rate_classifier(path)))
    return metrics


def compute_metrics(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """pairs is a list of (old_card, merged_card). Returns {metric: {old, new, delta, n}}."""
    n = len(pairs)
    out: Dict[str, Dict[str, float]] = {}
    if n == 0:
        return out
    inners = [(_inner(o), _inner(m)) for o, m in pairs]
    for name, cls in _classifiers():
        old_true = sum(1 for o, _ in inners if cls(o))
        new_true = sum(1 for _, m in inners if cls(m))
        old_rate = old_true / n
        new_rate = new_true / n
        out[name] = {
            "old": round(old_rate, 4),
            "new": round(new_rate, 4),
            "delta": round(new_rate - old_rate, 4),
            "n": n,
        }
    return out


def format_table(metrics: Dict[str, Dict[str, float]]) -> str:
    if not metrics:
        return "(no cards)"
    width = max(len(k) for k in metrics)
    lines = [f"{'metric'.ljust(width)}   old     new     delta"]
    for name in sorted(metrics):
        m = metrics[name]
        lines.append(
            f"{name.ljust(width)}  {m['old']:.4f}  {m['new']:.4f}  {m['delta']:+.4f}"
        )
    return "\n".join(lines)
