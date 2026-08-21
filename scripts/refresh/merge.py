"""The improve-only field merge and its harness-enforced floor.

merge() deep-copies the live card and, for each registered FieldSpec, decides
whether to overlay the refreshed value. The FLOOR is applied before any decide():
a blank / Not-specified / ungrounded `new` can never replace an existing value, so
good content is never regressed regardless of what a phase predicate returns.

The one guarded exception is a DEMOTE: a FieldSpec carrying an is_fake predicate may
lower provably-fake content to an honest "Not specified". When is_fake(old, ctx) is
True the harness bypasses ONLY the is_blank(new) clause of the floor for that one
spec; evidence is still required and decide() still runs, so genuine content is never
demoted and every demote is audited (decision="demoted").

With an empty registry the loop runs zero times, so merge returns a verbatim deep
copy of the live card. That is the null-refresh identity the acceptance gate checks.
"""

import copy
from typing import Any, Dict, List, Tuple

from auto_benchmarkcard.card_utils import extract_missing_fields, is_not_specified

from .registry import FieldRegistry, FieldSpec


def is_blank(value: Any) -> bool:
    """True when a value carries no information: None, empty string/list/dict, or the
    canonical Not-specified sentinel. Numbers and bools are never blank."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or is_not_specified(value)
    if isinstance(value, (list, dict)):
        return len(value) == 0 or is_not_specified(value)
    return False


def floor_allows_replace(old: Any, new: Any, evidence: Any, allow_blank_new: bool = False) -> bool:
    """The harness floor: a replacement is allowed only when `new` carries information
    and is grounded. This protects any non-blank `old` (the brief's minimum) and also
    refuses to adopt a blank-over-blank no-op. Independent of decide().

    allow_blank_new is the guarded demote bypass: when True (an authorized is_fake
    demote) a blank/Not-specified `new` is permitted, but evidence is still required."""
    if is_blank(new) and not allow_blank_new:
        return False
    if evidence is None:
        return False
    return True


def _get_path(card: Dict[str, Any], path: str) -> Any:
    node: Any = card
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_path(card: Dict[str, Any], path: str, value: Any) -> bool:
    """Set an EXISTING leaf in place (preserves key order). Returns False, changing
    nothing, when any parent segment is missing so the merge never adds new keys."""
    parts = path.split(".")
    node: Any = card
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False
    node[parts[-1]] = value
    return True


def _entry(spec: FieldSpec, old: Any, new: Any, decision: str, reason: str, evidence: Any) -> Dict[str, Any]:
    ev_ref = None
    if evidence is not None:
        ev_ref = evidence if isinstance(evidence, str) else repr(evidence)
        ev_ref = ev_ref[:300]
    return {
        "field": spec.path,
        "phase": spec.phase,
        "old": old,
        "new": new,
        "decision": decision,
        "reason": reason,
        "evidence_ref": ev_ref,
    }


def merge(
    live_card: Dict[str, Any],
    refreshed_card: Any,
    registry: FieldRegistry,
    ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Overlay grounded, strictly-better refreshed values onto a copy of live_card.

    Returns (merged_card, changelog_entries). card_info is never touched. When any
    field was replaced, missing_fields is re-derived so a newly filled field is not
    still listed as missing (mirrors the live finalize recompute).
    """
    merged = copy.deepcopy(live_card)
    inner = merged.get("benchmark_card", merged)
    refreshed_inner = None
    if isinstance(refreshed_card, dict):
        refreshed_inner = refreshed_card.get("benchmark_card", refreshed_card)

    kwargs = ctx.get("kwargs", {})
    changelog: List[Dict[str, Any]] = []
    any_replace = False

    for spec in registry.specs():
        old = _get_path(inner, spec.path)
        if not spec.eligible(ctx):
            changelog.append(_entry(spec, old, None, "kept", "ineligible", None))
            continue

        if spec.recompute is not None:
            new = spec.recompute(refreshed_card, kwargs, ctx)
        else:
            new = _get_path(refreshed_inner, spec.path) if refreshed_inner is not None else None

        evidence = spec.grounding(new, refreshed_card, ctx)

        # A demote is authorized only for a demote spec whose deterministic is_fake fires
        # on the current value. It bypasses just the is_blank(new) floor clause below.
        demote = spec.is_fake is not None and bool(spec.is_fake(old, ctx))

        if not floor_allows_replace(old, new, evidence, allow_blank_new=demote):
            changelog.append(_entry(spec, old, new, "kept", "floor_blocked", evidence))
            continue
        if spec.decide(old, new, evidence, ctx):
            if _set_path(inner, spec.path, new):
                any_replace = True
                decision = "demoted" if demote else "replaced"
                reason = "is_fake" if demote else "decide"
                changelog.append(_entry(spec, old, new, decision, reason, evidence))
            else:
                changelog.append(_entry(spec, old, new, "kept", "path_absent", evidence))
        else:
            changelog.append(_entry(spec, old, new, "kept", "decide_false", evidence))

    if any_replace:
        # Imported lazily: it lives in composer_tool, so a null refresh (no replacement)
        # never pulls the LLM stack just to keep the import graph honest.
        from auto_benchmarkcard.tools.composer.composer_tool import (
            drop_inapplicable_judge_missing,
        )
        inner["missing_fields"] = drop_inapplicable_judge_missing(
            inner, extract_missing_fields(inner)
        )

    return merged, changelog
