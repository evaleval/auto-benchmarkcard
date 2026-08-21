"""Offline invocation of the real composer + finalize (and a risk stub).

run_compose calls the pure composer function with reconstructed kwargs. compose_fn
is injectable so the null-refresh gate can pass a skip function (no LLM, returns
None) while the pilot path passes the real compose_benchmark_card.func.

finalize runs the extracted finalize_eee_card, the SAME post-composition step the
live pipeline applies, so a refreshed card is shaped exactly like a published one.
Heavy imports are lazy so the null-refresh path does not need the LLM stack.
"""

import os
import shutil
from typing import Any, Callable, Dict, Optional


def skip_compose(**_kwargs: Any) -> None:
    """A compose_fn that performs no composition (null-refresh gate)."""
    return None


def run_compose(
    kwargs: Dict[str, Any], compose_fn: Optional[Callable[..., Any]] = None
) -> Optional[Dict[str, Any]]:
    """Re-compose a card from reconstructed kwargs. Returns the composer result dict
    ({"benchmark_card", "provenance", "composition_metadata"}) or None when skipped."""
    if compose_fn is None:
        from auto_benchmarkcard.tools.composer.composer_tool import compose_benchmark_card
        compose_fn = compose_benchmark_card.func
    return compose_fn(**kwargs)


def make_output_manager(slug: str, base_dir: str):
    """An OutputManager rooted under base_dir whose sanitized name equals slug."""
    from auto_benchmarkcard.output import OutputManager
    return OutputManager(slug, base_dir)


def seed_output_manager(
    output_manager,
    slug: str,
    run_dir: str,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Seed the composer provenance and hf sidecars finalize_eee_card reads.

    The live finalize reads the provenance the composer just wrote and the cached hf
    json; reproduce both so the offline finalize behaves identically. Uses the fresh
    compose provenance when given, else the cached sidecar.
    """
    if provenance is not None:
        output_manager.save_tool_output(provenance, "composer", f"provenance_{slug}.json")
    else:
        src = os.path.join(run_dir, "tool_output", "composer", f"provenance_{slug}.json")
        if os.path.exists(src):
            dst_dir = output_manager.get_tool_output_path("composer")
            shutil.copyfile(src, os.path.join(dst_dir, f"provenance_{slug}.json"))

    hf_src = os.path.join(run_dir, "tool_output", "hf", f"{slug}.json")
    if os.path.exists(hf_src):
        dst_dir = output_manager.get_tool_output_path("hf")
        shutil.copyfile(hf_src, os.path.join(dst_dir, f"{slug}.json"))


def finalize(
    refreshed: Dict[str, Any],
    eee_metadata: Dict[str, Any],
    output_manager,
    benchmark_name: str,
) -> Dict[str, Any]:
    """Apply the live EEE finalize to a refreshed card; returns the finalized wrapper."""
    from auto_benchmarkcard.eee_workflow import finalize_eee_card
    return finalize_eee_card(refreshed, eee_metadata, output_manager, benchmark_name)


def run_risk(card: Dict[str, Any], enabled: bool = False) -> Dict[str, Any]:
    """Risk-identification hook for Phase C. Off by default (returns card unchanged).

    When enabled, mirrors workers.run_risk_identification's chain on the inner card.
    Phase C supplies the registry predicate that consumes the result; this only runs
    the identification so that predicate has fresh risks to ground against.
    """
    if not enabled:
        return card
    from auto_benchmarkcard.card_utils import apply_single_judge_bias_rule
    from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import (
        identify_and_integrate_risks,
        omit_empty_risk_urls,
    )

    inner = card.get("benchmark_card", card)
    enhanced = identify_and_integrate_risks(inner)
    enhanced = apply_single_judge_bias_rule(enhanced)
    enhanced = omit_empty_risk_urls(enhanced)
    if "benchmark_card" in card:
        card["benchmark_card"] = enhanced
        return card
    return enhanced
