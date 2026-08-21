#!/usr/bin/env python3
"""Refresh-from-cache CLI: reconstruct -> invoke -> improve-only merge -> changelog.

Usage:
  python scripts/refresh_from_cache.py --null-refresh                 # the byte gate (all 516)
  python scripts/refresh_from_cache.py --null-refresh bbq agieval     # gate on a few slugs
  python scripts/refresh_from_cache.py --pilot                        # stratified pilot (skip compose)
  python scripts/refresh_from_cache.py --pilot --compose --phases ab  # Pass 2: real compose (needs VPN)
  python scripts/refresh_from_cache.py all --compose --phases abc     # full refresh (needs VPN)
  python scripts/refresh_from_cache.py --eee-check 5                  # finalize_eee_card byte check
  python scripts/refresh_from_cache.py --rebuild-index

--null-refresh forces an empty registry and skips compose, so the merged card must
be byte-identical to the published card. Without --compose the run skips compose
for every card (build verification); --compose re-invokes the real composer.
"""

import argparse
import copy
import datetime
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from refresh import changelog as cl
from refresh import invoke, metrics as metrics_mod, pilot as pilot_mod
from refresh import merge as merge_mod
from refresh import reconstruct, resolve
from refresh.registry import build_registry


def _serialize(card):
    return json.dumps(card, indent=2, ensure_ascii=True)


def _inner(card):
    return card.get("benchmark_card", card)


def _write_card(run_out_dir, slug, card):
    cards_dir = os.path.join(run_out_dir, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    path = os.path.join(cards_dir, f"{slug}.json")
    with open(path, "w") as f:
        f.write(_serialize(card))
    return path


def _published_bytes(slug):
    with open(os.path.join(resolve.PUB_DIR, f"{slug}.json"), "rb") as f:
        return f.read()


def _run_id(mode):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{mode}_{ts}"


def run_refresh(targets, index, registry, run_out_dir, do_compose, null_refresh):
    """Drive reconstruct -> invoke -> merge for each slug. Returns (pairs, changelogs,
    byte_mismatches). compose is skipped unless do_compose; null_refresh additionally
    expects byte identity to the published card."""
    pairs = []
    changelogs = []
    mismatches = []
    compose_errors = 0
    om_base = os.path.join(run_out_dir, "compose")

    for i, slug in enumerate(targets, 1):
        run_dir = index[slug]
        live = resolve.load_live_card(slug)
        b_eligible = resolve.is_b_eligible(run_dir, slug)
        eee_metadata = reconstruct.load_eee_metadata(run_dir, slug)
        kwargs = reconstruct.reconstruct_kwargs(slug, run_dir)

        # compose_fn is injected: the real composer when do_compose, else the skip fn
        # (null-refresh / build verification) that returns None without an LLM call.
        compose_fn = None if do_compose else invoke.skip_compose
        refreshed = None
        try:
            # A compose failure (e.g. engine unreachable) leaves refreshed None, so the
            # merge floor keeps every live value: no regression from an offline card.
            result = invoke.run_compose(kwargs, compose_fn=compose_fn)
            if result is not None:
                om = invoke.make_output_manager(slug, om_base)
                invoke.seed_output_manager(om, slug, run_dir, result.get("provenance"))
                refreshed = invoke.finalize(result, eee_metadata, om, kwargs["query"])
        except Exception as e:
            compose_errors += 1
            print(f"  compose failed [{slug}]: {type(e).__name__}: {e}")

        ctx = {
            "slug": slug,
            "run_dir": run_dir,
            "b_eligible": b_eligible,
            "eee_metadata": eee_metadata,
            "kwargs": kwargs,
            "live_card": live,
            "refreshed_card": refreshed,
        }
        merged, fields = merge_mod.merge(live, refreshed, registry, ctx)
        _write_card(run_out_dir, slug, merged)

        if null_refresh and _serialize(merged).encode() != _published_bytes(slug):
            mismatches.append(slug)

        cc = cl.card_changelog(
            slug, run_dir, b_eligible, fields,
            _inner(live).get("missing_fields", []),
            _inner(merged).get("missing_fields", []),
        )
        cl.write_card_changelog(run_out_dir, slug, cc)
        changelogs.append(cc)
        pairs.append((live, merged))

        if i % 50 == 0:
            print(f"  ... {i}/{len(targets)}")

    return pairs, changelogs, mismatches, compose_errors


def cmd_eee_check(index, n):
    """Gate #2b: feed each cached risk-enhanced card (the finalize input) through the
    extracted finalize_eee_card and confirm the result matches the published card
    (ignoring the card_info timestamp). Limited to cards whose run had no FactReasoner
    pass, so the published card IS finalize(risk_enhanced)."""
    from auto_benchmarkcard.eee_workflow import finalize_eee_card

    touched = [
        "methodology.interpretation", "methodology.baseline_results",
        "methodology.human_baseline", "benchmark_details.benchmark_type",
        "benchmark_details.appears_in", "benchmark_details.contains",
        "methodology.judge_uses_llm",
    ]
    candidates = [
        s for s, rd in sorted(index.items())
        if not os.path.isdir(os.path.join(rd, "tool_output", "factreasoner"))
        and os.path.exists(os.path.join(rd, "tool_output", "risk_enhanced", f"{s}.json"))
    ]
    print(f"eee-check: {len(candidates)} no-FactReasoner candidates; testing {min(n, len(candidates))}")

    ok = 0
    for slug in candidates[:n]:
        run_dir = index[slug]
        with open(os.path.join(run_dir, "tool_output", "risk_enhanced", f"{slug}.json")) as f:
            risk_card = json.load(f)
        eee_metadata = reconstruct.load_eee_metadata(run_dir, slug)
        query = eee_metadata.get("benchmark_name") or slug

        base = tempfile.mkdtemp(prefix="eee_check_")
        om = invoke.make_output_manager(slug, base)
        invoke.seed_output_manager(om, slug, run_dir, None)
        finalized = finalize_eee_card(copy.deepcopy(risk_card), eee_metadata, om, query)

        pub_inner = copy.deepcopy(_inner(resolve.load_live_card(slug)))
        fin_inner = copy.deepcopy(_inner(finalized))
        pub_inner.pop("card_info", None)
        fin_inner.pop("card_info", None)

        field_ok = all(_field_eq(fin_inner, pub_inner, p) for p in touched)
        order_ok = list(fin_inner.keys()) == list(pub_inner.keys())
        full_ok = fin_inner == pub_inner
        status = "OK" if (field_ok and order_ok and full_ok) else "DIFF"
        if status == "OK":
            ok += 1
        print(f"  {slug}: {status} (fields={field_ok} order={order_ok} full={full_ok})")
    print(f"eee-check: {ok}/{min(n, len(candidates))} reproduced the published card")
    return ok


def _field_eq(a, b, dotted):
    def get(card, path):
        node = card
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node
    return get(a, dotted) == get(b, dotted)


def main():
    ap = argparse.ArgumentParser(description="Refresh published cards from cached tool_output.")
    ap.add_argument("slugs", nargs="*", help='slug(s), or "all"')
    ap.add_argument("--null-refresh", action="store_true",
                    help="empty registry + skip compose; expect byte identity (the gate)")
    ap.add_argument("--pilot", action="store_true", help="run the deterministic pilot subset")
    ap.add_argument("--compose", action="store_true",
                    help="re-invoke the real composer (needs the LLM engine / VPN)")
    ap.add_argument("--phases", default="", help='registered phases to apply, subset of "abc"')
    ap.add_argument("--rebuild-index", action="store_true", help="rebuild the slug->run_dir index")
    ap.add_argument("--eee-check", type=int, metavar="N", default=0,
                    help="run the finalize_eee_card byte check on N no-FactReasoner cards")
    args = ap.parse_args()

    index = resolve.build_index(rebuild=args.rebuild_index)
    all_slugs = resolve.published_slugs()
    missing = [s for s in all_slugs if s not in index]
    print(f"index: {len(index)}/{len(all_slugs)} published cards resolved to a run dir")
    if missing:
        print(f"UNRESOLVED ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    if args.eee_check:
        cmd_eee_check(index, args.eee_check)
        return

    if args.rebuild_index and not (args.slugs or args.pilot or args.null_refresh):
        return

    # Resolve targets.
    if args.pilot:
        targets = pilot_mod.select_pilot(index)
        print(f"pilot: {len(targets)} cards -> {targets}")
    elif args.slugs == ["all"] or (args.null_refresh and not args.slugs):
        targets = sorted(index)
    elif args.slugs:
        targets = [s for s in args.slugs if s in index]
        unknown = [s for s in args.slugs if s not in index]
        if unknown:
            print(f"WARNING unknown/unresolved slugs skipped: {unknown}")
    else:
        ap.error("give slug(s)/'all', or --null-refresh, or --pilot, or --eee-check")

    if args.null_refresh:
        registry = build_registry("")
        do_compose = False
    else:
        registry = build_registry(args.phases)
        do_compose = args.compose

    print(f"registry: {len(registry)} targeted field(s); compose={'real' if do_compose else 'skipped'}")

    mode = "null" if args.null_refresh else ("pilot" if args.pilot else "refresh")
    run_out_dir = os.path.join(resolve.REFRESH_OUT, _run_id(mode))
    os.makedirs(run_out_dir, exist_ok=True)

    pairs, changelogs, mismatches, compose_errors = run_refresh(
        targets, index, registry, run_out_dir, do_compose, args.null_refresh
    )

    if do_compose and compose_errors == len(targets):
        cmd = "python scripts/refresh_from_cache.py " + " ".join(sys.argv[1:])
        print("\nPASS-2 PENDING-ON-VPN: every compose failed (engine unreachable).")
        print("Reconstruct + merge path ran; no card was changed. Re-run on VPN with:")
        print(f"  {cmd}")

    rollup = cl.build_rollup(changelogs)
    cl.write_rollup(run_out_dir, rollup)
    m = metrics_mod.compute_metrics(pairs)
    with open(os.path.join(run_out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=True)

    print(f"\nrun dir: {run_out_dir}")
    print(f"rollup: {rollup['totals']}")
    print("\nmetrics (old -> new -> delta):")
    print(metrics_mod.format_table(m))

    if args.null_refresh:
        n = len(targets)
        identical = n - len(mismatches)
        print(f"\nNULL-REFRESH: {identical}/{n} merged cards byte-identical to published")
        if mismatches:
            print(f"MISMATCHES ({len(mismatches)}): {mismatches[:30]}")
            sys.exit(1)
        if any(v["delta"] != 0 for v in m.values()):
            print("WARNING: non-zero metric delta on null refresh")
            sys.exit(1)
        print("metrics null-delta == 0 confirmed")


if __name__ == "__main__":
    main()
