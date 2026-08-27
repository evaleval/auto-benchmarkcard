"""Paired before/after quality read on a RANDOM sample of frozen-screened cards.

The defect-selected gate answers "did we repair the known-bad cards". It cannot answer
"are the cards better now", because it is drawn from the confirmed-findings list and runs
2.5x worse than the rest of the corpus. This scores an unbiased draw instead: every card is
compared to its OWN frozen verdict, so the draw's mean does not bias the delta.

Reports the paired delta with a bootstrap interval (n=20 is small and a raw delta is easy to
over-read), the split by paper attachment (a card whose paper did not resolve cannot exercise
the grounding path, which is what made the phase 2 gate unattributable), and the regression
question that matters most: did cards with ZERO frozen defects get worse?

When --draw is given, the scored card set must equal the draw's selection exactly; a
partial screen (a card that never regenerated or was never screened) fails the run loudly
instead of silently scoring a smaller n. Per-category paired deltas carry their own
bootstrap interval so a class movement (splice, fabricated-fact) is never reported as a
bare count difference.

Usage:
  python scripts/score_random_gate.py --new-screen eval/random40_screen/screen/verdicts \
      --run-dir output/random40/output --draw eval/random40_draw.json
"""

import argparse
import collections
import glob
import json
import os
import random

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FROZEN_SCREEN = os.path.join(REPO, "eval", "s150", "screen", "verdicts")


def needs_fix(verdict):
    return [f for f in (verdict.get("findings") or []) if f.get("severity") == "needs-fix"]


def by_category(findings):
    return collections.Counter(f.get("category", "?") for f in findings)


def load_dir(d):
    return {f[:-5]: json.load(open(os.path.join(d, f)))
            for f in sorted(os.listdir(d)) if f.endswith(".json")}


def paper_attached(run_dir, slug):
    """Did a paper actually bind this run? Empty/absent docling means it did not."""
    dirs = sorted(glob.glob(os.path.join(run_dir, f"{slug}_*")))
    if not dirs:
        return None
    p = os.path.join(dirs[-1], "tool_output", "docling", f"{slug}.json")
    return os.path.exists(p) and os.path.getsize(p) > 0


def bootstrap_ci(pairs, iters=20000, seed=7):
    """Paired bootstrap over cards for the mean per-card delta (new minus frozen)."""
    rng = random.Random(seed)
    deltas = [n - f for f, n in pairs]
    n = len(deltas)
    means = []
    for _ in range(iters):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def sign_test(pairs):
    better = sum(1 for f, n in pairs if n < f)
    worse = sum(1 for f, n in pairs if n > f)
    same = sum(1 for f, n in pairs if n == f)
    return better, worse, same


def check_draw(scored_cards, draw):
    """The scored set must be exactly the drawn set. Returns the draw's card list.

    Failure class: partial-screen undercount. A regen or screen that silently dropped
    cards would otherwise score a smaller, self-selected n and bias the paired read."""
    drawn = draw.get("selection_order") or []
    if not drawn:
        raise SystemExit("draw file carries no selection_order")
    missing = sorted(set(drawn) - set(scored_cards))
    extra = sorted(set(scored_cards) - set(drawn))
    if missing or extra:
        raise SystemExit(
            f"scored cards do not match the draw (n={len(drawn)}): "
            f"missing={missing} extra={extra}")
    return drawn


def class_pairs(rows, cat):
    """Per-card (frozen, new) counts for one finding category."""
    return [(r["frozen_cats"].get(cat, 0), r["new_cats"].get(cat, 0)) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-screen", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--draw", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new = load_dir(args.new_screen)
    frozen = load_dir(FROZEN_SCREEN)
    cards = sorted(new)
    draw = None
    if args.draw:
        draw = json.load(open(args.draw))
        cards = sorted(check_draw(cards, draw))
    missing = [c for c in cards if c not in frozen]
    if missing:
        raise SystemExit(f"no frozen verdict for: {missing}")

    rows = []
    for c in cards:
        f, n = needs_fix(frozen[c]), needs_fix(new[c])
        rows.append({
            "card": c,
            "frozen": len(f), "new": len(n), "delta": len(n) - len(f),
            "frozen_cats": by_category(f), "new_cats": by_category(n),
            "attached": paper_attached(args.run_dir, c) if args.run_dir else None,
            "frozen_paper": frozen[c].get("paper_assessment"),
            "new_paper": new[c].get("paper_assessment"),
        })

    print(f"PAIRED READ over {len(rows)} randomly drawn cards")
    if draw is not None:
        pop = draw.get("population")
        pop_label = pop.get("path", pop) if isinstance(pop, dict) else pop
        print(f"  draw: seed {draw.get('seed')} from {pop_label} (n={draw.get('n')})")
    print()
    print(f"  {'card':32} {'frozen':>6} {'new':>5} {'delta':>6} {'paper':>7}")
    for r in rows:
        att = {True: "yes", False: "NO", None: "?"}[r["attached"]]
        print(f"  {r['card']:32} {r['frozen']:>6} {r['new']:>5} {r['delta']:>+6} {att:>7}")

    pairs = [(r["frozen"], r["new"]) for r in rows]
    tf, tn = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    lo, hi = bootstrap_ci(pairs)
    better, worse, same = sign_test(pairs)
    print(f"\n  TOTAL needs-fix {tf} -> {tn}   ({tf/len(pairs):.2f} -> {tn/len(pairs):.2f} per card)")
    print(f"  mean paired delta {(tn - tf)/len(pairs):+.2f} per card, 95% bootstrap CI "
          f"[{lo:+.2f}, {hi:+.2f}]")
    if lo < 0 < hi:
        print("  -> the interval spans zero: this sample does NOT establish a change.")
    elif hi <= 0:
        print("  -> interval entirely below zero: a real improvement on ordinary cards.")
    else:
        print("  -> interval entirely above zero: a real REGRESSION on ordinary cards.")
    print(f"  cards better {better} / worse {worse} / unchanged {same}")

    print("\n  BY PAPER ATTACHMENT (a card with no paper cannot exercise the grounding path)")
    for want, label in ((True, "paper attached"), (False, "NO paper")):
        sub = [r for r in rows if r["attached"] is want]
        if not sub:
            continue
        sf, sn = sum(r["frozen"] for r in sub), sum(r["new"] for r in sub)
        print(f"    {label:16} n={len(sub):>3}  needs-fix {sf:>3} -> {sn:<3} "
              f"({sf/len(sub):.2f} -> {sn/len(sub):.2f} per card)")

    clean = [r for r in rows if r["frozen"] == 0]
    if clean:
        cn = sum(r["new"] for r in clean)
        print(f"\n  REGRESSION CHECK, cards with ZERO frozen needs-fix findings: n={len(clean)}")
        print(f"    they now carry {cn} needs-fix findings ({cn/len(clean):.2f} per card)")
        for r in clean:
            if r["new"]:
                print(f"      {r['card']:32} 0 -> {r['new']}  {dict(r['new_cats'])}")

    fc, nc = collections.Counter(), collections.Counter()
    for r in rows:
        fc.update(r["frozen_cats"]); nc.update(r["new_cats"])
    print("\n  BY CATEGORY (paired bootstrap per class; a bare count delta over-reads)")
    by_class = {}
    for cat in sorted(set(fc) | set(nc)):
        cp = class_pairs(rows, cat)
        clo, chi = bootstrap_ci(cp)
        cmean = sum(n - f for f, n in cp) / len(cp)
        verdict = "spans zero" if clo < 0 < chi else (
            "BELOW zero" if chi <= 0 else "ABOVE zero")
        by_class[cat] = {"frozen": fc.get(cat, 0), "new": nc.get(cat, 0),
                         "mean_delta": cmean, "ci95": [clo, chi]}
        print(f"    {cat:24} {fc.get(cat,0):>3} -> {nc.get(cat,0):<3} "
              f"mean {cmean:+.2f}/card CI [{clo:+.2f}, {chi:+.2f}]  ({verdict})")

    if args.out:
        json.dump({"rows": [{k: (dict(v) if isinstance(v, collections.Counter) else v)
                             for k, v in r.items()} for r in rows],
                   "total_frozen": tf, "total_new": tn,
                   "mean_delta": (tn - tf)/len(pairs), "ci95": [lo, hi],
                   "better": better, "worse": worse, "same": same,
                   "by_class": by_class},
                  open(args.out, "w"), indent=1)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
