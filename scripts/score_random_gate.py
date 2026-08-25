"""Paired before/after quality read on a RANDOM sample of frozen-screened cards.

The defect-selected gate answers "did we repair the known-bad cards". It cannot answer
"are the cards better now", because it is drawn from the confirmed-findings list and runs
2.5x worse than the rest of the corpus. This scores an unbiased draw instead: every card is
compared to its OWN frozen verdict, so the draw's mean does not bias the delta.

Reports the paired delta with a bootstrap interval (n=20 is small and a raw delta is easy to
over-read), the split by paper attachment (a card whose paper did not resolve cannot exercise
the grounding path, which is what made the phase 2 gate unattributable), and the regression
question that matters most: did cards with ZERO frozen defects get worse?

Usage:
  python scripts/score_random_gate.py --new-screen eval/random20_screen/screen/verdicts \
      --run-dir output/random20/output [--draw eval/random20_draw.json]
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
    if args.draw and os.path.exists(args.draw):
        d = json.load(open(args.draw))
        print(f"  draw: seed {d.get('seed')} from {d.get('population')}")
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
    print("\n  BY CATEGORY")
    for cat in sorted(set(fc) | set(nc)):
        print(f"    {cat:24} {fc.get(cat,0):>3} -> {nc.get(cat,0):<3}")

    if args.out:
        json.dump({"rows": [{k: (dict(v) if isinstance(v, collections.Counter) else v)
                             for k, v in r.items()} for r in rows],
                   "total_frozen": tf, "total_new": tn,
                   "mean_delta": (tn - tf)/len(pairs), "ci95": [lo, hi],
                   "better": better, "worse": worse, "same": same},
                  open(args.out, "w"), indent=1)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
