"""Pure-numpy estimators for the stratified S evaluation (no I/O).

Design (locked): every S number is stratum-weighted; the primary estimator is
the stratified combined ratio over card-level counts

    R = (sum_h w_h sum_{i in h} num_i) / (sum_h w_h sum_{i in h} den_i)

card-level proportions are the den_i = 1 special case. CIs come from a
card-clustered stratified bootstrap (resample n_h cards with replacement
within each stratum, recompute the full weighted estimator); ONE shared
replicate index tensor is drawn per run so ratio/difference CIs between
metrics are paired-correct. Every proportion also gets an effective-sample-
size Wilson interval; the bootstrap is the primary CI only when the raw event
count is >= 10 and every stratum with a nonzero denominator has >= 1 event,
otherwise ci_method flips to wilson-neff (rare-event honesty).

Finite-population correction is deliberately ignored (sampling fractions
~0.3 make the CIs conservative by roughly 15-20%).
"""

import numpy as np

Z95 = 1.959963984540054


def make_strata(stratum_labels, weights_by_label):
    """stratum_labels: list of per-card labels aligned to the record arrays.
    weights_by_label: {label: w_h}. Returns {label: {"idx": array, "w": float}}."""
    labels = sorted(set(stratum_labels))
    out = {}
    for h in labels:
        idx = np.array([i for i, s in enumerate(stratum_labels) if s == h], dtype=int)
        out[h] = {"idx": idx, "w": float(weights_by_label[h])}
    return out


def make_replicates(strata, B=5000, seed=20260704):
    """One shared (B, n_h) position tensor per stratum, drawn once per run."""
    rng = np.random.default_rng(seed)
    reps = {}
    for h in sorted(strata):
        n_h = len(strata[h]["idx"])
        reps[h] = rng.integers(0, n_h, size=(B, n_h)) if n_h else np.zeros((B, 0), int)
    return reps


def _weighted_totals(arr, strata):
    return sum(strata[h]["w"] * arr[strata[h]["idx"]].sum() for h in sorted(strata))


def ratio_estimate(num, den, strata):
    n = _weighted_totals(np.asarray(num, float), strata)
    d = _weighted_totals(np.asarray(den, float), strata)
    return (n / d) if d > 0 else None


def bootstrap_totals(arr, strata, reps):
    """(B,) weighted totals of arr under the shared replicates."""
    arr = np.asarray(arr, float)
    B = next(iter(reps.values())).shape[0]
    tot = np.zeros(B)
    for h in sorted(strata):
        vals = arr[strata[h]["idx"]]
        if len(vals):
            tot += strata[h]["w"] * vals[reps[h]].sum(axis=1)
    return tot


def bootstrap_ratio(num, den, strata, reps):
    """(B,) replicate ratios; NaN where the replicate denominator is 0."""
    n = bootstrap_totals(num, strata, reps)
    d = bootstrap_totals(den, strata, reps)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / d, np.nan)


def percentile_ci(vals, alpha=0.05):
    vals = np.asarray(vals, float)
    ok = vals[~np.isnan(vals)]
    n_degenerate = int(len(vals) - len(ok))
    if not len(ok):
        return None, None, n_degenerate
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), n_degenerate


def neff_wilson_ci(p_hat, boot_var, raw_den, alpha=0.05):
    """Wilson interval at the bootstrap-implied effective sample size."""
    if p_hat is None:
        return None, None, None
    if boot_var and boot_var > 0 and 0 < p_hat < 1:
        n_eff = p_hat * (1 - p_hat) / boot_var
    else:
        n_eff = float(raw_den)  # boundary/degenerate: fall back to the raw count
    if n_eff <= 0:
        return None, None, 0.0
    z2 = Z95 * Z95
    denom = 1 + z2 / n_eff
    center = (p_hat + z2 / (2 * n_eff)) / denom
    half = (Z95 * np.sqrt(p_hat * (1 - p_hat) / n_eff + z2 / (4 * n_eff * n_eff))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half)), float(n_eff)


def mean_of_card_rates(num, den, strata):
    """Sensitivity estimator: stratified mean of per-card rates (den>0 cards)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    tot = cnt = 0.0
    for h in sorted(strata):
        idx = strata[h]["idx"]
        m = den[idx] > 0
        tot += strata[h]["w"] * (num[idx][m] / den[idx][m]).sum()
        cnt += strata[h]["w"] * m.sum()
    return (tot / cnt) if cnt > 0 else None


def make_metric(name, num, den, strata, reps, kind="proportion", is_rate=True):
    """Full metric object for the analysis JSON. kind: proportion | field_ratio."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    value = ratio_estimate(num, den, strata)
    boots = bootstrap_ratio(num, den, strata, reps)
    lo, hi, n_deg = percentile_ci(boots)
    boot_var = float(np.nanvar(boots)) if np.isfinite(np.nanvar(boots)) else None
    raw_num, raw_den = float(num.sum()), float(den.sum())
    by_stratum = {h: {"num": float(num[strata[h]["idx"]].sum()),
                      "den": float(den[strata[h]["idx"]].sum())}
                  for h in sorted(strata)}
    wl = wh = n_eff = None
    if is_rate and value is not None:
        wl, wh, n_eff = neff_wilson_ci(value, boot_var, raw_den)
    eligible = [h for h in by_stratum if by_stratum[h]["den"] > 0]
    bootstrap_primary = (raw_num >= 10
                         and all(by_stratum[h]["num"] >= 1 for h in eligible))
    out = {
        "name": name,
        "value": value,
        "estimator": "stratified-proportion" if kind == "proportion" else "combined-ratio",
        "provenance": "S-weighted",
        "ci_method": ("stratified-cluster-bootstrap-percentile" if bootstrap_primary
                      else "wilson-neff") if is_rate else "stratified-cluster-bootstrap-percentile",
        "ci95_bootstrap": [lo, hi],
        "ci95_wilson_neff": [wl, wh] if is_rate else None,
        "n_eff": n_eff,
        "counts": {"num": raw_num, "den": raw_den, "by_stratum": by_stratum},
        "n_bootstrap_degenerate": n_deg,
    }
    out["ci95"] = out["ci95_bootstrap"] if out["ci_method"].startswith("stratified") \
        else out["ci95_wilson_neff"]
    if kind == "field_ratio":
        out["sensitivity_mean_of_card_rates"] = mean_of_card_rates(num, den, strata)
    return out


def derived_metric(name, fn, boot_args, point_args, is_rate=False):
    """Metric derived elementwise from replicate vectors (F1, catch ratio, ...).

    fn(*vectors) -> vector; boot_args = replicate vectors, point_args = scalars."""
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = fn(*boot_args)
    try:
        value = fn(*[np.array([a], float) for a in point_args])
        value = float(value[0]) if np.isfinite(value[0]) else None
    except Exception:
        value = None
    lo, hi, n_deg = percentile_ci(boots)
    return {"name": name, "value": value, "provenance": "S-weighted",
            "estimator": "derived-from-replicates",
            "ci_method": "stratified-cluster-bootstrap-percentile",
            "ci95": [lo, hi], "n_bootstrap_degenerate": n_deg}


def restrict(arr, mask):
    """Zero out non-members so domain estimates flow through the same machinery."""
    return np.asarray(arr, float) * np.asarray(mask, float)
