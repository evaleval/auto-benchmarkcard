"""Unit tests for the S-analysis statistics (scripts/s_stats.py + analyze_s join).

Known-answer fixtures for the stratified combined ratio estimator, bootstrap
sanity, degenerate handling, the weights-cancel and budget identities, kappa,
and the v31b replay regression that pins the weighted machinery to the
existing dev-set numbers where definitions coincide.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import s_stats  # noqa: E402
from score_calibration import cohens_kappa, normalize_human_label, read_ratings  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


def two_strata(labels, w_f, w_c):
    return s_stats.make_strata(labels, {"F": w_f, "C": w_c})


def test_ratio_estimator_hand_computed():
    # stratum F: w=4, cards (num, den) = (2,10), (0,10); stratum C: w=2, (1,20)
    labels = ["F", "F", "C"]
    strata = two_strata(labels, 4.0, 2.0)
    num, den = np.array([2, 0, 1.0]), np.array([10, 10, 20.0])
    # (4*2 + 4*0 + 2*1) / (4*10 + 4*10 + 2*20) = 10/120
    assert s_stats.ratio_estimate(num, den, strata) == pytest.approx(10 / 120)


def test_homogeneous_strata_exact():
    # any stratified sample from homogeneous strata reproduces the corpus rate:
    # F: N=200, 10 fields, 2 unsupported each; C: N=300, 20 fields, 1 each
    n_f, n_c = 5, 4
    labels = ["F"] * n_f + ["C"] * n_c
    strata = two_strata(labels, 200 / n_f, 300 / n_c)
    num = np.array([2.0] * n_f + [1.0] * n_c)
    den = np.array([10.0] * n_f + [20.0] * n_c)
    truth = (200 * 2 + 300 * 1) / (200 * 10 + 300 * 20)
    assert s_stats.ratio_estimate(num, den, strata) == pytest.approx(truth)


def test_consistency_monte_carlo():
    rng = np.random.default_rng(1234)
    N_f, N_c = 200, 300
    pop_num_f = rng.binomial(10, 0.2, N_f)
    pop_num_c = rng.binomial(20, 0.05, N_c)
    truth = (pop_num_f.sum() + pop_num_c.sum()) / (N_f * 10 + N_c * 20)
    n_f = n_c = 40
    ests, mocr = [], []
    for _ in range(300):
        i_f = rng.choice(N_f, n_f, replace=False)
        i_c = rng.choice(N_c, n_c, replace=False)
        labels = ["F"] * n_f + ["C"] * n_c
        strata = two_strata(labels, N_f / n_f, N_c / n_c)
        num = np.concatenate([pop_num_f[i_f], pop_num_c[i_c]]).astype(float)
        den = np.array([10.0] * n_f + [20.0] * n_c)
        ests.append(s_stats.ratio_estimate(num, den, strata))
        mocr.append(s_stats.mean_of_card_rates(num, den, strata))
    assert np.mean(ests) == pytest.approx(truth, abs=0.003)
    # mean-of-card-rates estimates a different estimand here (F cards have
    # 4x the rate but half the fields) - documents the Q1 primary choice
    assert abs(np.mean(mocr) - truth) > 0.01


def test_bootstrap_coverage_sanity():
    rng = np.random.default_rng(99)
    N_f, N_c = 150, 350
    pop_f = rng.binomial(15, 0.15, N_f)
    pop_c = rng.binomial(15, 0.05, N_c)
    truth = (pop_f.sum() + pop_c.sum()) / (15 * (N_f + N_c))
    n_h = 50
    covered = 0
    sims = 120
    for s in range(sims):
        i_f = rng.choice(N_f, n_h, replace=False)
        i_c = rng.choice(N_c, n_h, replace=False)
        labels = ["F"] * n_h + ["C"] * n_h
        strata = two_strata(labels, N_f / n_h, N_c / n_h)
        num = np.concatenate([pop_f[i_f], pop_c[i_c]]).astype(float)
        den = np.full(2 * n_h, 15.0)
        reps = s_stats.make_replicates(strata, B=400, seed=1000 + s)
        boots = s_stats.bootstrap_ratio(num, den, strata, reps)
        lo, hi, _ = s_stats.percentile_ci(boots)
        covered += (lo <= truth <= hi)
    assert covered / sims >= 0.85  # loose: percentile bootstrap, no FPC


def test_degenerate_zero_event_stratum():
    labels = ["F"] * 4 + ["C"] * 4
    strata = two_strata(labels, 2.0, 3.0)
    num = np.array([1, 0, 2, 0] + [0, 0, 0, 0], float)  # C has zero events
    den = np.full(8, 10.0)
    reps = s_stats.make_replicates(strata, B=300, seed=7)
    metric = s_stats.make_metric("t", num, den, strata, reps, kind="field_ratio")
    assert metric["ci_method"] == "wilson-neff"  # raw events 3 < 10
    lo, hi = metric["ci95"]
    assert 0.0 <= lo <= metric["value"] <= hi <= 1.0


def test_den_zero_replicates_are_nan_dropped():
    labels = ["F", "F"]
    strata = s_stats.make_strata(labels, {"F": 1.0})
    num, den = np.array([1.0, 0.0]), np.array([2.0, 0.0])  # one card has den 0
    reps = {"F": np.array([[1, 1], [0, 1], [0, 0]])}  # first replicate: only card 1
    boots = s_stats.bootstrap_ratio(num, den, strata, reps)
    assert np.isnan(boots[0]) and not np.isnan(boots[1]) and not np.isnan(boots[2])
    lo, hi, n_deg = s_stats.percentile_ci(boots)
    assert n_deg == 1


def test_weights_cancel_for_precision():
    # flags exist only in stratum F -> weighted precision equals raw F precision
    labels = ["F"] * 3 + ["C"] * 3
    strata = two_strata(labels, 5.0, 9.0)
    tp = np.array([1, 0, 2, 0, 0, 0], float)
    flagged = np.array([2, 1, 3, 0, 0, 0], float)
    weighted = s_stats.ratio_estimate(tp, flagged, strata)
    assert weighted == pytest.approx(3 / 6)


def test_budget_identity_catch_ratio():
    labels = ["F"] * 3 + ["C"] * 3
    strata = two_strata(labels, 2.0, 4.0)
    tp = np.array([1, 1, 0, 0, 0, 0], float)
    flagged = np.array([2, 1, 1, 0, 0, 0], float)
    pos = np.array([2, 1, 0, 1, 0, 1], float)
    filled = np.full(6, 20.0)
    precision = s_stats.ratio_estimate(tp, flagged, strata)
    unsup = s_stats.ratio_estimate(pos, filled, strata)
    w = lambda arr: sum(strata[h]["w"] * arr[strata[h]["idx"]].sum() for h in strata)
    catch_ratio = (w(tp) / w(flagged)) / (w(pos) / w(filled))
    assert catch_ratio == pytest.approx(precision / unsup)


def test_wilson_neff_boundaries():
    lo, hi, n_eff = s_stats.neff_wilson_ci(0.0, 0.0, 100)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi > 0.01 and n_eff == 100
    lo, hi, _ = s_stats.neff_wilson_ci(0.5, 0.0025, 100)  # var -> n_eff = 100
    assert 0.4 < lo < 0.5 < hi < 0.6


def test_cohens_kappa_fixtures():
    k, po = cohens_kappa([("a", "a"), ("b", "b")], ["a", "b"])
    assert k == pytest.approx(1.0) and po == 1.0
    # hand-computed 2x2: pairs with po=0.6, marginals a:(0.6,0.6) -> pe=0.52
    pairs = [("a", "a")] * 3 + [("a", "b")] * 3 + [("b", "a")] * 1 + [("b", "b")] * 3
    k, po = cohens_kappa(pairs, ["a", "b"])
    pe = 0.6 * 0.4 + 0.4 * 0.6
    assert po == pytest.approx(0.6)
    assert k == pytest.approx((0.6 - pe) / (1 - pe))


@pytest.mark.parametrize("packet_label, internal_label", [
    ("supported_registry_only", "supported_by_eee_only"),
    ("yes_outside_registry", "yes_primary"),
    ("yes_registry_only", "yes_eee_only"),
    ("supported", "supported"),
    (" partial ", "partial"),
    ("", ""),
])
def test_human_label_aliases(packet_label, internal_label):
    assert normalize_human_label(packet_label) == internal_label


def test_read_ratings_normalizes_packet_labels(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "item_id,human_label\n"
        "c001,supported_registry_only\n"
        "c002,yes_outside_registry\n"
        "c003,yes_registry_only\n"
        "c004,supported_by_eee_only\n"
        "c005,\n"
    )
    assert read_ratings(str(ratings)) == {
        "c001": "supported_by_eee_only",
        "c002": "yes_primary",
        "c003": "yes_eee_only",
        "c004": "supported_by_eee_only",
    }


def test_replay_v31b_regression():
    """The weighted machinery must reduce to the pooled dev-set numbers on the
    22 gold cards (weights 1) with fr2_nofill flags."""
    judge_path = os.path.join(REPO, "gold_set", "judge_v31b.json")
    report_path = os.path.join(REPO, "gold_set", "report_fr2_nofill_v31b.json")
    if not os.path.exists(judge_path) or not os.path.exists(report_path):
        pytest.skip("private development gold-set artifacts are not distributed")
    judge = json.load(open(judge_path))
    report = json.load(open(report_path))
    flags = {c["name"]: set(c.get("flagged_paths", [])) for c in report["cards"]}

    names = list(judge["per_card"])
    labels, filled, unsup, tp, flagged_lab, outside = [], [], [], [], [], 0
    for n in names:
        fvs = judge["per_card"][n]["field_verdicts"]
        lab = {f["path"]: f["status"] for f in fvs}
        fill = {p for p, s in lab.items()
                if s in ("supported", "partial", "unsupported", "supported_by_eee_only")}
        pos = {p for p, s in lab.items() if s == "unsupported"}
        fl = flags.get(n, set())
        labels.append("all")
        filled.append(len(fill))
        unsup.append(len(pos))
        flagged_lab.append(len(fl & fill))
        tp.append(len(fl & fill & pos))
        outside += len(fl - fill)
    strata = s_stats.make_strata(labels, {"all": 1.0})
    filled, unsup = np.array(filled, float), np.array(unsup, float)
    tp, flagged_lab = np.array(tp, float), np.array(flagged_lab, float)

    assert s_stats.ratio_estimate(unsup, filled, strata) == pytest.approx(9 / 320)
    assert s_stats.ratio_estimate(tp, flagged_lab, strata) == pytest.approx(0.5)
    pos_arr = unsup
    assert s_stats.ratio_estimate(tp, pos_arr, strata) == pytest.approx(5 / 9)
    assert outside == 7
    agg = report["aggregate"]["flag_quality"]["unsupported"]
    assert agg["tp"] == 5 and agg["fp"] == 5 and agg["fn"] == 4
    assert agg["flagged_unlabeled"] == 7
