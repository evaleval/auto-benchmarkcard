#!/usr/bin/env python3
"""Run the frozen exploratory source-complexity analysis for S150.

The exposure table must have been created outcome-free by
build_source_complexity_exposures.py. This script joins the frozen outcomes,
fits design-weighted descriptive logistic models, and writes machine-readable
and human-readable reports. It makes no causal or defect-prevalence claim.

Paper correspondence: this produces the exploratory documentary-source-count
analysis summarized in the main paper's Discussion and documented under
Supplement Section K (Reproducibility).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2, norm


RELEASE = Path(__file__).resolve().parents[1]
DEFAULT_DIR = RELEASE / "eval" / "s150" / "source_complexity"
DEFAULT_CORPUS_CARDS = RELEASE / "eval" / "corpus" / "cards"
EXPECTED_EXPOSURE_SHA256 = "b251490c1db9ee5898dea1fc9175f4e347dd588c1f5243a1505f3fb8ccc46722"
STRATUM_POPULATION = {"flagged": 152, "unflagged": 378}


class AnalysisError(RuntimeError):
    """Raised when a frozen input or model invariant fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisError(f"missing input: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot load JSON {path}: {exc}") from exc


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise AnalysisError(f"cannot load CSV {path}: {exc}") from exc


def expit(value: np.ndarray | float) -> np.ndarray | float:
    value = np.asarray(value)
    out = np.empty_like(value, dtype=float)
    pos = value >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-value[pos]))
    neg_value = np.exp(value[~pos])
    out[~pos] = neg_value / (1.0 + neg_value)
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = q * sorted_weights.sum()
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def kish_weight_only_effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective n from weight dispersion only, not a design-effect estimate."""
    return float(weights.sum() ** 2 / np.square(weights).sum())


def taylor_score_variance(
    unweighted_scores: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    """Variance of a Horvitz-Thompson score total under stratified SRSWOR."""
    if unweighted_scores.ndim == 1:
        unweighted_scores = unweighted_scores[:, None]
    width = unweighted_scores.shape[1]
    variance = np.zeros((width, width), dtype=float)
    for stratum, population in STRATUM_POPULATION.items():
        mask = strata == stratum
        scores = unweighted_scores[mask]
        sample_n = scores.shape[0]
        if sample_n < 2:
            raise AnalysisError(f"too few observations in stratum {stratum}")
        sample_cov = np.cov(scores, rowvar=False, ddof=1)
        if width == 1:
            sample_cov = np.array([[float(sample_cov)]])
        fpc = 1.0 - sample_n / population
        variance += (population**2) * fpc * sample_cov / sample_n
    return variance


@dataclass
class LogisticFit:
    beta: np.ndarray
    covariance: np.ndarray
    converged: bool
    iterations: int
    fitted: np.ndarray
    design_matrix: np.ndarray
    outcome: np.ndarray
    weights: np.ndarray
    strata: np.ndarray


def fit_survey_logistic(
    design_matrix: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> LogisticFit:
    x = np.asarray(design_matrix, dtype=float)
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or len(y) != len(x) or len(w) != len(x):
        raise AnalysisError("incompatible logistic model arrays")
    if set(np.unique(y)) - {0.0, 1.0}:
        raise AnalysisError("logistic outcome is not binary")
    if len(np.unique(y)) < 2:
        raise AnalysisError("logistic outcome has no variation")

    beta = np.zeros(x.shape[1], dtype=float)
    weighted_mean = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
    beta[0] = math.log(weighted_mean / (1 - weighted_mean))
    converged = False
    information = None

    for iteration in range(1, max_iter + 1):
        probability = np.asarray(expit(x @ beta), dtype=float)
        variance = np.maximum(probability * (1 - probability), 1e-10)
        score = x.T @ (w * (y - probability))
        information = x.T @ ((w * variance)[:, None] * x)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ score
        beta += step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
        if np.max(np.abs(beta)) > 50:
            break

    probability = np.asarray(expit(x @ beta), dtype=float)
    variance = np.maximum(probability * (1 - probability), 1e-10)
    information = x.T @ ((w * variance)[:, None] * x)
    bread = np.linalg.pinv(information)
    score_rows = x * (y - probability)[:, None]
    score_variance = taylor_score_variance(score_rows, strata)
    covariance = bread @ score_variance @ bread.T

    return LogisticFit(
        beta=beta,
        covariance=covariance,
        converged=converged,
        iterations=iteration,
        fitted=probability,
        design_matrix=x,
        outcome=y,
        weights=w,
        strata=strata,
    )


def coefficient_summary(fit: LogisticFit, index: int) -> dict[str, Any]:
    estimate = float(fit.beta[index])
    standard_error = float(math.sqrt(max(fit.covariance[index, index], 0.0)))
    z = estimate / standard_error if standard_error > 0 else float("nan")
    p = 2 * norm.sf(abs(z)) if math.isfinite(z) else float("nan")
    low = estimate - 1.96 * standard_error
    high = estimate + 1.96 * standard_error
    return {
        "log_odds": estimate,
        "standard_error": standard_error,
        "z": z,
        "p_value": float(p),
        "odds_ratio": math.exp(estimate),
        "odds_ratio_ci95": [math.exp(low), math.exp(high)],
    }


def standardized_probability_difference(
    fit: LogisticFit,
    predictor_index: int,
    low: float,
    high: float,
) -> dict[str, Any]:
    x_low = fit.design_matrix.copy()
    x_high = fit.design_matrix.copy()
    x_low[:, predictor_index] = low
    x_high[:, predictor_index] = high
    p_low = np.asarray(expit(x_low @ fit.beta), dtype=float)
    p_high = np.asarray(expit(x_high @ fit.beta), dtype=float)
    weight_norm = fit.weights / fit.weights.sum()
    low_mean = float(np.sum(weight_norm * p_low))
    high_mean = float(np.sum(weight_norm * p_high))
    difference = high_mean - low_mean
    gradient = np.sum(
        weight_norm[:, None]
        * (
            (p_high * (1 - p_high))[:, None] * x_high
            - (p_low * (1 - p_low))[:, None] * x_low
        ),
        axis=0,
    )
    variance = float(gradient @ fit.covariance @ gradient)
    standard_error = math.sqrt(max(variance, 0.0))
    return {
        "probability_low": low_mean,
        "probability_high": high_mean,
        "risk_difference": difference,
        "risk_difference_ci95": [
            difference - 1.96 * standard_error,
            difference + 1.96 * standard_error,
        ],
        "standard_error": standard_error,
    }


def design_weighted_domain_mean(
    outcome: np.ndarray,
    domain: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
) -> dict[str, Any]:
    domain = np.asarray(domain, dtype=float)
    denominator = float(np.sum(weights * domain))
    if denominator <= 0:
        return {"n": 0, "estimate": None, "ci95": [None, None]}
    estimate = float(np.sum(weights * domain * outcome) / denominator)
    linearized = domain * (outcome - estimate)
    variance_total = taylor_score_variance(linearized, strata)[0, 0]
    standard_error = math.sqrt(max(float(variance_total), 0.0)) / denominator
    return {
        "n": int(domain.sum()),
        "weighted_denominator": denominator,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci95": [
            max(0.0, estimate - 1.96 * standard_error),
            min(1.0, estimate + 1.96 * standard_error),
        ],
    }


def domain_mean_contrasts(
    outcome: np.ndarray,
    domains: dict[str, np.ndarray],
    weights: np.ndarray,
    strata: np.ndarray,
) -> dict[str, Any]:
    labels = list(domains)
    estimates: dict[str, float] = {}
    linearized_columns = []
    for label in labels:
        domain = np.asarray(domains[label], dtype=float)
        denominator = float(np.sum(weights * domain))
        if denominator <= 0:
            raise AnalysisError(f"empty domain for {label}")
        estimate = float(np.sum(weights * domain * outcome) / denominator)
        estimates[label] = estimate
        linearized_columns.append(domain * (outcome - estimate) / denominator)
    covariance = taylor_score_variance(
        np.column_stack(linearized_columns), strata
    )
    requested = [("0", "1"), ("1", "2"), ("2", "3_or_4")]
    result: dict[str, Any] = {}
    for low_label, high_label in requested:
        low_index = labels.index(low_label)
        high_index = labels.index(high_label)
        difference = estimates[high_label] - estimates[low_label]
        variance = (
            covariance[high_index, high_index]
            + covariance[low_index, low_index]
            - 2 * covariance[high_index, low_index]
        )
        standard_error = math.sqrt(max(float(variance), 0.0))
        z = difference / standard_error if standard_error > 0 else float("nan")
        result[f"{high_label}_minus_{low_label}"] = {
            "risk_difference": difference,
            "standard_error": standard_error,
            "ci95": [
                difference - 1.96 * standard_error,
                difference + 1.96 * standard_error,
            ],
            "p_value": float(2 * norm.sf(abs(z))) if math.isfinite(z) else None,
        }
    return result


def exponential_coefficient_summary(
    beta: np.ndarray,
    covariance: np.ndarray,
    index: int,
    ratio_name: str,
) -> dict[str, Any]:
    estimate = float(beta[index])
    standard_error = float(math.sqrt(max(covariance[index, index], 0.0)))
    z = estimate / standard_error if standard_error > 0 else float("nan")
    low = estimate - 1.96 * standard_error
    high = estimate + 1.96 * standard_error
    return {
        "log_ratio": estimate,
        "standard_error": standard_error,
        "z": z,
        "p_value": float(2 * norm.sf(abs(z))) if math.isfinite(z) else None,
        ratio_name: math.exp(estimate),
        f"{ratio_name}_ci95": [math.exp(low), math.exp(high)],
    }


def fit_survey_poisson(
    design_matrix: np.ndarray,
    count: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    offset: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(design_matrix, dtype=float)
    y = np.asarray(count, dtype=float)
    w = np.asarray(weights, dtype=float)
    offset = np.asarray(offset, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    rate = max(float(np.sum(w * y) / np.sum(w * np.exp(offset))), 1e-8)
    beta[0] = math.log(rate)
    converged = False
    for iteration in range(1, 101):
        mean = np.exp(np.clip(x @ beta + offset, -30, 30))
        score = x.T @ (w * (y - mean))
        information = x.T @ ((w * mean)[:, None] * x)
        step = np.linalg.pinv(information) @ score
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    mean = np.exp(np.clip(x @ beta + offset, -30, 30))
    information = x.T @ ((w * mean)[:, None] * x)
    bread = np.linalg.pinv(information)
    score_variance = taylor_score_variance(x * (y - mean)[:, None], strata)
    covariance = bread @ score_variance @ bread.T
    return {
        "beta": beta,
        "covariance": covariance,
        "converged": converged,
        "iterations": iteration,
    }


def fit_survey_grouped_binomial(
    design_matrix: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(design_matrix, dtype=float)
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.any(successes > trials) or np.any(trials <= 0):
        raise AnalysisError("invalid grouped-binomial counts")
    proportion = np.clip(
        float(np.sum(w * successes) / np.sum(w * trials)), 1e-8, 1 - 1e-8
    )
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = math.log(proportion / (1 - proportion))
    converged = False
    for iteration in range(1, 101):
        probability = np.asarray(expit(x @ beta), dtype=float)
        score = x.T @ (w * (successes - trials * probability))
        information = x.T @ (
            (w * trials * probability * (1 - probability))[:, None] * x
        )
        step = np.linalg.pinv(information) @ score
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    probability = np.asarray(expit(x @ beta), dtype=float)
    information = x.T @ (
        (w * trials * probability * (1 - probability))[:, None] * x
    )
    bread = np.linalg.pinv(information)
    score_variance = taylor_score_variance(
        x * (successes - trials * probability)[:, None], strata
    )
    covariance = bread @ score_variance @ bread.T
    return {
        "beta": beta,
        "covariance": covariance,
        "converged": converged,
        "iterations": iteration,
    }


def is_not_specified(value: Any) -> bool:
    empty = {
        "not specified",
        "not specified.",
        "no information found",
        "no information found.",
    }
    if value is None or value == "" or value == [] or value == {}:
        return True
    if isinstance(value, str):
        return value.strip().lower() in empty
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], str)
    ):
        return value[0].strip().lower() in empty
    return False


def count_complete_card_filled_fields(card_path: Path) -> tuple[int, int]:
    value = load_json(card_path)
    card = value.get("benchmark_card", value)
    sections = (
        "benchmark_details",
        "purpose_and_intended_users",
        "data",
        "methodology",
        "ethical_and_legal_considerations",
    )
    total = 0
    filled = 0
    for section in sections:
        fields = card.get(section)
        if not isinstance(fields, dict):
            raise AnalysisError(f"missing card section {section}: {card_path}")
        for field_value in fields.values():
            total += 1
            filled += int(not is_not_specified(field_value))
    if total not in {39, 40}:
        raise AnalysisError(f"unexpected complete-card field count {total}: {card_path}")
    return filled, total


def unweighted_logistic(
    predictor: np.ndarray,
    outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([np.ones(len(predictor)), predictor])
    weights = np.ones(len(predictor))
    strata = np.array(["flagged"] * len(predictor))
    beta = np.zeros(2)
    mean = np.clip(outcome.mean(), 1e-6, 1 - 1e-6)
    beta[0] = math.log(mean / (1 - mean))
    for _ in range(100):
        p = np.asarray(expit(x @ beta), dtype=float)
        v = np.maximum(p * (1 - p), 1e-10)
        score = x.T @ (outcome - p)
        information = x.T @ (v[:, None] * x)
        step = np.linalg.pinv(information) @ score
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.asarray(expit(x @ beta), dtype=float)
    information = x.T @ ((p * (1 - p))[:, None] * x)
    covariance = np.linalg.pinv(information)
    return beta, covariance


def model_result(
    predictor: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    low_raw: float,
    high_raw: float,
    lineage: np.ndarray | None = None,
    predictor_label: str = "log2(1 + documentary_channel_count)",
    contrast_label: str = "contrast_raw_channel_counts",
) -> dict[str, Any]:
    transformed = np.log2(1 + predictor)
    columns = [np.ones(len(predictor)), transformed]
    if lineage is not None:
        columns.append((lineage == "regen").astype(float))
    x = np.column_stack(columns)
    fit = fit_survey_logistic(x, outcome, weights, strata)
    effect = standardized_probability_difference(
        fit,
        predictor_index=1,
        low=math.log2(1 + low_raw),
        high=math.log2(1 + high_raw),
    )
    result = {
        "n": len(outcome),
        "events": int(outcome.sum()),
        "converged": fit.converged,
        "iterations": fit.iterations,
        "predictor": predictor_label,
        "coefficient": coefficient_summary(fit, 1),
        "standardized_probability_contrast": effect,
        "_fit": fit,
    }
    result[contrast_label] = [low_raw, high_raw]
    return result


def clean_model_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "_fit"}


def leave_one_out_range(
    predictor: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    low_raw: float,
    high_raw: float,
) -> dict[str, Any]:
    estimates = []
    failures = 0
    for index in range(len(outcome)):
        keep = np.arange(len(outcome)) != index
        try:
            result = model_result(
                predictor[keep],
                outcome[keep],
                weights[keep],
                strata[keep],
                low_raw,
                high_raw,
            )
            estimates.append(
                result["standardized_probability_contrast"]["risk_difference"]
            )
        except (AnalysisError, FloatingPointError, ValueError):
            failures += 1
    return {
        "successful_fits": len(estimates),
        "failed_fits": failures,
        "risk_difference_min": float(min(estimates)) if estimates else None,
        "risk_difference_max": float(max(estimates)) if estimates else None,
    }


def percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exposures",
        type=Path,
        default=DEFAULT_DIR / "exposures_outcome_free.csv",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=RELEASE / "eval" / "s150" / "sample.json",
    )
    parser.add_argument(
        "--judge",
        type=Path,
        default=RELEASE / "eval" / "s150" / "judge" / "analysis_frame.json",
    )
    parser.add_argument(
        "--verifier",
        type=Path,
        default=RELEASE / "eval" / "s150" / "screen" / "verifier_ratings.csv",
    )
    parser.add_argument(
        "--corpus-cards",
        type=Path,
        default=DEFAULT_CORPUS_CARDS,
        help="Directory containing the 530 distributed card JSON files.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    exposure_hash = sha256(args.exposures)
    if exposure_hash != EXPECTED_EXPOSURE_SHA256:
        raise AnalysisError(
            "outcome-free exposure table hash drifted: "
            f"expected {EXPECTED_EXPOSURE_SHA256}, got {exposure_hash}"
        )

    exposures = load_csv(args.exposures)
    sample = load_json(args.sample)
    judge = load_json(args.judge)
    verifier = load_csv(args.verifier)
    if len(exposures) != 150 or len({row["name"] for row in exposures}) != 150:
        raise AnalysisError("exposure table must have 150 unique cards")
    sample_names = {row["name"] for row in sample["cards"]}
    if sample_names != {row["name"] for row in exposures}:
        raise AnalysisError("exposure and sample card sets differ")
    if sample_names != set(judge["per_card"]):
        raise AnalysisError("judge and sample card sets differ")
    sample_by_name = {row["name"]: row for row in sample["cards"]}

    material_by_card: dict[str, int] = {}
    wrong_splice_by_card: dict[str, int] = {}
    for row in verifier:
        if row["verifier_label"] != "confirmed-material":
            continue
        card = row["card"]
        material_by_card[card] = material_by_card.get(card, 0) + 1
        if row["category"] == "wrong-section-splice":
            wrong_splice_by_card[card] = wrong_splice_by_card.get(card, 0) + 1

    joined: list[dict[str, Any]] = []
    for exposure in exposures:
        name = exposure["name"]
        aggregate = judge["per_card"][name]["aggregate"]
        card_path = (args.corpus_cards / f"{name}.json").resolve()
        try:
            card_path.relative_to(args.corpus_cards.resolve())
        except ValueError as exc:
            raise AnalysisError(f"card path escapes --corpus-cards: {name}") from exc
        complete_filled, complete_total = count_complete_card_filled_fields(card_path)
        row = dict(exposure)
        row.update(
            {
                "complete_card_filled_fields": complete_filled,
                "complete_card_present_fields": complete_total,
                "judged23_filled_fields": aggregate["n_filled"],
                "judged23_not_specified_fields": aggregate["n_ns"],
                "confirmed_material_count": material_by_card.get(name, 0),
                "has_confirmed_material": int(material_by_card.get(name, 0) > 0),
                "confirmed_wrong_section_splice_count": wrong_splice_by_card.get(name, 0),
                "has_confirmed_wrong_section_splice": int(
                    wrong_splice_by_card.get(name, 0) > 0
                ),
                "primary_source_fillable_abstention_count": aggregate[
                    "ns_missed_primary"
                ],
                "has_primary_source_fillable_abstention": int(
                    aggregate["ns_missed_primary"] > 0
                ),
                "unsupported_filled_count": aggregate["unsupported"],
                "has_unsupported_filled": int(aggregate["unsupported"] > 0),
                "partial_or_unsupported_filled_count": (
                    aggregate["partial"] + aggregate["unsupported"]
                ),
                "has_partial_or_unsupported_filled": int(
                    aggregate["partial"] + aggregate["unsupported"] > 0
                ),
            }
        )
        joined.append(row)

    joined.sort(key=lambda row: row["name"])
    joined_path = args.out_dir / "analysis_joined_card_table.csv"
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    with joined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    predictor = np.array(
        [float(row["documentary_channel_count"]) for row in joined], dtype=float
    )
    weights = np.array([float(row["weight"]) for row in joined], dtype=float)
    strata = np.array([row["stratum"] for row in joined])
    lineage = np.array([row["generation_lineage"] for row in joined])
    low_raw = weighted_quantile(predictor, weights, 0.25)
    high_raw = weighted_quantile(predictor, weights, 0.75)

    outcome_fields = {
        "confirmed_material": "has_confirmed_material",
        "primary_source_fillable_abstention": (
            "has_primary_source_fillable_abstention"
        ),
        "unsupported_filled": "has_unsupported_filled",
        "partial_or_unsupported_filled": "has_partial_or_unsupported_filled",
        "confirmed_wrong_section_splice": "has_confirmed_wrong_section_splice",
    }

    models: dict[str, Any] = {}
    for label, field in outcome_fields.items():
        outcome = np.array([int(row[field]) for row in joined], dtype=float)
        result = model_result(
            predictor, outcome, weights, strata, low_raw, high_raw
        )
        models[label] = clean_model_result(result)

    primary_outcome = np.array(
        [int(row["has_confirmed_material"]) for row in joined], dtype=float
    )
    primary_adjusted = model_result(
        predictor,
        primary_outcome,
        weights,
        strata,
        low_raw,
        high_raw,
        lineage=lineage,
    )

    domain_estimates: dict[str, Any] = {}
    domain_arrays: dict[str, np.ndarray] = {}
    domain_definitions: list[tuple[str, Callable[[float], bool]]] = [
        ("0", lambda value: value == 0),
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3_or_4", lambda value: value >= 3),
    ]
    for label, predicate in domain_definitions:
        domain = np.array([predicate(value) for value in predictor], dtype=float)
        domain_arrays[label] = domain
        domain_estimates[label] = design_weighted_domain_mean(
            primary_outcome, domain, weights, strata
        )
    domain_contrasts = domain_mean_contrasts(
        primary_outcome, domain_arrays, weights, strata
    )

    complete_filled = np.array(
        [float(row["complete_card_filled_fields"]) for row in joined]
    )
    judged_filled = np.array(
        [float(row["judged23_filled_fields"]) for row in joined]
    )
    opportunity_design = np.column_stack(
        [
            np.ones(len(predictor)),
            np.log2(1 + predictor),
            np.log2(1 + complete_filled),
        ]
    )
    opportunity_fit = fit_survey_logistic(
        opportunity_design, primary_outcome, weights, strata
    )
    opportunity_adjusted = {
        "note": (
            "post-hoc opportunity diagnostic; complete-card filled-field count "
            "is an output/mediator, not a baseline confounder"
        ),
        "n": len(primary_outcome),
        "events": int(primary_outcome.sum()),
        "source_count_coefficient": coefficient_summary(opportunity_fit, 1),
        "card_fullness_coefficient": coefficient_summary(opportunity_fit, 2),
        "source_count_standardized_probability_contrast": (
            standardized_probability_difference(
                opportunity_fit,
                predictor_index=1,
                low=math.log2(1 + low_raw),
                high=math.log2(1 + high_raw),
            )
        ),
    }

    categorical_design = np.column_stack(
        [
            np.ones(len(predictor)),
            (predictor == 0).astype(float),
            (predictor == 2).astype(float),
            (predictor >= 3).astype(float),
        ]
    )
    categorical_fit = fit_survey_logistic(
        categorical_design, primary_outcome, weights, strata
    )
    categorical_beta = categorical_fit.beta[1:4]
    categorical_covariance = categorical_fit.covariance[1:4, 1:4]
    categorical_wald = float(
        categorical_beta @ np.linalg.pinv(categorical_covariance) @ categorical_beta
    )
    categorical_summary = {
        "reference_group": "1 documentary source channel",
        "zero_vs_one": coefficient_summary(categorical_fit, 1),
        "two_vs_one": coefficient_summary(categorical_fit, 2),
        "three_or_four_vs_one": coefficient_summary(categorical_fit, 3),
        "omnibus_wald_chi_square": categorical_wald,
        "omnibus_df": 3,
        "omnibus_p_value": float(chi2.sf(categorical_wald, 3)),
    }

    categorical_opportunity_design = np.column_stack(
        [categorical_design, np.log2(1 + complete_filled)]
    )
    categorical_opportunity_fit = fit_survey_logistic(
        categorical_opportunity_design, primary_outcome, weights, strata
    )
    categorical_opportunity_summary = {
        "note": "post-hoc opportunity diagnostic",
        "reference_group": "1 documentary source channel",
        "zero_vs_one": coefficient_summary(categorical_opportunity_fit, 1),
        "two_vs_one": coefficient_summary(categorical_opportunity_fit, 2),
        "three_or_four_vs_one": coefficient_summary(
            categorical_opportunity_fit, 3
        ),
        "card_fullness_coefficient": coefficient_summary(
            categorical_opportunity_fit, 4
        ),
    }

    trend_design = np.column_stack(
        [np.ones(len(predictor)), np.log2(1 + predictor)]
    )
    confirmed_count = np.array(
        [float(row["confirmed_material_count"]) for row in joined]
    )
    wrong_splice_count = np.array(
        [float(row["confirmed_wrong_section_splice_count"]) for row in joined]
    )
    finding_rate_fit = fit_survey_poisson(
        trend_design,
        confirmed_count,
        weights,
        strata,
        offset=np.log(complete_filled),
    )
    wrong_splice_rate_fit = fit_survey_poisson(
        trend_design,
        wrong_splice_count,
        weights,
        strata,
        offset=np.log(complete_filled),
    )
    opportunity_normalized_counts = {
        "note": (
            "post-hoc screen-opportunity diagnostic; the denominator is filled "
            "card fields, but the screen was not an exhaustive field-level audit"
        ),
        "confirmed_material_findings_per_filled_field": {
            "events": int(confirmed_count.sum()),
            "source_count": exponential_coefficient_summary(
                finding_rate_fit["beta"],
                finding_rate_fit["covariance"],
                1,
                "incidence_rate_ratio",
            ),
        },
        "confirmed_wrong_section_splices_per_filled_field": {
            "events": int(wrong_splice_count.sum()),
            "source_count": exponential_coefficient_summary(
                wrong_splice_rate_fit["beta"],
                wrong_splice_rate_fit["covariance"],
                1,
                "incidence_rate_ratio",
            ),
        },
    }

    unsupported_count = np.array(
        [float(row["unsupported_filled_count"]) for row in joined]
    )
    partial_or_unsupported_count = np.array(
        [float(row["partial_or_unsupported_filled_count"]) for row in joined]
    )
    unsupported_rate_fit = fit_survey_grouped_binomial(
        trend_design,
        unsupported_count,
        judged_filled,
        weights,
        strata,
    )
    partial_rate_fit = fit_survey_grouped_binomial(
        trend_design,
        partial_or_unsupported_count,
        judged_filled,
        weights,
        strata,
    )
    judged_field_rate_diagnostics = {
        "note": (
            "field-opportunity analysis over the frozen 23-field source-judge "
            "frame; partial-support labels were less stable under human assessment"
        ),
        "unsupported_per_filled_field": {
            "events": int(unsupported_count.sum()),
            "trials": int(judged_filled.sum()),
            "source_count": exponential_coefficient_summary(
                unsupported_rate_fit["beta"],
                unsupported_rate_fit["covariance"],
                1,
                "odds_ratio",
            ),
        },
        "partial_or_unsupported_per_filled_field": {
            "events": int(partial_or_unsupported_count.sum()),
            "trials": int(judged_filled.sum()),
            "source_count": exponential_coefficient_summary(
                partial_rate_fit["beta"],
                partial_rate_fit["covariance"],
                1,
                "odds_ratio",
            ),
        },
    }

    unweighted_beta, unweighted_covariance = unweighted_logistic(
        np.log2(1 + predictor), primary_outcome
    )
    unweighted_se = math.sqrt(max(unweighted_covariance[1, 1], 0.0))

    within_stratum: dict[str, Any] = {}
    for stratum in STRATUM_POPULATION:
        mask = strata == stratum
        beta, covariance = unweighted_logistic(
            np.log2(1 + predictor[mask]), primary_outcome[mask]
        )
        se = math.sqrt(max(covariance[1, 1], 0.0))
        within_stratum[stratum] = {
            "n": int(mask.sum()),
            "events": int(primary_outcome[mask].sum()),
            "log_odds": float(beta[1]),
            "standard_error_model_based": float(se),
            "odds_ratio": float(math.exp(beta[1])),
            "odds_ratio_ci95_model_based": [
                float(math.exp(beta[1] - 1.96 * se)),
                float(math.exp(beta[1] + 1.96 * se)),
            ],
        }

    length_proxy = np.array(
        [float(row["total_stage_a_documentary_chars"]) for row in joined]
    )
    length_low = weighted_quantile(length_proxy, weights, 0.25)
    length_high = weighted_quantile(length_proxy, weights, 0.75)
    length_result = model_result(
        length_proxy,
        primary_outcome,
        weights,
        strata,
        length_low,
        length_high,
        predictor_label="log2(1 + budgeted_stage_a_documentary_chars)",
        contrast_label="contrast_raw_budgeted_stage_a_documentary_chars",
    )
    length_result["predictor_note"] = (
        "budgeted-character proxy only; exact long-paper passages were not persisted"
    )

    count_values = np.array(
        [float(row["confirmed_material_count"]) for row in joined]
    )
    weighted_finding_count_by_domain: dict[str, Any] = {}
    for label, predicate in domain_definitions:
        domain = np.array([predicate(value) for value in predictor], dtype=float)
        denominator = np.sum(weights * domain)
        weighted_finding_count_by_domain[label] = {
            "n": int(domain.sum()),
            "weighted_mean_confirmed_findings": (
                float(np.sum(weights * domain * count_values) / denominator)
                if denominator
                else None
            ),
        }

    result_payload = {
        "analysis_version": "source-complexity-v2-2026-07-28",
        "scope": (
            "exploratory association with observed screen-detected and "
            "verifier-confirmed findings in the finite 530-card collection"
        ),
        "claim_guards": [
            "No causal effect of source count is estimated.",
            "The primary outcome is not overall defect prevalence.",
            "A screen-negative card was not manually verified error-free.",
            "Documentary source channels are source types, not necessarily unique documents.",
            "Budgeted character measures are proxies, not exact Composer token counts.",
            (
                "Complete-card fullness is a post-generation opportunity measure; "
                "fullness-adjusted results answer a different estimand and are diagnostic."
            ),
            (
                "The paper-cap exclusion sensitivity is disabled because deleting "
                "domain-external sampled cards before variance estimation does not "
                "preserve the original stratified sample design."
            ),
        ],
        "input_hashes": {
            "exposures_outcome_free_csv": exposure_hash,
            "sample_json": sha256(args.sample),
            "judge_analysis_frame_json": sha256(args.judge),
            "verifier_ratings_csv": sha256(args.verifier),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
        "sample": {
            "n": len(joined),
            "strata": {
                stratum: int(np.sum(strata == stratum))
                for stratum in STRATUM_POPULATION
            },
            "kish_weight_only_effective_n": (
                kish_weight_only_effective_sample_size(weights)
            ),
            "primary_events": int(primary_outcome.sum()),
            "weighted_exposure_quartiles_raw": [low_raw, high_raw],
        },
        "primary_model": models["confirmed_material"],
        "primary_model_lineage_adjusted_sensitivity": clean_model_result(
            primary_adjusted
        ),
        "primary_domain_estimates": domain_estimates,
        "primary_domain_contrasts": domain_contrasts,
        "primary_categorical_model": categorical_summary,
        "primary_opportunity_adjusted_diagnostic": opportunity_adjusted,
        "primary_categorical_opportunity_adjusted_diagnostic": (
            categorical_opportunity_summary
        ),
        "opportunity_normalized_count_diagnostics": opportunity_normalized_counts,
        "judged_field_rate_diagnostics": judged_field_rate_diagnostics,
        "secondary_models": {
            key: value
            for key, value in models.items()
            if key != "confirmed_material"
        },
        "sensitivities": {
            "unweighted": {
                "log_odds": float(unweighted_beta[1]),
                "standard_error_model_based": float(unweighted_se),
                "odds_ratio": float(math.exp(unweighted_beta[1])),
                "odds_ratio_ci95_model_based": [
                    float(math.exp(unweighted_beta[1] - 1.96 * unweighted_se)),
                    float(math.exp(unweighted_beta[1] + 1.96 * unweighted_se)),
                ],
            },
            "within_stratum": within_stratum,
            "leave_one_out": leave_one_out_range(
                predictor,
                primary_outcome,
                weights,
                strata,
                low_raw,
                high_raw,
            ),
            "exclude_paper_budget_reached": {
                "status": "disabled_after_statistical_audit",
                "reason": (
                    "The previous implementation subset sampled cards before Taylor "
                    "variance estimation, causing stratum sample sizes and finite-"
                    "population corrections to describe the subset rather than the "
                    "original sampling design. No estimate from that implementation "
                    "is reported."
                ),
            },
            "budgeted_character_proxy": clean_model_result(length_result),
        },
        "weighted_confirmed_finding_count_by_source_count": (
            weighted_finding_count_by_domain
        ),
        "joined_table": joined_path.name,
        "joined_table_sha256": sha256(joined_path),
    }
    result_path = args.out_dir / "analysis_results.json"
    result_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    primary = result_payload["primary_model"]
    contrast = primary["standardized_probability_contrast"]
    report_lines = [
        "# Exploratory source-complexity analysis",
        "",
        "This analysis was specified before the outcome join. It is descriptive, "
        "non-causal, and limited to the finite 530-card collection represented by "
        "the stratified 150-card sample.",
        "",
        "## Primary result",
        "",
        f"- Sample: {primary['n']} cards; {primary['events']} cards with at least "
        "one screen-detected, verifier-confirmed material finding.",
        f"- Kish weight-only effective sample size: "
        f"{result_payload['sample']['kish_weight_only_effective_n']:.1f}.",
        f"- Weighted exposure quartiles: {low_raw:.0f} versus {high_raw:.0f} "
        "documentary source channels.",
        f"- Standardized weighted probabilities: {percent(contrast['probability_low'])} "
        f"versus {percent(contrast['probability_high'])}.",
        f"- Risk difference: {percent(contrast['risk_difference'])} "
        f"(95% CI {percent(contrast['risk_difference_ci95'][0])} to "
        f"{percent(contrast['risk_difference_ci95'][1])}).",
        f"- Odds ratio per unit increase in log2(1 + channel count): "
        f"{primary['coefficient']['odds_ratio']:.2f} "
        f"(95% CI {primary['coefficient']['odds_ratio_ci95'][0]:.2f} to "
        f"{primary['coefficient']['odds_ratio_ci95'][1]:.2f}; "
        f"p={primary['coefficient']['p_value']:.3f}).",
        "",
        "## Weighted descriptive probabilities by available documentary channels",
        "",
    ]
    for label, estimate in domain_estimates.items():
        report_lines.append(
            f"- {label}: n={estimate['n']}, {percent(estimate['estimate'])} "
            f"(95% CI {percent(estimate['ci95'][0])} to "
            f"{percent(estimate['ci95'][1])})."
        )
    one_vs_two = domain_contrasts["2_minus_1"]
    fullness_contrast = opportunity_adjusted[
        "source_count_standardized_probability_contrast"
    ]
    unsupported_rate = judged_field_rate_diagnostics[
        "unsupported_per_filled_field"
    ]["source_count"]
    splice_rate = opportunity_normalized_counts[
        "confirmed_wrong_section_splices_per_filled_field"
    ]["source_count"]
    report_lines.extend(
        [
            "",
            "## Robustness of the source-volume interpretation",
            "",
            f"- The observed one-versus-two-channel risk difference was "
            f"{percent(one_vs_two['risk_difference'])} "
            f"(95% CI {percent(one_vs_two['ci95'][0])} to "
            f"{percent(one_vs_two['ci95'][1])}).",
            f"- After conditioning on complete-card fullness as a post-hoc "
            f"opportunity diagnostic, the modeled one-to-two-channel risk "
            f"difference was {percent(fullness_contrast['risk_difference'])} "
            f"(95% CI {percent(fullness_contrast['risk_difference_ci95'][0])} "
            f"to {percent(fullness_contrast['risk_difference_ci95'][1])}).",
            f"- In the exact 23-field judge frame, the source-count odds ratio "
            f"for unsupported content per filled field was "
            f"{unsupported_rate['odds_ratio']:.2f} "
            f"(95% CI {unsupported_rate['odds_ratio_ci95'][0]:.2f} to "
            f"{unsupported_rate['odds_ratio_ci95'][1]:.2f}).",
            f"- For confirmed wrong-section-splice findings per filled complete-card "
            f"field, the opportunity-normalized incidence-rate ratio was "
            f"{splice_rate['incidence_rate_ratio']:.2f} "
            f"(95% CI {splice_rate['incidence_rate_ratio_ci95'][0]:.2f} to "
            f"{splice_rate['incidence_rate_ratio_ci95'][1]:.2f}).",
            "- The planned paper-cap exclusion sensitivity is disabled after "
            "statistical audit because subsetting cards before Taylor variance "
            "estimation did not preserve the original sampling design.",
            "",
            "## Interpretation guard",
            "",
            "The positive prespecified log-linear trend is not a stable monotonic "
            "one-source-to-two-source contrast. Much of the endpoint pattern also "
            "coincides with card fullness, which is itself produced by the pipeline. "
            "Accordingly, the analysis does not establish that adding sources makes "
            "generation harder. The outcome records detected-and-confirmed findings, "
            "not true defect prevalence. Source count may also affect the screen's "
            "opportunity or ability to detect discrepancies. The count denotes "
            "available source types, not necessarily unique documents. The length "
            "analysis is only a budgeted-character sensitivity because nearly all "
            "full papers reached the extraction cap and exact retrieved passages were "
            "not persisted.",
            "",
        ]
    )
    report_path = args.out_dir / "ANALYSIS_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "results": str(result_path),
                "report": str(report_path),
                "joined_table": str(joined_path),
                "primary_events": int(primary_outcome.sum()),
                "contrast": contrast,
                "odds_ratio": primary["coefficient"]["odds_ratio"],
                "p_value": primary["coefficient"]["p_value"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
