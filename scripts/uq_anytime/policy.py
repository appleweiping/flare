"""Finite-family early classification with simultaneous selective-risk bounds.

The support gate is a ranking/decision heuristic, not a calibrated probability.
Each policy sees one complete object trajectory during calibration and makes at
most one prediction. For a family fixed independently of the calibration data,
Clopper--Pearson bounds and a union bound certify the error probability conditional
on selection. This requires iid calibration and future object trajectories from
the same population, with the predictor and policy family frozen beforehand.
It is not a distribution-shift, per-class, online-FDR, or novelty guarantee.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import beta


DEFAULT_HORIZONS = (5.0, 10.0, 20.0, 100.0)
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
SUPPORT_REGULARIZERS = (0.0, 5.0, 15.0)
EARLIEST_DAYS = (5.0, 10.0, 20.0)


def policy_grid() -> list[dict[str, float]]:
    """Return the prespecified 45 policies in deterministic tie-break order."""
    return [
        {"tau": tau, "support_lambda": regularizer, "earliest_day": day}
        for tau in THRESHOLDS
        for regularizer in SUPPORT_REGULARIZERS
        for day in EARLIEST_DAYS
    ]


def _probability(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in (0, 1)") from exc
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    return value


def _validate_inputs(probs, nobs, horizons, labels):
    try:
        probs = np.asarray(probs, dtype=float)
        nobs = np.asarray(nobs, dtype=float)
        horizons = np.asarray(horizons, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probs, nobs, and horizons must be numeric arrays") from exc
    if probs.ndim != 3 or probs.shape[1] == 0 or probs.shape[2] < 2:
        raise ValueError("probs must have shape (objects, looks, classes), with looks > 0 and classes >= 2")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must be finite probabilities in [0, 1]")
    if not np.allclose(probs.sum(axis=2), 1, rtol=0, atol=1e-7):
        raise ValueError("probs must sum to one over classes at every look")
    if nobs.shape != probs.shape[:2] or not np.isfinite(nobs).all() or (nobs < 0).any():
        raise ValueError("nobs must be finite, nonnegative, and have shape (objects, looks)")
    if (horizons.shape != (probs.shape[1],) or not np.isfinite(horizons).all()
            or (horizons <= 0).any() or (np.diff(horizons) <= 0).any()):
        raise ValueError("horizons must be finite, positive, strictly increasing, and match the number of looks")
    if labels is None:
        labels = probs.argmax(axis=2)
    else:
        labels = np.asarray(labels)
        if (labels.shape != probs.shape[:2] or not np.issubdtype(labels.dtype, np.integer)
                or (labels < 0).any() or (labels >= probs.shape[2]).any()):
            raise ValueError("labels must be integer class indices with shape (objects, looks)")
    return probs, nobs, horizons, labels


def _validate_policy(policy: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(policy, Mapping) or set(policy) != {"tau", "support_lambda", "earliest_day"}:
        raise ValueError("policy must contain exactly tau, support_lambda, and earliest_day")
    try:
        result = {key: float(value) for key, value in policy.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("policy values must be finite numbers") from exc
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("policy values must be finite numbers")
    if not 0 <= result["tau"] <= 1:
        raise ValueError("tau must lie in [0, 1]")
    if result["support_lambda"] < 0 or result["earliest_day"] <= 0:
        raise ValueError("support_lambda must be nonnegative and earliest_day must be positive")
    return result


def _evaluate_validated(probs, nobs, policy, horizons, labels):
    regularizer = policy["support_lambda"]
    # The lambda=0 baseline has gate one even when nobs=0; avoid 0/0.
    gate = np.ones_like(nobs) if regularizer == 0 else nobs / (nobs + regularizer)
    label_probability = np.take_along_axis(probs, labels[..., None], axis=2)[..., 0]
    eligible = (horizons[None, :] >= policy["earliest_day"])
    qualifies = eligible & (label_probability * gate >= policy["tau"])
    selected = qualifies.any(axis=1)
    first = qualifies.argmax(axis=1)
    stop_index = np.where(selected, first, -1)
    prediction = np.where(selected, labels[np.arange(len(probs)), first], -1)
    stop_day = np.where(selected, horizons[first], np.nan)
    return {"selected": selected, "prediction": prediction,
            "stop_index": stop_index, "stop_day": stop_day}


def evaluate_policy(
    probs, nobs, policy: Mapping[str, float], *,
    horizons: Sequence[float] = DEFAULT_HORIZONS, labels=None,
) -> dict[str, np.ndarray]:
    """Select the first eligible look, or abstain with class/index -1 and day NaN.

    If hierarchy labels are provided, the gate uses the probability of that
    label, which may differ from the maximum-probability class. Evaluation never
    reads true labels or future probabilities to revise an earlier decision.
    """
    validated = _validate_inputs(probs, nobs, horizons, labels)
    return _evaluate_validated(*validated[:2], _validate_policy(policy), *validated[2:])


def binomial_upper_bound(errors: int, selected: int, tail_probability: float) -> float | None:
    """One-sided exact Clopper--Pearson upper bound, None without observations."""
    if (not isinstance(errors, (int, np.integer)) or isinstance(errors, (bool, np.bool_))
            or not isinstance(selected, (int, np.integer)) or isinstance(selected, (bool, np.bool_))
            or selected < 0 or errors < 0 or errors > selected):
        raise ValueError("errors and selected must be integers with 0 <= errors <= selected")
    tail_probability = _probability(tail_probability, "tail_probability")
    if selected == 0:
        return None
    if errors == selected:
        return 1.0
    # isf avoids loss of precision from forming 1-tail_probability.
    return float(beta.isf(tail_probability, errors + 1, selected - errors))


def certify_policies(
    probs, nobs, y, *, horizons: Sequence[float] = DEFAULT_HORIZONS,
    labels=None, risk_limit: float = 0.05, delta: float = 0.05,
) -> dict:
    """Certify all 45 frozen policies on a separate labelled calibration sample.

    Both winners share the same simultaneous delta/45 confidence correction:
    `winner_index` ranges over all policies; `baseline_winner_index` ranges over
    lambda=0. A None winner means deploy abstain-all, not an uncertified fallback.
    Selection maximizes calibration selection fraction, then minimizes mean
    decision day, then grid index. Records are JSON serializable without NaN.
    The simultaneous risk certificate survives this winner selection; selection
    fraction and mean delay estimates themselves are not confidence guarantees.
    """
    probs, nobs, horizons, labels = _validate_inputs(probs, nobs, horizons, labels)
    y = np.asarray(y)
    if (y.shape != (len(probs),) or (len(y) and not np.issubdtype(y.dtype, np.integer))
            or (y < 0).any() or (y >= probs.shape[2]).any()):
        raise ValueError("y must contain one integer true class index per object")
    risk_limit = _probability(risk_limit, "risk_limit")
    delta = _probability(delta, "delta")
    policies = policy_grid()
    multiplicity = len(policies)
    per_policy_delta = delta / multiplicity
    records = []
    for index, policy in enumerate(policies):
        outcome = _evaluate_validated(probs, nobs, policy, horizons, labels)
        mask = outcome["selected"]
        count = int(mask.sum())
        errors = int((outcome["prediction"][mask] != y[mask]).sum())
        upper = binomial_upper_bound(errors, count, per_policy_delta)
        records.append({
            "index": index, **policy, "n_selected": count, "n_errors": errors,
            "selection_fraction": float(count / len(y)) if len(y) else 0.0,
            "empirical_risk": float(errors / count) if count else None,
            "risk_upper_bound": upper,
            "mean_stop_day": float(outcome["stop_day"][mask].mean()) if count else None,
            "certified": bool(upper is not None and upper <= risk_limit),
            "per_policy_delta": per_policy_delta,
        })

    def winner(baseline_only):
        candidates = [r for r in records if r["certified"]
                      and (not baseline_only or r["support_lambda"] == 0)]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (-r["selection_fraction"], r["mean_stop_day"], r["index"]))["index"]

    return {
        "records": records,
        "winner_index": winner(False),
        "baseline_winner_index": winner(True),
        "multiplicity": multiplicity,
        "risk_limit": risk_limit,
        "delta": delta,
        "n_calibration_objects": len(y),
        "guarantee": "simultaneous conditional-on-selection misclassification risk under iid object trajectories",
        "no_certified_policy_action": "abstain_all",
    }
