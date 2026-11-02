"""Research calibration primitives for a fixed cohort and observation schedule.

These routines do not fit or tune scores. Rank guarantees require exchangeable
calibration/query objects scored with the same frozen, independently trained
procedure. A table of arbitrary model outputs is not sufficient evidence that
those assumptions hold. A previously inspected test population is not made
confirmatory by using these routines.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np


def _finite(values, name: str, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values; missing looks need an explicit policy")
    return out


def _count(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _looks(values, name: str) -> np.ndarray:
    out = _finite(values, name, ndim=2)
    if out.shape[1] == 0:
        raise ValueError(f"{name} must have at least one look")
    return out


def _budget(total_looks, observed_looks: int) -> int:
    total = _count(total_looks, "total_looks", minimum=1)
    if total < observed_looks:
        raise ValueError("observed looks exceed the prespecified total_looks")
    return total


def rank_pvalues(calibration_scores, query_scores) -> np.ndarray:
    """Conservative upper-tail ranks: (1 + # calibration >= query)/(n + 1).

    Calibration scores must be one-dimensional; queries may have any shape.
    Ties are counted against rejection. An empty calibration set returns ones.
    The finite-sample superuniform statement is marginal over an exchangeable
    calibration/new-object sample, not conditional on a realized calibration
    table and not valid after arbitrary data-dependent score selection.
    """
    cal = _finite(calibration_scores, "calibration_scores", ndim=1)
    query = _finite(query_scores, "query_scores")
    ge = len(cal) - np.searchsorted(np.sort(cal), query, side="left")
    return np.asarray((1.0 + ge) / (len(cal) + 1), dtype=float)


@lru_cache(maxsize=128)
def _inverse_square_normalizer(rank_count: int) -> float:
    ranks = np.arange(1, rank_count + 1, dtype=float)
    return float(np.sum(ranks ** -2))


def rank_evalues(p_values, n_calibration: int) -> np.ndarray:
    """Convert valid rank-grid p-values into mean-at-most-one e-values.

    For N = n_calibration + 1 and p = r/N, return
    N / (sum_{j=1}^N j**-2) / r**2. The mean over uniform discrete ranks
    1,...,N is exactly one. Since this function decreases in the rank, ties or
    other superuniformity on the same grid make it conservative. Continuous
    p-values or p-values on a different grid are rejected, not rounded into a
    purported guarantee. The maximal evidence is N / sum_{j=1}^N j**-2.
    """
    n = _count(n_calibration, "n_calibration")
    p = _finite(p_values, "p_values")
    N = n + 1
    scaled = p * N
    ranks = np.rint(scaled)
    if np.any((p < 1.0 / N) | (p > 1)) or np.any((ranks < 1) | (ranks > N)) or not np.all(
        np.isclose(scaled, ranks, rtol=1e-12, atol=1e-8)
    ):
        raise ValueError("p_values must lie on the rank grid {1/(n+1), ..., 1}")
    return (N / _inverse_square_normalizer(N)) / ranks**2


def trajectory_pvalues(calibration_trajectories, query_trajectories) -> np.ndarray:
    """Rank query running maxima against calibration full-trajectory maxima.

    Inputs have shape (objects, looks), with larger scores more anomalous.
    Calibration contains the entire fixed schedule; query contains a nonempty
    prefix of that same schedule. Horizon-specific score functions may differ
    but must be frozen and used identically on calibration and query objects.
    Object trajectories, including eligibility/context policies, must be
    exchangeable. No NaNs are silently skipped.

    The returned p-path is nonincreasing and bounded below by the valid rank
    p-value of the query's full-trajectory maximum. Consequently its rejection
    probability at any stopping rule is <= alpha. This is a full-trajectory
    envelope argument, not a claim that per-look p-values form a martingale.
    Empty calibration returns ones. Objects are never pooled across looks.
    """
    cal = _looks(calibration_trajectories, "calibration_trajectories")
    query = _looks(query_trajectories, "query_trajectories")
    if query.shape[1] > cal.shape[1]:
        raise ValueError("query has more looks than the fixed calibration schedule")
    return rank_pvalues(np.max(cal, axis=1), np.maximum.accumulate(query, axis=1))


def trajectory_evalues(calibration_trajectories, query_trajectories) -> np.ndarray:
    """Discrete-rank e-values of the full-trajectory envelope p-path.

    Each prefix value is bounded by the final full-trajectory e-value, whose
    expectation is <=1 under the assumptions of :func:`trajectory_pvalues`.
    Thus the evidence is valid at arbitrary stopping without assuming that
    successive increments are conditionally valid e-factors. It is not a
    product process; no conditional-mean or supermartingale claim is made.
    """
    cal = _looks(calibration_trajectories, "calibration_trajectories")
    return rank_evalues(trajectory_pvalues(cal, query_trajectories), len(cal))


def bonferroni_path(per_look_pvalues, total_looks: int) -> np.ndarray:
    """Any-look p-bound for a prespecified number of marginally valid looks.

    Shape is (objects, observed prefix looks). All p-values must be in [0,1].
    `total_looks` is mandatory and must remain fixed when more looks arrive;
    replacing it by the number observed so far invalidates the union bound.
    Arbitrary dependence among the fixed looks is allowed.
    """
    p = _looks(per_look_pvalues, "per_look_pvalues")
    total = _budget(total_looks, p.shape[1])
    if np.any((p < 0) | (p > 1)):
        raise ValueError("per_look_pvalues must lie in [0, 1]")
    return np.minimum(1.0, np.minimum.accumulate(p, axis=1) * total)


def e_budget_path(per_look_evalues, total_looks: int) -> np.ndarray:
    """Cumulative evidence with fixed uniform weights 1/total_looks.

    Shape is (objects, observed prefix looks). If each nonnegative per-look
    e-value has unconditional null mean <=1, the path is bounded by the full
    schedule's weighted sum, which has null mean <=1, irrespective of look
    dependence. This also bounds evidence at arbitrary stopping. It does not
    assert conditional mean validity or a supermartingale property. Unobserved
    looks keep their budget; weights are never renormalized after stopping.
    """
    e = _looks(per_look_evalues, "per_look_evalues")
    total = _budget(total_looks, e.shape[1])
    if np.any(e < 0):
        raise ValueError("per_look_evalues must be nonnegative")
    # Scaling first also avoids overflow in a large sum of finite e-values.
    return np.cumsum(e / total, axis=1)


def e_bh(e_values, alpha: float) -> np.ndarray:
    """e-BH rejection mask for one fixed family of hypotheses at FDR alpha.

    Reject the largest k sorted values satisfying e_(k) >= m/(alpha*k).
    Every null input must be a genuine mean-at-most-one nonnegative e-value;
    arbitrary dependence is permitted. The caller must fix the hypothesis
    family before reading its evidence. This function does not justify
    evidence-dependent prefiltering, repeatedly accumulating discoveries from
    new families, or a claim about false alarms per week. Empty input yields
    an empty mask. NaN/inf and negative e-values are rejected explicitly.
    """
    e = _finite(e_values, "e_values", ndim=1)
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha must be a scalar strictly between 0 and 1")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if np.any(e < 0):
        raise ValueError("e_values must be nonnegative")
    if not len(e):
        return np.zeros(0, dtype=bool)
    ordered = np.sort(e)[::-1]
    eligible = ordered >= len(e) / (alpha * np.arange(1, len(e) + 1))
    if not eligible.any():
        return np.zeros(len(e), dtype=bool)
    k = np.flatnonzero(eligible)[-1] + 1
    return e >= ordered[k - 1]
