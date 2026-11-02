"""Behavior and exact finite-sample checks for the frozen early-stop policy study."""
import json

import numpy as np
import pytest
from scipy.stats import binom

from scripts.uq_anytime.policy import (
    binomial_upper_bound, certify_policies, evaluate_policy, policy_grid,
)


def example(probabilities=(0.6, 0.8, 0.95, 0.99), count=10):
    p = np.asarray(probabilities, dtype=float)
    return np.stack([p, 1 - p], axis=-1)[None, :, :], np.full((1, len(p)), count, dtype=float)


def policy(tau=0.8, support_lambda=0, earliest_day=5):
    return dict(tau=tau, support_lambda=support_lambda, earliest_day=earliest_day)


def test_first_eligible_look_and_no_future_revision():
    probs, nobs = example()
    out = evaluate_policy(probs, nobs, policy())
    assert out["selected"].tolist() == [True]
    assert out["stop_index"].tolist() == [1]
    assert out["stop_day"].tolist() == [10]
    assert out["prediction"].tolist() == [0]
    probs[:, 2:] = [0, 1]
    later = evaluate_policy(probs, nobs, policy())
    for key in out:
        np.testing.assert_equal(later[key], out[key])
    out = evaluate_policy(probs, nobs, policy(earliest_day=20))
    assert out["stop_index"].tolist() == [2]
    assert out["prediction"].tolist() == [1]


def test_hierarchy_label_probability_is_used():
    probs, nobs = example((0.9, 0.9, 0.9, 0.9))
    out = evaluate_policy(probs, nobs, policy(), labels=np.ones((1, 4), dtype=int))
    assert not out["selected"][0]
    assert out["prediction"][0] == out["stop_index"][0] == -1
    assert np.isnan(out["stop_day"][0])


def test_support_gate_zero_and_exact_threshold():
    probs, nobs = example((0.8, 0.8, 0.8, 0.8), count=0)
    assert evaluate_policy(probs, nobs, policy())["stop_index"][0] == 0
    assert not evaluate_policy(probs, nobs, policy(support_lambda=5))["selected"][0]
    nobs[:] = 5
    assert evaluate_policy(probs, nobs, policy(tau=0.4, support_lambda=5))["stop_index"][0] == 0
    # More information can make the same frozen policy first stop later.
    nobs[:] = [1, 5, 10, 15]
    assert evaluate_policy(probs, nobs, policy(tau=0.5, support_lambda=5))["stop_index"][0] == 2


def test_all_policies_share_correction_and_winner_tie_break():
    probs, nobs = example((0.99, 0.99, 0.99, 0.99), count=100)
    probs = np.repeat(probs, 200, axis=0)
    nobs = np.repeat(nobs, 200, axis=0)
    result = certify_policies(probs, nobs, np.zeros(200, dtype=int))
    assert len(policy_grid()) == result["multiplicity"] == len(result["records"]) == 45
    assert len({tuple(p.values()) for p in policy_grid()}) == 45
    assert result["winner_index"] == result["baseline_winner_index"] == 0
    assert all(r["per_policy_delta"] == 0.05 / 45 for r in result["records"])
    assert sum(r["support_lambda"] == 0 for r in result["records"]) == 15
    expected = 1 - (0.05 / 45) ** (1 / 200)
    assert result["records"][0]["risk_upper_bound"] == pytest.approx(expected)
    json.dumps(result, allow_nan=False)


def test_no_selections_and_no_certified_policy_abstain():
    probs = np.full((100, 4, 6), 1 / 6)
    nobs = np.zeros((100, 4))
    result = certify_policies(probs, nobs, np.zeros(100, dtype=int))
    assert result["winner_index"] is result["baseline_winner_index"] is None
    assert all(r["n_selected"] == 0 and r["risk_upper_bound"] is None
               and not r["certified"] for r in result["records"])
    json.dumps(result, allow_nan=False)
    # Zero observed errors alone is not a certificate with a small sample.
    probs, nobs = example((0.99, 0.99, 0.99, 0.99))
    result = certify_policies(probs, nobs, np.array([0]))
    assert result["records"][0]["n_errors"] == 0
    assert result["winner_index"] is None


def test_empty_calibration_and_all_errors():
    result = certify_policies(np.empty((0, 4, 2)), np.empty((0, 4)), np.array([], dtype=int))
    assert result["winner_index"] is None
    assert binomial_upper_bound(0, 0, 0.05) is None
    assert binomial_upper_bound(5, 5, 0.05) == 1


@pytest.mark.parametrize("n", [1, 5, 30, 135, 200])
@pytest.mark.parametrize("tail", [0.05, 0.05 / 45])
def test_exact_binomial_coverage_enumerated(n, tail):
    """Enumerate all error counts, not a Monte Carlo/implementation mirror.

    For every fixed n and population risk p, the exact binomial probability
    that the upper bound misses p must not exceed the declared tail level.
    This is the ingredient for the 45-policy union bound even if dependent.
    """
    counts = np.arange(n + 1)
    uppers = np.array([binomial_upper_bound(int(k), n, tail) for k in counts])
    for risk in [0.001, 0.01, 0.05, 0.051, 0.1, 0.3, 0.8, 0.99]:
        failure_probability = binom.pmf(counts, n, risk)[uppers < risk].sum()
        assert failure_probability <= tail + 1e-12
    # For any unsafe true risk, false certification is controlled as well.
    for unsafe_risk in [0.0501, 0.06, 0.1, 0.5]:
        false_certification = binom.pmf(counts, n, unsafe_risk)[uppers <= 0.05].sum()
        assert false_certification <= tail + 1e-12


@pytest.mark.parametrize("which", ["negative", "nan", "unnormalized", "shape", "nobs", "horizons", "labels"])
def test_invalid_trajectory_inputs(which):
    probs, nobs = example()
    kwargs = {}
    if which == "negative":
        probs[0, 0] = [-0.1, 1.1]
    elif which == "nan":
        probs[0, 0, 0] = np.nan
    elif which == "unnormalized":
        probs[0, 0] = [0.8, 0.8]
    elif which == "shape":
        probs = probs[0]
    elif which == "nobs":
        nobs[0, 0] = -1
    elif which == "horizons":
        kwargs["horizons"] = [5, 5, 20, 100]
    else:
        kwargs["labels"] = np.full((1, 4), 2)
    with pytest.raises(ValueError):
        evaluate_policy(probs, nobs, policy(), **kwargs)


@pytest.mark.parametrize("bad", [policy(tau=-0.1), policy(tau=np.nan), policy(support_lambda=-1),
                                  policy(earliest_day=0), {"tau": 0.5}])
def test_invalid_policy(bad):
    probs, nobs = example()
    with pytest.raises(ValueError):
        evaluate_policy(probs, nobs, bad)


def test_invalid_certification_labels_and_limits():
    probs, nobs = example()
    for y in [np.array([2]), np.array([0.5]), np.array([[0]]), np.array([0, 1])]:
        with pytest.raises(ValueError):
            certify_policies(probs, nobs, y)
    for kwargs in [{"delta": 0}, {"delta": 1}, {"risk_limit": np.nan}, {"risk_limit": -1}]:
        with pytest.raises(ValueError):
            certify_policies(probs, nobs, np.array([0]), **kwargs)
    for args in [(2, 1, 0.05), (-1, 1, 0.05), (0.5, 10, 0.05), (0, 1, 0), (0, 1, 1)]:
        with pytest.raises(ValueError):
            binomial_upper_bound(*args)
