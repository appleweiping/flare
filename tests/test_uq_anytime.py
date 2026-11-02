"""Finite enumeration and adversarial checks of research UQ guarantees."""
from itertools import combinations, permutations

import numpy as np
import pytest

from scripts.uq_anytime.calibration import (
    bonferroni_path,
    e_bh,
    e_budget_path,
    rank_evalues,
    rank_pvalues,
    trajectory_evalues,
    trajectory_pvalues,
)


def test_upper_tail_ties_and_empty_calibration():
    assert np.allclose(rank_pvalues([1, 1, 2], [0, 1, 1.5, 2, 3]), [1, 1, .5, .5, .25])
    assert np.array_equal(rank_pvalues([], [[0, 100], [-3, 7]]), np.ones((2, 2)))
    assert np.array_equal(rank_evalues([1, 1], 0), np.ones(2))
    assert np.array_equal(trajectory_pvalues(np.empty((0, 3)), [[1, 2]]), np.ones((1, 2)))
    assert np.array_equal(trajectory_evalues(np.empty((0, 3)), [[1, 2]]), np.ones((1, 2)))


@pytest.mark.parametrize("scores", [[0., 1., 2., 3.], [0., 0., 1., 1.], [2., 2., 2., 2.]])
def test_exhaustive_exchangeable_rank_validity(scores):
    # All assignments of an exchangeable finite population, including repeated
    # values, rather than a Monte Carlo check under one convenient distribution.
    ps, es = [], []
    for perm in permutations(scores):
        p = rank_pvalues(perm[:-1], perm[-1])
        ps.append(float(p))
        es.append(float(rank_evalues(p, len(scores) - 1)))
    ps = np.asarray(ps)
    for alpha in np.linspace(0, 1, 101):
        assert np.mean(ps <= alpha) <= alpha + 1e-14
    assert np.mean(es) <= 1 + 1e-14
    if len(set(scores)) == len(scores):
        assert np.mean(es) == pytest.approx(1)


@pytest.mark.parametrize("n", [0, 1, 3, 19, 999])
def test_rank_evalue_exact_discrete_normalization(n):
    ranks = np.arange(1, n + 2, dtype=float)
    e = rank_evalues(ranks / (n + 1), n)
    assert e.mean() == pytest.approx(1, abs=1e-14)
    assert np.all(np.diff(e) <= 0)
    assert e[0] == pytest.approx((n + 1) / np.sum(ranks**-2))


def test_trajectory_envelope_and_stopping_exhaustive():
    # Dependent looks and deliberately different score scales. The maximum
    # across looks is exchangeable at the object level; prefixes are not extra
    # calibration observations.
    population = np.array([[8., 0., 2.], [0., 3., 9.], [4., 6., 5.], [2., 7., 10.]])
    stopped_p, stopped_e, final_e = [], [], []
    for perm in permutations(range(len(population))):
        cal = population[list(perm[:-1])]
        query = population[[perm[-1]]]
        p = trajectory_pvalues(cal, query)[0]
        e = trajectory_evalues(cal, query)[0]
        assert np.all(np.diff(p) <= 0)
        assert np.all(np.diff(e) >= 0)
        for h in range(1, 4):
            assert np.array_equal(trajectory_pvalues(cal, query[:, :h]), p[None, :h])
            assert np.array_equal(trajectory_evalues(cal, query[:, :h]), e[None, :h])
        # Stop on evidence crossing or at the final look, whichever comes first.
        crossed = np.flatnonzero(e >= 1.5)
        t = int(crossed[0]) if len(crossed) else 2
        stopped_p.append(p[t]); stopped_e.append(e[t]); final_e.append(e[-1])
        assert e[t] <= e[-1]
    assert np.mean(final_e) == pytest.approx(1)
    assert np.mean(stopped_e) <= 1 + 1e-14
    for alpha in np.linspace(0, 1, 101):
        assert np.mean(np.asarray(stopped_p) <= alpha) <= alpha + 1e-14


def test_fixed_budgets_do_not_renormalize_with_observed_looks():
    e = np.array([[4., 0., 2., 6.], [0., 4., 0., 0.]])
    full = e_budget_path(e, total_looks=4)
    assert np.array_equal(full, [[1., 1., 1.5, 3.], [0., 1., 1., 1.]])
    p = np.array([[.1, .8, .02, .1], [.9, .4, .5, .7]])
    full_p = bonferroni_path(p, total_looks=4)
    assert np.allclose(full_p, [[.4, .4, .08, .08], [1., 1., 1., 1.]])
    for h in range(1, 5):
        assert np.array_equal(e_budget_path(e[:, :h], 4), full[:, :h])
        assert np.array_equal(bonferroni_path(p[:, :h], 4), full_p[:, :h])
    assert e_budget_path(e[:, :1], 4)[0, 0] != e[0, 0]


def test_adversarial_dependent_looks_stopping_bound():
    # Each state is equally likely. Every look has mean-one evidence, but the
    # looks are maximally dependent and taking an unbudgeted max has mean 4.
    e = np.eye(4) * 4
    path = e_budget_path(e, 4)
    stops = np.argmax(path >= .75, axis=1)
    stopped = path[np.arange(4), stops]
    assert stopped.mean() == pytest.approx(1)
    assert np.all(stopped <= path[:, -1])
    assert e.max(axis=1).mean() == 4
    # Marginal p-values each put mass 1/4 at 1/4, otherwise at one.
    p = np.where(e > 0, .25, 1.)
    assert np.all(bonferroni_path(p, 4) == 1)


def test_e_bh_under_arbitrary_dependence_exact_all_null_fdr():
    # A mixture over common shocks and mutually exclusive/subset shocks.
    # Null means are exactly one; dependence is neither independence nor an
    # assumption in e-BH. Enumerate every state to check FDR exactly.
    m, alpha = 4, .25
    mean = np.zeros(m); fdr = 0.0
    for k in [1, 2, 4]:
        subsets = list(combinations(range(m), k))
        state_prob = alpha / (3 * len(subsets))
        for subset in subsets:
            e = np.zeros(m)
            e[list(subset)] = m / (alpha * k)
            rejected = e_bh(e, alpha)
            assert rejected.sum() == k
            mean += state_prob * e
            fdr += state_prob * bool(rejected.any())  # all hypotheses are null
    assert not e_bh(np.zeros(m), alpha).any()  # remaining probability 1-alpha
    assert np.allclose(mean, 1)
    assert fdr == pytest.approx(alpha)


def test_e_bh_step_up_ties_and_family_size():
    assert np.array_equal(e_bh([0, 20, 20, 0], .1), [False, True, True, False])
    assert not e_bh([9, 9, 9], .1).any()
    assert e_bh([], .1).shape == (0,)
    # The same evidence can fail in a larger family: no prefiltering allowed.
    assert e_bh([10], .1)[0]
    assert not e_bh([10, 0, 0, 0], .1).any()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_data_rejected(bad):
    with pytest.raises(ValueError): rank_pvalues([bad], [1])
    with pytest.raises(ValueError): rank_pvalues([1], [bad])
    with pytest.raises(ValueError): rank_evalues([bad], 3)
    with pytest.raises(ValueError): trajectory_pvalues([[1, bad]], [[1]])
    with pytest.raises(ValueError): trajectory_pvalues([[1, 2]], [[bad]])
    with pytest.raises(ValueError): trajectory_evalues([[1, 2]], [[bad]])
    with pytest.raises(ValueError): bonferroni_path([[bad]], 2)
    with pytest.raises(ValueError): e_budget_path([[bad]], 2)
    with pytest.raises(ValueError): e_bh([bad], .1)


def test_invalid_shapes_grids_and_budgets_rejected():
    with pytest.raises(ValueError): rank_pvalues([[1, 2]], [1])
    with pytest.raises(ValueError): trajectory_pvalues([1, 2], [[1]])
    with pytest.raises(ValueError): trajectory_pvalues([[1, 2]], [1])
    with pytest.raises(ValueError): trajectory_pvalues(np.empty((2, 0)), [[1]])
    with pytest.raises(ValueError): trajectory_pvalues([[1]], [[1, 2]])
    with pytest.raises(ValueError): trajectory_pvalues([[1]], np.empty((2, 0)))
    for n in [-1, 1.5, True]:
        with pytest.raises(ValueError): rank_evalues([1], n)
    for p in [[0], [-.1], [1.1], [1 + 1e-10], [.3]]:
        with pytest.raises(ValueError): rank_evalues(p, 3)
    for total in [0, -1, 1, 2.5, True]:
        with pytest.raises(ValueError): e_budget_path([[1, 1]], total)
        with pytest.raises(ValueError): bonferroni_path([[.5, .5]], total)
    with pytest.raises(ValueError): e_budget_path([1, 1], 2)
    with pytest.raises(ValueError): e_budget_path([[-1, 1]], 2)
    with pytest.raises(ValueError): bonferroni_path([[0, 1.1]], 2)
    with pytest.raises(ValueError): bonferroni_path([[-.1, .5]], 2)
    with pytest.raises(ValueError): e_bh([[1, 2]], .1)
    with pytest.raises(ValueError): e_bh([-1], .1)
    for alpha in [0, 1, -1, np.nan, np.inf, True, [.1]]:
        with pytest.raises(ValueError): e_bh([1], alpha)
