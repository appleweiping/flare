"""Candidate M5: class-conditional (Mondrian) APS and RAPS scores in place of LAC.

Romano, Sesia & Candes (2020), "Classification with valid and adaptive coverage" (APS);
Angelopoulos, Bates, Malik & Jordan (2021), "Uncertainty sets for image classifiers using
conformal prediction" (RAPS). Registration: docs/uq_preregistration.md, "Candidates", M5
("class-conditional APS / RAPS scores in place of LAC (efficiency and empty sets)").

Scores, for a row with probabilities P (6 classes) and a candidate class c:
    order:  classes sorted by decreasing P, ties broken by increasing class index
            (stable argsort of -P), rank(c) in 1..6
    APS:    s(x, c) = sum of P_j over classes with rank(j) <= rank(c)
            (= sum of P_j over classes with P_j > P_c, plus P_c, ties by class index)
    RAPS:   s(x, c) + lam * max(0, rank(c) - k_reg)
    APS-r:  sum of P_j over rank(j) < rank(c), plus u * P_c, with u ~ U(0, 1)   (exploratory)
Mondrian thresholds, per class c:
    q_c = harness.lac_quantile( s(x_i, c) over calibration rows with y_i == c )
    c in set(x)  iff  s(x, c) <= q_c

Design choices, all fixed before any M5 variant was scored:
- Variants and names. M5_aps = deterministic APS (lam = 0); M5_raps001 = RAPS lam = 0.01,
  k_reg = 1; M5_raps01 = RAPS lam = 0.1, k_reg = 1 (the two registered RAPS settings);
  M5_aps_rand = randomised APS (exploratory, not in the registration).
- Registered candidate designated before scoring: M5_aps (the plain registered score, no
  penalty parameter). M5_raps001 and M5_raps01 are registered alternates (three registered score
  variants -> multiplicity caveat on any single one passing); M5_aps_rand is exploratory.
- Ranks start at 1; k_reg = 1 means the top class is never penalised.
- Ties in P are broken by class index (the lower index ranks first), in calibration and at test
  time alike. (There are no exact ties in the h100 pattern-0 out-of-fold probabilities.)
- Randomisation (APS-r only): one u per row, shared by the six candidate classes of that row
  (the single-U construction of Romano et al.). Calibration rows draw u in the row order of the
  calibration View from np.random.default_rng([31415, 0]); test rows draw u in the row order of
  the View passed to sets() from np.random.default_rng([31415, 1]). The seed 31415 is fixed and
  deliberately different from the harness seeds (2026, 2026 + 7k) so u is independent of the
  scenario removal patterns. u never depends on labels.
- Quantile: harness.lac_quantile unweighted (alpha = harness.ALPHA = 0.10). M5 is registered on
  pattern-0 calibration only, so wants_augmented = 0 (all row weights are 1; asserted).
- No fallback rule is needed or used: every class has >= 46 calibration objects per fold, so q_c is
  finite (lac_quantile would return +inf for a class with fewer than 9 calibration objects, i.e.
  that class is then always included). No non-empty rule, no score smoothing, no tuning of lam,
  k_reg or alpha. Probabilities are used exactly as given (rows sum to 1 within 1e-15).
- fit() takes only the labelled calibration View; in the shift protocol the method therefore
  never sees deployment-side rows.
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    import harness
except ImportError:                       # running this file directly from methods/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import harness
import numpy as np
from harness import View, lac_quantile

RAND_SEED = 31415


def aps_scores(P, lam=0.0, k_reg=1, u=None):
    """(m, 6) conformity-violation scores s(x, c) for every row and every candidate class c.

    u: None for the deterministic score; else an (m,) array of U(0,1) draws, one per row,
    giving the randomised APS score (cumulative mass strictly before c, plus u * P_c)."""
    P = np.asarray(P, float)
    m, K = P.shape
    order = np.argsort(-P, axis=1, kind="stable")          # ties -> lower class index first
    Ps = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Ps, axis=1)                              # inclusive cumulative mass by rank
    if u is not None:
        u = np.asarray(u, float).reshape(m, 1)
        cum = (cum - Ps) + u * Ps                            # mass strictly before, plus u * own
    rank = np.broadcast_to(np.arange(1, K + 1), (m, K))
    if lam:
        cum = cum + lam * np.maximum(0, rank - k_reg)
    s = np.empty_like(cum)
    np.put_along_axis(s, order, cum, axis=1)                 # back to class-index columns
    return s


class M5:
    wants_augmented = 0

    def __init__(self, lam=0.0, k_reg=1, randomized=False, seed=RAND_SEED):
        self.lam, self.k_reg = float(lam), int(k_reg)
        self.randomized, self.seed = bool(randomized), int(seed)
        if self.randomized and self.lam:
            raise ValueError("only randomised APS (lam = 0) is part of this study")

    def _u(self, n, stream):
        if not self.randomized:
            return None
        return np.random.default_rng([self.seed, stream]).random(n)

    def fit(self, cal: View):
        y = cal.y
        assert (y >= 0).all(), "calibration labels required"
        assert np.allclose(cal.w, 1.0), "M5 is registered without augmented calibration views"
        S = aps_scores(cal.P, self.lam, self.k_reg, self._u(len(y), 0))
        self.n_cal = np.array([(y == c).sum() for c in range(6)])
        self.q = np.array([lac_quantile(S[y == c, c]) for c in range(6)])
        return self

    def sets(self, v: View):
        S = aps_scores(v.P, self.lam, self.k_reg, self._u(len(v.P), 1))
        return S <= self.q[None, :]


# ------------------------------------------------------------------ self-test on synthetic data
def _brute(P, lam=0.0, k_reg=1, u=None):
    m, K = P.shape
    s = np.zeros((m, K))
    for i in range(m):
        for c in range(K):
            before = [j for j in range(K) if P[i, j] > P[i, c] or (P[i, j] == P[i, c] and j < c)]
            r = len(before) + 1
            own = P[i, c] if u is None else u[i] * P[i, c]
            s[i, c] = sum(P[i, j] for j in before) + own + lam * max(0, r - k_reg)
    return s


def _synthetic(n, rng):
    y = rng.integers(0, 6, n)
    logits = rng.normal(0, 1, (n, 6)) * rng.uniform(0.5, 2.5, (n, 1))
    logits[np.arange(n), y] += rng.uniform(0, 3, n)
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    z = np.zeros(n)
    return View(P=P, L=P.argmax(1), E=z, nobs=z + 20, present=np.ones((n, 4), bool),
                pattern=np.zeros(n, int), y=y, obj=np.arange(n), w=np.ones(n))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. Hand example with a tie (classes 1 and 2): ranks 1, 2, 3, 4, 5, 6.
    P = np.array([[0.5, 0.2, 0.2, 0.05, 0.03, 0.02]])
    assert np.allclose(aps_scores(P), [[0.5, 0.7, 0.9, 0.95, 0.98, 1.0]])
    assert np.allclose(aps_scores(P, 0.1, 1), [[0.5, 0.8, 1.1, 1.25, 1.38, 1.5]])
    assert np.allclose(aps_scores(P, 0.01, 1), [[0.5, 0.71, 0.92, 0.98, 1.02, 1.05]])
    assert np.allclose(aps_scores(P, u=np.array([0.5])), [[0.25, 0.6, 0.8, 0.925, 0.965, 0.99]])
    Pt = np.array([[0.2, 0.3, 0.2, 0.3, 0.0, 0.0]])        # ties: 1 before 3, 0 before 2, 4 before 5
    assert np.allclose(aps_scores(Pt), [[0.8, 0.3, 1.0, 0.6, 1.0, 1.0]])
    assert np.allclose(aps_scores(Pt, 0.1, 1), [[1.0, 0.3, 1.3, 0.7, 1.4, 1.5]])

    # 2. Vectorised scores equal a brute-force loop, for every variant (including tied rows).
    Pr = rng.dirichlet(np.ones(6) * 0.7, 300); Pr[:20, 1] = Pr[:20, 4]
    Pr /= Pr.sum(1, keepdims=True); ur = rng.random(300)
    for lam in (0.0, 0.01, 0.1):
        assert np.allclose(aps_scores(Pr, lam, 1), _brute(Pr, lam, 1)), lam
    assert np.allclose(aps_scores(Pr, u=ur), _brute(Pr, u=ur))
    assert np.allclose(aps_scores(Pr, u=np.ones(300)), aps_scores(Pr))      # u = 1 -> deterministic
    assert np.allclose(aps_scores(Pr).max(1), 1.0)                          # last-ranked class: full mass
    assert np.allclose(aps_scores(Pr)[np.arange(300), Pr.argmax(1)], Pr.max(1))

    # 3. Thresholds are the finite-sample quantile of the class's own scores; labels unused in sets().
    cal, te = _synthetic(6000, rng), _synthetic(60000, rng)
    for kw in (dict(), dict(lam=0.01), dict(lam=0.1), dict(randomized=True)):
        m = M5(**kw).fit(cal)
        S = aps_scores(cal.P, m.lam, m.k_reg, m._u(len(cal.y), 0))
        for c in range(6):
            sc = np.sort(S[cal.y == c, c]); n = len(sc)
            assert m.q[c] == sc[int(np.ceil((n + 1) * 0.9)) - 1], (kw, c)
        s = m.sets(te.blind())
        assert s.shape == (len(te.y), 6) and s.dtype == bool and np.array_equal(s, m.sets(te)), kw
        # 4. class-conditional coverage at the nominal level on exchangeable data
        cov = s[np.arange(len(te.y)), te.y]
        for c in range(6):
            nc = m.n_cal[c]; tol = 4.5 * np.sqrt(0.09 / (nc + 1) + 0.09 / (te.y == c).sum())
            assert abs(cov[te.y == c].mean() - 0.90) < tol + 0.01, (kw, c, cov[te.y == c].mean())
        assert m.wants_augmented == 0

    # 5. Fixed seed: the randomised variant is reproducible across instances and calls.
    a, b = M5(randomized=True).fit(cal), M5(randomized=True).fit(cal)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.sets(te.blind()), b.sets(te.blind()))
    try:
        M5(lam=0.1, randomized=True); raise AssertionError("randomised RAPS should be refused")
    except ValueError:
        pass
    print("m5_aps self-test: all asserts passed")
