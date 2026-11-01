"""Candidate M4: class-conditional weighted conformal prediction under covariate shift
(Tibshirani, Barber, Candes & Ramdas 2019), docs/uq_preregistration.md "Candidates" and Amendment 1.

For class c, calibration rows i with y_i == c, scores s_i = 1 - P_c(x_i), weights w_i = r(z_i), and a
test point x with weight r(z):

    q_c(x) = inf{ s : sum_{i: s_i <= s} w_i >= (1 - alpha) * (sum_i w_i + r(z)) }   (+inf if no such s)
    c in set(x)  iff  1 - P_c(x) <= q_c(x)

i.e. the (1 - alpha) quantile of sum_i p_i delta_{s_i} + p_x delta_{+inf}, p_i = w_i / (sum_j w_j + r(z)),
p_x = r(z) / (sum_j w_j + r(z)). Vectorised: per class the scores are sorted once and the weights
cumulated; for each test point the extra mass r(z) only moves the target (1 - alpha)(W_n + r(z)), so
the threshold index is one searchsorted per class.

Why the marginal ratio is used inside a class-conditional (Mondrian) construction: under covariate
shift (p(y | z) the same in both domains), p_dep(z | y=c) / p_cal(z | y=c) = r(z) * pi_cal(c) / pi_dep(c),
a per-class constant times r(z); the constant cancels in p_i and p_x, so the marginal ratio is the
correct class-conditional ratio (only the clipping below does not commute with that constant; in
lofo_shift the cal and deployment objects are the same objects, so pi_cal = pi_dep and the constant is 1).

Design choices, all fixed before any M4 variant was scored (written with this file, before any run on
the OOF arrays; the only real-data numbers seen beforehand were the B0/B1 baseline results and the
covariate summaries: shapes, class counts, median nobs / energy / max P, finiteness of Z):
- alpha = harness.ALPHA = 0.10; wants_augmented = 0 (registered: M4 is a shift-scenario candidate on
  pattern-0 rows; the calibration View must have unit row weights, which fit() asserts).
- No unlabeled View (every non-shift scenario): r == 1 for calibration rows and test points. The same
  engine then gives index ceil((n+1)(1-alpha)) - 1, i.e. exactly B0's lac_quantile (asserted in the
  self-test and verified on every non-shift scenario by m4_diagnostics.py: identical sets).
- Density ratio (registered variant, ratio="logistic"): features = View.Z (log1p nobs, energy, 4 context
  presence indicators, max P), standardised with a StandardScaler fitted on the stacked calibration +
  unlabeled rows; sklearn LogisticRegression with its defaults (L2, C=1.0, lbfgs) except max_iter=1000
  (convergence only); label 0 = calibration row, 1 = unlabeled deployment row. r(z) = odds(z) *
  n_cal / n_unl with odds = exp(decision_function); fitted once on all rows and evaluated in-sample on
  the calibration rows (7 coefficients on ~14,000 rows: no cross-fitting). The evaluated fold never
  enters the ratio (harness lofo_shift passes the other folds' objects on deployment photometry).
- Clipping: r clipped to [1/20, 20], for calibration rows and test points alike, after the n_cal/n_unl
  factor (which is 1 in lofo_shift, where n_cal = n_unl).
- Ties: scores sorted with a stable sort; the threshold is the score value at the first cumulative
  weight that reaches the target, so tied scores are all in or all out (membership uses <=).
- A class with no calibration rows (does not occur here) gets q = +inf.
- Exploratory variant (ratio="lgbm"), chosen a priori over a class-conditional ratio. Reasons:
  (i) under covariate shift the marginal ratio already is the class-conditional ratio up to a constant
  (above), so a class-conditional estimate only helps where p(y|z) differs between photometries, and the
  only label-free way to get p_dep(z|c) is to weight deployment rows by the classifier's own P_c, the
  very probabilities whose miscalibration under the shift is the problem; per-class domain classifiers
  for TDE (~46 calibration objects per fold) would also be noisy. (ii) The registered logistic ratio is
  log-linear in log nobs, energy and max P; random thinning to 30% with a floor of 3 detections per band
  compresses the nobs distribution and couples nobs with energy / max P non-linearly, so a log-linear
  ratio is plausibly misspecified, and a flexible classifier is the standard alternative (Tibshirani et
  al. use a random forest as well as logistic regression). LightGBM settings, fixed now and not tuned:
  LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100,
  reg_lambda=1.0, subsample=1.0, colsample_bytree=1.0, random_state=0, n_jobs=1, verbose=-1), same
  standardised Z, cross-fitted over 5 object folds (seed 0; an object's calibration row and its
  unlabeled row share a fold) so that calibration rows get out-of-fold log odds (trees would otherwise
  be overconfident in-sample); test points get the mean over the 5 fold models of log odds +
  log(n_cal_train / n_unl_train). Same clipping, same quantile engine. Labelled exploratory.
- Diagnostics kept on the instance (no effect on the sets): per class n, effective sample size
  ESS_c = (sum w_i)^2 / sum w_i^2 of the calibration weights, fractions clipped; per sets() call the
  test weights and the thresholds.
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
from harness import View, ALPHA

CLIP = 20.0
N_CROSS = 5
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, reg_lambda=1.0,
                   subsample=1.0, colsample_bytree=1.0, random_state=0, n_jobs=1, verbose=-1)


def weighted_thresholds(s_sorted, W_cum, w_test, alpha=ALPHA):
    """Per test point, inf{s : sum_{s_i <= s} w_i >= (1-alpha)(W_n + w_x)} over sorted scores s_sorted with
    cumulative weights W_cum; +inf when the point mass at +inf is needed to reach the level."""
    w_test = np.asarray(w_test, float)
    n = len(s_sorted)
    if n == 0:
        return np.full(len(w_test), np.inf)
    t = (1 - alpha) * (W_cum[-1] + w_test)
    k = np.searchsorted(W_cum, t, side="left")
    q = np.full(len(w_test), np.inf)
    ok = k < n
    q[ok] = s_sorted[k[ok]]
    return q


# ------------------------------------------------------------------ density-ratio estimators
class _LogisticRatio:
    def fit(self, Zc, Zu, obj_c=None, obj_u=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = np.vstack([Zc, Zu]); d = np.r_[np.zeros(len(Zc)), np.ones(len(Zu))]
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=1.0, max_iter=1000).fit(self.scaler.transform(X), d)
        self.log_prior = np.log(len(Zc) / len(Zu))
        self.log_r_cal = self.log_ratio(Zc)
        return self

    def log_ratio(self, Z):
        return self.clf.decision_function(self.scaler.transform(Z)) + self.log_prior


class _LGBMRatio:
    def fit(self, Zc, Zu, obj_c, obj_u):
        from lightgbm import LGBMClassifier
        from sklearn.preprocessing import StandardScaler
        X = np.vstack([Zc, Zu]); d = np.r_[np.zeros(len(Zc)), np.ones(len(Zu))]
        self.scaler = StandardScaler().fit(X); Xs = self.scaler.transform(X)
        objs = np.r_[obj_c, obj_u]
        uo = np.unique(objs)
        fold_of = dict(zip(uo, np.random.default_rng(0).permutation(len(uo)) % N_CROSS))
        f = np.array([fold_of[o] for o in objs])
        is_cal = d == 0
        self.models, self.log_priors = [], []
        self.log_r_cal = np.empty(len(Zc))
        for k in range(N_CROSS):
            tr = f != k
            m = LGBMClassifier(**LGBM_PARAMS).fit(Xs[tr], d[tr])
            lp = np.log((tr & is_cal).sum() / (tr & ~is_cal).sum())
            self.models.append(m); self.log_priors.append(lp)
            te_cal = (f == k) & is_cal
            self.log_r_cal[te_cal[:len(Zc)]] = m.predict(Xs[te_cal], raw_score=True) + lp
        return self

    def log_ratio(self, Z):
        Xs = self.scaler.transform(Z)
        return np.mean([m.predict(Xs, raw_score=True) + lp for m, lp in zip(self.models, self.log_priors)], axis=0)


RATIOS = dict(logistic=_LogisticRatio, lgbm=_LGBMRatio)


class M4:
    """Class-conditional weighted LAC conformal sets; reduces to B0 when fit() gets no unlabeled View."""
    wants_augmented = 0

    def __init__(self, ratio="logistic", clip=CLIP, alpha=ALPHA):
        if ratio not in RATIOS:
            raise ValueError(f"ratio must be one of {tuple(RATIOS)}, got {ratio!r}")
        self.ratio_name, self.clip, self.alpha = ratio, float(clip), float(alpha)

    def _clip(self, log_r):
        return np.clip(np.exp(log_r), 1.0 / self.clip, self.clip)

    def fit(self, cal: View, unlabeled: View | None = None):
        y = cal.y
        assert (y >= 0).all(), "calibration labels required"
        assert np.allclose(cal.w, 1.0), "M4 is registered without augmented calibration views"
        if unlabeled is None:
            self.ratio = None
            w = np.ones(len(y))
        else:
            self.ratio = RATIOS[self.ratio_name]().fit(cal.Z, unlabeled.Z, cal.obj, unlabeled.obj)
            w = self._clip(self.ratio.log_r_cal)
        self.w_cal = w
        S = 1.0 - cal.P
        self.s, self.W = [], []
        self.diag = dict(n=np.zeros(6, int), ess=np.zeros(6), sum_w=np.zeros(6),
                         clip_hi=np.zeros(6), clip_lo=np.zeros(6))
        for c in range(6):
            m = y == c
            o = np.argsort(S[m, c], kind="stable")
            self.s.append(S[m, c][o]); self.W.append(np.cumsum(w[m][o]))
            wc = w[m]
            self.diag["n"][c] = m.sum()
            self.diag["sum_w"][c] = wc.sum()
            self.diag["ess"][c] = wc.sum() ** 2 / (wc ** 2).sum() if m.any() else 0.0
            self.diag["clip_hi"][c] = np.mean(wc >= self.clip) if m.any() else 0.0
            self.diag["clip_lo"][c] = np.mean(wc <= 1.0 / self.clip) if m.any() else 0.0
        return self

    def test_weights(self, v: View):
        if self.ratio is None:
            return np.ones(len(v.P))
        return self._clip(self.ratio.log_ratio(v.Z))

    def thresholds(self, v: View):
        wx = self.test_weights(v)
        return np.column_stack([weighted_thresholds(self.s[c], self.W[c], wx, self.alpha) for c in range(6)]), wx

    def sets(self, v: View):
        q, wx = self.thresholds(v)
        self.last_q, self.last_wx = q, wx          # diagnostics only
        return (1.0 - v.P) <= q


# ------------------------------------------------------------------ self-test on synthetic data
def _synthetic(n, rng, mu_lognobs, sd=0.6):
    """Toy 6-class data whose scores sharpen with nobs; the domain differs only in the nobs distribution
    (covariate shift: given nobs, labels and logits have the same law), log1p(nobs) ~ N(mu, sd)."""
    y = rng.integers(0, 6, n)
    ln = rng.normal(mu_lognobs, sd, n)
    nobs = np.expm1(ln)
    present = rng.random((n, 4)) < 0.7
    sharp = -1.0 + 1.0 * ln
    logits = rng.normal(0, 1, (n, 6)); logits[np.arange(n), y] += sharp
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    from scipy.special import logsumexp
    return View(P=P, L=P.argmax(1), E=-logsumexp(logits[:, :5], axis=1), nobs=nobs, present=present,
                pattern=np.zeros(n, int), y=y, obj=np.arange(n), w=np.ones(n))


def _brute_threshold(s, w, wx, alpha=ALPHA):
    """Direct definition: quantile of sum_i p_i delta_{s_i} + p_x delta_inf, p normalised."""
    vals = np.r_[s, np.inf]; p = np.r_[w, wx] / (w.sum() + wx)
    o = np.argsort(vals, kind="stable"); c = np.cumsum(p[o])
    # smallest value whose cumulative mass reaches 1-alpha (tolerance for the float normalisation)
    k = np.flatnonzero(c >= (1 - alpha) - 1e-12)[0]
    return vals[o][k]


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. Unit weights + unit test weight reproduce harness.lac_quantile exactly (with ties), for every n.
    for n in range(1, 400):
        s = np.round(rng.random(n), 2)                       # many ties
        o = np.argsort(s, kind="stable")
        q = weighted_thresholds(s[o], np.cumsum(np.ones(n)), np.ones(3))
        assert (q == harness.lac_quantile(s)).all(), n

    # 2. Vectorised weighted threshold equals the direct definition, random weights and test weights.
    for _ in range(300):
        n = rng.integers(1, 60); s = np.round(rng.random(n), 1); w = rng.exponential(1, n)
        wx = np.r_[rng.exponential(1, 5), 0.0, 1e3]
        o = np.argsort(s, kind="stable")
        q = weighted_thresholds(s[o], np.cumsum(w[o]), wx)
        qb = np.array([_brute_threshold(s, w, x) for x in wx])
        assert np.array_equal(q, qb), (s, w, wx, q, qb)

    cal = _synthetic(20000, rng, 3.2); unl = _synthetic(20000, rng, 2.4).blind(); te = _synthetic(60000, rng, 2.4)

    # 3. Without unlabeled data M4 is B0 (identical sets); sets() ignores labels; shape / dtype.
    b0 = harness.B0(); b0.fit(cal)
    m0 = M4().fit(cal)
    assert np.array_equal(m0.sets(te.blind()), b0.sets(te.blind()))
    assert np.array_equal(m0.sets(te), m0.sets(te.blind()))
    s0 = m0.sets(te.blind()); assert s0.shape == (len(te.y), 6) and s0.dtype == bool

    # 4. Under a pure covariate shift, B0 under-covers and weighted M4 (both ratios) restores ~0.90 per class.
    #    Weighted conformal is valid marginally over calibration draws; conditional on one calibration set the
    #    coverage has sd ~ sqrt(.09 / ESS) (ESS of the class weights), plus test noise: tolerance 4 sd.
    cb = b0.sets(te.blind())[np.arange(len(te.y)), te.y]
    assert cb.mean() < 0.85, cb.mean()
    for ratio in ("logistic", "lgbm"):
        m = M4(ratio).fit(cal, unlabeled=unl)
        assert m.w_cal.min() >= 1 / CLIP and m.w_cal.max() <= CLIP
        s = m.sets(te.blind()); cov = s[np.arange(len(te.y)), te.y]
        assert np.array_equal(s, m.sets(te)), ratio
        ess = m.diag["ess"]
        assert ess.min() > 200 and (ess <= m.diag["n"] + 1e-9).all(), m.diag
        tol = 4 * np.sqrt(0.09 / ess.sum() + 0.09 / len(te.y))
        assert abs(cov.mean() - 0.90) < tol, (ratio, cov.mean(), tol)
        for c in range(6):
            tol = 4 * np.sqrt(0.09 / ess[c] + 0.09 / (te.y == c).sum())
            assert abs(cov[te.y == c].mean() - 0.90) < tol, (ratio, c, cov[te.y == c].mean(), tol)
        print(f"  synthetic shift, ratio={ratio}: B0 {cb.mean():.4f}  M4 {cov.mean():.4f}  ESS/n {np.round(ess / m.diag['n'], 3)}")
        assert m.last_wx.min() >= 1 / CLIP and m.last_wx.max() <= CLIP

    # 5. Logistic ratio recovers the true log ratio slope on log1p(nobs): N(2.4,.6)/N(3.2,.6) -> -0.8/.36 per unit.
    m = M4().fit(cal, unlabeled=unl)
    coef = m.ratio.clf.coef_[0] / m.ratio.scaler.scale_
    assert abs(coef[0] - (-0.8 / 0.36)) < 0.25, coef

    # 6. A rare class with a large test weight gets the trivial threshold (point mass at +inf > alpha).
    s = np.sort(rng.random(30)); W = np.cumsum(np.ones(30))
    assert np.isinf(weighted_thresholds(s, W, np.array([20.0]))[0])          # 20 / 50 > 0.1
    assert np.isfinite(weighted_thresholds(s, W, np.array([1 / 20]))[0])
    print("m4_weighted self-test: all asserts passed")
