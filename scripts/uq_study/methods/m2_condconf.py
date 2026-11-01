"""Candidate M2 (and M3-for-M2): conditional conformal over overlapping groups.

Gibbs, Cherian & Candes (2023), "Conformal prediction with conditional guarantees", in its
unaugmented / split quantile-regression form (the "vanilla" fit of their Sec. 2.3, Jung et al. 2023):
the test point is NOT added to the fit. Per class c, on the calibration rows with y == c,

    beta_c = argmin_beta  sum_i w_i * rho_tau( s_i - phi_c(x_i)^T beta ),   s_i = 1 - P_c(x_i),
    rho_tau(r) = tau * r  if r > 0  else (tau - 1) * r,          w_i = cal.w (1 / views per object),
    c in set(x)  iff  1 - P_c(x) <= tau_c(x) = phi_c(x)^T beta_c.

All design choices below were fixed and written down before any M2 variant was scored.

Design matrix (registered). phi(x) = [1, nobs tercile dummies (2), atyp quintile dummies (4),
presence bits of host, gaia_wise, ps1_pos, M_pseudo (4)] = 11 columns, with the tercile / quintile
edges from harness.group_edges(cal) on the calibration View exactly as passed to fit() (row-weighted;
with aug > 0 the atyp edges include the augmented views) and labels from harness.groups(view, edges).
Every registered group (each nobs tercile, each atyp quintile, each block present / absent) is an
indicator in the span of phi, so the first-order conditions of the fit target all of them at once.
For a multi-level family the reference level (absorbed by the intercept) is, per class, the level
with the largest class-c calibration count (first one on ties); this does not change the span.

Rare-class rule (fixed, per class c and per LOFO fold):
 1. Screening. "count" = sum of cal.w over class-c rows (= number of class-c calibration objects;
    with aug > 0 an object contributes its views' fractional weights, so it counts once in total).
    A dummy for level k of nobs / atyp is kept only if count(level k) >= 10 and count(not k) >= 10;
    failing levels merge into the reference level. A presence bit is kept only if count(present)
    >= 10 and count(absent) >= 10.
 2. Collinearity. Columns linearly dependent on earlier ones (order: intercept, nobs, atyp, ctx
    [, nctx]) on the class-c calibration rows are dropped, so beta is identified and thresholds at
    unseen covariate combinations are deterministic. p = number of remaining columns (= rank).
 3. Budget. Require n_c >= 20 * p, n_c = class-c object count (sum of w). Otherwise drop whole
    families in the fixed order ctx, then nobs, then atyp (exploratory nctx family first of all),
    re-applying 2 after each drop.
 4. Intercept only (p == 1): the class uses B0's threshold exactly, harness.lac_quantile of the
    class-c scores (unweighted when aug == 0, i.e. identical to B0; weighted by cal.w when aug > 0,
    the same convention as M1's fallback).
Level. Registered: tau = 1 - alpha = 0.90 for every class with p > 1 (the literal unaugmented fit).
Exploratory "adjust": tau_c = (1 - alpha) * (n_c + p_c) / n_c, the split-conformal (n+1)/n
finite-sample correction applied once per fitted parameter. Why this formula: for an intercept-only
fit the pinball minimiser at that level is the ceil((n+1)(1-alpha))-th order statistic, i.e. B0
exactly; for a partition into p equal cells it reproduces the per-cell split-conformal order
statistic; the unaugmented fit's in-sample optimism is one interpolated point per parameter,
O(p/n). n_c counts objects (sum of w) also when aug > 0 (the conservative effective sample size),
except in the exploratory count='rows' variant, where it counts view rows.
Solver. The weighted pinball problem is the LP  min sum w_i (tau u_i + (1-tau) v_i)  s.t.
X beta + u - v = s, u, v >= 0. It is solved exactly with scipy.optimize.linprog(method='highs') in
its LP-dual form (max s^T a s.t. X^T a = 0, -(1-tau) w <= a <= tau w; beta = -multipliers of the
equality rows), which is ~30x faster on the augmented SN Ia fit; optimality of the returned beta
is certified by a zero duality gap (primal pinball loss at beta == dual optimum, rel. 1e-8),
and the primal LP is solved instead if the certificate fails. Membership uses the exact comparison
1 - P_c <= tau_c(x), as B0. No clipping of tau_c(x) (it cannot change membership for s in [0, 1]).
alpha = harness.ALPHA = 0.10; no non-empty rule; nothing tuned on results.

Variants scored (names fixed before scoring):
  M2               aug=0, tau=0.90                      registered candidate M2
  M2_aug3          aug=3, tau=0.90                      registered M3-for-M2
  M2_adj           aug=0, adjusted level                exploratory
  M2_aug3_adj      aug=3, adjusted level                exploratory
  M2_aug3_adj_nctx aug=3, adjusted, phi + n_ctx level dummies (family nctx, dropped first)
                   exploratory: harness.evaluate also scores n_ctx cells, which are not in the
                   registered span (n_ctx == k is not linear in the 4 bits)
  M2_aug3_adj_rows aug=3, adjusted, every count (screening, budget and the n of the level
                   adjustment) in view rows instead of objects; exploratory: lets ctx bits survive
                   for the small classes when every object is seen under several removal patterns.
                   (Fixed before any scoring: the first draft kept objects in the adjustment, which
                   gives tau > 1 for TDE once the row budget admits p > n_obj / 9.)
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
from scipy import sparse
from scipy.optimize import linprog
from harness import View, lac_quantile, group_edges, groups, BLOCKS, ALPHA

LEVELS = dict(nobs=3, atyp=5, nctx=5)
GROUP_KEY = dict(nobs="nobs", atyp="atyp", nctx="n_ctx")
FAMILY_ORDER = ("nobs", "atyp", "ctx", "nctx")     # column order (collinearity is resolved in this order)
DROP_ORDER = ("nctx", "ctx", "nobs", "atyp")        # budget: whole families dropped in this order


# ---------------------------------------------------------------- weighted linear quantile regression
def pinball(fitted, s, w, tau):
    r = s - fitted
    return float(np.sum(w * np.where(r > 0, tau * r, (tau - 1.0) * r)))


def _qr_primal(X, s, w, tau):
    n, p = X.shape
    A = sparse.hstack([sparse.csr_matrix(X), sparse.identity(n), -sparse.identity(n)]).tocsc()
    cost = np.concatenate([np.zeros(p), tau * w, (1.0 - tau) * w])
    r = linprog(cost, A_eq=A, b_eq=s, bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs")
    if r.status != 0:
        raise RuntimeError(f"primal QR LP failed: {r.message}")
    return r.x[:p], float(r.fun)


def qr_fit(X, s, w, tau, return_info=False):
    """Exact minimiser of sum_i w_i rho_tau(s_i - X_i beta) via the HiGHS LP (dual form, certified)."""
    X = np.asarray(X, float); s = np.asarray(s, float); w = np.asarray(w, float)
    n, p = X.shape
    assert 0.0 < tau < 1.0 and (w > 0).all() and n > 0
    rd = linprog(-s, A_eq=X.T, b_eq=np.zeros(p), bounds=np.column_stack([-(1.0 - tau) * w, tau * w]),
                 method="highs")
    info = dict(solver="dual")
    if rd.status == 0:
        dual = -float(rd.fun)
        for beta in (-rd.eqlin.marginals, rd.eqlin.marginals):
            prim = pinball(X @ beta, s, w, tau)
            if prim - dual <= 1e-8 * max(1.0, abs(dual)):
                info.update(objective=prim, gap=prim - dual)
                return (beta, info) if return_info else beta
    beta, obj = _qr_primal(X, s, w, tau)
    info.update(solver="primal", objective=obj, gap=np.nan)
    return (beta, info) if return_info else beta


# ---------------------------------------------------------------- the method
class M2:
    def __init__(self, aug=0, adjust=False, nctx=False, count="objects", alpha=ALPHA,
                 min_side=10, per_param=20):
        if count not in ("objects", "rows"):
            raise ValueError("count must be 'objects' or 'rows'")
        self.aug, self.adjust, self.nctx, self.count = int(aug), bool(adjust), bool(nctx), count
        self.alpha, self.min_side, self.per_param = float(alpha), float(min_side), float(per_param)
        self.wants_augmented = self.aug
        self.families = FAMILY_ORDER if self.nctx else FAMILY_ORDER[:3]

    # -------------------------------------------------------------- columns
    @staticmethod
    def _column(lab, col):
        fam, key, val = col
        if fam == "int":
            return np.ones(len(lab["nobs"]))
        return (lab[key] == val).astype(float)

    def _design(self, lab, spec):
        return np.column_stack([self._column(lab, col) for col in spec])

    def _screen(self, lab, cnt):
        """Candidate columns per family for one class (lab, cnt restricted to that class's rows)."""
        n_c = cnt.sum(); tol = 1e-9; fams = {}
        for fam in self.families:
            if fam == "ctx":
                cols = []
                for b in BLOCKS:
                    n1 = cnt[lab[f"ctx_{b}"] == 1].sum()
                    if min(n1, n_c - n1) >= self.min_side - tol:
                        cols.append(("ctx", f"ctx_{b}", 1))
            else:
                key = GROUP_KEY[fam]
                counts = np.bincount(lab[key], weights=cnt, minlength=LEVELS[fam])
                ref = int(np.argmax(counts))
                cols = [(fam, key, k) for k in range(LEVELS[fam])
                        if k != ref and min(counts[k], n_c - counts[k]) >= self.min_side - tol]
            fams[fam] = cols
        return fams

    @staticmethod
    def _independent(X, spec):
        """Greedy: keep a column only if it raises the rank of the kept ones (fixed column order)."""
        U = np.unique(X, axis=0)
        keep, dropped = [0], []
        for j in range(1, X.shape[1]):
            if np.linalg.matrix_rank(U[:, keep + [j]]) > len(keep):
                keep.append(j)
            else:
                dropped.append(spec[j])
        return keep, dropped

    # -------------------------------------------------------------- fit / sets
    def fit(self, cal: View, unlabeled: View | None = None):     # unlabeled is ignored
        y = cal.y
        assert (y >= 0).all(), "calibration labels required"
        self.edges = group_edges(cal)
        lab = groups(cal, self.edges)
        tau0 = 1.0 - self.alpha
        self.beta, self.spec, self.q, self.info = [None] * 6, [None] * 6, np.full(6, np.nan), [None] * 6
        for c in range(6):
            m = np.flatnonzero(y == c)
            assert len(m), f"no calibration rows for class {c}"
            lab_c = {k: v[m] for k, v in lab.items()}
            w = cal.w[m].astype(float)
            s = 1.0 - cal.P[m, c]
            n_obj = float(w.sum())
            cnt = w if self.count == "objects" else np.ones(len(m))
            n_budget = float(cnt.sum())
            fams = self._screen(lab_c, cnt)
            active = [f for f in self.families]
            budget_dropped = []
            while True:
                spec = [("int", None, None)] + [col for f in self.families if f in active for col in fams[f]]
                X = self._design(lab_c, spec)
                keep, collinear = self._independent(X, spec)
                spec, X = [spec[j] for j in keep], X[:, keep]
                p = len(spec)
                if n_budget >= self.per_param * p or p == 1:
                    break
                nxt = next(f for f in DROP_ORDER if f in active)
                active.remove(nxt); budget_dropped.append(nxt)
            info = dict(n=n_obj, rows=len(m), p=p, cols=[f"{k}={v}" if k else "1" for _, k, v in spec],
                        screened_out={f: LEVELS.get(f, 4) - (0 if f == "ctx" else 1) - len(fams[f]) for f in self.families},
                        collinear=[f"{k}={v}" for _, k, v in collinear], budget_dropped=budget_dropped)
            if p == 1:                                   # intercept only: B0 for this class
                self.q[c] = lac_quantile(s) if self.aug == 0 else lac_quantile(s, w)
                info.update(kind="B0", tau=tau0)
            else:
                tau = tau0 * (n_budget + p) / n_budget if self.adjust else tau0   # n_budget = n_obj unless count='rows'
                assert tau < 1.0, (c, tau)
                beta, qi = qr_fit(X, s, w, tau, return_info=True)
                self.beta[c], self.spec[c] = beta, spec
                info.update(kind="QR", tau=tau, solver=qi["solver"], gap=qi["gap"])
            self.info[c] = info
        return self

    def thresholds(self, v: View):
        lab = groups(v, self.edges)
        T = np.empty((len(v.nobs), 6))
        for c in range(6):
            T[:, c] = self.q[c] if self.beta[c] is None else self._design(lab, self.spec[c]) @ self.beta[c]
        return T

    def sets(self, v: View):
        return (1.0 - v.P) <= self.thresholds(v)


# ------------------------------------------------------------------ self-test on synthetic data
def _synthetic(n, rng, n_aug=0, effect=True, class_p=None):
    """Toy 6-class data whose true-class score shifts additively with nobs, energy and context, so
    that a single class threshold (B0) mis-covers the groups."""
    class_p = np.full(6, 1 / 6) if class_p is None else np.asarray(class_p) / np.sum(class_p)
    y = rng.choice(6, n, p=class_p)
    nobs = rng.integers(3, 120, n).astype(float)
    present = rng.random((n, 4)) < np.array([0.9, 0.6, 0.45, 0.7])
    E = rng.normal(0, 1, n)
    sharp = 2.0 + ((np.log1p(nobs) - 3.3) * 0.9 - E * 0.6 + (present.sum(1) - 2.6) * 0.5 if effect else 0.0)
    logits = rng.normal(0, 1, (n, 6)); logits[np.arange(n), y] += sharp
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    obj = np.arange(n)
    v = View(P=P, L=P.argmax(1), E=E, nobs=nobs, present=present, pattern=np.zeros(n, int), y=y, obj=obj,
             w=np.ones(n))
    if n_aug:
        k = 1 + n_aug
        v = View(P=np.tile(P, (k, 1)), L=np.tile(v.L, k), E=np.tile(E, k), nobs=np.tile(nobs, k),
                 present=np.tile(present, (k, 1)), pattern=np.zeros(n * k, int), y=np.tile(y, k),
                 obj=np.tile(obj, k), w=np.full(n * k, 1.0 / k))
    return v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tau0 = 1 - ALPHA

    # 1. LP exactness. Intercept only: the pinball minimiser is the ceil(n tau)-th order statistic, and
    #    at the adjusted level tau (n+1)/n it is B0's lac_quantile (n chosen with n*tau non-integer).
    for n in (23, 46, 105, 171, 1001):
        s = rng.random(n); X = np.ones((n, 1)); w = np.ones(n)
        b = qr_fit(X, s, w, tau0)[0]
        assert abs(b - np.sort(s)[int(np.ceil(n * tau0)) - 1]) < 1e-9, n
        b_adj = qr_fit(X, s, w, tau0 * (n + 1) / n)[0]
        assert abs(b_adj - lac_quantile(s)) < 1e-9, n
    #    General design: the dual-form solution attains the primal LP optimum and no perturbation helps.
    n, p = 3000, 11
    X = np.column_stack([np.ones(n), rng.random((n, p - 1)) < 0.4]).astype(float)
    s = rng.random(n) + 0.3 * X[:, 1] - 0.2 * X[:, 5]; w = rng.choice([0.25, 0.5, 1.0], n)
    beta, qi = qr_fit(X, s, w, tau0, return_info=True)
    _, obj_primal = _qr_primal(X, s, w, tau0)
    assert qi["solver"] == "dual" and abs(qi["objective"] - obj_primal) < 1e-7 * obj_primal
    for _ in range(200):
        assert pinball(X @ (beta + rng.normal(0, 1e-3, p)), s, w, tau0) >= qi["objective"] - 1e-9
    #    First-order conditions: weighted in-sample coverage of every indicator group (and complement) in
    #    the span lies in [tau, tau + weight on the fit / weight of the group].
    fit = X @ beta; on = np.abs(s - fit) < 1e-9; cov_in = s <= fit + 1e-9
    for j in range(1, p):
        for G in (X[:, j] == 1, X[:, j] == 0):
            cg = (w * cov_in)[G].sum() / w[G].sum()
            assert tau0 - 1e-7 <= cg <= tau0 + (w * on)[G].sum() / w[G].sum() + 1e-7, (j, cg)

    # 2. Rare-class rule.
    cal = _synthetic(20000, rng, class_p=[40, 15, 1, 1.6, 0.45, 2])
    m = M2().fit(cal)
    n_c = np.array([inf["n"] for inf in m.info])
    for c in range(6):
        inf = m.info[c]
        assert inf["n"] >= 20 * inf["p"] or inf["p"] == 1, inf
        if inf["kind"] == "B0":                       # fallback identical to B0
            assert m.q[c] == lac_quantile(1 - cal.P[cal.y == c, c])
        if "ctx" not in inf["budget_dropped"]:        # nobs is never dropped while ctx is still in
            assert "nobs" not in inf["budget_dropped"] and "atyp" not in inf["budget_dropped"], inf
    assert m.info[0]["p"] == 11 and m.info[0]["kind"] == "QR"
    small = M2().fit(cal.take(np.flatnonzero(cal.y == 4)[:30].tolist() + np.flatnonzero(cal.y != 4).tolist()))
    assert small.info[4]["kind"] == "B0"
    b0 = harness.B0(); b0.fit(cal)
    allB0 = M2(per_param=1e9).fit(cal)               # budget never met -> intercept only = B0 everywhere
    te = _synthetic(60000, rng)
    assert np.array_equal(allB0.sets(te.blind()), b0.sets(te.blind()))
    #    screening: exactly 9 objects on one side drops the bit, 10 keeps it
    base = _synthetic(3000, np.random.default_rng(3), class_p=[1, 1, 1, 1, 1, 1])
    for k, keep in ((9, False), (10, True)):
        v = base.take(np.arange(3000)); v.present[:, 1] = True
        idx0 = np.flatnonzero(v.y == 0); v.present[idx0[:k], 1] = False
        mm = M2().fit(v)
        assert ("ctx_gaia_wise=1" in mm.info[0]["cols"]) == keep, (k, mm.info[0]["cols"])
    #    collinearity: a bit identical to another in calibration is dropped (later column), sets stay defined
    v = base.take(np.arange(3000)); v.present[:, 3] = v.present[:, 1]
    mm = M2().fit(v)
    assert all("ctx_M_pseudo=1" in inf["collinear"] or "ctx_M_pseudo=1" not in inf["cols"] for inf in mm.info)

    # 3. sets() never reads labels; shape and dtype.
    for kw in (dict(), dict(adjust=True), dict(nctx=True, adjust=True)):
        mm = M2(**kw).fit(cal)
        S = mm.sets(te.blind())
        assert S.shape == (len(te.y), 6) and S.dtype == bool and np.array_equal(S, mm.sets(te)), kw

    # 4. Group-conditional coverage on fresh exchangeable data: B0 is off by > 0.05 in some registered
    #    group, M2 (adjusted) is within sampling error in every large (class, group) cell.
    cal = _synthetic(30000, rng); te = _synthetic(90000, rng)
    edges = group_edges(cal); g = groups(te, edges)
    b0.fit(cal)
    cb = b0.sets(te.blind())[np.arange(len(te.y)), te.y]
    for kw in (dict(), dict(adjust=True)):
        mm = M2(**kw).fit(cal)
        cm = mm.sets(te.blind())[np.arange(len(te.y)), te.y]
        dev_b, dev_m = [], []
        for name in ["nobs", "atyp"] + [f"ctx_{b}" for b in BLOCKS]:
            for c in range(6):
                for val in np.unique(g[name]):
                    msk = (te.y == c) & (g[name] == val)
                    if msk.sum() >= 1000:
                        dev_b.append(cb[msk].mean() - tau0); dev_m.append(cm[msk].mean() - tau0)
        dev_b, dev_m = np.array(dev_b), np.array(dev_m)
        assert dev_b.min() < -0.05 and dev_m.min() > -0.03 and abs(dev_m.mean()) < 0.01, (kw, dev_b.min(), dev_m.min(), dev_m.mean())
        assert abs(cm.mean() - tau0) < 0.01, (kw, cm.mean())

    # 5. Augmented. (a) 4 identical rows with weights 1/4 are the same weighted LP as one row (same fit;
    #    exact at the solver level -- at the M2 level harness.group_edges can put a data value exactly on
    #    an edge and float rounding then flips that row's group, moving the QR to a neighbouring vertex).
    #    (b) counts are in objects, rows are 4x.
    X1 = np.column_stack([np.ones(2000), rng.random((2000, 6)) < 0.5]).astype(float)
    s1 = rng.random(2000) + 0.2 * X1[:, 2]
    f1 = X1 @ qr_fit(X1, s1, np.ones(2000), 0.9137)
    f4 = np.tile(X1, (4, 1)) @ qr_fit(np.tile(X1, (4, 1)), np.tile(s1, 4), np.full(8000, 0.25), 0.9137)
    assert np.abs(np.tile(f1, 4) - f4).max() < 1e-7
    one = _synthetic(4000, np.random.default_rng(5)); four = _synthetic(4000, np.random.default_rng(5), n_aug=3)
    m1, m4 = M2(aug=0, adjust=True).fit(one), M2(aug=3, adjust=True).fit(four)
    assert m4.wants_augmented == 3
    for c in range(6):
        assert abs(m1.info[c]["n"] - m4.info[c]["n"]) < 1e-9 and m4.info[c]["rows"] == 4 * m1.info[c]["rows"]
        assert m1.info[c]["cols"] == m4.info[c]["cols"] and abs(m1.info[c]["tau"] - m4.info[c]["tau"]) < 1e-12
    # rows counting gives the augmented fit at least as many columns as object counting
    mr = M2(aug=3, adjust=True, count="rows").fit(four)
    assert all(mr.info[c]["p"] >= m4.info[c]["p"] for c in range(6))
    #    adjusted level formula
    for c in range(6):
        if m1.info[c]["kind"] == "QR":
            assert abs(m1.info[c]["tau"] - tau0 * (m1.info[c]["n"] + m1.info[c]["p"]) / m1.info[c]["n"]) < 1e-12
        if mr.info[c]["kind"] == "QR":
            assert abs(mr.info[c]["tau"] - tau0 * (mr.info[c]["rows"] + mr.info[c]["p"]) / mr.info[c]["rows"]) < 1e-12
            assert mr.info[c]["tau"] <= tau0 * (1 + 1 / 20) + 1e-12
    print("m2_condconf self-test: all asserts passed")
