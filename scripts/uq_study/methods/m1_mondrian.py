"""Candidate M1 (and M3-for-M1): class x group Mondrian LAC conformal sets.

The AD-MCP of ASTRANet (Sasli et al. 2026, Sec. VI H) generalised to any one partition of the
inputs (docs/uq_preregistration.md, "Candidates"). For class c and group value g:

    q[c, g] = lac_quantile(1 - P_c  over calibration rows with y == c and group == g)
    c in set(x)  iff  1 - P_c(x) <= q[c, g(x)]

Design choices, all fixed before any M1 variant was scored:
- Partitions. 'nobs' = n_obs tercile and 'atyp' = energy quintile, both from
  harness.group_edges(cal) / harness.groups(view, edges), with the edges computed on the calibration
  View exactly as passed to fit() (row-weighted, so with aug > 0 each object weighs one; nobs edges
  are then identical to the pattern-0 edges, energy edges include the augmented views).
  'ctx' = presence pattern sum_j present[:, j] << j over [host, gaia_wise, ps1_pos, M_pseudo]
  (16 values; the presence after removal, not the removal pattern, so it is an input-only quantity).
  'nctx' = number of blocks present (0..4) = harness.groups(...)['n_ctx'].
- Cell size. "Fewer than min_cell calibration OBJECTS" counts distinct cal.obj in the cell (with
  aug > 0 an object that lands in a cell under several views counts once).
- Fallback. A (class, group) cell under min_cell objects, or a group value never seen in
  calibration, uses the class threshold q_c computed on all class-c calibration rows.
- Weights. aug == 0: the unweighted finite-sample quantile (identical to B0's for the fallback).
  aug > 0: harness.lac_quantile with weights cal.w, both for cells and for the class fallback q_c.
- No other tuning: alpha = harness.ALPHA = 0.10, min_cell = 10 (registered), no non-empty rule,
  no smoothing or merging of cells.
- Registered variant designated before scoring: M1_atyp (aug = 0), the literal AD-MCP stratification
  (registration: atyp is "the classifier-native score ... as in ASTRANet's AD-MCP"). M1_nobs and
  M1_ctx are the other two partitions the registration lists ("one partition at a time"), reported as
  registered alternates with a multiplicity caveat; M1_nctx is exploratory (not in the registration).
  The *_aug3 runs are the registered M3 construction applied to M1 (3 extra random-removal views).
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
from harness import View, lac_quantile, group_edges, groups

PARTITIONS = ("nobs", "atyp", "ctx", "nctx")
N_VALUES = dict(nobs=3, atyp=5, ctx=16, nctx=5)


class M1:
    def __init__(self, partition, min_cell=10, aug=0):
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
        self.partition, self.min_cell, self.aug = partition, int(min_cell), int(aug)
        self.wants_augmented = self.aug

    # -------------------------------------------------------------- group labels (inputs only)
    def _labels(self, v: View):
        if self.partition in ("nobs", "atyp"):
            return groups(v, self.edges)[self.partition].astype(int)
        if self.partition == "ctx":
            return (v.present.astype(int) << np.arange(4)[None, :]).sum(1)
        return v.present.sum(1).astype(int)

    def _q(self, scores, weights, objs=None):
        if len(scores) == 0:
            return np.inf
        if weights is not None and objs is not None:
            # amendment 2: weights per pool, one per distinct object, test point = one object
            weights = harness.pool_weights(objs)
        return lac_quantile(scores, None if weights is None else weights)

    def fit(self, cal: View, unlabeled: View | None = None):   # unlabeled is ignored
        y = cal.y
        assert (y >= 0).all(), "calibration labels required"
        self.edges = group_edges(cal)
        g = self._labels(cal)
        wt = cal.w if self.aug > 0 else None
        S = 1.0 - cal.P
        G = N_VALUES[self.partition]
        self.q_class = np.empty(6)
        self.q = np.empty((6, G))
        self.n_obj = np.zeros((6, G), int)
        self.own = np.zeros((6, G), bool)          # True = cell has its own threshold
        for c in range(6):
            mc = y == c
            self.q_class[c] = self._q(S[mc, c], None if wt is None else wt[mc], cal.obj[mc])
            self.q[c] = self.q_class[c]
            for gv in range(G):
                m = mc & (g == gv)
                self.n_obj[c, gv] = len(np.unique(cal.obj[m]))
                if self.n_obj[c, gv] >= self.min_cell:
                    self.q[c, gv] = self._q(S[m, c], None if wt is None else wt[m], cal.obj[m])
                    self.own[c, gv] = True
        return self

    def sets(self, v: View):
        g = self._labels(v)
        assert g.min() >= 0 and g.max() < self.q.shape[1]
        return (1.0 - v.P) <= self.q[:, g].T


# ------------------------------------------------------------------ self-test on synthetic data
def _synthetic(n, rng, n_aug=0, shift=True):
    """Toy 6-class data whose B0 coverage differs by group (scores depend on nobs and context)."""
    y = rng.integers(0, 6, n)
    nobs = rng.integers(3, 120, n).astype(float)
    present = rng.random((n, 4)) < 0.7
    E = rng.normal(0, 1, n)
    sharp = 0.5 + (np.log1p(nobs) - 1) * 0.6 + present.sum(1) * 0.4 if shift else np.full(n, 2.0)
    logits = rng.normal(0, 1, (n, 6)); logits[np.arange(n), y] += sharp
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    L = P.argmax(1)
    obj = np.arange(n)
    v = View(P=P, L=L, E=E, nobs=nobs, present=present, pattern=np.zeros(n, int), y=y, obj=obj, w=np.ones(n))
    if n_aug:
        k = 1 + n_aug
        v = View(P=np.tile(P, (k, 1)), L=np.tile(L, k), E=np.tile(E, k), nobs=np.tile(nobs, k),
                 present=np.tile(present, (k, 1)), pattern=np.zeros(n * k, int), y=np.tile(y, k),
                 obj=np.tile(obj, k), w=np.full(n * k, 1.0 / k))
    return v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cal, te = _synthetic(20000, rng), _synthetic(60000, rng)

    # 1. min_cell larger than any cell -> every cell falls back -> identical to B0.
    b0 = harness.B0(); b0.fit(cal)
    for p in PARTITIONS:
        m = M1(p, min_cell=10 ** 9).fit(cal)
        assert not m.own.any() and np.array_equal(m.sets(te.blind()), b0.sets(te.blind())), p

    # 2. sets() never reads labels; output shape / dtype.
    for p in PARTITIONS:
        m = M1(p).fit(cal)
        s = m.sets(te.blind())
        assert s.shape == (len(te.y), 6) and s.dtype == bool
        assert np.array_equal(s, m.sets(te)), p

    # 3. Per-cell coverage close to 0.90 on fresh exchangeable data, where B0 is off in some cells.
    #    Conditional on calibration, cell coverage ~ Beta(k, n+1-k) (sd ~ sqrt(.09/n)), plus test noise:
    #    per-cell tolerance 4.5 sigma, and the average over cells must sit at the nominal level.
    for p in PARTITIONS:
        m = M1(p).fit(cal)
        s = m.sets(te.blind()); cov = s[np.arange(len(te.y)), te.y]
        g = m._labels(te); dev = []
        for c in range(6):
            for gv in range(N_VALUES[p]):
                msk = (te.y == c) & (g == gv)
                if m.own[c, gv] and msk.sum() >= 300:
                    tol = 4.5 * np.sqrt(0.09 / (m.n_obj[c, gv] + 1) + 0.09 / msk.sum())
                    assert abs(cov[msk].mean() - 0.90) < tol, (p, c, gv, cov[msk].mean(), tol)
                    dev.append(cov[msk].mean() - 0.90)
        assert len(dev) >= 12 and abs(np.mean(dev)) < 0.01, (p, np.mean(dev))
    m = M1("nobs").fit(cal); bs = b0.sets(te.blind()); ms = m.sets(te.blind())
    gl = m._labels(te); cb = bs[np.arange(len(te.y)), te.y]; cm = ms[np.arange(len(te.y)), te.y]
    db = max(abs(cb[(te.y == c) & (gl == k)].mean() - 0.9) for c in range(6) for k in range(3))
    dm = max(abs(cm[(te.y == c) & (gl == k)].mean() - 0.9) for c in range(6) for k in range(3))
    assert db > 0.05 and dm < db / 2, ("toy data should break B0 by nobs and M1_nobs should repair it", db, dm)

    # 4. Fallback threshold: a cell with exactly min_cell - 1 objects uses q_c; min_cell objects gets its own.
    small = _synthetic(400, rng, shift=False)
    small.present[:] = True; small.present[small.y == 0, 0] = False
    idx0 = np.flatnonzero(small.y == 0)
    small.present[idx0[9:], 0] = True                       # exactly 9 class-0 objects without host
    m = M1("ctx").fit(small)
    assert m.n_obj[0, 0b1110] == 9 and not m.own[0, 0b1110] and m.q[0, 0b1110] == m.q_class[0]
    small.present[idx0[9], 0] = False                       # now 10
    m = M1("ctx").fit(small)
    assert m.n_obj[0, 0b1110] == 10 and m.own[0, 0b1110]
    sc = 1 - small.P[(small.y == 0) & ~small.present[:, 0], 0]
    assert m.q[0, 0b1110] == np.sort(sc)[int(np.ceil(11 * 0.9)) - 1] == sc.max()

    # 5. Augmented: duplicated views with weights 1/(1+aug) count objects once and give the unweighted quantile.
    aug = _synthetic(3000, np.random.default_rng(5), n_aug=3)
    one = _synthetic(3000, np.random.default_rng(5), n_aug=0)
    for p in PARTITIONS:
        ma, m0 = M1(p, aug=3).fit(aug), M1(p).fit(one)
        assert ma.wants_augmented == 3 and ma.n_obj.sum() == m0.n_obj.sum() == 3000, p
        if p in ("ctx", "nctx"):                             # edge-free partitions: exact
            assert np.array_equal(ma.n_obj, m0.n_obj) and np.array_equal(ma.own, m0.own), p
        else:   # harness.group_edges can move an edge by float rounding (~1e-10) onto a tied integer nobs
            la, l0 = ma._labels(one), m0._labels(one)
            assert (la == l0).mean() > 0.98, (p, (la == l0).mean())
        # weighted quantile of 4 identical copies differs from the unweighted one by at most one order statistic
        for c in range(6):
            assert abs(ma.q_class[c] - m0.q_class[c]) < 0.05, (p, c)
    # weighted lac_quantile with unit weights equals the unweighted one
    sc = rng.random(57)
    assert lac_quantile(sc, np.ones(57)) == lac_quantile(sc)
    print("m1_mondrian self-test: all asserts passed")
