"""Candidate M6 (POST HOC, preregistration amendment 2): atypicality-stratified class-conditional LAC
on dropout-augmented calibration, with the non-empty rule.

    q[c, g] = weighted finite-sample 0.9 quantile of 1 - P_c over calibration rows of class c whose
              classifier energy falls in quintile g (edges from the calibration objects as observed);
    set(x)  = {c : 1 - P_c(x) <= q[c, g(x)]}, and an empty set becomes {hierarchy-rule label}.

It is M1(partition='atyp', aug=3) (the registered M3 form of M1, verified in the first round) with two
changes fixed before it was scored: the quintile edges come from pattern-0 calibration rows, so that
the cells are those of the objects as they arrive, and the non-empty rule of B1, because exact coverage
in the confident strata otherwise turns correct, ~99%-confident predictions into empty sets.
"""
from __future__ import annotations
import sys
from pathlib import Path
try:
    import harness
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); import harness
import numpy as np
from harness import View, group_edges
from m1_mondrian import M1


class M6(M1):
    def __init__(self, aug=3, min_cell=10, nonempty=True):
        super().__init__("atyp", min_cell=min_cell, aug=aug)
        self.nonempty = nonempty

    def fit(self, cal: View, unlabeled: View | None = None):
        super().fit(cal)
        # recompute the cells with edges from the objects as observed, then refit the thresholds
        self.edges = group_edges(cal.take(np.flatnonzero(cal.pattern == 0)))
        g = self._labels(cal); S = 1.0 - cal.P; wt = cal.w if self.aug > 0 else None
        for c in range(6):
            mc = cal.y == c
            self.q[c] = self.q_class[c]; self.own[c] = False
            for gv in range(self.q.shape[1]):
                m = mc & (g == gv)
                self.n_obj[c, gv] = len(np.unique(cal.obj[m]))
                if self.n_obj[c, gv] >= self.min_cell:
                    self.q[c, gv] = self._q(S[m, c], None if wt is None else wt[m], cal.obj[m]); self.own[c, gv] = True
        return self

    def sets(self, v: View):
        s = super().sets(v)
        if self.nonempty:
            e = ~s.any(1); s[np.flatnonzero(e), v.L[e]] = True
        return s


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 3000
    P = rng.dirichlet(np.ones(6) * .5, n); y = np.array([rng.choice(6, p=p) for p in P])
    v = View(P=P, L=P.argmax(1), E=-np.log(P.max(1)), nobs=rng.integers(5, 60, n).astype(float),
             present=rng.random((n, 4)) < .7, pattern=np.zeros(n, int), y=y, obj=np.arange(n), w=np.ones(n))
    m = M6(aug=0).fit(v.take(np.arange(2000)))
    s = m.sets(v.take(np.arange(2000, n)).blind())
    assert s.any(1).all(), "non-empty rule"
    s0 = M6(aug=0, nonempty=False).fit(v.take(np.arange(2000))).sets(v.take(np.arange(2000, n)).blind())
    assert (s >= s0).all() and (s.sum(1) - s0.sum(1) <= 1).all()
    cov = s[np.arange(1000), y[2000:]].mean(); print("synthetic coverage", round(cov, 3)); assert cov > .85
    print("self-test ok")
