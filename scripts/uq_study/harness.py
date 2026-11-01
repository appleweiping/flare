"""Evaluation harness for the conditional-calibration study (docs/uq_preregistration.md).

A calibration method is a class with
    wants_augmented: int            number of extra random-removal views per calibration object (0 = none)
    fit(cal: View) -> None
    sets(test: View) -> (m, 6) bool membership
A View carries, for a set of (object, context pattern) rows:
    P (m,6) probabilities, L (m,) hierarchy-rule label, E (m,) classifier energy (-logsumexp top margins),
    nobs (m,), present (m,4) bool context blocks present after removal, pattern (m,), y (m,) labels
    (-1 unknown), obj (m,) object index, Z (m,k) compact covariates (log nobs, energy, present, max P),
    w (m,) row weight (1 / views per object, so that each calibration object weighs one).
Methods must never read test labels: `sets` receives a View whose y is set to -1.

Protocol: leave-one-fold-out over the five development folds (calibrate on the other four folds'
out-of-fold rows, evaluate on the held-out fold), the construction of scripts/update_coverage.py.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logsumexp

SIX = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]
BLOCKS = ["host", "gaia_wise", "ps1_pos", "M_pseudo"]
ALPHA = 0.10
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results
OOF = WORK / "oofs"
CACHE = WORK / "cache"
MIN_CELL = 30


@dataclass
class View:
    P: np.ndarray; L: np.ndarray; E: np.ndarray; nobs: np.ndarray; present: np.ndarray
    pattern: np.ndarray; y: np.ndarray; obj: np.ndarray; w: np.ndarray

    @property
    def Z(self):
        return np.column_stack([np.log1p(self.nobs), self.E, self.present.astype(float), self.P.max(1)])

    def take(self, idx):
        return View(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})

    def blind(self):
        return replace(self, y=np.full_like(self.y, -1))


def presence_from_cache(tag, obj_ids):
    from flare.config import HOST_COLS, GW_COLS, POS_COLS  # noqa
    D = pd.read_parquet(CACHE / f"X_{tag}.parquet").drop_duplicates("obj_id").set_index("obj_id").reindex(obj_ids)
    cols = [HOST_COLS, GW_COLS, POS_COLS, ["M_pseudo"]]
    return np.column_stack([D[c].notna().any(axis=1).to_numpy() for c in cols])


class Setting:
    """Out-of-fold (or deployed) predictions under all 16 context patterns for one cache tag."""

    def __init__(self, npz_path, tag):
        z = np.load(npz_path, allow_pickle=True)
        self.P, self.L, self.M, self.y = z["P"], z["L"], z["M"], z["y"]
        self.fold = z["fold"] if "fold" in z else None
        self.obj_id = z["obj_id"]; self.nobs = z["X_nobs"].astype(float)
        self.bts_type = z["bts_type"] if "bts_type" in z else None
        self.present0 = presence_from_cache(tag, self.obj_id)
        self.n = len(self.y)

    def view(self, idx, patterns, weight=None):
        idx = np.asarray(idx); patterns = np.asarray(patterns)
        removed = ((patterns[:, None] >> np.arange(4)[None, :]) & 1).astype(bool)
        present = self.present0[idx] & ~removed
        M = self.M[patterns, idx]
        return View(P=self.P[patterns, idx], L=self.L[patterns, idx], E=-logsumexp(M, axis=1),
                    nobs=self.nobs[idx], present=present, pattern=patterns, y=self.y[idx], obj=idx,
                    w=np.ones(len(idx)) if weight is None else weight)


def scenario_patterns(n, scenario, seed=2026):
    if scenario == "observed":
        return np.zeros(n, int)
    if scenario == "missing":
        return np.random.default_rng(seed).integers(0, 16, n)
    raise ValueError(scenario)


def calibration_view(S: Setting, idx, n_aug, seed):
    """Pattern 0 for every calibration object, plus n_aug random removal patterns each (weights 1/(1+n_aug))."""
    if n_aug <= 0:
        return S.view(idx, np.zeros(len(idx), int))
    rng = np.random.default_rng(seed)
    pats = np.concatenate([np.zeros(len(idx), int)] + [rng.integers(1, 16, len(idx)) for _ in range(n_aug)])
    ids = np.tile(idx, 1 + n_aug)
    return S.view(ids, pats, weight=np.full(len(ids), 1.0 / (1 + n_aug)))


# ---------------------------------------------------------------- groups and metrics
def group_edges(cal: View):
    w = cal.w / cal.w.sum()
    def wq(x, qs):
        o = np.argsort(x); c = np.cumsum(w[o]); return np.interp(qs, c, x[o])
    return dict(nobs=wq(cal.nobs, [1 / 3, 2 / 3]), atyp=wq(cal.E, [.2, .4, .6, .8]))


def groups(v: View, edges):
    g = dict(nobs=np.searchsorted(edges["nobs"], v.nobs, side="right"),
             atyp=np.searchsorted(edges["atyp"], v.E, side="right"))
    for j, b in enumerate(BLOCKS):
        g[f"ctx_{b}"] = v.present[:, j].astype(int)
    return g          # registered families only (amendment 2 dropped the number-of-blocks family)


def pool_weights(objs):
    """Row weights for one calibration pool: 1 / (number of rows of the same object in this pool), so
    that every distinct object in the pool weighs one (hierarchical conformal, Lee, Barber & Willett 2023).
    An object with one augmented view in a cell counts fully there; its four views in a class pool
    count one quarter each."""
    objs = np.asarray(objs)
    _, inv, cnt = np.unique(objs, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def lac_quantile(scores, weights=None, alpha=ALPHA):
    """Finite-sample (1-alpha) quantile; with weights, the weighted version (unit total = n)."""
    s = np.asarray(scores, float)
    if weights is None:
        v = np.sort(s); k = int(np.ceil((len(v) + 1) * (1 - alpha)))
        return float(v[k - 1]) if 0 < k <= len(v) else np.inf
    o = np.argsort(s); s, w = s[o], np.asarray(weights, float)[o]
    # weights are in units of calibration OBJECTS (calibration_view gives each object total weight 1),
    # so the test point carries the weight of one object (audit correction, amendment 2)
    c = np.cumsum(w) / (w.sum() + 1.0)
    k = np.searchsorted(c, 1 - alpha, side="left")
    return float(s[k]) if k < len(s) else np.inf


def evaluate(sets, v: View, g):
    """Per-row outcome arrays plus the summary of the registered endpoints."""
    y = v.y; m = len(y)
    cov = sets[np.arange(m), y]; size = sets.sum(1)
    out = dict(marginal=float(cov.mean()), size=float(size.mean()), singleton=float((size == 1).mean()),
               empty=float((size == 0).mean()),
               singleton_acc=float(((size == 1) & cov).sum() / max((size == 1).sum(), 1)),
               per_class={SIX[c]: float(cov[y == c].mean()) for c in range(6) if (y == c).any()})
    cells = {}
    for name, lab in g.items():
        for c in range(6):
            for val in np.unique(lab):
                msk = (y == c) & (lab == val)
                if msk.sum():
                    cells[(name, SIX[c], int(val))] = (int(msk.sum()), float(cov[msk].mean()))
    big = {k: v_ for k, v_ in cells.items() if v_[0] >= MIN_CELL}
    out["U"] = float(max([max(0.0, 1 - ALPHA - c) for _, c in big.values()], default=0.0))
    out["worst_cell"] = min(big.items(), key=lambda kv: kv[1][1])[0] if big else None
    out["dispersion"] = float(np.std([c for _, c in big.values()])) if big else 0.0
    out["cells"] = cells
    return out, cov, size


def lofo(S: Setting, make_method, scenario="observed", seed=2026, cal_scenario="observed"):
    """Leave-one-fold-out sets for every development object in `scenario`; returns per-object arrays."""
    pats = scenario_patterns(S.n, scenario, seed)
    sets = np.zeros((S.n, 6), bool)
    gl = {k: np.zeros(S.n, int) for k in ["nobs", "atyp"] + [f"ctx_{b}" for b in BLOCKS]}
    for k in range(5):
        cal_idx, te = np.flatnonzero(S.fold != k), np.flatnonzero(S.fold == k)
        m = make_method()
        cal = calibration_view(S, cal_idx, getattr(m, "wants_augmented", 0), seed + 10007 + 7 * k)
        m.fit(cal)
        tv = S.view(te, pats[te])
        sets[te] = m.sets(tv.blind())
        e = group_edges(S.view(cal_idx, np.zeros(len(cal_idx), int)))   # evaluation groups: same edges for all methods
        for name, lab in groups(tv, e).items():
            gl[name][te] = lab
    v = S.view(np.arange(S.n), pats)
    res, cov, size = evaluate(sets, v, gl)
    return dict(res=res, cov=cov, size=size, groups=gl, y=S.y, sets=sets)


def lofo_shift(S_cal: Setting, S_dep: Setting, make_method, seed=2026):
    """Photometry-shift protocol. For fold k: calibrate on S_cal rows of the other folds (sparse
    photometry, labelled); the method may also see S_dep rows of the other folds WITHOUT labels
    (`fit(cal, unlabeled=...)` when it accepts the keyword); evaluate on S_dep rows of fold k
    (full photometry). Group edges for evaluation come from the deployment-side other folds."""
    import inspect
    assert (S_cal.obj_id == S_dep.obj_id).all() and (S_cal.fold == S_dep.fold).all()
    n = S_dep.n; sets = np.zeros((n, 6), bool)
    gl = {k: np.zeros(n, int) for k in ["nobs", "atyp"] + [f"ctx_{b}" for b in BLOCKS]}
    for k in range(5):
        cal_idx, te = np.flatnonzero(S_cal.fold != k), np.flatnonzero(S_cal.fold == k)
        m = make_method()
        cal = calibration_view(S_cal, cal_idx, getattr(m, "wants_augmented", 0), seed + 10007 + 7 * k)
        unl = S_dep.view(cal_idx, np.zeros(len(cal_idx), int)).blind()
        if "unlabeled" in inspect.signature(m.fit).parameters:
            m.fit(cal, unlabeled=unl)
        else:
            m.fit(cal)
        tv = S_dep.view(te, np.zeros(len(te), int))
        sets[te] = m.sets(tv.blind())
        e = group_edges(S_dep.view(cal_idx, np.zeros(len(cal_idx), int)))
        for name, lab in groups(tv, e).items():
            gl[name][te] = lab
    v = S_dep.view(np.arange(n), np.zeros(n, int))
    res, cov, size = evaluate(sets, v, gl)
    return dict(res=res, cov=cov, size=size, groups=gl, y=S_dep.y, sets=sets)


def cell_matrix(groups_, y):
    """Sparse (cells x objects) membership of every (group family, class, value) cell."""
    from scipy import sparse
    rows, cols, keys = [], [], []
    for name, lab in groups_.items():
        for c in range(6):
            for val in np.unique(lab):
                idx = np.flatnonzero((y == c) & (lab == val))
                if len(idx):
                    rows.append(np.full(len(idx), len(keys))); cols.append(idx); keys.append((name, SIX[c], int(val)))
    r, c_ = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(r)), (r, c_)), shape=(len(keys), len(y))), keys


def U_from_counts(C, cov, mult):
    den = C @ mult; num = C @ (mult * cov)
    ok = den >= MIN_CELL
    return float(np.max(np.maximum(0.0, 1 - ALPHA - num[ok] / den[ok]), initial=0.0))


def paired_bootstrap_U(a, b, n_boot=2000, seed=0):
    """Bootstrap of U(a) - U(b) (negative = a better), objects resampled within class, same draws for both.
    Cells are defined by the evaluation groups of `a` (identical for every method under lofo)."""
    y = a["y"]; n = len(y); rng = np.random.default_rng(seed)
    C, _ = cell_matrix(a["groups"], y)
    byc = [np.flatnonzero(y == c) for c in range(6)]
    ca, cb = a["cov"].astype(float), b["cov"].astype(float)
    d = np.empty(n_boot)
    for t in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in byc])
        mult = np.bincount(idx, minlength=n).astype(float)
        d[t] = U_from_counts(C, ca, mult) - U_from_counts(C, cb, mult)
    one = np.ones(n)
    return dict(point=U_from_counts(C, ca, one) - U_from_counts(C, cb, one),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


# ---------------------------------------------------------------- baselines
class B0:
    """Shipped construction: class-conditional LAC, (1 - P_c) <= q_c."""
    wants_augmented = 0

    def fit(self, cal: View):
        self.q = np.array([lac_quantile(1 - cal.P[cal.y == c, c]) for c in range(6)])

    def sets(self, v: View):
        return (1 - v.P) <= self.q[None, :]


class B1(B0):
    """B0 with the non-empty rule: an empty set becomes {hierarchy-rule label}."""

    def sets(self, v: View):
        s = super().sets(v)
        e = ~s.any(1); s[np.flatnonzero(e), v.L[e]] = True
        return s
