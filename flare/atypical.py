"""Atypicality-stratified class-conditional conformal sets: an opt-in alternative to the class-only sets.

The shipped sets threshold 1 - P_c per class (flare.conformal, Mondrian over the class). Their 90% holds
on average over a class but not across how typical an object is: out of fold on the BTS benchmark, the
SN Ia in the most atypical fifth of the classifier energy are covered about half of the time and the most
typical ones almost always (docs/uq_conditional_calibration.md). This module calibrates one threshold per
(class, energy quintile) cell instead, the AD-stratified Mondrian of ASTRANet (Sasli et al. 2026, Sec. VI H)
with the classifier's own energy as the stratifier:

    q[c, g]  = weighted finite-sample (1 - alpha) quantile of 1 - P_c over calibration objects of class c
               whose energy -logsumexp(top-level margins) falls in quintile g;
    set(x)   = {c : 1 - P_c(x) <= q[c, g(x)]}, and an empty set becomes {predicted class}.

Calibration objects are the out-of-fold predictions of the five-fold cross-fit, each seen as observed and
under three random removals of context blocks (the classifier is trained with blocks blanked, so missing
context must have calibration members too); within every pool each object weighs one and the new object
weighs one (hierarchical conformal, Lee, Barber & Willett 2023). Quintile edges come from the calibration
objects as observed; a cell with fewer than `min_cell` objects uses its class threshold. The non-empty
rule never costs set size: without it, exact coverage in the confident cells turns correct, nearly certain
predictions into empty sets.

Status: a RESEARCH option, not adopted (docs/uq_conditional_calibration.md). It restores coverage across energy
strata and under sparser photometry, but for larger sets and fewer singletons, and against the class-only sets run
at a lower alpha with the same mean set size its advantage is confined to the energy strata and is paid for partly
by SN CC and SLSN coverage. No table ships, so the mode is off unless conformal_atypical.json is exported
(scripts/export_review_threshold.py --atypical); it is then selected with prediction_sets(..., mode="atypicality")
or FLARE_SET_MODE=atypicality. Its intended use is the evaluation the FLARE paper (Sec. 9.2) asks for: calibrating
and scoring the sets on representative Fritz photometry.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILENAME = "conformal_atypical.json"
MODES = ("class", "atypicality")


def classifier_energy(raw_top_margins: np.ndarray) -> np.ndarray:
    """-logsumexp of the top-level margins; higher = less typical."""
    from scipy.special import logsumexp
    return -logsumexp(np.atleast_2d(raw_top_margins), axis=1)


def pool_weights(objs) -> np.ndarray:
    """1 / (rows of the same object in this pool): every distinct object of the pool weighs one."""
    _, inv, cnt = np.unique(np.asarray(objs), return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def weighted_quantile(scores, weights=None, alpha: float = 0.10) -> float:
    """Finite-sample (1 - alpha) quantile with the new object weighing one; +inf when unattainable.
    With unit weights this is the order statistic ceil((n + 1)(1 - alpha)) of flare.conformal."""
    s = np.asarray(scores, float)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, float)
    if not len(s):
        return np.inf
    o = np.argsort(s, kind="mergesort"); s, w = s[o], w[o]
    c = np.cumsum(w) / (w.sum() + 1.0)
    k = int(np.searchsorted(c, 1 - alpha, side="left"))
    return float(s[k]) if k < len(s) else np.inf


def _edges(energy, weights, n_cells):
    o = np.argsort(energy, kind="mergesort"); c = np.cumsum(weights[o]) / weights.sum()
    return np.interp(np.arange(1, n_cells) / n_cells, c, energy[o])


def fit_table(P, y, energy, obj, observed, alpha: float = 0.10, n_cells: int = 5, min_cell: int = 10,
              meta: dict | None = None) -> dict:
    """Calibration table from out-of-fold rows.

    P (n, C) probabilities, y (n,) labels, energy (n,), obj (n,) object index (an object appears once
    as observed and possibly under further context removals), observed (n,) bool for the as-observed row.
    """
    P, y, energy, obj = np.asarray(P, float), np.asarray(y, int), np.asarray(energy, float), np.asarray(obj)
    observed = np.asarray(observed, bool)
    C = P.shape[1]
    w_obs = pool_weights(obj[observed])
    edges = _edges(energy[observed], w_obs, n_cells)
    g = np.searchsorted(edges, energy, side="right")
    S = 1.0 - P
    q_class = np.empty(C); q = np.empty((C, n_cells)); n_obj = np.zeros((C, n_cells), int)
    for c in range(C):
        mc = y == c
        q_class[c] = weighted_quantile(S[mc, c], pool_weights(obj[mc]), alpha)
        for k in range(n_cells):
            m = mc & (g == k)
            n_obj[c, k] = len(np.unique(obj[m]))
            q[c, k] = weighted_quantile(S[m, c], pool_weights(obj[m]), alpha) if n_obj[c, k] >= min_cell else q_class[c]
    return dict(method="atypicality-stratified class-conditional LAC, dropout-augmented, non-empty", alpha=alpha,
                n_cells=n_cells, min_cell=min_cell, energy_edges=edges.tolist(),
                q=[[None if not np.isfinite(v) else float(v) for v in row] for row in q],
                q_class=[None if not np.isfinite(v) else float(v) for v in q_class],
                n_objects=n_obj.tolist(), n_calibration_objects=int(len(np.unique(obj))), **(meta or {}))


class AtypicalConformal:
    def __init__(self, table: dict, classes):
        self.classes = list(classes)
        self.alpha = float(table["alpha"])
        self.edges = np.asarray(table["energy_edges"], float)
        self.q = np.array([[np.inf if v is None else v for v in row] for row in table["q"]], float)
        self.meta = {k: v for k, v in table.items() if k not in ("q", "q_class", "n_objects")}

    def cells(self, energy) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(energy, float), side="right")

    def membership(self, P, energy, label) -> np.ndarray:
        """(n, C) bool sets; `label` (n,) is the predicted class index used by the non-empty rule."""
        P = np.atleast_2d(np.asarray(P, float)); g = self.cells(energy)
        s = (1.0 - P) <= self.q[:, g].T
        e = ~s.any(1)
        s[np.flatnonzero(e), np.asarray(label, int)[e]] = True
        return s


def load(path: str | Path, classes) -> AtypicalConformal | None:
    path = Path(path)
    return AtypicalConformal(json.loads(path.read_text()), classes) if path.exists() else None
