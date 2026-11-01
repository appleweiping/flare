"""Review-trigger study T1 (docs/uq_preregistration.md, "Review triggers" and "Endpoints").

Deployed rule (flare/skyportal.py rule_verdict, the anomaly_review branch, sets and p-values exactly as
flare/skyportal.classify builds them):
    pair(x) = argmax_excluded_from_set(x)  OR  round(credibility(x), 4) < 0.1
    credibility = max_c p_c(x),  p_c(x) = (1 + #{cal scores of class c >= 1 - P_c(x)}) / (n_c + 1)
    (flare/hierarchical.py p_values; the deployed set is {c : p_c > alpha}, alpha = 0.1, which is the
    B0 set {c : 1 - P_c <= q_c} exactly; "argmax" in the payload is the hierarchy-rule label L).
Candidate T1: one statistic, p_L(x) = p_{L(x)}(x); flag iff p_L(x) <= tau, tau set on KNOWN calibration
objects for a 1% per-analysis rate.

Design choices, all fixed before anything below was run on the study data:
- Calibration scores: pattern-0 out-of-fold scores 1 - P_y of the calibration objects (as shipped),
  one sorted list per class, no weights (every calibration row is one object).
- Calibration objects' own statistic: leave-one-out p-values (the object's own pattern-0 score is
  removed from its class list, n_c - 1 others), so that the calibration statistics are exchangeable
  with a new object's p_L and are not inflated by the object counting itself.
- tau rule (finite-sample, conservative): tau = the largest calibration value v of p_L with
  (1 + #{i : p_L,i <= v}) <= 0.01 (n + 1); tau = 0 (no flags) if none. Flag iff p_L(x) <= tau.
  Under exchangeability P(flag) <= 1%; this is the conformal p-value of p_L among known objects <= 0.01.
- Development pool (LOFO): for held-out fold k, the conformal scores, the B0 sets of the deployed pair
  and tau all come from the other four folds; the held-out fold is predicted under the scenario
  pattern (observed: 0; missing: harness.scenario_patterns(n, 'missing'), seed 2026). tau is always
  set on pattern-0 calibration objects (registered; the shipped system has only pattern-0 scores).
- Out-of-taxonomy recall: final_<tag>_on_anomaly.npz (final model, 53 hold-out rarities) calibrated with
  ALL development out-of-fold pattern-0 scores (shipped cross-conformal); tau from the leave-one-out p_L
  of all 8,922 development objects at pattern 0. In 'missing' the 53 objects take
  harness.scenario_patterns(53, 'missing') patterns (seed 2026).
- ROC (primary, "the same ROC" for both rules): negatives = the 8,922 development objects with
  leave-one-out p-values against all development scores (the known population of the shipped
  calibration, the one tau is set on), under the scenario patterns; positives = the 53 rarities.
  Scores: 1 - p_L, 1 - credibility, and the pair boolean (ROC (0,0)-(r,R)-(1,1)). Reading at a rate
  between ROC vertices is linear (randomised tie-breaking); the pair at a lower rate is its chord,
  i.e. its recall scaled by rate ratio. A secondary ROC uses the LOFO held-out negatives instead.
- Adoption (registration): (1) T1's realised known-class flag rate on LOFO held-out folds, h100
  observed, in [0.5%, 2%]; (2) #rarities flagged by T1 at its tau >= #flagged by the pair x
  (r_T1 / r_pair), both rates read on the primary ROC's negatives. Decided in h100 'observed'; 'missing'
  and horizons 20/10/5 are reported as secondary and do not change the verdict.
- A paired bootstrap (2,000 draws; rarities resampled, negatives resampled within class) of
  count_T1 - count_pair x r_T1 / r_pair is reported descriptively (not part of the rule).
- Exploratory variant T1x_aug (labelled exploratory, not a candidate): identical, except that tau is
  set on the calibration objects each taken under ONE removal pattern drawn uniformly from the 16
  (rng seed 2026 + 7k + 1 for LOFO fold k, 2026 + 99 for the all-development calibration); the
  conformal scores stay pattern-0. Scored on h100 observed and missing only.

Run:  python t1_trigger.py            (self-test only)
      python t1_trigger.py --study    (self-test, then the study; writes results/T1.json)
Never reads the sealed test predictions.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
UQ = HERE.parent
import os as _os
REPO = HERE.parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))
for p in (str(UQ), str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
import harness as H  # noqa: E402

SET_ALPHA = H.ALPHA          # 0.10, the deployed set level and the credibility cut
CRED_CUT = 0.1
RATE = 0.01
N_BOOT = 2000


# ---------------------------------------------------------------- core statistics
def class_scores(P, y):
    """Sorted calibration scores 1 - P_y per class (list of 6 arrays)."""
    return [np.sort(1.0 - P[y == c, c]) for c in range(6)]


def p_values(P, scores, loo_y=None, loo_own=None):
    """(m,6) class-conditional conformal p-values (flare/hierarchical.py p_values).

    loo_y / loo_own: for calibration rows, the object's class (>= 0) and its own pattern-0 score; that
    score is removed from its class list for the column of its class (leave-one-out)."""
    S = 1.0 - np.asarray(P, float)
    out = np.empty_like(S)
    for c in range(6):
        cal = scores[c]; n = len(cal)
        ge = n - np.searchsorted(cal, S[:, c], side="left")
        out[:, c] = (1.0 + ge) / (n + 1.0)
        if loo_y is not None:
            r = np.flatnonzero(loo_y == c)
            if len(r):
                ge_r = ge[r] - (loo_own[r] >= S[r, c])
                out[r, c] = (1.0 + ge_r) / n            # n - 1 others
    return out


def choose_tau(stat, rate=RATE):
    """Largest calibration value v with (1 + #{stat <= v}) <= rate (n + 1); 0.0 if none."""
    v = np.sort(np.asarray(stat, float)); n = len(v)
    cnt = np.searchsorted(v, v, side="right")
    ok = (1 + cnt) <= rate * (n + 1)
    return float(v[ok].max()) if ok.any() else 0.0


def deployed_pair(pv, sets, L):
    """anomaly_review branch of rule_verdict: label excluded from its set OR rounded credibility < 0.1."""
    i = np.arange(len(L))
    excluded = ~sets[i, L]
    cred = np.round(pv.max(1), 4)
    return excluded | (cred < CRED_CUT), excluded, cred < CRED_CUT


class T1Trigger:
    """p-value of the predicted class, flagged at a 1% known-class rate."""
    wants_augmented = 0

    def __init__(self, rate=RATE):
        self.rate = rate

    def fit(self, cal, tau_view=None):
        """cal: pattern-0 calibration View (labels used). tau_view: optional View of the SAME objects
        (other patterns) on which tau is set; default cal itself."""
        assert (cal.pattern == 0).all() and (cal.y >= 0).all()
        self.scores = class_scores(cal.P, cal.y)
        self.n_c = np.array([len(s) for s in self.scores])
        own = np.full(int(cal.obj.max()) + 1, np.nan); own[cal.obj] = 1.0 - cal.P[np.arange(len(cal.y)), cal.y]
        ycl = np.full(len(own), -1); ycl[cal.obj] = cal.y
        self._own, self._ycl = own, ycl
        tv = cal if tau_view is None else tau_view
        pv = self.p_values_loo(tv)
        self.tau = choose_tau(pv[np.arange(len(tv.L)), tv.L], self.rate)
        return self

    def p_values(self, v):
        """p-values of new (blind) rows: no labels used."""
        return p_values(v.P, self.scores)

    def p_values_loo(self, v):
        """p-values of calibration objects (any pattern), own pattern-0 score left out."""
        return p_values(v.P, self.scores, loo_y=self._ycl[v.obj], loo_own=self._own[v.obj])

    def flags(self, v, pv=None):
        pv = self.p_values(v) if pv is None else pv
        return pv[np.arange(len(v.L)), v.L] <= self.tau


# ---------------------------------------------------------------- ROC reading
def roc(neg, pos):
    """ROC of a score (higher = flag) with ties as linear segments; returns fpr, tpr from (0,0) to (1,1)."""
    neg = np.asarray(neg, float); pos = np.asarray(pos, float)
    thr = np.unique(np.concatenate([neg, pos]))[::-1]
    ns, ps = np.sort(neg), np.sort(pos)
    fpr = (len(ns) - np.searchsorted(ns, thr, side="left")) / len(ns)
    tpr = (len(ps) - np.searchsorted(ps, thr, side="left")) / len(ps)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def read_at(fpr, tpr, r):
    """TPR at FPR = r on the piecewise-linear ROC (top of a vertical segment at exactly r)."""
    j = np.flatnonzero(fpr <= r + 1e-15)[-1]
    if j == len(fpr) - 1 or fpr[j] >= r - 1e-15:
        return float(tpr[j])
    k = j + 1
    return float(tpr[j] + (tpr[k] - tpr[j]) * (r - fpr[j]) / (fpr[k] - fpr[j]))


def auc(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def compress(fpr, tpr, max_fpr=0.25):
    """ROC vertices up to max_fpr where tpr changes (for the JSON)."""
    keep = np.r_[True, np.diff(tpr) > 0] | np.r_[np.diff(tpr) > 0, True]
    keep &= fpr <= max_fpr
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(fpr[keep], tpr[keep])]


# ---------------------------------------------------------------- LOFO on the development pool
def lofo_triggers(S, scenario, aug_tau=False, seed=2026):
    n = S.n; pats = H.scenario_patterns(n, scenario, seed)
    pL = np.zeros(n); cred = np.zeros(n); t1 = np.zeros(n, bool); pair = np.zeros(n, bool)
    excl = np.zeros(n, bool); credlow = np.zeros(n, bool); empty = np.zeros(n, bool); L = np.zeros(n, int)
    cov = np.zeros(n, bool); nob = np.zeros(n, int); taus = []; floors = []
    for k in range(5):
        ci, te = np.flatnonzero(S.fold != k), np.flatnonzero(S.fold == k)
        cal = S.view(ci, np.zeros(len(ci), int))
        tv_tau = None
        if aug_tau:
            tv_tau = S.view(ci, np.random.default_rng(seed + 7 * k + 1).integers(0, 16, len(ci)))
        m = T1Trigger().fit(cal, tv_tau); taus.append(m.tau); floors.append((1.0 / (m.n_c + 1)).tolist())
        b0 = H.B0(); b0.fit(cal)
        tv = S.view(te, pats[te]); blind = tv.blind()
        pv = m.p_values(blind); sets = b0.sets(blind)
        assert ((pv > SET_ALPHA) == sets).all(), "deployed set {p_c > alpha} != B0 set"
        i = np.arange(len(te))
        pL[te] = pv[i, blind.L]; cred[te] = pv.max(1); L[te] = blind.L
        t1[te] = m.flags(blind, pv)
        pair[te], excl[te], credlow[te] = deployed_pair(pv, sets, blind.L)
        empty[te] = ~sets.any(1); cov[te] = sets[i, S.y[te]]
        nob[te] = H.groups(tv, H.group_edges(cal))["nobs"]
    assert (pair == (pL <= SET_ALPHA)).all(), "pair is not p_L <= 0.1"
    return dict(pL=pL, cred=cred, t1=t1, pair=pair, excl=excl, credlow=credlow, empty=empty, L=L,
                cov=cov, nobs_grp=nob, y=S.y, taus=taus, floors=floors, pats=pats)


def rate_block(r, flag):
    y, L = r["y"], r["L"]
    return dict(rate=float(flag.mean()), n_flag=int(flag.sum()),
                per_true_class={H.SIX[c]: float(flag[y == c].mean()) for c in range(6)},
                per_pred_class={H.SIX[c]: float(flag[L == c].mean()) for c in range(6) if (L == c).any()},
                per_nobs_tercile={int(g): float(flag[r["nobs_grp"] == g].mean()) for g in range(3)},
                per_fold=None)


def summarize_lofo(r, S):
    out = {}
    for name, flag in (("T1", r["t1"]), ("pair", r["pair"])):
        b = rate_block(r, flag)
        b["per_fold"] = [float(flag[S.fold == k].mean()) for k in range(5)]
        out[name] = b
    pf = r["pair"]
    out["pair_breakdown"] = dict(
        frac_flags_empty_set=float(r["empty"][pf].mean()),
        frac_flags_credibility_lt_0p1=float(r["credlow"][pf].mean()),
        frac_flags_nonempty_label_excluded=float((pf & ~r["empty"])[pf].mean()),
        credibility_flags_subset_of_exclusion=bool((~r["credlow"] | r["excl"]).all()),
        pair_identical_to_pL_le_0p1=bool((r["pair"] == (r["pL"] <= SET_ALPHA)).all()),
        frac_flags_label_correct=float((r["L"] == r["y"])[pf].mean()))
    out["T1_breakdown"] = dict(frac_flags_empty_set=float(r["empty"][r["t1"]].mean()) if r["t1"].any() else None,
                               T1_subset_of_pair=bool((~r["t1"] | r["pair"]).all()),
                               frac_flags_label_correct=float((r["L"] == r["y"])[r["t1"]].mean()) if r["t1"].any() else None)
    out["tau_per_fold"] = r["taus"]
    out["p_value_floor_per_class_fold0"] = dict(zip(H.SIX, r["floors"][0]))
    out["unflaggable_frac_pred_floor_gt_tau"] = _unflaggable(r, S)
    out["per_pred_class_min_pL"] = {H.SIX[c]: float(r["pL"][r["L"] == c].min()) for c in range(6) if (r["L"] == c).any()}
    out["B0_marginal_coverage_check"] = float(r["cov"].mean())
    out["rate_le_0p01_check_pL"] = float((r["pL"] <= 0.01).mean())
    return out


def _unflaggable(r, S):
    """Fraction of held-out objects whose predicted class's p-value floor 1/(n_L+1) exceeds tau."""
    u = np.zeros(S.n, bool)
    for k in range(5):
        te = S.fold == k
        n_c = np.bincount(S.y[S.fold != k], minlength=6)
        u[te] = 1.0 / (n_c[r["L"][te]] + 1) > r["taus"][k]
    return float(u.mean())


# ---------------------------------------------------------------- out-of-taxonomy recall
def anomaly_triggers(S, tag, scenario, aug_tau=False, seed=2026):
    """All-development calibration (shipped cross-conformal) applied to the 53 rarities."""
    n = S.n; cal = S.view(np.arange(n), np.zeros(n, int))
    tv_tau = S.view(np.arange(n), np.random.default_rng(seed + 99).integers(0, 16, n)) if aug_tau else None
    m = T1Trigger().fit(cal, tv_tau)
    # negatives: development objects, leave-one-out p-values, scenario patterns
    neg_v = S.view(np.arange(n), H.scenario_patterns(n, scenario, seed))
    pvn = m.p_values_loo(neg_v); i = np.arange(n)
    neg = dict(pL=pvn[i, neg_v.L], cred=pvn.max(1))
    neg["pair"] = (neg["pL"] <= SET_ALPHA) | (np.round(neg["cred"], 4) < CRED_CUT)
    neg["t1"] = neg["pL"] <= m.tau
    # positives
    z = np.load(H.OOF / f"final_{tag}_on_anomaly.npz", allow_pickle=True)
    na = len(z["y"]); pa = H.scenario_patterns(na, scenario, seed); ia = np.arange(na)
    va = SimpleNamespace(P=z["P"][pa, ia], L=z["L"][pa, ia])
    b0 = H.B0(); b0.fit(cal)
    pva = m.p_values(va); sets = b0.sets(va)
    assert ((pva > SET_ALPHA) == sets).all()
    pos = dict(pL=pva[ia, va.L], cred=pva.max(1), L=va.L, empty=~sets.any(1), sets=sets)
    pos["pair"], pos["excl"], pos["credlow"] = deployed_pair(pva, sets, va.L)
    pos["t1"] = m.flags(va, pva)
    assert (pos["pair"] == (pos["pL"] <= SET_ALPHA)).all()
    return m, neg, pos, z["bts_type"]


def matched_rate(neg, pos, r_T1, r_pair, n_pos):
    """ROC of 1-p_L, 1-credibility, pair boolean; recall at the pair's rate and at 1%."""
    res = {}
    curves = {"pL": roc(-neg["pL"], -pos["pL"]), "credibility": roc(-neg["cred"], -pos["cred"]),
              "pair": roc(neg["pair"].astype(float), pos["pair"].astype(float))}
    for name, (f, t) in curves.items():
        res[name] = dict(auc=auc(f, t),
                         recall_at_pair_rate=read_at(f, t, r_pair), count_at_pair_rate=read_at(f, t, r_pair) * n_pos,
                         recall_at_1pct=read_at(f, t, RATE), count_at_1pct=read_at(f, t, RATE) * n_pos,
                         recall_at_T1_rate=read_at(f, t, r_T1), count_at_T1_rate=read_at(f, t, r_T1) * n_pos,
                         roc_vertices_fpr_le_0p25=compress(f, t))
    return res


def boot_diff(neg, pos, y, n_boot=N_BOOT, seed=0):
    """count_T1 - count_pair * r_T1 / r_pair, rarities resampled and negatives resampled within class."""
    rng = np.random.default_rng(seed); n = len(y); na = len(pos["t1"])
    byc = [np.flatnonzero(y == c) for c in range(6)]
    nt, np_ = neg["t1"].astype(float), neg["pair"].astype(float)
    pt, pp = pos["t1"].astype(float), pos["pair"].astype(float)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in byc])
        w = np.bincount(idx, minlength=n)
        ja = rng.integers(0, na, na)
        rt, rp = (w @ nt) / n, (w @ np_) / n
        d[b] = pt[ja].sum() - pp[ja].sum() * (rt / rp if rp > 0 else 0.0)
    return dict(lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), frac_ge_0=float((d >= 0).mean()))


def anomaly_block(S, tag, scenario, lofo_r, aug_tau=False):
    m, neg, pos, btype = anomaly_triggers(S, tag, scenario, aug_tau)
    na = len(pos["pL"])
    r_T1, r_pair = float(neg["t1"].mean()), float(neg["pair"].mean())
    c_T1, c_pair = int(pos["t1"].sum()), int(pos["pair"].sum())
    out = dict(tau_all_dev=m.tau, n_anomaly=na, n_cal_per_class=m.n_c.tolist(),
               p_value_floor_per_class=dict(zip(H.SIX, (1.0 / (m.n_c + 1)).tolist())),
               primary_negatives="development objects, leave-one-out p-values vs all development scores",
               negative_rate_T1=r_T1, negative_rate_pair=r_pair,
               T1_count=c_T1, T1_recall=c_T1 / na, pair_count=c_pair, pair_recall=c_pair / na,
               pair_count_scaled_to_T1_rate=c_pair * r_T1 / r_pair if r_pair > 0 else None,
               pair_breakdown=dict(frac_flags_empty_set=float(pos["empty"][pos["pair"]].mean()) if pos["pair"].any() else None,
                                   n_credibility_lt_0p1=int(pos["credlow"].sum()), n_label_excluded=int(pos["excl"].sum()),
                                   n_empty_sets=int(pos["empty"].sum())),
               predicted_class_counts={H.SIX[c]: int((pos["L"] == c).sum()) for c in range(6)},
               unflaggable_by_T1_floor_gt_tau=int((1.0 / (m.n_c[pos["L"]] + 1) > m.tau).sum()),
               min_pL=float(pos["pL"].min()), median_pL=float(np.median(pos["pL"])),
               # p_c at P_c = 1/6, the smallest probability an argmax label can have: a structural lower
               # bound of p_L for objects whose label is the argmax (diagnostic, not used by any rule)
               argmax_bound_pL_per_class=dict(zip(H.SIX, p_values(np.full((1, 6), 1 / 6), m.scores)[0].tolist())),
               per_pred_class={H.SIX[c]: dict(n=int((pos["L"] == c).sum()), T1=int(pos["t1"][pos["L"] == c].sum()),
                                              pair=int(pos["pair"][pos["L"] == c].sum()),
                                              min_pL=float(pos["pL"][pos["L"] == c].min()))
                               for c in range(6) if (pos["L"] == c).any()},
               per_type={t: dict(n=int((btype == t).sum()), T1=int(pos["t1"][btype == t].sum()),
                                 pair=int(pos["pair"][btype == t].sum())) for t in np.unique(btype)})
    out["roc_primary"] = matched_rate(neg, pos, r_T1, r_pair, na)
    out["boot_count_T1_minus_scaled_pair"] = boot_diff(neg, pos, S.y)
    out["T1_ge_scaled_pair_primary"] = bool(c_T1 >= out["pair_count_scaled_to_T1_rate"] - 1e-9)
    if lofo_r is not None:   # secondary ROC: LOFO held-out negatives
        lt, lp = float(lofo_r["t1"].mean()), float(lofo_r["pair"].mean())
        negl = dict(pL=lofo_r["pL"], cred=lofo_r["cred"], pair=lofo_r["pair"], t1=lofo_r["t1"])
        sec = matched_rate(negl, pos, lt, lp, na)
        for v in sec.values():
            v.pop("roc_vertices_fpr_le_0p25")
        out["roc_secondary_lofo_negatives"] = dict(
            rate_T1=lt, rate_pair=lp, pair_count_scaled_to_T1_rate=c_pair * lt / lp if lp > 0 else None,
            T1_ge_scaled_pair=bool(c_T1 >= c_pair * lt / lp - 1e-9) if lp > 0 else None, curves=sec)
    return out


# ---------------------------------------------------------------- study
def main():
    t0 = time.time()
    out = dict(study="T1 review trigger", rate_target=RATE, set_alpha=SET_ALPHA, credibility_cut=CRED_CUT,
               registered="T1", scenarios={}, exploratory={})
    b0json = json.loads((WORK / "results" / "B0.json").read_text())["scenarios"]
    for tag, scen in [("h100", "observed"), ("h100", "missing"), ("h20", "observed"), ("h10", "observed"), ("h5", "observed")]:
        S = H.Setting(H.OOF / f"oof_{tag}.npz", tag)
        key = f"{tag}_{scen}"
        r = lofo_triggers(S, scen)
        blk = dict(lofo=summarize_lofo(r, S))
        assert abs(blk["lofo"]["B0_marginal_coverage_check"] - b0json[key]["marginal"]) < 1e-12, key
        blk["anomaly"] = anomaly_block(S, tag, scen, r)
        out["scenarios"][key] = blk
        L, A = blk["lofo"], blk["anomaly"]
        print(f"{key:14s} LOFO rate T1 {L['T1']['rate']:.4f} pair {L['pair']['rate']:.4f} (empty {L['pair_breakdown']['frac_flags_empty_set']:.3f}) "
              f"| tau_all {A['tau_all_dev']:.5f} anomalies T1 {A['T1_count']}/{A['n_anomaly']} pair {A['pair_count']} "
              f"scaled {A['pair_count_scaled_to_T1_rate']:.2f} AUC pL {A['roc_primary']['pL']['auc']:.3f} "
              f"cred {A['roc_primary']['credibility']['auc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if tag == "h100":   # exploratory: tau on randomly-removed calibration views
            rx = lofo_triggers(S, scen, aug_tau=True)
            bx = dict(lofo=summarize_lofo(rx, S), anomaly=anomaly_block(S, tag, scen, rx, aug_tau=True))
            out["exploratory"][f"T1x_aug_{key}"] = bx
            print(f"  [exploratory T1x_aug] LOFO rate {bx['lofo']['T1']['rate']:.4f} anomalies {bx['anomaly']['T1_count']}", flush=True)
    o = out["scenarios"]["h100_observed"]
    rate_ok = 0.005 <= o["lofo"]["T1"]["rate"] <= 0.02
    recall_ok = o["anomaly"]["T1_ge_scaled_pair_primary"]
    out["adoption"] = dict(scenario="h100_observed", realised_rate=o["lofo"]["T1"]["rate"], rate_in_window=bool(rate_ok),
                           T1_count=o["anomaly"]["T1_count"], pair_count_scaled=o["anomaly"]["pair_count_scaled_to_T1_rate"],
                           recall_ok=bool(recall_ok), adopted=bool(rate_ok and recall_ok),
                           secondary_missing=dict(
                               realised_rate=out["scenarios"]["h100_missing"]["lofo"]["T1"]["rate"],
                               rate_in_window=bool(0.005 <= out["scenarios"]["h100_missing"]["lofo"]["T1"]["rate"] <= 0.02),
                               recall_ok=out["scenarios"]["h100_missing"]["anomaly"]["T1_ge_scaled_pair_primary"]))
    (WORK / "results").mkdir(exist_ok=True)
    (WORK / "results" / "T1.json").write_text(json.dumps(out, indent=1, default=float))
    print("adoption", out["adoption"]); print("wrote", WORK / "results" / "T1.json")


# ---------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.default_rng(0)
    # 1. p-values against brute force, with ties (scores on a coarse grid)
    P = rng.dirichlet(np.ones(6), 400); P = np.round(P, 2); P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(6, p=p) for p in P])
    sc = class_scores(P, y)
    Pt = np.round(rng.dirichlet(np.ones(6), 50), 2)
    pv = p_values(Pt, sc)
    for c in range(6):
        brute = (1 + (sc[c][None, :] >= (1 - Pt[:, c])[:, None]).sum(1)) / (len(sc[c]) + 1)
        assert np.allclose(pv[:, c], brute)
    # 2. deployed set {p > 0.1} is the B0 LAC set
    q = np.array([H.lac_quantile(1 - P[y == c, c]) for c in range(6)])
    assert ((pv > SET_ALPHA) == ((1 - Pt) <= q[None, :])).all()
    # 3. leave-one-out p-values equal recomputation without the object
    own = 1 - P[np.arange(len(y)), y]
    pl = p_values(P, sc, loo_y=y, loo_own=own)
    for i in rng.choice(len(y), 25, replace=False):
        keep = np.arange(len(y)) != i
        ref = p_values(P[i:i + 1], class_scores(P[keep], y[keep]))[0]
        assert np.allclose(pl[i], ref), i
    # 4. tau rule: largest admissible value, next value inadmissible
    st = rng.random(1000); tau = choose_tau(st, 0.01)
    assert 1 + (st <= tau).sum() <= 0.01 * 1001 and 1 + (st <= np.sort(st)[(st <= tau).sum()]).sum() > 0.01 * 1001
    assert choose_tau(np.ones(50), 0.01) == 0.0
    # 5. the pair is p_L <= 0.1 (credibility < 0.1 implies an empty set)
    Lt = Pt.argmax(1); sets = pv > SET_ALPHA
    pair, excl, cl = deployed_pair(pv, sets, Lt)
    assert (pair == (pv[np.arange(len(Lt)), Lt] <= SET_ALPHA)).all() and (~cl | ~sets.any(1)).all()
    # 6. ROC reading
    f, t = roc(np.array([0., 0, 0, 1]), np.array([1., 1, 0, 0]))   # boolean: (0.25, 0.5)
    assert abs(read_at(f, t, 0.25) - 0.5) < 1e-12 and abs(read_at(f, t, 0.125) - 0.25) < 1e-12
    assert abs(auc(f, t) - 0.625) < 1e-12
    f, t = roc(np.arange(100.), np.array([1000.]))
    assert read_at(f, t, 0.0) == 1.0 and auc(f, t) == 1.0
    # 7. realised rate on exchangeable synthetic data (calibrated model, labels drawn from P)
    n = 20000; P = rng.dirichlet(np.full(6, 0.6), n); y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    v = H.View(P=P, L=P.argmax(1), E=np.zeros(n), nobs=np.ones(n), present=np.ones((n, 4), bool),
               pattern=np.zeros(n, int), y=y, obj=np.arange(n), w=np.ones(n))
    cal, te = v.take(np.arange(0, n, 2)), v.take(np.arange(1, n, 2))
    m = T1Trigger().fit(cal)
    r = m.flags(te.blind()).mean()
    assert 0.006 <= r <= 0.014, r
    print(f"self-test ok (synthetic realised rate {r:.4f}, tau {m.tau:.4f})")


if __name__ == "__main__":
    _selftest()
    if "--study" in sys.argv:
        main()
