"""Nested (deployment-faithful) evaluation, preregistration amendment 3.

For each outer fold k (oofs/nested_<tag>_k<k>.npz): calibration = inner out-of-fold rows of the other four
folds; prediction = the refitted final model (D0, D0ne, M6 on final) or the inner fold models (CC, CC6).
Evaluation cells are method-independent: nobs tercile and the four context bits as before, and the
atypicality quintile of the deployed final model's energy ranked within the evaluated outer fold.
    python nested_eval.py <tag> [--boot 2000]
Writes results_nested/<tag>.json.
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "methods")); sys.path.insert(0, str(REPO))
import numpy as np
from scipy.special import logsumexp
from harness import (View, lac_quantile, pool_weights, presence_from_cache, scenario_patterns, evaluate,
                     paired_bootstrap_U, ALPHA, BLOCKS, MIN_CELL)
from m6_atyp_nonempty import M6

N_AUG, MIN_OBJ = 3, 10


def removed(pats):
    return ((np.asarray(pats)[:, None] >> np.arange(4)[None, :]) & 1).astype(bool)


def energy(M):
    return -logsumexp(M, axis=-1)


class Fold:
    def __init__(self, z, present0, pats_test, seed):
        self.z = z; self.pk, self.te = z["pool_idx"], z["test_idx"]
        self.yin, self.yte, self.fin = z["y_in"], z["y_te"], z["fold_in"]
        self.nobs = z["nobs"]; self.present0 = present0
        self.pt = pats_test                                   # test patterns (scenario)
        rng = np.random.default_rng(seed)
        nk = len(self.pk)
        # calibration rows: pattern 0 plus N_AUG random removal patterns per inner OOF object
        self.cal_pat = np.concatenate([np.zeros(nk, int)] + [rng.integers(1, 16, nk) for _ in range(N_AUG)])
        self.cal_row = np.tile(np.arange(nk), 1 + N_AUG)

    # ---- views
    def cal_view(self, aug):
        r = self.cal_row if aug else np.arange(len(self.pk)); p = self.cal_pat if aug else np.zeros(len(self.pk), int)
        Z = self.z
        pres = self.present0[self.pk[r]] & ~removed(p)
        return View(P=Z["Pin"][p, r], L=Z["Lin"][p, r], E=energy(Z["Min"][p, r]), nobs=self.nobs[self.pk[r]], present=pres,
                    pattern=p, y=self.yin[r], obj=r, w=np.full(len(r), 1.0 / (1 + (N_AUG if aug else 0))))

    def final_view(self):
        Z, i = self.z, np.arange(len(self.te))
        pres = self.present0[self.te] & ~removed(self.pt)
        return View(P=Z["Pf"][self.pt, i], L=Z["Lf"][self.pt, i], E=energy(Z["Mf"][self.pt, i]), nobs=self.nobs[self.te],
                    present=pres, pattern=self.pt, y=self.yte, obj=i, w=np.ones(len(i)))


def eval_groups(f: Fold):
    v = f.final_view()
    g = dict(nobs=np.searchsorted(np.quantile(v.nobs, [1 / 3, 2 / 3]), v.nobs, side="right"),
             atyp=np.searchsorted(np.quantile(v.E, [.2, .4, .6, .8]), v.E, side="right"))
    for j, b in enumerate(BLOCKS):
        g[f"ctx_{b}"] = v.present[:, j].astype(int)
    return g, v


# ---------------------------------------------------------------- set constructions
def d0(f, nonempty=False):
    cal = f.cal_view(False); v = f.final_view()
    q = np.array([lac_quantile(1 - cal.P[cal.y == c, c]) for c in range(6)])
    s = (1 - v.P) <= q[None, :]
    if nonempty:
        e = ~s.any(1); s[np.flatnonzero(e), v.L[e]] = True
    return s


def m6_on_final(f):
    m = M6(aug=N_AUG); m.fit(f.cal_view(True)); return m.sets(f.final_view().blind())


def cross_conformal(f, cells=False, aug=False, nonempty=False):
    """p_c(x) = (w_x + sum_j sum_{i in I_j, y_i=c, cell_j(i)=cell_j(x)} w_i [s_i >= s_j(x,c)]) / (w_x + sum of those w_i)."""
    Z = f.z; nt = len(f.te); it = np.arange(nt)
    rows = f.cal_row if aug else np.arange(len(f.pk)); pats = f.cal_pat if aug else np.zeros(len(f.pk), int)
    yr = f.yin[rows]; jr = f.fin[rows]
    sc = 1 - Z["Pin"][pats, rows, yr]                                   # calibration scores (own model j = fold of the row)
    Er = energy(Z["Min"][pats, rows])
    G = 5 if cells else 1
    # quintile edges per inner model from its own out-of-fold objects as observed (pattern 0)
    edges = [np.quantile(energy(Z["Min"][0, np.flatnonzero(f.fin == j)]), [.2, .4, .6, .8]) for j in range(5)]
    gr = np.array([np.searchsorted(edges[j], e, side="right") for j, e in zip(jr, Er)]) if cells else np.zeros(len(rows), int)
    # per-pool object weights, pool = (class, cell)
    w = np.empty(len(rows))
    for c in range(6):
        for g in range(G):
            m = (yr == c) & (gr == g)
            if m.any():
                w[m] = pool_weights(rows[m]) if aug else 1.0
    wclass = np.empty(len(rows))
    for c in range(6):
        m = yr == c; wclass[m] = pool_weights(rows[m]) if aug else 1.0
    num = np.zeros((nt, 6)); den = np.zeros((nt, 6))
    numc = np.zeros((nt, 6)); denc = np.zeros((nt, 6))
    nobj = np.zeros((nt, 6))
    for j in range(5):
        Pj = Z["Pj"][j][f.pt, it]; Ej = energy(Z["Mj"][j][f.pt, it])
        gx = np.searchsorted(edges[j], Ej, side="right") if cells else np.zeros(nt, int)
        sx = 1 - Pj
        for c in range(6):
            mj = (jr == j) & (yr == c)
            # class level (fallback)
            o = np.argsort(sc[mj]); s_sorted = sc[mj][o]; w_sorted = wclass[mj][o]; cw = np.r_[0, np.cumsum(w_sorted)]
            k = np.searchsorted(s_sorted, sx[:, c], side="left"); numc[:, c] += cw[-1] - cw[k]; denc[:, c] += cw[-1]
            if cells:
                for g in range(G):
                    mg = mj & (gr == g); sel = gx == g
                    if not sel.any():
                        continue
                    o = np.argsort(sc[mg]); s_sorted = sc[mg][o]; w_sorted = w[mg][o]; cw = np.r_[0, np.cumsum(w_sorted)]
                    k = np.searchsorted(s_sorted, sx[sel, c], side="left")
                    num[sel, c] += cw[-1] - cw[k]; den[sel, c] += cw[-1]
                    nobj[sel, c] += len(np.unique(rows[mg]))
    if cells:
        use_cell = nobj >= MIN_OBJ
        p = np.where(use_cell, (1 + num) / (1 + den), (1 + numc) / (1 + denc))
    else:
        p = (1 + numc) / (1 + denc)
    s = p > ALPHA
    if nonempty:
        Lf = Z["Lf"][f.pt, it]; e = ~s.any(1); s[np.flatnonzero(e), Lf[e]] = True
    return s


METHODS = {
    "D0": lambda f: d0(f),
    "D0_nonempty": lambda f: d0(f, nonempty=True),
    "M6_on_final": m6_on_final,
    "CC": lambda f: cross_conformal(f),
    "CC_nonempty": lambda f: cross_conformal(f, nonempty=True),
    "CC6": lambda f: cross_conformal(f, cells=True, aug=True, nonempty=True),
    "CC6_noaug": lambda f: cross_conformal(f, cells=True, aug=False, nonempty=True),
}


def summary(res):
    x = dict(res); x.pop("cells"); x["worst_cell"] = list(x["worst_cell"]) if x["worst_cell"] else None; return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    z0 = np.load(WORK / "oofs" / f"nested_{a.tag}_k0.npz", allow_pickle=True)
    present0 = presence_from_cache(a.tag, z0["obj_id"]); n = len(z0["obj_id"])
    out = {}
    for sc in ("observed", "missing"):
        pats_all = scenario_patterns(n, sc, 2026)
        per = {m: [] for m in METHODS}; G = {}; Y = []; V = []
        for k in range(5):
            z = np.load(WORK / "oofs" / f"nested_{a.tag}_k{k}.npz", allow_pickle=True)
            f = Fold(z, present0, pats_all[z["test_idx"]], seed=31337 + 11 * k)
            g, v = eval_groups(f)
            for name, fn in METHODS.items():
                per[name].append(fn(f))
            for kk, lab in g.items():
                G.setdefault(kk, []).append(lab)
            Y.append(f.yte); V.append(v)
        y = np.concatenate(Y); groups_ = {kk: np.concatenate(l) for kk, l in G.items()}
        vv = View(**{fld: np.concatenate([getattr(x, fld) for x in V]) for fld in View.__dataclass_fields__})
        res = {}
        for name in METHODS:
            sets = np.concatenate(per[name]); r, cov, size = evaluate(sets, vv, groups_)
            res[name] = dict(res=r, cov=cov, size=size, groups=groups_, y=y)
        out[sc] = {}
        for name in METHODS:
            x = summary(res[name]["res"])
            if name != "D0":
                x["boot_U_vs_D0"] = paired_bootstrap_U(res[name], res["D0"], a.boot)
            out[sc][name] = x
            print(f"{a.tag} {sc:8s} {name:12s} marg {x['marginal']:.4f} size {x['size']:.3f} single {x['singleton']:.3f} "
                  f"sacc {x['singleton_acc']:.3f} empty {x['empty']:.4f} U {x['U']:.3f} worst {x['worst_cell']} "
                  + (f"dU [{x['boot_U_vs_D0']['lo']:+.3f},{x['boot_U_vs_D0']['hi']:+.3f}]" if 'boot_U_vs_D0' in x else ""), flush=True)
            print(f"{'':32s} per class " + " ".join(f"{c}:{v:.3f}" for c, v in x["per_class"].items()), flush=True)
    (WORK / "results_nested").mkdir(exist_ok=True)
    (WORK / "results_nested" / f"{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
