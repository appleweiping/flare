"""Nested cross-fit that mirrors the deployed construction (preregistration amendment 3).

The shipped models are calibrated on the out-of-fold scores of a five-fold cross-fit and then REFITTED
on the whole pool; the refit predicts new objects. The development protocol of the first two rounds
scored each fold model on its own held-out fold, so it could not see a calibration/deployment mismatch.
Here, for each outer fold k of the same five folds (seed 42):
  * pool_k = the other four folds; an inner StratifiedKFold(5, seed 42) cross-fit on pool_k with the
    repository recipe (scripts/train_early_cc.hierarchy_nocal) gives inner out-of-fold predictions;
  * the "final" model of pool_k is the same recipe on all of pool_k (85/15, random_state 42), exactly as
    scripts/train_full_cc.py builds the shipped weights; it predicts outer fold k;
  * each inner fold model also predicts outer fold k (for proper cross-conformal p-values).
All predictions under the 16 context patterns. Output oofs/nested_<tag>_k<k>.npz.
Run: python nested.py <repo> <tag> [--threads 4]
"""
import argparse, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("tag"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, a.repo); sys.path.insert(0, str(Path(a.repo) / "scripts")); sys.path.insert(0, str(HERE))
    import oof
    from train_early_cc import hierarchy_nocal
    from evaluate_unified import SIX
    from flare.config import HOST_COLS, GW_COLS, POS_COLS
    oof.SIX = list(SIX); oof.BLOCKS = [HOST_COLS, GW_COLS, POS_COLS, ["M_pseudo"]]
    args = SimpleNamespace(threads=a.threads, rounds=2000)
    D = pd.read_parquet(WORK / "cache" / f"X_{a.tag}.parquet")
    p = D[D.split.isin(["train", "val"])].reset_index(drop=True)
    pool = pd.concat([p[p.split == "train"], p[p.split == "val"]], ignore_index=True)
    feats = [c for c in pool.columns if c not in ("obj_id", "split", "cls", "bts_type")]
    X, y = pool[feats].astype(float), np.array([SIX.index(c) for c in pool.cls]); n = len(y)
    outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    t0 = time.time()
    for k, (pk, te) in enumerate(outer):
        yk = y[pk]; nk = len(pk)
        inner = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(nk), yk))
        Pin = np.empty((16, nk, 6)); Lin = np.empty((16, nk), int); Min = np.empty((16, nk, 5)); fin = np.full(nk, -1)
        Pj = np.empty((5, 16, len(te), 6)); Lj = np.empty((5, 16, len(te)), int); Mj = np.empty((5, 16, len(te), 5))
        for j, (tr0, he) in enumerate(inner):
            tr, es = train_test_split(tr0, test_size=.15, stratify=yk[tr0], random_state=j)
            m = hierarchy_nocal(X.iloc[pk], yk, tr, es, args)
            Pin[:, he], Lin[:, he], Min[:, he] = oof.predict_views(m, X.iloc[pk].iloc[he]); fin[he] = j
            Pj[j], Lj[j], Mj[j] = oof.predict_views(m, X.iloc[te])
        tr, es = train_test_split(np.arange(nk), test_size=.15, stratify=yk, random_state=42)
        fm = hierarchy_nocal(X.iloc[pk], yk, tr, es, args)
        Pf, Lf, Mf = oof.predict_views(fm, X.iloc[te])
        np.savez_compressed(WORK / "oofs" / f"nested_{a.tag}_k{k}.npz", pool_idx=pk, test_idx=te,
                            Pin=Pin, Lin=Lin, Min=Min, fold_in=fin, y_in=yk,
                            Pf=Pf, Lf=Lf, Mf=Mf, Pj=Pj, Lj=Lj, Mj=Mj, y_te=y[te],
                            obj_id=pool.obj_id.values, nobs=X["n_obs_total"].to_numpy())
        print(f"{a.tag} outer {k}: inner acc {np.mean(Lin[0] == yk):.4f} final acc {np.mean(Lf[0] == y[te]):.4f} "
              f"fold-avg acc {np.mean(Pj[:, 0].mean(0).argmax(1) == y[te]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", a.tag)


if __name__ == "__main__":
    main()
