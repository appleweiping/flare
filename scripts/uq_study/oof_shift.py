"""Development side of the photometry-shift simulation.

Refits the five thin100 fold models of oof.py (identical folds, splits and seeds) and predicts each
held-out fold on the FULL light curves (cache h100), under the 16 context patterns. The result,
oofs/oof_thin100_deploy_h100.npz, pairs with oofs/oof_thin100.npz (same objects, same fold models,
sparse photometry): calibrate on the latter, evaluate on the former. Labels are stored for evaluation
only. A check confirms the refitted fold models reproduce oof_thin100.npz on the thin features.
Run: python oof_shift.py <repo> [fit_tag thin100] [deploy_tag h100]  (also h100 thin100 for the reverse shift)
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results


def main():
    repo = sys.argv[1]
    fit_tag = sys.argv[2] if len(sys.argv) > 2 else "thin100"
    dep_tag = sys.argv[3] if len(sys.argv) > 3 else "h100"
    sys.path.insert(0, repo); sys.path.insert(0, str(Path(repo) / "scripts")); sys.path.insert(0, str(HERE))
    import oof
    from train_early_cc import hierarchy_nocal
    from evaluate_unified import SIX
    from flare.config import HOST_COLS, GW_COLS, POS_COLS
    oof.SIX = list(SIX); oof.BLOCKS = [HOST_COLS, GW_COLS, POS_COLS, ["M_pseudo"]]
    args = SimpleNamespace(threads=4, rounds=2000)

    def pool_of(tag):
        D = pd.read_parquet(WORK / "cache" / f"X_{tag}.parquet")
        p = D[D.split.isin(["train", "val"])].reset_index(drop=True)
        return pd.concat([p[p.split == "train"], p[p.split == "val"]], ignore_index=True)

    thin, full = pool_of(fit_tag), pool_of(dep_tag)
    assert (thin.obj_id.values == full.obj_id.values).all()
    feats = [c for c in thin.columns if c not in ("obj_id", "split", "cls", "bts_type")]
    Xt, Xf = thin[feats].astype(float), full[feats].astype(float)
    y = np.array([SIX.index(c) for c in thin.cls]); n = len(y)
    ref = np.load(WORK / "oofs" / f"oof_{fit_tag}.npz")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    P = np.empty((16, n, 6)); L = np.empty((16, n), int); M = np.empty((16, n, 5)); fold_id = np.full(n, -1)
    for k, (tr0, te) in enumerate(folds):
        tr, es = train_test_split(tr0, test_size=.15, stratify=y[tr0], random_state=k)
        m = hierarchy_nocal(Xt, y, tr, es, args)
        chk = m.predict_proba(Xt.iloc[te][m.feature_names])
        dev = float(np.abs(chk - ref["P"][0, te]).max())
        P[:, te], L[:, te], M[:, te] = oof.predict_views(m, Xf.iloc[te]); fold_id[te] = k
        print(f"fold {k}: refit reproduces thin OOF to {dev:.2e}; acc on full curves {np.mean(L[0, te] == y[te]):.4f}", flush=True)
    assert (fold_id == ref["fold"]).all()
    np.savez_compressed(WORK / "oofs" / f"oof_{fit_tag}_deploy_{dep_tag}.npz", P=P, L=L, M=M, y=y, fold=fold_id,
                        obj_id=full.obj_id.values, X_nobs=Xf["n_obs_total"].to_numpy(), feature_names=np.array(feats))
    print("DONE")


if __name__ == "__main__":
    main()
