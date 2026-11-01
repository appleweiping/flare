"""Cross-fit out-of-fold predictions for the UQ study (the repo's own recipe, BTS benchmark only).

For a feature cache tag (h5, h10, h20, h100, thin100):
  * pool = benchmark train + val (8,922 known-class objects), in that order;
  * five StratifiedKFold(shuffle, seed 42) folds, as scripts/train_early_cc.py and train_full_cc.py;
    each fold model is scripts/train_early_cc.hierarchy_nocal (85/15 fit / early stop, random_state=k,
    context blocks blanked on 30% of fitting rows, class weights, tuned SLSN branch threshold);
  * each held-out object is predicted under all 16 context patterns: bit j of the pattern removes
    block j of (host, gaia_wise, ps1_position, M_pseudo); pattern 0 is the object as observed;
  * the final model is the same recipe fitted on the whole pool (85/15, random_state 42) and predicts
    the test split and the anomaly hold-out under the same 16 patterns;
  * with --deploy-tag, the final model also predicts the test split from another cache (the
    photometry-shift simulation: fitted and calibrated on thin100, deployed on h100).
Stored per row: P6 (16, n, 6) probabilities, the label (-1 for anomaly), the fold id, the
predicted label of the hierarchy rule, and the raw top-level margins (16, n, 5).
Run: python oof.py <repo> <cache_dir> <tag> <out_dir> [--deploy-tag h100] [--threads 4]
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

BLOCKS = None


def predict_views(model, X):
    """P6, rule labels and top margins under the 16 context-removal patterns."""
    n = len(X); P = np.empty((16, n, 6)); L = np.empty((16, n), int); M = np.empty((16, n, 5))
    for pat in range(16):
        Xp = X.copy()
        for j, cols in enumerate(BLOCKS):
            if pat >> j & 1:
                Xp[cols] = np.nan
        Xf = Xp[model.feature_names]
        P[pat] = model.predict_proba(Xf)
        L[pat] = [SIX.index(c) for c in model.predict(Xf)]
        M[pat] = model.top.predict(Xf, raw_score=True)
    return P, L, M


def main():
    global BLOCKS, SIX
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("cache"); ap.add_argument("tag"); ap.add_argument("out")
    ap.add_argument("--deploy-tag", default=None); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, a.repo); sys.path.insert(0, str(Path(a.repo) / "scripts"))
    from train_early_cc import hierarchy_nocal
    from evaluate_unified import SIX as _SIX
    from flare.config import HOST_COLS, GW_COLS, POS_COLS, BTS6
    SIX = list(_SIX); BLOCKS = [HOST_COLS, GW_COLS, POS_COLS, ["M_pseudo"]]
    args = SimpleNamespace(threads=a.threads, rounds=2000)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(Path(a.cache) / f"X_{a.tag}.parquet")
    feats = [c for c in D.columns if c not in ("obj_id", "split", "cls", "bts_type")]
    pool = D[D.split.isin(["train", "val"])].reset_index(drop=True)
    pool = pd.concat([pool[pool.split == "train"], pool[pool.split == "val"]], ignore_index=True)
    X, y = pool[feats].astype(float), np.array([SIX.index(c) for c in pool.cls]); n = len(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    P = np.empty((16, n, 6)); L = np.empty((16, n), int); M = np.empty((16, n, 5)); fold_id = np.full(n, -1)
    t0 = time.time()
    for k, (tr0, te) in enumerate(folds):
        tr, es = train_test_split(tr0, test_size=.15, stratify=y[tr0], random_state=k)
        m = hierarchy_nocal(X, y, tr, es, args)
        P[:, te], L[:, te], M[:, te] = predict_views(m, X.iloc[te]); fold_id[te] = k
        print(f"{a.tag} fold {k}: acc {np.mean(L[0, te] == y[te]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out / f"oof_{a.tag}.npz", P=P, L=L, M=M, y=y, fold=fold_id, obj_id=pool.obj_id.values,
                        X_nobs=X["n_obs_total"].to_numpy(), feature_names=np.array(feats))
    tr, es = train_test_split(np.arange(n), test_size=.15, stratify=y, random_state=42)
    final = hierarchy_nocal(X, y, tr, es, args)
    final.top.save_model(str(out / f"final_{a.tag}_top.txt")); final.branch.save_model(str(out / f"final_{a.tag}_branch.txt"))
    (out / f"final_{a.tag}_card.json").write_text(json.dumps({"branch_threshold": final.branch_threshold}))
    for split, tag in [("test", a.tag), ("anomaly", a.tag)] + ([("test", a.deploy_tag)] if a.deploy_tag else []):
        Dd = D if tag == a.tag else pd.read_parquet(Path(a.cache) / f"X_{tag}.parquet")
        S = Dd[Dd.split == split].reset_index(drop=True)
        Pt, Lt, Mt = predict_views(final, S[feats].astype(float))
        yt = np.array([SIX.index(c) if isinstance(c, str) and c in SIX else -1 for c in S.cls])
        name = f"final_{a.tag}_on_{split}" + ("" if tag == a.tag else f"_{tag}")
        np.savez_compressed(out / f"{name}.npz", P=Pt, L=Lt, M=Mt, y=yt, obj_id=S.obj_id.values,
                            X_nobs=S["n_obs_total"].to_numpy(), bts_type=S.bts_type.values.astype(str))
        print(name, "acc", np.mean(Lt[0][yt >= 0] == yt[yt >= 0]).round(4), flush=True)
    print("DONE", a.tag, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
