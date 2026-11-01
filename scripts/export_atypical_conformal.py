"""Export the atypicality-stratified sets (flare.atypical) and the review-trigger threshold for a model.

Same cross-fit as scripts/train_full_cc.py: five StratifiedKFold(shuffle, seed 42) folds of train+val, each
fold model the repository hierarchy (scripts/train_early_cc.hierarchy_nocal, 85/15 fit/early stop with
random_state = fold). For every held-out object the fold model predicts it as observed and under three
random removals of the context blocks (seed 2026); these out-of-fold rows give
  * conformal_atypical.json: class x energy-quintile thresholds (flare.atypical.fit_table);
  * review_p_threshold (written into conformal.json): the largest value tau whose leave-one-out
    known-class rate of "p-value of the predicted class <= tau" is at most 1%, on the as-observed rows,
    against the class calibration scores that conformal.json already ships.
Features are rebuilt from the light curves through the inference path (flare.analysis.observed_events and
feature_frame), so early horizons are prefix-causal; the full-curve LC columns equal features_all.parquet
up to the prefix-colour rebuild. The thresholds are as valid as the match between these fold models and the
weights in --models: run it with the same pool those weights were fitted on (the public benchmark pool is
the full-curve model's; the shipped early weights were fitted on a pool that also holds Fritz objects).
Run: python scripts/export_atypical_conformal.py --models models/bts6 [--horizon 100] [--threads 4]
"""
import argparse, hashlib, json, os, sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python scripts/setup_data.py
N_AUG, AUG_SEED, RATE = 3, 2026, 0.01


def _features(job):
    from flare.analysis import feature_frame, observed_events
    from flare.data import load_events
    fp, ctx, h = job
    f, _ = feature_frame(observed_events(load_events(str(ROOT / "data" / fp)), h), ctx)
    return f.iloc[0].to_dict()


def review_threshold(P, y, L, rate=RATE):
    """Largest tau with (1 + #{p_L <= tau}) <= rate (n + 1), p_L leave-one-out against the class scores."""
    S = 1.0 - P; n_obj = len(y)
    scores = [np.sort(S[y == c, c]) for c in range(P.shape[1])]
    pL = np.empty(n_obj)
    for c in range(P.shape[1]):
        r = np.flatnonzero(L == c)
        cal = scores[c]; n = len(cal)
        ge = n - np.searchsorted(cal, S[r, c], side="left")
        own = y[r] == c                                     # the object's own score is in its class list
        ge = ge - own.astype(int)
        pL[r] = (1.0 + ge) / np.where(own, n, n + 1.0)
    v = np.sort(pL); cnt = np.searchsorted(v, v, side="right")
    ok = (1 + cnt) <= rate * (n_obj + 1)
    return float(v[ok].max()) if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--horizon", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2000)
    a = ap.parse_args()
    from evaluate_unified import context_table, SIX
    from train_early_cc import hierarchy_nocal
    from flare.config import HOST_COLS, GW_COLS, POS_COLS
    from flare import atypical
    train = pd.read_csv(ROOT / "data/splits/train.csv"); val = pd.read_csv(ROOT / "data/splits/val.csv")
    pool = pd.concat([train, val], ignore_index=True); n = len(pool)
    y = np.array([SIX.index(c) for c in pool.cls])
    ctx = context_table(ROOT, pool.obj_id.values)
    jobs = [(r.filepath, {k: (None if pd.isna(v) else float(v)) for k, v in ctx.loc[r.obj_id].items()}, a.horizon)
            for r in pool.itertuples()]
    with Pool(a.workers) as p:
        X = pd.DataFrame(p.map(_features, jobs, chunksize=50))
    blocks = [HOST_COLS, GW_COLS, POS_COLS, ["M_pseudo"]]
    rng = np.random.default_rng(AUG_SEED)
    pats = np.stack([np.zeros(n, int)] + [rng.integers(1, 16, n) for _ in range(N_AUG)])     # (1 + N_AUG, n)
    args = SimpleNamespace(threads=a.threads, rounds=a.rounds)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    P = np.empty((1 + N_AUG, n, 6)); E = np.empty((1 + N_AUG, n)); L0 = np.empty(n, int)
    for k, (tr0, te) in enumerate(folds):
        tr, es = train_test_split(tr0, test_size=.15, stratify=y[tr0], random_state=k)
        m = hierarchy_nocal(X, y, tr, es, args)
        for v in range(1 + N_AUG):
            Xv = X.iloc[te].copy()
            for j, cols in enumerate(blocks):
                Xv.loc[(pats[v, te] >> j & 1).astype(bool), cols] = np.nan
            Xv = Xv[m.feature_names]
            P[v, te] = m.predict_proba(Xv); E[v, te] = m.classifier_energy(Xv)
            if v == 0:
                L0[te] = [SIX.index(c) for c in m.predict(Xv)]
        print(f"fold {k}: accuracy {np.mean(L0[te] == y[te]):.4f}", flush=True)
    obj = np.tile(np.arange(n), 1 + N_AUG); observed = np.r_[np.ones(n, bool), np.zeros(n * N_AUG, bool)]
    table = atypical.fit_table(P.reshape(-1, 6), np.tile(y, 1 + N_AUG), E.reshape(-1), obj, observed,
                               meta=dict(horizon_days=a.horizon, n_augmented_views=N_AUG, augmentation_seed=AUG_SEED,
                                         note="out-of-fold rows of the five-fold cross-fit of scripts/train_full_cc.py; "
                                              "see docs/uq_conditional_calibration.md",
                                         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    out = a.models
    (out / atypical.FILENAME).write_text(json.dumps(table, indent=1))
    tau = review_threshold(P[0], y, L0)
    conf_path = out / "conformal.json"; conf = json.loads(conf_path.read_text())
    conf["review_p_threshold"] = tau
    conf["review_p_threshold_note"] = ("anomaly_review fires when the p-value of the predicted class is <= min(this, alpha): "
                                       f"a {RATE:.0%} leave-one-out known-class rate per analysis on the {n} out-of-fold objects "
                                       "(scripts/export_atypical_conformal.py); the deployed rule before it was p <= alpha")
    conf_path.write_text(json.dumps(conf, indent=2, allow_nan=False))
    man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
    for f in (atypical.FILENAME, "conformal.json"):
        man[f] = hashlib.sha256((out / f).read_bytes()).hexdigest()
    man_path.write_text(json.dumps(dict(sorted(man.items())), indent=1))
    print(f"wrote {out / atypical.FILENAME}; review_p_threshold {tau:.5f}", flush=True)


if __name__ == "__main__":
    main()
