"""Prefix-causal feature caches for the UQ study, built from the public benchmark light curves.

For every benchmark object (train / val / test / anomaly), and for horizons 5, 10, 20 and 100 days,
the events are truncated with flare.analysis.observed_events and the features (LC + context, with
M_pseudo recomputed from the observed peak) come from flare.analysis.feature_frame, i.e. exactly the
inference path of the unified API, so colours are rebuilt from the observed prefix.

A "thin" variant at h=100 keeps a seeded random 30% of the detections of each object (at least 3 per
g/r band where the object has them). It is the sparse-calibration side of the photometry-shift
simulation: the benchmark light curves have a median of 26 detections, the photometry that Fritz
supplies has 93 (FLARE paper Sec. 9.2), a ratio of about 0.28.

Output: <out>/X_<tag>.parquet with obj_id, split, cls (NaN-free label for known classes) columns.
Run: python build_cache.py <repo> <paper_data> <out> [--workers 4]
"""
import argparse, sys, hashlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

A = None


def init(repo, root):
    global A
    sys.path.insert(0, str(repo))
    from flare.analysis import feature_frame, observed_events
    from flare.data import load_events, COL
    sys.path.insert(0, str(Path(repo) / "scripts"))
    from evaluate_unified import context_table
    A = dict(ff=feature_frame, oe=observed_events, le=load_events, COL=COL, root=Path(root), ctx_table=context_table)


def thin(arr, frac, seed):
    COL = A["COL"]
    rng = np.random.default_rng(seed)
    keep = rng.random(len(arr)) < frac
    for b in (0, 1):
        idx = np.flatnonzero(arr[:, COL["band_id"]] == b)
        if len(idx) and keep[idx].sum() < min(3, len(idx)):
            keep[rng.choice(idx, size=min(3, len(idx)), replace=False)] = True
    a = arr[keep]
    a = a[np.argsort(a[:, COL["dt"]], kind="stable")].copy()
    # a sparse survey's clock starts at its own first detection (audit correction, amendment 2)
    a[:, COL["dt"]] -= a[0, COL["dt"]]
    a[:, COL["dt_prev"]] = np.r_[0.0, np.diff(a[:, COL["dt"]])]
    return a


def one(job):
    obj_id, fp, ctx, tags = job
    arr = A["le"](str(A["root"] / "data" / fp))
    out = {}
    for tag in tags:
        try:
            if tag.startswith("thin"):
                seed = int(hashlib.sha256(obj_id.encode()).hexdigest()[:8], 16)
                a = thin(A["oe"](arr, 100.0), 0.30, seed)
            else:
                a = A["oe"](arr, float(tag[1:]))
            f, _ = A["ff"](a, ctx)
            r = f.iloc[0].to_dict()
        except ValueError:
            r = None
        out[tag] = r
    return obj_id, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--tags", nargs="+", default=["h5", "h10", "h20", "h100", "thin100"])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    init(a.repo, a.root)
    root = Path(a.root)
    frames = []
    for s in ("train", "val", "test", "anomaly"):
        d = pd.read_csv(root / "data/splits" / f"{s}.csv"); d["split"] = s; frames.append(d)
    objs = pd.concat(frames, ignore_index=True)
    ctx = A["ctx_table"](root, objs.obj_id.values)
    jobs = [(r.obj_id, r.filepath, {k: (None if pd.isna(v) else float(v)) for k, v in ctx.loc[r.obj_id].items()}, a.tags)
            for r in objs.itertuples()]
    with Pool(a.workers, initializer=init, initargs=(a.repo, a.root)) as pool:
        res = dict(pool.imap_unordered(one, jobs, chunksize=20))
    for tag in a.tags:
        rows, keep = [], []
        for i, r in enumerate(objs.itertuples()):
            f = res[r.obj_id][tag]
            if f is not None:
                rows.append(f); keep.append(i)
        X = pd.DataFrame(rows)
        meta = objs.iloc[keep][["obj_id", "split", "cls", "bts_type"]].reset_index(drop=True)
        X = pd.concat([meta, X.reset_index(drop=True)], axis=1)
        X.to_parquet(out / f"X_{tag}.parquet")
        print(tag, X.shape, "dropped", len(objs) - len(keep), flush=True)


if __name__ == "__main__":
    main()
