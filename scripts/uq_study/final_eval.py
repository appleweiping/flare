"""One-time evaluation of frozen candidates on the 1,575 benchmark test objects (docs/uq_preregistration.md).

Calibration: all development out-of-fold rows (the five folds together, as the shipped cross-conformal
calibration); predictions: the final model of the same recipe fitted on the whole development pool.
Scenarios mirror the development ones: observed, missing (seeded random context removal), horizons 5/10/20,
shift_sparse (final day-100 model on the test objects' thinned curves) and shift_dense (final thin model
on the test objects' full curves). For shift scenarios the unlabelled deployment sample of a method that
accepts one is the OTHER half of the test objects on the deployment photometry (halves fixed by a seeded
permutation, then swapped and pooled), so no evaluated object enters its own density ratio.
    python final_eval.py <module>:<Class> --name NAME [--kw '{}']
"""
import argparse, importlib, inspect, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "methods")); sys.path.insert(0, str(REPO))
import numpy as np
from harness import (Setting, calibration_view, group_edges, groups, evaluate, scenario_patterns, paired_bootstrap_U,
                     B0, OOF, BLOCKS)
SEALED = WORK / "sealed"


def run(make, cal_set, test_set, scenario="observed", unl_set=None, seed=2026):
    m = make()
    cal_idx = np.arange(cal_set.n)
    cal = calibration_view(cal_set, cal_idx, getattr(m, "wants_augmented", 0), seed + 10007)
    e = group_edges(cal_set.view(cal_idx, np.zeros(cal_set.n, int)))
    n = test_set.n; pats = scenario_patterns(n, scenario, seed + 1)
    sets = np.zeros((n, 6), bool)
    takes_unl = "unlabeled" in inspect.signature(m.fit).parameters
    if unl_set is None or not takes_unl:
        m.fit(cal); sets = m.sets(test_set.view(np.arange(n), pats).blind())
    else:
        half = np.random.default_rng(seed + 2).permutation(n) < n // 2
        for h in (True, False):
            mm = make(); evl, unl = np.flatnonzero(half == h), np.flatnonzero(half != h)
            mm.fit(calibration_view(cal_set, cal_idx, getattr(mm, "wants_augmented", 0), seed + 10007),
                   unlabeled=unl_set.view(unl, np.zeros(len(unl), int)).blind())
            sets[evl] = mm.sets(test_set.view(evl, pats[evl]).blind())
    v = test_set.view(np.arange(n), pats)
    g = groups(v, e)
    res, cov, size = evaluate(sets, v, g)
    return dict(res=res, cov=cov, size=size, groups=g, y=test_set.y, sets=sets)


def summary(r):
    x = dict(r["res"]); x.pop("cells"); x["worst_cell"] = list(x["worst_cell"]) if x["worst_cell"] else None
    return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--name"); ap.add_argument("--kw", default="{}")
    ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    mod, cls = a.target.split(":"); C = getattr(importlib.import_module(mod), cls); kw = json.loads(a.kw)
    make = lambda: C(**kw); name = a.name or cls
    out = dict(method=a.target, kwargs=kw, scenarios={})
    dev = {t: Setting(OOF / f"oof_{t}.npz", t) for t in ("h100", "h20", "h10", "h5", "thin100")}
    test = {t: Setting(SEALED / f"final_{t}_on_test.npz", t) for t in ("h100", "h20", "h10", "h5")}
    plan = [("test_h100_observed", dev["h100"], test["h100"], "observed", None),
            ("test_h100_missing", dev["h100"], test["h100"], "missing", None),
            ("test_h20_observed", dev["h20"], test["h20"], "observed", None),
            ("test_h10_observed", dev["h10"], test["h10"], "observed", None),
            ("test_h5_observed", dev["h5"], test["h5"], "observed", None)]
    sp = Setting(SEALED / "final_h100_on_test_thin100.npz", "thin100")
    sd = Setting(SEALED / "final_thin100_on_test_h100.npz", "h100")
    plan += [("test_shift_sparse", dev["h100"], sp, "observed", sp), ("test_shift_dense", dev["thin100"], sd, "observed", sd)]
    for key, cs, ts, sc, us in plan:
        r = run(make, cs, ts, sc, us); x = summary(r)
        if key in ("test_h100_observed", "test_h100_missing", "test_shift_sparse"):
            b = run(B0, cs, ts, sc, us); x["boot_U_vs_B0"] = paired_bootstrap_U(r, b, a.boot); x["B0"] = summary(b)
        out["scenarios"][key] = x
        print(f"{key:20s} marg {x['marginal']:.4f} size {x['size']:.3f} single {x['singleton']:.3f} empty {x['empty']:.4f} U {x['U']:.3f} "
              f"worst {x['worst_cell']} " + (f"dU {x['boot_U_vs_B0']}" if "boot_U_vs_B0" in x else ""), flush=True)
        print("      per class", {k: round(v, 3) for k, v in x["per_class"].items()}, flush=True)
    (WORK / "results_test").mkdir(exist_ok=True)
    (WORK / "results_test" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
