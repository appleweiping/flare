"""Score one calibration method on every registered development scenario (docs/uq_preregistration.md).

    python run_method.py <module>:<Class> [--name NAME] [--boot 2000] [--kw '{"param": 1}']

The class is instantiated with the keyword arguments of --kw. Writes results/<NAME>.json with, per
scenario: marginal, size, singleton, singleton_acc, empty, U, worst_cell, dispersion, per_class,
and for the primary scenarios the paired object bootstrap of U(method) - U(B0) (negative = better).
Never reads the sealed test predictions.
"""
import argparse, importlib, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "methods")); sys.path.insert(0, str(REPO))
import numpy as np
from harness import Setting, lofo, lofo_shift, paired_bootstrap_U, B0, OOF

SCEN = [
    ("h100_observed", "h100", "observed"), ("h100_missing", "h100", "missing"),
    ("h20_observed", "h20", "observed"), ("h10_observed", "h10", "observed"), ("h5_observed", "h5", "observed"),
    ("h20_missing", "h20", "missing"),
]
SHIFT = [("shift_sparse", ("oof_h100.npz", "h100"), ("oof_h100_deploy_thin100.npz", "thin100")),
         ("shift_dense", ("oof_thin100.npz", "thin100"), ("oof_thin100_deploy_h100.npz", "h100"))]
PRIMARY = {"h100_observed", "h100_missing"}


def summary(r):
    x = dict(r["res"]); x.pop("cells")
    x["worst_cell"] = list(x["worst_cell"]) if x["worst_cell"] else None
    return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--name"); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--kw", default="{}"); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    mod, cls = a.target.split(":"); C = getattr(importlib.import_module(mod), cls); kw = json.loads(a.kw)
    make = lambda: C(**kw)
    name = a.name or cls
    out = dict(method=a.target, kwargs=kw, scenarios={})
    cache = {}
    t0 = time.time()
    for key, tag, sc in SCEN:
        if a.only and key not in a.only:
            continue
        S = cache.setdefault(tag, Setting(OOF / f"oof_{tag}.npz", tag))
        r = lofo(S, make, sc); x = summary(r)
        if key in PRIMARY:
            base = lofo(S, B0, sc)
            x["boot_U_vs_B0"] = paired_bootstrap_U(r, base, a.boot)
            x["B0"] = summary(base)
        out["scenarios"][key] = x
        print(f"{key:14s} marg {x['marginal']:.4f} size {x['size']:.3f} single {x['singleton']:.3f} empty {x['empty']:.4f} "
              f"U {x['U']:.3f} worst {x['worst_cell']}" + (f"  dU vs B0 {x['boot_U_vs_B0']}" if 'boot_U_vs_B0' in x else "")
              + f"  ({time.time()-t0:.0f}s)", flush=True)
    for key, (fc, tc), (fd, td) in SHIFT:
        if a.only and key not in a.only:
            continue
        r = lofo_shift(Setting(OOF / fc, tc), Setting(OOF / fd, td), make); x = summary(r)
        out["scenarios"][key] = x
        print(f"{key:14s} marg {x['marginal']:.4f} size {x['size']:.3f} single {x['singleton']:.3f} empty {x['empty']:.4f} "
              f"U {x['U']:.3f} per_class {{{', '.join(f'{k}: {v:.3f}' for k, v in x['per_class'].items())}}}", flush=True)
    (WORK / "results").mkdir(exist_ok=True)
    (WORK / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", WORK / "results" / f"{name}.json")


if __name__ == "__main__":
    main()
