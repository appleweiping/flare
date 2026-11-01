"""One-time test check of the review trigger: tau and calibration scores from all development out-of-fold
rows, realised known-class rate on the 1,575 test objects (final models) for the deployed pair, T1 and its
deployed form T1' = p_L <= min(tau, alpha)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "methods")); sys.path.insert(0, str(REPO))
import numpy as np
from harness import Setting, OOF, scenario_patterns, B0, ALPHA
from t1_trigger import T1Trigger, deployed_pair
out = {}
for tag in ("h100", "h20", "h10", "h5"):
    dev = Setting(OOF / f"oof_{tag}.npz", tag); test = Setting(WORK / "sealed" / f"final_{tag}_on_test.npz", tag)
    cal = dev.view(np.arange(dev.n), np.zeros(dev.n, int))
    t1 = T1Trigger().fit(cal); b0 = B0(); b0.fit(cal)
    for sc in (("observed", "missing") if tag == "h100" else ("observed",)):
        v = test.view(np.arange(test.n), scenario_patterns(test.n, sc, 2027)).blind()
        pv = t1.p_values(v); pL = pv[np.arange(test.n), v.L]
        pair, _, _ = deployed_pair(pv, b0.sets(v), v.L)
        r = dict(tau=t1.tau, pair=float(pair.mean()), T1=float((pL <= t1.tau).mean()),
                 T1_deployed=float((pL <= min(t1.tau, ALPHA)).mean()), n=int(test.n))
        out[f"{tag}_{sc}"] = r; print(tag, sc, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
(WORK / "results_test").mkdir(exist_ok=True); (WORK / "results_test" / "T1.json").write_text(json.dumps(out, indent=1))
