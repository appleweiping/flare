"""Apply the registered adoption rules (docs/uq_preregistration.md, amendments 1-2) to results/*.json.
Shift rule, closeness reading (stated by the M4 implementer before scoring and kept for every candidate):
marginal |m - 0.90| <= 0.5 |B0 - 0.90| and every class |c - 0.90| <= 0.05 or no farther from 0.90 than B0.
Writes results/decision.json and prints a table."""
import json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results
R = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(WORK / "results" / "*.json")) if "decision" not in f and "T1" not in f}
B = R["B0"]["scenarios"]
POST_HOC = {"M6"}
EXPLORATORY = {"M1_nctx", "M1_nctx_aug3", "M2_adj", "M2_aug3_adj", "M4_lgbm_explore", "M5_aps_rand"}


def shift_ok(x, b):
    half = abs(x["marginal"] - .9) <= .5 * abs(b["marginal"] - .9)
    cls = all(abs(x["per_class"][c] - .9) <= .05 or abs(x["per_class"][c] - .9) <= abs(b["per_class"][c] - .9) for c in b["per_class"])
    return half and cls


rows, out = [], {}
for name, r in sorted(R.items()):
    s = r["scenarios"]; o, m = s["h100_observed"], s["h100_missing"]
    r1 = all((B[k]["U"] - s[k]["U"]) >= .03 and s[k]["boot_U_vs_B0"]["hi"] < 0 for k in ("h100_observed", "h100_missing"))
    r2 = .895 <= o["marginal"] <= .915
    r3 = o["size"] <= 1.10 * B["h100_observed"]["size"] and o["singleton"] >= B["h100_observed"]["singleton"] - .03
    r4 = all(o["per_class"][c] >= B["h100_observed"]["per_class"][c] - .02 for c in o["per_class"])
    hz = all(s[k]["U"] <= B[k]["U"] + .01 for k in ("h20_observed", "h10_observed", "h5_observed"))
    sh_s, sh_d = shift_ok(s["shift_sparse"], B["shift_sparse"]), shift_ok(s["shift_dense"], B["shift_dense"])
    tag = "post hoc" if name in POST_HOC else ("exploratory" if name in EXPLORATORY else "registered")
    adopt = r1 and r2 and r3 and r4 and hz and tag == "registered"
    out[name] = dict(tag=tag, rule1=r1, rule2=r2, rule3=r3, rule4=r4, horizons=hz, shift_sparse=sh_s, shift_dense=sh_d,
                     adopted_as_default=adopt,
                     obs=dict(marginal=o["marginal"], size=o["size"], singleton=o["singleton"], singleton_acc=o["singleton_acc"],
                              empty=o["empty"], U=o["U"], dU=o["boot_U_vs_B0"]),
                     miss=dict(marginal=m["marginal"], size=m["size"], U=m["U"], dU=m["boot_U_vs_B0"]),
                     shift_sparse_marginal=s["shift_sparse"]["marginal"], shift_sparse_size=s["shift_sparse"]["size"],
                     shift_dense_marginal=s["shift_dense"]["marginal"], shift_dense_size=s["shift_dense"]["size"])
    f = lambda b: "Y" if b else "-"
    rows.append(f"{name:18s} {tag:11s} R1 {f(r1)} R2 {f(r2)} R3 {f(r3)} R4 {f(r4)} Hz {f(hz)} Ss {f(sh_s)} Sd {f(sh_d)} | "
                f"obs m {o['marginal']:.3f} sz {o['size']:.3f} sg {o['singleton']:.3f} sacc {o['singleton_acc']:.3f} e {o['empty']:.3f} "
                f"U {o['U']:.3f} [{o['boot_U_vs_B0']['lo']:+.3f},{o['boot_U_vs_B0']['hi']:+.3f}] | miss m {m['marginal']:.3f} sz {m['size']:.3f} U {m['U']:.3f} "
                f"| sparse {s['shift_sparse']['marginal']:.3f}/{s['shift_sparse']['size']:.2f} dense {s['shift_dense']['marginal']:.3f}/{s['shift_dense']['size']:.2f}")
(WORK / "results" / "decision.json").write_text(json.dumps(out, indent=1))
print("\n".join(rows))
