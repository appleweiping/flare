"""Markdown tables for docs/uq_conditional_calibration.md, generated from the result JSONs (no hand copying)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os as _os
REPO = Path(__file__).resolve().parents[2]
WORK = Path(_os.environ.get("FLARE_UQ_WORK", REPO / "paper_data" / "uq"))  # caches, oofs, sealed, results


def f3(x):
    return f"{x:.3f}"


def dev_table(names):
    rows = ["| candidate | status | marginal | size | singleton | singleton acc. | empty | U observed [95%] | U missing | sparse shift | dense shift | rules 1/2/3/4 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    D = json.load(open(WORK / "results" / "decision.json"))
    for n in names:
        d = D[n]; o, m = d["obs"], d["miss"]
        b = o["dU"]
        rule = "/".join("Y" if d[f"rule{i}"] else "-" for i in (1, 2, 3, 4))
        rows.append(f"| {n} | {d['tag']} | {f3(o['marginal'])} | {f3(o['size'])} | {f3(o['singleton'])} | {f3(o['singleton_acc'])} | {f3(o['empty'])} | "
                    f"{f3(o['U'])} [{b['lo']:+.3f}, {b['hi']:+.3f}] | {f3(m['U'])} | {f3(d['shift_sparse_marginal'])} ({d['shift_sparse_size']:.2f}) | "
                    f"{f3(d['shift_dense_marginal'])} ({d['shift_dense_size']:.2f}) | {rule} |")
    return "\n".join(rows)


def nested_table(tag, names):
    R = json.load(open(WORK / "results_nested" / f"{tag}.json"))
    rows = [f"| {tag} | scenario | marginal | size | singleton | singleton acc. | empty | U | U change vs D0 [95%] |", "|---|---|---|---|---|---|---|---|---|"]
    for sc in ("observed", "missing"):
        for n in names:
            x = R[sc][n]; b = x.get("boot_U_vs_D0")
            rows.append(f"| {n} | {sc} | {f3(x['marginal'])} | {f3(x['size'])} | {f3(x['singleton'])} | {f3(x['singleton_acc'])} | {f3(x['empty'])} | {f3(x['U'])} | "
                        + (f"{b['point']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]" if b else "reference") + " |")
    return "\n".join(rows)


def test_table(names):
    rows = ["| candidate | marginal | size | singleton | empty | U observed | U missing | day 20 / 10 / 5 U | sparse shift | dense shift |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = json.load(open(WORK / "results_test" / f"{n}.json"))["scenarios"]; o = s["test_h100_observed"]
        rows.append(f"| {n} | {f3(o['marginal'])} | {f3(o['size'])} | {f3(o['singleton'])} | {f3(o['empty'])} | {f3(o['U'])} | {f3(s['test_h100_missing']['U'])} | "
                    f"{f3(s['test_h20_observed']['U'])} / {f3(s['test_h10_observed']['U'])} / {f3(s['test_h5_observed']['U'])} | "
                    f"{f3(s['test_shift_sparse']['marginal'])} ({s['test_shift_sparse']['size']:.2f}) | {f3(s['test_shift_dense']['marginal'])} ({s['test_shift_dense']['size']:.2f}) |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(dev_table(["B0", "B1", "M1_atyp", "M1_atyp_aug3", "M1_nobs_aug3", "M1_ctx_aug3", "M2", "M2_aug3", "M4", "M5_aps", "M5_raps01", "M6"]))
    print()
    for t in ("h100", "h20", "h10", "h5"):
        if (WORK / "results_nested" / f"{t}.json").exists():
            print(nested_table(t, ["D0", "D0_nonempty", "M6_on_final", "CC", "CC6"])); print()
    print(test_table(["B0", "B1", "M1_atyp", "M1_atyp_aug3", "M2_aug3", "M6"]))
