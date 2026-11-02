"""Summarize a completed, frozen UQ development run without selecting methods.

python scripts/uq_anytime/summarize.py --study /path/to/study-v1 --out /new/output

The COMPLETE checksum is verified before parsing results.json. No prediction
arrays, test data, or model files are read. Fold ranges are descriptive: models
overlap in training data, and the same 53 anomaly objects recur in every refit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "naive_per_look", "bonferroni", "trajectory", "e_budget", "trajectory_e"
)
LABELS = {
    "naive_per_look": "Naive per-look",
    "bonferroni": "Bonferroni",
    "trajectory": "Trajectory envelope",
    "e_budget": "Fixed e-budget",
    "trajectory_e": "Trajectory e-value",
}
POLICIES = ("confidence_only", "support_family")
CLASSES = ("SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def read_completed(study):
    expected = (study / "COMPLETE").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("COMPLETE must contain the SHA256 of results.json")
    payload = (study / "results.json").read_bytes()
    if sha256(payload) != expected:
        raise ValueError("results.json does not match COMPLETE; refusing to report")
    data = json.loads(payload)
    provenance, folds = data["provenance"], data["folds"]
    if provenance.get("test_predictions_read") is not False:
        raise ValueError("Expected a development-only study with test_predictions_read=false")
    if len(folds) != 5 or sorted(f["fold"] for f in folds) != list(range(5)):
        raise ValueError("Expected five distinct development folds")
    if provenance["known_development"] != 8922 or provenance["descriptive_anomalies"] != 53:
        raise ValueError("This report schema is fixed to study-v1: 8922 known, 53 anomalies")
    if sum(f["n_evaluation"] for f in folds) != provenance["known_development"]:
        raise ValueError("Evaluation counts do not match the known development denominator")
    for fold in folds:
        if set(fold["alarms"]) != set(METHODS):
            raise ValueError("Unexpected alarm methods; do not silently omit an arm")
        for method in METHODS:
            alarm = fold["alarms"][method]
            if alarm["known"]["n"] != fold["n_evaluation"] or alarm["anomaly"]["n"] != 53:
                raise ValueError("Inconsistent alarm denominators")
        for name in POLICIES:
            policy = fold["policies"][name]
            if policy["n"] != fold["n_evaluation"] or set(policy["per_class"]) != set(CLASSES):
                raise ValueError("Inconsistent policy denominator or class schema")
    return data, expected


def summarize(data, source_sha):
    folds = sorted(data["folds"], key=lambda f: f["fold"])
    n_known = data["provenance"]["known_development"]
    n_anomalies = data["provenance"]["descriptive_anomalies"]
    summary = {
        "schema": "flare-uq-study-v1-summary",
        "source_results_sha256": source_sha,
        "source_provenance": data["provenance"],
        "n_refits": len(folds),
        "n_known_evaluation_objects": n_known,
        "n_unique_descriptive_anomalies": n_anomalies,
        "nominal_alarm_level": 0.01,
        "alarm_comparison": "Nominal 1% thresholds, not matched achieved false-alarm rates; naive uses 1% per look",
        "uncertainty": "Fold ranges are descriptive, not confidence intervals; training/calibration overlap across folds",
        "anomaly_denominator": "Mean recovered per refit divided by the same 53 unique anomalies; not 265 independent objects",
        "interpretation": "Previously inspected development data; no external validation or deployment claim",
        "alarms": {},
        "policies": {},
        "fixed_batch_e_bh": {},
    }
    for method in METHODS:
        known = [f["alarms"][method]["known"] for f in folds]
        anomaly = [f["alarms"][method]["anomaly"] for f in folds]
        known_count = sum(v["alarm_count"] for v in known)
        known_rates = [v["alarm_count"] / v["n"] for v in known]
        recovered = [v["alarm_count"] for v in anomaly]
        summary["alarms"][method] = {
            "known_alarm_count": known_count,
            "known_denominator": n_known,
            "known_lifetime_alarm_fraction": known_count / n_known,
            "known_fold_fractions": known_rates,
            "known_fold_range": [min(known_rates), max(known_rates)],
            "anomalies_recovered_per_refit": recovered,
            "mean_anomalies_recovered": float(np.mean(recovered)),
            "anomaly_denominator_per_refit": n_anomalies,
            "mean_anomaly_recovery_fraction": float(np.mean(recovered)) / n_anomalies,
            "anomaly_recovery_fold_range": [min(recovered) / n_anomalies, max(recovered) / n_anomalies],
            "mean_anomalies_by_first_day": {
                str(day): float(np.mean([v["counts_by_first_day"][str(day)] for v in anomaly]))
                for day in data["provenance"]["horizons"]
            },
        }
    class_rows = []
    for name in POLICIES:
        policies = [f["policies"][name] for f in folds]
        selected = sum(p["selected_count"] for p in policies)
        errors = sum(p["error_count"] for p in policies)
        summary["policies"][name] = {
            "known_denominator": n_known,
            "selected_count": selected,
            "error_count": errors,
            "selected_fraction": selected / n_known,
            "selective_error": errors / selected if selected else None,
            "selected_mean_day": (
                sum(p["mean_day"] * p["selected_count"] for p in policies if p["selected_count"]) / selected
                if selected else None
            ),
            "policy_indices_per_fold": [p["policy_index"] for p in policies],
            "policies_per_fold": [p["policy"] for p in policies],
            "per_true_class": {},
        }
        for cls in CLASSES:
            rows = [p["per_class"][cls] for p in policies]
            n = sum(v["n"] for v in rows)
            ns = sum(v["selected_count"] for v in rows)
            ne = sum(v["error_count"] for v in rows)
            row = dict(policy_family=name, true_class=cls, n_objects=n,
                       selected_count=ns, error_count=ne,
                       selected_fraction=ns / n if n else None,
                       selected_error=ne / ns if ns else None,
                       lower_support_class=cls in ("SLSN", "AGN", "TDE", "CV"))
            summary["policies"][name]["per_true_class"][cls] = row
            class_rows.append(row)
    summary["policy_families_identical_in_every_fold"] = all(
        f["policies"]["confidence_only"] == f["policies"]["support_family"] for f in folds
    )
    summary["policy_risk_scope"] = "Aggregate selected error; no per-class risk certificate"
    for name in ("e_budget", "trajectory_e"):
        rows = [f["fixed_batch_e_bh"][name] for f in folds]
        summary["fixed_batch_e_bh"][name] = {
            "nominal_fdr_level": 0.1,
            "family_sizes_per_fold": [r["n_family"] for r in rows],
            "discoveries_per_fold": [r["discoveries"] for r in rows],
            "known_rejected_per_fold": [r["known_rejected"] for r in rows],
            "anomalies_rejected_per_fold": [r["anomalies_rejected"] for r in rows],
            "observed_fdp_per_fold": [r["fdp"] for r in rows],
            "interpretation": "Descriptive fixed batches with shared anomalies; FDP is not an estimate of population FDR",
        }
    return summary, class_rows


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white", metadata={"Software": "FLARE UQ summary"})
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                metadata={"Creator": "FLARE UQ summary", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def plot_alarms(summary, out):
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    colors = ("#555555", "#0072B2", "#009E73", "#D55E00", "#CC79A7")
    markers = ("o", "s", "D", "^", "v")
    for method, color, marker in zip(METHODS, colors, markers):
        row = summary["alarms"][method]
        x = 100 * row["known_lifetime_alarm_fraction"]
        y = 100 * row["mean_anomaly_recovery_fraction"]
        lo_x, hi_x = 100 * np.asarray(row["known_fold_range"])
        lo_y, hi_y = 100 * np.asarray(row["anomaly_recovery_fold_range"])
        ax.errorbar(x, y, xerr=[[x - lo_x], [hi_x - x]], yerr=[[y - lo_y], [hi_y - y]],
                    fmt=marker, color=color, markersize=8, capsize=4, elinewidth=1.5,
                    markeredgecolor="white", markeredgewidth=0.7, label=LABELS[method])
    ax.axvline(1, color="#888888", linestyle="--", linewidth=1.1, label="Nominal lifetime target (1%)")
    ax.set(xlabel="Known-source lifetime alarm fraction (%)",
           ylabel="Mean recovery of the same 53 anomalies (%)",
           title="Four-look alarm comparison: days 5, 10, 20 and 100",
           xlim=(-0.08, 3.5), ylim=(-0.65, 8.8))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.text(0.08, 0.115, "Points: pooled rates for 8,922 held-out known objects; mean anomaly recovery over five refits.", fontsize=9)
    fig.text(0.08, 0.085, "Bars: descriptive fold minima–maxima, not confidence intervals; refits share training/calibration data.", fontsize=9)
    fig.text(0.08, 0.055, "Same 53 anomalies reused. Nominal 1% thresholds; achieved FARs are not matched. Naive: 1% per look.", fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.24)
    save_figure(fig, out / "alarm_tradeoff")


def plot_policy(summary, out):
    # Both arms stay visible if a future completed v1 run selects different policies.
    identical = summary["policy_families_identical_in_every_fold"]
    names = ["confidence_only"] if identical else list(POLICIES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    x = np.arange(len(CLASSES))
    width = 0.68 if identical else 0.34
    for index, name in enumerate(names):
        rows = summary["policies"][name]["per_true_class"]
        shift = 0 if identical else (index - 0.5) * width
        color = "#0072B2" if index == 0 else "#009E73"
        label = "Both policy families" if identical else name.replace("_", " ")
        selection = [100 * rows[c]["selected_fraction"] for c in CLASSES]
        risk = [np.nan if rows[c]["selected_error"] is None else 100 * rows[c]["selected_error"] for c in CLASSES]
        axes[0].bar(x + shift, selection, width=width, color=color, label=label)
        bars = axes[1].bar(x + shift, risk, width=width, color=color, label=label)
        if identical:
            for bar, cls in zip(bars, CLASSES):
                row = rows[cls]
                if row["selected_count"]:
                    axes[1].annotate(f'{row["error_count"]}/{row["selected_count"]}',
                                     (bar.get_x() + bar.get_width()/2, bar.get_height()),
                                     ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    for ax in axes:
        ax.set_xticks(x, CLASSES)
        ax.set_ylim(0, 108)
        ax.set_xlabel("True class")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set(title="Selected fraction within each true class", ylabel="Selected / all objects (%)")
    axes[1].set(title="Errors among selected objects", ylabel="Errors / selected objects (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Aggregate risk control does not imply rare-class accuracy", fontsize=13)
    note = "The two policy families chose identical policies in all five folds." if identical else "Policy families are reported separately; no post-hoc selection."
    fig.text(0.075, 0.09, note, fontsize=9)
    fig.text(0.075, 0.05, "Pooled held-out development counts. Class estimates are descriptive; no classwise risk guarantee.", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.23, wspace=0.27)
    save_figure(fig, out / "selected_class_errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="New directory; default is STUDY-summary beside the study")
    args = parser.parse_args()
    source, source_sha = read_completed(args.study)
    summary, rows = summarize(source, source_sha)
    out = args.out if args.out is not None else args.study.with_name(args.study.name + "-summary")
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "selected_error_by_true_class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    plot_alarms(summary, out)
    plot_policy(summary, out)
    manifest = {path.name: sha256(path.read_bytes()) for path in sorted(out.iterdir()) if path.is_file()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "COMPLETE").write_text(manifest["summary.json"] + "\n")
    print(json.dumps({"output": str(out), "source_results_sha256": source_sha,
                      "known_objects": summary["n_known_evaluation_objects"],
                      "unique_anomalies_reused": summary["n_unique_descriptive_anomalies"],
                      "policy_families_identical": summary["policy_families_identical_in_every_fold"]}))


if __name__ == "__main__":
    main()
