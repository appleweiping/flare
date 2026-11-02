"""Plot the frozen Mondrian allocation comparison from a completed artifact.

python scripts/uq_anytime/summarize_budget.py --study ALLOCATION --out FIGURE_DIR

Reads completed summary/provenance and archived code only; never refits models
or reads per-object predictions. Random ranges describe algorithmic seeds at
equal allocated quotas, not confidence intervals or matched realized costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np


CLASSES = ("SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV")
ALPHAS = (.1, .2)
BUDGETS = (0, .25, .5, 1, 2, 3)
POLICIES = ("round_robin", "confidence", "entropy", "rare_mass",
            "macro_output", "macro_quality", "aggregate_output")
RANDOM = ("random_seed0", "random_seed1", "random_seed2")
STYLES = (
    ("Round robin", "#686868", "o", "--"),
    ("Confidence", "#0072B2", "s", "-"),
    ("Entropy", "#E69F00", "^", "--"),
    ("Rare-class mass", "#CC79A7", "v", "-."),
    ("Macro output", "#009E73", "D", "-"),
    ("Macro quality / temporal", "#D55E00", "+", "--"),
    ("Aggregate output", "#56B4E9", "x", "-."),
)
STEM = "budget_allocation"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read_completed(study):
    """Verify the completion/provenance chain before using plotted metrics."""
    marker = (study / "COMPLETE").read_bytes()
    expected = marker.decode().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("COMPLETE must contain the SHA-256 of results.json")
    payload = (study / "results.json").read_bytes()
    if sha(payload) != expected:
        raise ValueError("results.json does not match COMPLETE")
    data = json.loads(payload)
    provenance_bytes = (study / "provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    if (provenance != data["provenance"]
            or sha(provenance_bytes) != data["artifacts_sha256"]["provenance.json"]):
        raise ValueError("source provenance differs from completed results")
    for flag in ("winner_selection", "classifier_retrained", "test_outcomes_read", "anomaly_rows_used"):
        if provenance.get(flag) is not False:
            raise ValueError(f"expected exploratory development provenance with {flag}=false")
    if (provenance["classes"] != list(CLASSES) or provenance["alphas"] != list(ALPHAS)
            or provenance["budget_multipliers"] != list(BUDGETS)
            or provenance["random_seeds"] != [0, 1, 2]):
        raise ValueError("source differs from the frozen figure grid")
    input_hashes = {"COMPLETE": sha(marker), "results.json": expected,
                    "provenance.json": sha(provenance_bytes)}
    for name, digest in provenance["code_sha256"].items():
        relative = f"code/{name}"
        if sha((study / relative).read_bytes()) != digest:
            raise ValueError(f"archived source mismatch: {relative}")
        input_hashes[relative] = digest
    protocol = "code/EXPERIMENT_PLAN.md"
    if sha((study / protocol).read_bytes()) != provenance["protocol_sha256"]:
        raise ValueError("archived protocol differs from provenance")
    input_hashes[protocol] = provenance["protocol_sha256"]
    folds = data["folds"]
    if len(folds) != 5 or sorted(f["fold"] for f in folds) != list(range(5)):
        raise ValueError("expected five distinct development folds")
    n = sum(fold["n_evaluation"] for fold in folds)
    if n != 8922:
        raise ValueError("this figure requires the 8,922-object development cohort")
    expected_grid = {(regime, alpha, policy, budget)
                     for regime in ("pooled", "mondrian") for alpha in ALPHAS
                     for policy in (*POLICIES, *RANDOM, "candidate_count") for budget in BUDGETS}
    records = {(r["regime"], r["alpha"], r["policy"], r["budget_multiplier"]): r
               for r in data["pooled"]}
    if set(records) != expected_grid or len(records) != len(data["pooled"]):
        raise ValueError("pooled allocation cells differ from the frozen complete grid")
    class_counts = None
    for record in records.values():
        metrics = record["metrics"]
        groups = metrics["by_true_class"]
        if set(groups) != set(CLASSES):
            raise ValueError("unexpected true-class macro schema")
        counts = tuple(groups[c]["n_objects"] for c in CLASSES)
        if min(counts) <= 0 or sum(counts) != n or (class_counts is not None and counts != class_counts):
            raise ValueError("true-class denominators changed between plotted cells")
        class_counts = counts
        macro = np.mean([groups[c]["n_correct_automatic"] / groups[c]["n_objects"] for c in CLASSES])
        if not np.isclose(macro, metrics["macro_true_class_correct_automatic_rate"], rtol=0, atol=1e-12):
            raise ValueError("saved macro rate differs from six-class count calculation")
        resource = metrics["resources"]
        used, allocated = resource["consumed_extra_tokens"], resource["allocated_extra_tokens"]
        if (metrics["overall"]["n_objects"] != n or resource["initial_checkpoint_queries"] != n
                or not 0 <= used <= allocated or allocated - used != resource["unused_extra_tokens"]):
            raise ValueError("inconsistent query-resource denominator or accounting")
    return data, records, n, dict(zip(CLASSES, class_counts)), input_hashes


def coordinates(records, alpha, policy, n):
    rows = [records[("mondrian", alpha, policy, budget)] for budget in BUDGETS]
    x = np.array([r["metrics"]["resources"]["consumed_extra_tokens"] / n for r in rows])
    y = np.array([r["metrics"]["macro_true_class_correct_automatic_rate"] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (np.diff(x) < 0).any():
        raise ValueError("invalid or nonmonotone realized query coordinates")
    if (y < 0).any() or (y > 1).any():
        raise ValueError("invalid macro rate")
    return x, y


def plot(records, n, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7), sharex=True, sharey=True)
    plotted = {}
    for ax, alpha in zip(axes, ALPHAS):
        panel = {}
        for policy, (label, color, marker, linestyle) in zip(POLICIES, STYLES):
            x, y = coordinates(records, alpha, policy, n)
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.8, markersize=6, markerfacecolor="none", markeredgewidth=1.3)
            panel[policy] = dict(x=x.tolist(), y=y.tolist())
        seeds = np.array([coordinates(records, alpha, policy, n) for policy in RANDOM])
        xs, ys = seeds[:, 0], seeds[:, 1]
        x, y = xs.mean(axis=0), ys.mean(axis=0)
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color="#333333", alpha=.13)
        ax.errorbar(x, y, xerr=[np.maximum(0, x - xs.min(axis=0)), np.maximum(0, xs.max(axis=0) - x)],
                    color="#222222", marker=".", linestyle=":", linewidth=1.6,
                    markersize=7, capsize=3, label="Random: 3-seed mean")
        panel["random"] = dict(x_mean=x.tolist(), y_mean=y.tolist(),
                               x_min=xs.min(axis=0).tolist(), x_max=xs.max(axis=0).tolist(),
                               y_min=ys.min(axis=0).tolist(), y_max=ys.max(axis=0).tolist())
        plotted[str(alpha)] = panel
        ax.set_title(f"Mondrian envelope · α = {alpha:.2f}")
        ax.set_xlim(-.03, 3.03)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 3.1, .5))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Actual consumed extra queries / 8,922 objects")
    axes[0].set_ylabel("Macro true-class correct commitment rate (%)")
    fig.suptitle("Exploratory archived-checkpoint playback", x=.07, ha="left", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color="#333333", alpha=.13))
    labels.append("Random: seed min–max")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.51, .175),
               ncol=3, frameon=False, fontsize=9.5, columnspacing=2.5)
    notes = (
        "Points follow equal allocated quotas b = 0, 0.25, 0.5, 1, 2, 3; closure can make actual consumption differ by policy.",
        "Random means/ranges pair seeds at each quota: shaded y range and horizontal x bars are algorithmic min–max, not confidence intervals.",
        "Macro = mean correct/all over six true classes. Initial checkpoints excluded from x; reused development data, no telescope intervention.",
    )
    for y, note in zip((.125, .09, .055), notes):
        fig.text(.07, y, note, fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.87, bottom=.39, wspace=.12)
    png, pdf = out / f"{STEM}.png", out / f"{STEM}.pdf"
    fig.savefig(png, dpi=200, facecolor="white", metadata={"Software": "FLARE budget summary"})
    fig.savefig(pdf, facecolor="white", metadata={"Creator": "FLARE budget summary",
                                               "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return plotted, (png, pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Completed run_budget.py output")
    parser.add_argument("--out", type=Path, required=True, help="Figure directory; existing unrelated files are retained")
    args = parser.parse_args()
    outputs = [args.out / f"{STEM}.{suffix}" for suffix in ("png", "pdf")]
    manifest_path = args.out / "budget_figure_manifest.json"
    if any(path.exists() for path in (*outputs, manifest_path)):
        raise ValueError("figure or manifest already exists; choose another output directory")
    data, records, n, class_counts, input_hashes = read_completed(args.study)
    args.out.mkdir(parents=True, exist_ok=True)
    plotted, outputs = plot(records, n, args.out)
    manifest = dict(
        command=shlex.join([sys.executable, *sys.argv]),
        working_directory=str(Path.cwd()), source_directory=str(args.study.resolve()),
        input_sha256=input_hashes, script_sha256=sha(Path(__file__).read_bytes()),
        figure_sha256={path.name: sha(path.read_bytes()) for path in outputs},
        package_versions={"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        n_evaluation=n, true_class_denominators=class_counts, regimes=["mondrian"], alphas=ALPHAS,
        included_policies=POLICIES, random_seeds=[0, 1, 2], omitted_from_figure_only=["candidate_count"],
        x="actual consumed extra checkpoint queries / 8922; common initial checkpoints excluded",
        y="unweighted mean correct automatic / all objects within each of six true classes",
        random_range="seed min/max at equal allocated quota; x/y means; horizontal x range and shaded y range; not confidence intervals",
        interpretation=data["provenance"]["interpretation"], plotted_coordinates=plotted)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=[str(p) for p in (*outputs, manifest_path)],
                          source_results_sha256=input_hashes["results.json"], n_evaluation=n)))


if __name__ == "__main__":
    main()
