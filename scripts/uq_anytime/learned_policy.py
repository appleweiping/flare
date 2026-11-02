"""Post-hoc exploratory reliability-ranking study, without classifier retraining.

Run only after the prospective amendment is saved:
  python scripts/uq_anytime/learned_policy.py --study STUDY --cache CACHE --out NEWDIR

The new family was designed after the pilot: the exact binomial calculations
below are nominal simultaneous bounds under fixed-family iid assumptions, not
new confirmatory evidence on a previously inspected calibration/evaluation set.
Gate outputs are learned ranking scores, not certified correctness probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.uq_anytime.policy import (
    DEFAULT_HORIZONS, EARLIEST_DAYS, THRESHOLDS, _validate_inputs,
    binomial_upper_bound, evaluate_policy, policy_grid,
)

HORIZONS = tuple(int(h) for h in DEFAULT_HORIZONS)
CLASSES = ("SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV")
RANKERS = ("output_only", "quality_temporal")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_study_complete(study):
    """Require the source study's completion marker to match its result bytes.

    This hashes the result artifact without parsing or selecting on its outcomes.
    Individual saved models and caches are checked separately before use.
    """
    study = Path(study)
    marker, results = study / "COMPLETE", study / "results.json"
    if not marker.is_file() or not results.is_file():
        raise ValueError("source study is incomplete: COMPLETE and results.json are required")
    expected = marker.read_text().strip()
    actual = sha(results)
    if expected != actual:
        raise ValueError("source COMPLETE marker does not match results.json SHA256")
    return actual


def _ids(values, role):
    ids = np.asarray(values)
    if ids.ndim != 1 or any(not isinstance(v, (str, np.str_)) or not v for v in ids):
        raise ValueError(f"{role} IDs must be a one-dimensional list of nonempty strings")
    out = ids.astype(str).tolist()
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate {role} IDs")
    return out


def validate_metadata(metadata_by_horizon, calibration_ids, evaluation_ids):
    """Reject any gate-training/calibration/evaluation object contamination.

    All look-specific models must have the same proper-training and early-stop
    memberships and the same ordered feature schema. Order within metadata
    membership lists may differ; gate predictions use the first look's ID order.
    """
    cal = _ids(calibration_ids, "calibration")
    ev = _ids(evaluation_ids, "evaluation")
    if set(cal) & set(ev):
        raise ValueError("calibration/evaluation IDs overlap")
    if set(metadata_by_horizon) != set(HORIZONS):
        raise ValueError("metadata must cover exactly horizons 5, 10, 20, 100")
    first = metadata_by_horizon[HORIZONS[0]]
    training = _ids(first["training_ids"], "training")
    gate = _ids(first["early_stop_ids"], "early-stop gate training")
    features = first["features"]
    if not gate or not training or len(set(features)) != len(features) or not features:
        raise ValueError("training IDs, early-stop IDs and unique features are required")
    if set(training) & set(gate):
        raise ValueError("training/early-stop IDs overlap")
    if (set(training) | set(gate)) & (set(cal) | set(ev)):
        raise ValueError("proper training/gate IDs overlap calibration or evaluation")
    for horizon, meta in metadata_by_horizon.items():
        tr = _ids(meta["training_ids"], f"h{horizon} training")
        es = _ids(meta["early_stop_ids"], f"h{horizon} early-stop")
        if set(tr) != set(training) or set(es) != set(gate):
            raise ValueError("training or early-stop memberships differ across horizons")
        if meta["features"] != features:
            raise ValueError("ordered feature schema differs across horizons")
    return dict(training_ids=training, gate_training_ids=gate,
                calibration_ids=cal, evaluation_ids=ev, features=features)


def reliability_features(probs, nobs, labels, *, quality_temporal=False, horizons=HORIZONS):
    """Return (objects, looks, features), ordered names; use only current/past looks."""
    probs, nobs, horizons, labels = _validate_inputs(probs, nobs, horizons, labels)
    if probs.shape[2] != 6:
        raise ValueError("FLARE reliability features require six class probabilities")
    confidence = np.take_along_axis(probs, labels[..., None], axis=2)
    entropy = -(probs * np.log(np.maximum(probs, np.finfo(float).tiny))).sum(axis=2, keepdims=True) / np.log(6)
    ordered = np.sort(probs, axis=2)
    margin = (ordered[..., -1] - ordered[..., -2])[..., None]
    onehot = np.eye(6)[labels]
    blocks = [probs, confidence, entropy, margin, onehot]
    names = ([f"probability_{c}" for c in CLASSES]
             + ["predicted_label_probability", "normalized_entropy", "top_two_margin"]
             + [f"predicted_label_{c}" for c in CLASSES])
    if quality_temporal:
        change = np.zeros_like(nobs)
        switch = np.zeros_like(nobs)
        change[:, 1:] = np.abs(probs[:, 1:] - probs[:, :-1]).sum(axis=2)
        switch[:, 1:] = labels[:, 1:] != labels[:, :-1]
        log_horizon = np.broadcast_to(np.log1p(horizons)[None, :], nobs.shape)
        blocks += [np.log1p(nobs)[..., None], log_horizon[..., None], change[..., None], switch[..., None]]
        names += ["log1p_nobs", "log1p_horizon", "previous_look_probability_l1", "previous_look_label_switch"]
    return np.concatenate(blocks, axis=2), names


def _validate_y(y, count):
    y = np.asarray(y)
    if (y.shape != (count,) or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= 6).any()):
        raise ValueError("y must contain one integer six-class label per object")
    return y


def fit_rankers(probs, nobs, labels, y):
    """Fit the two fixed logistic rankers on early-stop objects only.

    Four look rows have weight 1/4 each, so each object's total weight is one.
    No tuning, balanced class weighting, feature selection or calibration occurs.
    """
    y = _validate_y(y, len(probs))
    if np.asarray(probs).shape[1] != 4:
        raise ValueError("ranker training requires four fixed looks")
    # Feature validation must precede correctness computation.
    base, _ = reliability_features(probs, nobs, labels)
    del base
    correct = (np.asarray(labels) == y[:, None]).astype(int).ravel()
    if len(np.unique(correct)) != 2:
        raise ValueError("logistic rankers require both correct and incorrect early-stop predictions")
    weights = np.full(len(correct), 0.25)
    fitted = {}
    for name in RANKERS:
        features, names = reliability_features(probs, nobs, labels, quality_temporal=name == "quality_temporal")
        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("ranker", LogisticRegression(C=1, max_iter=2000))])
        pipeline.fit(features.reshape(-1, features.shape[-1]), correct,
                     scaler__sample_weight=weights, ranker__sample_weight=weights)
        fitted[name] = (pipeline, names)
    return fitted


def score_rankers(fitted, probs, nobs, labels):
    scores = {}
    for name in RANKERS:
        features, names = reliability_features(probs, nobs, labels, quality_temporal=name == "quality_temporal")
        pipeline, fitted_names = fitted[name]
        if names != fitted_names:
            raise ValueError("ranker feature schema mismatch")
        class_index = np.flatnonzero(pipeline.named_steps["ranker"].classes_ == 1)
        if len(class_index) != 1:
            raise ValueError("ranker has no correctness class")
        scores[name] = pipeline.predict_proba(features.reshape(-1, features.shape[-1]))[:, class_index[0]].reshape(features.shape[:2])
    return scores


def first_stop(scores, labels, tau, earliest_day, *, horizons=HORIZONS):
    """First eligible score crossing; later scores never change an earlier stop."""
    scores, labels, horizons = np.asarray(scores, float), np.asarray(labels), np.asarray(horizons, float)
    if (scores.ndim != 2 or scores.shape[1] == 0 or labels.shape != scores.shape
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= 6).any()):
        raise ValueError("scores and integer labels must be valid (objects, looks) arrays")
    if (horizons.shape != (scores.shape[1],) or not np.isfinite(horizons).all()
            or (horizons <= 0).any() or (np.diff(horizons) <= 0).any()):
        raise ValueError("invalid horizons")
    if not np.isfinite(tau) or not 0 <= tau <= 1 or not np.isfinite(earliest_day) or earliest_day <= 0:
        raise ValueError("invalid score threshold or earliest day")
    hit = (scores >= tau) & (horizons[None, :] >= earliest_day)
    selected = hit.any(axis=1)
    index = hit.argmax(axis=1)
    return dict(selected=selected, prediction=np.where(selected, labels[np.arange(len(scores)), index], -1),
                stop_index=np.where(selected, index, -1), stop_day=np.where(selected, horizons[index], np.nan))


def full_policy_grid():
    original = [dict(family="original", **p) for p in policy_grid()]
    return original + [dict(family=name, tau=tau, earliest_day=day)
                       for name in RANKERS for tau in THRESHOLDS for day in EARLIEST_DAYS]


def evaluate_family_policy(probs, nobs, labels, ranker_scores, index):
    if index is None:
        n = len(probs)
        return dict(selected=np.zeros(n, bool), prediction=np.full(n, -1),
                    stop_index=np.full(n, -1), stop_day=np.full(n, np.nan))
    policy = full_policy_grid()[index]
    family = policy["family"]
    if family == "original":
        return evaluate_policy(probs, nobs, {k: v for k, v in policy.items() if k != "family"}, labels=labels)
    return first_stop(ranker_scores[family], labels, policy["tau"], policy["earliest_day"])


def summarize_decisions(outcome, y):
    y = _validate_y(y, len(outcome["selected"]))
    selected = outcome["selected"]
    errors = selected & (outcome["prediction"] != y)
    count = int(selected.sum())
    days = outcome["stop_day"][selected]
    return dict(n=len(y), selected_count=count, error_count=int(errors.sum()),
                selected_fraction=float(count / len(y)) if len(y) else 0.0,
                selective_error=float(errors.sum() / count) if count else None,
                mean_day=float(days.mean()) if count else None,
                median_day=float(np.median(days)) if count else None,
                selected_counts_by_day={str(h): int((selected & (outcome["stop_day"] == h)).sum()) for h in HORIZONS},
                per_class={c: dict(n=int((y == j).sum()), selected_count=int(selected[y == j].sum()),
                                   error_count=int(errors[y == j].sum())) for j, c in enumerate(CLASSES)})


def certify_family(probs, nobs, labels, y, ranker_scores, *, risk_limit=0.05, delta=0.05 / 5):
    """Certify all 75 then choose four separate winners, never a pooled new winner."""
    probs, nobs, _, labels = _validate_inputs(probs, nobs, HORIZONS, labels)
    y = _validate_y(y, len(probs))
    if not np.isfinite(risk_limit) or not 0 < risk_limit < 1 or not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("risk_limit and delta must lie in (0, 1)")
    if set(ranker_scores) != set(RANKERS):
        raise ValueError("both prespecified ranker score matrices are required")
    for name in RANKERS:
        if np.asarray(ranker_scores[name]).shape != labels.shape:
            raise ValueError("ranker scores must match the object/look dimensions")
        first_stop(ranker_scores[name], labels, 0.5, 5)  # validate the full matrix, without outcomes
    grid = full_policy_grid()
    per_policy_delta = delta / len(grid)
    records = []
    for index, policy in enumerate(grid):
        decisions = evaluate_family_policy(probs, nobs, labels, ranker_scores, index)
        summary = summarize_decisions(decisions, y)
        bound = binomial_upper_bound(summary["error_count"], summary["selected_count"], per_policy_delta)
        records.append(dict(index=index, **policy, **summary, risk_upper_bound=bound,
                            certified=bool(bound is not None and bound <= risk_limit),
                            per_policy_delta=per_policy_delta))
    subsets = {
        "confidence_only": lambda r: r["family"] == "original" and r["support_lambda"] == 0,
        "support_family": lambda r: r["family"] == "original",
        "output_only": lambda r: r["family"] == "output_only",
        "quality_temporal": lambda r: r["family"] == "quality_temporal",
    }
    winners = {}
    for name, predicate in subsets.items():
        candidates = [r for r in records if predicate(r) and r["certified"]]
        winners[name] = (min(candidates, key=lambda r: (-r["selected_fraction"], r["mean_day"], r["index"]))["index"]
                         if candidates else None)
    return dict(records=records, winner_indices=winners, multiplicity=len(grid), delta=delta,
                per_policy_delta=per_policy_delta, risk_limit=risk_limit,
                posthoc_exploratory=True,
                interpretation="Nominal simultaneous CP under iid/frozen-family assumptions; reused pilot data is exploratory")


def _model_predictions(folder, h, metadata, frame, threads):
    import lightgbm as lgb
    from flare.hierarchical import FIVE, I_CCP, I_SLSN, I_SNCC
    top_path, branch_path = folder / f"h{h}_top.txt", folder / f"h{h}_branch.txt"
    if sha(top_path) != metadata["top_sha256"] or sha(branch_path) != metadata["branch_sha256"]:
        raise ValueError("saved classifier hash mismatch")
    top, branch = lgb.Booster(model_file=str(top_path)), lgb.Booster(model_file=str(branch_path))
    if top.feature_name() != metadata["features"] or branch.feature_name() != metadata["features"]:
        raise ValueError("saved classifier feature schema mismatch")
    x = frame[metadata["features"]].astype(float)
    p5, pb = top.predict(x, num_threads=threads), branch.predict(x, num_threads=threads)
    p6 = np.empty((len(frame), 6))
    for j, name in enumerate(CLASSES):
        p6[:, j] = (p5[:, I_CCP] * (1 - pb) if name == "SN_CC" else
                    p5[:, I_CCP] * pb if name == "SLSN" else p5[:, FIVE.index(name)])
    top_label = p5.argmax(axis=1)
    labels = np.empty(len(frame), dtype=int)
    for j, name in enumerate(FIVE):
        if name != "SN_CC+":
            labels[top_label == j] = CLASSES.index(name)
    cc = top_label == I_CCP
    labels[cc & (pb > metadata["branch_threshold"])] = I_SLSN
    labels[cc & (pb <= metadata["branch_threshold"])] = I_SNCC
    return p6, labels


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--amendment", type=Path, default=REPO / "docs/uq_learned_policy_amendment.md")
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error("--threads must be positive")
    source_results_sha = validate_study_complete(args.study)
    if not args.amendment.is_file():
        raise ValueError("the saved prospective exploratory amendment is required")
    amendment_sha = sha(args.amendment)
    args.out.mkdir(parents=True, exist_ok=False)
    source_provenance = json.loads((args.study / "provenance.json").read_text())
    cache_hashes = {f"h{h}": sha(args.cache / f"X_h{h}.parquet") for h in HORIZONS}
    if cache_hashes != source_provenance["cache_sha256"]:
        raise ValueError("cache hashes differ from the original study")
    provenance = dict(posthoc_exploratory=True, source_study=str(args.study.resolve()),
                      source_results_sha256=source_results_sha,
                      source_completion_sha256=sha(args.study / "COMPLETE"),
                      source_provenance_sha256=sha(args.study / "provenance.json"),
                      source_protocol_sha256=source_provenance["protocol_sha256"],
                      amendment_path=str(args.amendment.resolve()), amendment_sha256=amendment_sha,
                      commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                      code_sha256=sha(__file__), policy_code_sha256=sha(Path(__file__).with_name("policy.py")),
                      cache_sha256=cache_hashes, classifier_retrained=False,
                      gate_fit_role="proper-training early-stop IDs only; one total weight per object",
                      test_outcomes_read=False, anomaly_rows_used=False,
                      fixed_rankers=dict(C=1, max_iter=2000, scaler="StandardScaler", row_weight=0.25),
                      package_versions={name: importlib.metadata.version(name)
                                        for name in ["numpy", "scipy", "scikit-learn", "lightgbm", "pandas"]},
                      fixed_policy_grid=full_policy_grid(),
                      family_size=75, number_folds=5, global_delta=0.05,
                      per_policy_delta=0.05 / 5 / 75,
                      interpretation="Posthoc exploratory on reused pilot development objects, not confirmatory validation")
    write_json(args.out / "provenance.json", provenance)
    results = []
    for fold in range(5):
        folder = args.study / f"fold{fold}"
        out = args.out / f"fold{fold}"
        out.mkdir()
        metadata = {h: json.loads((folder / f"h{h}_metadata.json").read_text()) for h in HORIZONS}
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            calibration_ids, evaluation_ids = saved["calibration_ids"], saved["evaluation_ids"]
            roles = validate_metadata(metadata, calibration_ids, evaluation_ids)
            nc, ne = len(calibration_ids), len(evaluation_ids)
            stop = nc + ne
            # The archive also contains anomaly rows. They never enter feature
            # construction, ranker scoring, certification, or evaluation here.
            obj_ids = saved["obj_id"][:stop]
            if not np.array_equal(obj_ids, np.concatenate([calibration_ids, evaluation_ids])):
                raise ValueError("saved prediction ID ordering mismatch")
            probs = saved["probabilities"][:stop].copy()
            labels = saved["labels"][:stop].copy()
            nobs = saved["nobs"][:stop].copy()
            calibration_y, evaluation_y = saved["calibration_y"], saved["evaluation_y"]
        gate_ids = roles["gate_training_ids"]
        gate_probs = np.empty((len(gate_ids), 4, 6))
        gate_labels = np.empty((len(gate_ids), 4), dtype=int)
        gate_nobs = np.empty((len(gate_ids), 4))
        gate_y = None
        for look, h in enumerate(HORIZONS):
            columns = ["obj_id", "split", "cls"] + roles["features"]
            frame = pd.read_parquet(args.cache / f"X_h{h}.parquet", columns=columns,
                                    filters=[("obj_id", "in", gate_ids)])
            if frame.obj_id.duplicated().any() or set(frame.obj_id) != set(gate_ids):
                raise ValueError("gate-training cache ID mismatch")
            frame = frame.set_index("obj_id").loc[gate_ids]
            if not frame.split.isin(["train", "val"]).all():
                raise ValueError("gate-training cache includes a test/anomaly object")
            current_y = np.array([CLASSES.index(c) for c in frame.cls], dtype=int)
            if gate_y is not None and not np.array_equal(current_y, gate_y):
                raise ValueError("gate-training class labels differ across horizons")
            gate_y = current_y
            gate_probs[:, look], gate_labels[:, look] = _model_predictions(folder, h, metadata[h], frame, args.threads)
            gate_nobs[:, look] = frame.n_obs_total.to_numpy()
        fitted = fit_rankers(gate_probs, gate_nobs, gate_labels, gate_y)
        scores = score_rankers(fitted, probs, nobs, labels)
        with (out / "rankers.pkl").open("wb") as stream:
            pickle.dump(fitted, stream, protocol=pickle.HIGHEST_PROTOCOL)
        coefficients = {}
        for name, (pipeline, names) in fitted.items():
            scaler, ranker = pipeline.named_steps["scaler"], pipeline.named_steps["ranker"]
            coefficients[name] = dict(features=names, scaler_mean=scaler.mean_.tolist(),
                                      scaler_scale=scaler.scale_.tolist(), coef=ranker.coef_.tolist(),
                                      intercept=ranker.intercept_.tolist(), classes=ranker.classes_.tolist(),
                                      n_iter=ranker.n_iter_.tolist())
        write_json(out / "coefficients.json", coefficients)
        np.savez_compressed(out / "gate_training.npz", obj_id=np.asarray(gate_ids), probabilities=gate_probs,
                            labels=gate_labels, nobs=gate_nobs, y=gate_y)
        fold_provenance = dict(fold=fold, **roles,
                               source_predictions_sha256=sha(folder / "predictions.npz"),
                               metadata_sha256={str(h): sha(folder / f"h{h}_metadata.json") for h in HORIZONS},
                               model_sha256={f"h{h}_{part}": sha(folder / f"h{h}_{part}.txt")
                                             for h in HORIZONS for part in ("top", "branch")},
                               rankers_sha256=sha(out / "rankers.pkl"),
                               gate_training_sha256=sha(out / "gate_training.npz"),
                               gate_training_correct_count=int((gate_labels == gate_y[:, None]).sum()),
                               gate_training_rows=int(gate_labels.size), calibration_evaluation_rows=int(stop))
        write_json(out / "provenance.json", fold_provenance)
        certified = certify_family(probs[:nc], nobs[:nc], labels[:nc], calibration_y,
                                   {name: s[:nc] for name, s in scores.items()})
        write_json(out / "certification.json", certified)
        outputs = {}
        saved_decisions = {}
        for family, winner in certified["winner_indices"].items():
            decision = evaluate_family_policy(probs[nc:], nobs[nc:], labels[nc:],
                                               {name: s[nc:] for name, s in scores.items()}, winner)
            outputs[family] = dict(policy_index=winner,
                                   policy=None if winner is None else full_policy_grid()[winner],
                                   **summarize_decisions(decision, evaluation_y))
            for name, values in decision.items():
                saved_decisions[f"{family}_{name}"] = values
        np.savez_compressed(out / "scores_decisions.npz", calibration_ids=calibration_ids,
                            evaluation_ids=evaluation_ids,
                            **{f"calibration_{name}": s[:nc] for name, s in scores.items()},
                            **{f"evaluation_{name}": s[nc:] for name, s in scores.items()}, **saved_decisions)
        result = dict(fold=fold, n_gate_training=len(gate_ids), n_calibration=nc, n_evaluation=ne,
                      policies=outputs, posthoc_exploratory=True)
        write_json(out / "result.json", result)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    write_json(args.out / "results.json", dict(provenance=provenance, folds=results))
    (args.out / "COMPLETE").write_text(sha(args.out / "results.json") + "\n")


if __name__ == "__main__":
    main()
