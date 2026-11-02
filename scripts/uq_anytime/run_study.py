"""Run the frozen source-level development protocol; never scores the old test split.

python scripts/uq_anytime/run_study.py --cache DIR --out NEW_DIR --threads 4
See docs/uq_anytime_protocol.md. This is an experiment, not a deployed calibrator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import KFold, train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from train_early_cc import hierarchy_nocal
from evaluate_unified import SIX
from calibration import (rank_pvalues, rank_evalues, trajectory_pvalues,
                         bonferroni_path, e_budget_path, e_bh)
from policy import policy_grid, evaluate_policy, certify_policies

HORIZONS = np.array([5, 10, 20, 100])
META = {'obj_id', 'split', 'cls', 'bts_type'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize_alarm(path, levels, y=None):
    hit = path <= levels
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    result = dict(n=len(path), alarm_count=int(any_hit.sum()),
                  rate=float(any_hit.mean()),
                  median_first_day=float(np.median(HORIZONS[first[any_hit]])) if any_hit.any() else None,
                  counts_by_first_day={str(h): int(np.sum(any_hit & (first == j))) for j, h in enumerate(HORIZONS)})
    if y is not None:
        result['per_class'] = {c: dict(n=int(np.sum(y == j)), alarm_count=int(any_hit[y == j].sum()),
                                     rate=float(any_hit[y == j].mean()) if np.any(y == j) else None)
                               for j, c in enumerate(SIX)}
    return result


def summarize_policy(probs, labels, nobs, y, index):
    if index is None:
        selected = np.zeros(len(y), bool)
        pred = np.full(len(y), -1)
        days = np.zeros(len(y))
    else:
        result = evaluate_policy(probs, nobs, policy_grid()[index], labels=labels)
        selected, pred, days = result['selected'], result['prediction'], result['stop_day']
    wrong = selected & (pred != y)
    return dict(policy_index=index, policy=None if index is None else policy_grid()[index],
                n=len(y), selected_count=int(selected.sum()), error_count=int(wrong.sum()),
                selected_fraction=float(selected.mean()),
                selective_error=float(wrong.sum() / selected.sum()) if selected.any() else None,
                mean_day=float(np.mean(days[selected])) if selected.any() else None,
                median_day=float(np.median(days[selected])) if selected.any() else None,
                per_class={c: dict(n=int(np.sum(y == j)), selected_count=int(selected[y == j].sum()),
                                   error_count=int(wrong[y == j].sum())) for j, c in enumerate(SIX)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = {h: pd.read_parquet(args.cache / f'X_h{h}.parquet') for h in HORIZONS}
    for h, d in data.items():
        if d.obj_id.duplicated().any():
            raise ValueError(f'duplicate object IDs at {h}')
    master = data[100]
    dev_ids = master.loc[master.split.isin(['train', 'val']), 'obj_id'].to_numpy(dtype=str)
    novel_ids = master.loc[master.split == 'anomaly', 'obj_id'].to_numpy(dtype=str)
    old_test_ids = set(master.loc[master.split == 'test', 'obj_id'])
    if old_test_ids & (set(dev_ids) | set(novel_ids)):
        raise ValueError('test/development IDs overlap')
    ids = np.concatenate([dev_ids, novel_ids])
    if len(set(ids)) != len(ids):
        raise ValueError('known/anomaly IDs overlap')
    features = [c for c in master if c not in META]
    for h, d in data.items():
        if set(d.obj_id) != set(master.obj_id):
            raise ValueError(f'horizon {h} has a different object universe')
        if [c for c in d if c not in META] != features:
            raise ValueError('feature order changed across horizons')
        data[h] = d.set_index('obj_id').loc[ids]
    y = np.array([SIX.index(c) for c in data[100].loc[dev_ids, 'cls']])
    for h in HORIZONS:
        if not np.array_equal(data[h].loc[dev_ids, 'cls'], data[100].loc[dev_ids, 'cls']):
            raise ValueError('inconsistent class labels')
    folds = list(KFold(5, shuffle=True, random_state=20261102).split(dev_ids))
    fold_id = np.empty(len(dev_ids), int)
    for k, (_, idx) in enumerate(folds):
        fold_id[idx] = k
    membership = pd.DataFrame(dict(obj_id=dev_ids, fold=fold_id, cls=[SIX[j] for j in y]))
    membership.to_csv(args.out / 'membership.csv', index=False)
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                      protocol_sha256=sha(REPO / 'docs/uq_anytime_protocol.md'),
                      code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                      cache_sha256={f'h{h}': sha(args.cache / f'X_h{h}.parquet') for h in HORIZONS},
                      horizons=HORIZONS.tolist(), seed=20261102, rounds=2000,
                      known_development=len(dev_ids), descriptive_anomalies=len(novel_ids),
                      excluded_test_count=len(old_test_ids), test_predictions_read=False,
                      interpretation='Previously inspected development benchmark; no external validation')
    write_json(args.out / 'provenance.json', provenance)
    ceilings = {}
    for k in range(5):
        ncal = int(np.sum(fold_id == (k+1) % 5))
        maximum = float((ncal+1)/np.sum(np.arange(1, ncal+2, dtype=float)**-2))
        ceilings[str(k)] = dict(n_calibration=ncal, rank_e_max=maximum,
                               first_look_e_budget_max=maximum/4)
    write_json(args.out / 'power_preflight.json', ceilings)
    print('Analytic evidence ceilings: ' + json.dumps(ceilings), flush=True)
    start = time.monotonic()
    results = []
    for k, (_, ev) in enumerate(folds):
        cal = np.flatnonzero(fold_id == (k + 1) % 5)
        proper = np.flatnonzero((fold_id != k) & (fold_id != (k + 1) % 5))
        tr, es = train_test_split(proper, test_size=.15, stratify=y[proper], random_state=100+k)
        assert not set(tr) & set(cal) and not set(es) & set(ev) and not set(proper) & (set(cal) | set(ev))
        model_dir = args.out / f'fold{k}'
        model_dir.mkdir()
        selected = np.concatenate([cal, ev, np.arange(len(dev_ids), len(ids))])
        scores = np.empty((len(selected), 4))
        probs = np.empty((len(selected), 4, 6))
        labels = np.empty((len(selected), 4), int)
        nobs = np.empty((len(selected), 4))
        for look, h in enumerate(HORIZONS):
            X = data[h][features].astype(float)
            model = hierarchy_nocal(X.iloc[:len(dev_ids)], y, tr, es,
                                    SimpleNamespace(threads=args.threads, rounds=2000))
            # Only proper-training early-stop rows define normalization.
            reference = -logsumexp(model.top.predict(X.iloc[es], raw_score=True,
                                                    num_threads=args.threads), axis=1)
            loc = float(np.median(reference))
            scale = max(float(np.quantile(reference, .75)-np.quantile(reference, .25)), 1e-6)
            Xs = X.iloc[selected]
            raw = model.top.predict(Xs, raw_score=True, num_threads=args.threads)
            scores[:, look] = (-logsumexp(raw, axis=1) - loc) / scale
            probs[:, look] = model.predict_proba(Xs)
            labels[:, look] = [SIX.index(c) for c in model.predict(Xs)]
            nobs[:, look] = Xs.n_obs_total
            model.top.save_model(str(model_dir / f'h{h}_top.txt'))
            model.branch.save_model(str(model_dir / f'h{h}_branch.txt'))
            write_json(model_dir / f'h{h}_metadata.json', dict(location=loc, scale=scale,
                       branch_threshold=float(model.branch_threshold), features=features,
                       training_ids=dev_ids[tr].tolist(), early_stop_ids=dev_ids[es].tolist(),
                       top_sha256=sha(model_dir / f'h{h}_top.txt'),
                       branch_sha256=sha(model_dir / f'h{h}_branch.txt')))
            print(f'fold={k} h={h} models saved elapsed={time.monotonic()-start:.0f}s', flush=True)
        nc, ne = len(cal), len(ev)
        np.savez_compressed(model_dir / 'predictions.npz', obj_id=ids[selected],
                            calibration_ids=dev_ids[cal], evaluation_ids=dev_ids[ev], anomaly_ids=novel_ids,
                            scores=scores, probabilities=probs, labels=labels, nobs=nobs,
                            calibration_y=y[cal], evaluation_y=y[ev])
        if not np.isfinite(scores).all() or not np.isfinite(probs).all():
            raise ValueError('nonfinite score/probability')
        cp = scores[:nc]
        qp = scores[nc:]
        p = np.column_stack([rank_pvalues(cp[:, j], qp[:, j]) for j in range(4)])
        envelope = trajectory_pvalues(cp, qp)
        e_budget = e_budget_path(rank_evalues(p, nc), total_looks=4)
        envelope_e = rank_evalues(envelope, nc)
        alarm_paths = dict(naive_per_look=np.minimum.accumulate(p, axis=1),
                           bonferroni=bonferroni_path(p, total_looks=4), trajectory=envelope,
                           e_budget=1 / e_budget, trajectory_e=1 / envelope_e)
        alarms = {name: dict(known=summarize_alarm(path[:ne], .01, y[ev]),
                            anomaly=summarize_alarm(path[ne:], .01)) for name, path in alarm_paths.items()}
        bh = {}
        for name, values in dict(e_budget=e_budget, trajectory_e=envelope_e).items():
            rejected = e_bh(values[:, -1], alpha=.1)
            false = int(rejected[:ne].sum())
            true = int(rejected[ne:].sum())
            bh[name] = dict(n_family=len(rejected), known_rejected=false, anomalies_rejected=true,
                            discoveries=false+true, fdp=false/max(false+true, 1))
        certified = certify_policies(probs[:nc], nobs[:nc], y[cal], labels=labels[:nc],
                                     risk_limit=.05, delta=.05/5)
        write_json(model_dir / 'certification.json', certified)
        policies = {name: summarize_policy(probs[nc:nc+ne], labels[nc:nc+ne], nobs[nc:nc+ne], y[ev],
                                           certified[key]) for name, key in
                    [('confidence_only', 'baseline_winner_index'), ('support_family', 'winner_index')]}
        # Save continuous scores and per-object decisions for reproducible paired analyses.
        np.savez_compressed(model_dir / 'alarm_paths.npz', obj_id=ids[selected][nc:],
                            **alarm_paths, e_budget_values=e_budget, trajectory_e_values=envelope_e)
        result = dict(fold=k, n_train=len(proper), n_calibration=nc, n_evaluation=ne,
                      e_ceiling=float((nc+1)/np.sum(np.arange(1, nc+2, dtype=float)**-2)),
                      alarms=alarms, fixed_batch_e_bh=bh, policies=policies)
        write_json(model_dir / 'result.json', result)
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = dict(provenance=provenance, folds=results, elapsed_seconds=time.monotonic()-start)
    write_json(args.out / 'results.json', summary)
    (args.out / 'COMPLETE').write_text(sha(args.out / 'results.json')+'\n')


if __name__ == '__main__':
    main()
