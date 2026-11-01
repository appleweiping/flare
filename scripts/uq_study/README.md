# Conditional-calibration study: code

Reproduces [docs/uq_conditional_calibration.md](../../docs/uq_conditional_calibration.md) under the design
of [docs/uq_preregistration.md](../../docs/uq_preregistration.md). Data and intermediate arrays go to
`$FLARE_UQ_WORK` (default `paper_data/uq`); the benchmark working copy is `$FLARE_PAPER_DATA`
(`python scripts/setup_data.py`). CPU only; about 30 min on 4 cores, most of it LightGBM fits.

```bash
export FLARE_PAPER_DATA=$PWD/paper_data FLARE_UQ_WORK=$PWD/paper_data/uq
cd scripts/uq_study
# 1. prefix-causal feature caches at days 5/10/20/100 and the 30%-thinned day-100 curves
python build_cache.py ../.. $FLARE_PAPER_DATA $FLARE_UQ_WORK/cache
# 2. five-fold cross-fits of the repository recipe, all 16 context patterns; final models on test
for t in h100 h20 h10 h5; do python oof.py ../.. $FLARE_UQ_WORK/cache $t $FLARE_UQ_WORK/oofs; done
python oof.py ../.. $FLARE_UQ_WORK/cache thin100 $FLARE_UQ_WORK/oofs --deploy-tag h100
python oof_shift.py ../.. thin100 h100 && python oof_shift.py ../.. h100 thin100
mkdir -p $FLARE_UQ_WORK/sealed && mv $FLARE_UQ_WORK/oofs/final_*_on_test*.npz $FLARE_UQ_WORK/sealed/
#    (final_h100_on_test_thin100.npz: the final day-100 model on the thinned test curves, see final_eval.py)
# 3. development scoring of every candidate and the registered decisions
./rescore_all.sh && python decide.py
# 4. the one-time test read (already spent for this benchmark)
python final_eval.py m6_atyp_nonempty:M6 --name M6 --kw '{"aug": 3}'; python final_eval_T1.py
# 5. nested, deployment-faithful protocol (amendment 3)
for t in h100 h20; do python nested.py ../.. $t && python nested_eval.py $t; done
python make_report.py
```

| file | role |
|---|---|
| `harness.py` | views, leave-one-fold-out and shift protocols, groups, endpoint U, paired object bootstrap, B0/B1 |
| `methods/` | M1 (class x group Mondrian), M2 (conditional conformal), M4 (weighted conformal), M5 (APS/RAPS), M6, T1 |
| `run_method.py`, `decide.py` | standard scoring of one candidate on every registered scenario; the registered rules |
| `nested.py`, `nested_eval.py` | nested cross-fit mirroring `scripts/train_full_cc.py`; D0, CC, CC6, M6 on the refit |
| `final_eval.py`, `final_eval_T1.py` | the test read |

Result files of the run reported in the document are in `docs/uq_study_results/`.
