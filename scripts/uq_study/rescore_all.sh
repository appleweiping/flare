#!/bin/bash
# Amendment-2 rescoring of every candidate on the corrected harness (first scoring kept in results_v1_preaudit/)
cd "$(dirname "$0")"
PY="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ${PYTHON:-python}"
run(){ echo "== $1"; $PY run_method.py "$2" --name "$1" --kw "$3" 2>&1 | grep -v -i warn | tail -9; }
run B0 harness:B0 '{}'
run B1 harness:B1 '{}'
for p in atyp nobs ctx nctx; do run M1_$p m1_mondrian:M1 "{\"partition\": \"$p\", \"min_cell\": 10, \"aug\": 0}"; run M1_${p}_aug3 m1_mondrian:M1 "{\"partition\": \"$p\", \"min_cell\": 10, \"aug\": 3}"; done
run M2 m2_condconf:M2 '{"aug": 0}'
run M2_aug3 m2_condconf:M2 '{"aug": 3}'
run M2_adj m2_condconf:M2 '{"aug": 0, "adjust": true}'
run M2_aug3_adj m2_condconf:M2 '{"aug": 3, "adjust": true}'
run M2_aug3_adj_nctx m2_condconf:M2 '{"aug": 3, "adjust": true, "nctx": true}'
run M4 m4_weighted:M4 '{"ratio": "logistic"}'
run M4_lgbm_explore m4_weighted:M4 '{"ratio": "lgbm"}'
run M5_aps m5_aps:M5 '{}'
run M5_raps001 m5_aps:M5 '{"lam": 0.01, "k_reg": 1}'
run M5_raps01 m5_aps:M5 '{"lam": 0.1, "k_reg": 1}'
run M5_aps_rand m5_aps:M5 '{"randomized": true, "seed": 31415}'
run M6 m6_atyp_nonempty:M6 '{"aug": 3}'
echo "== T1"; (cd methods && $PY t1_trigger.py --study 2>&1 | grep -v -i warn | tail -3)
echo ALLDONE
