"""Sampled replay of frozen boosters from original feature caches; no study imports."""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path('/workspace/.research/flare-uq'); checks=[]
for fold in [0,3]:
    inp=np.load(ROOT/f'rare-budget-v1/fold{fold}/inputs.npz',allow_pickle=False)
    gate=np.load(ROOT/f'learned-v1/fold{fold}/gate_training.npz',allow_pickle=False)
    nc=len(inp['calibration_ids']);ne=len(inp['evaluation_ids'])
    cal=np.linspace(0,nc-1,15,dtype=int);ev=nc+np.linspace(0,ne-1,15,dtype=int);gg=np.linspace(0,len(gate['obj_id'])-1,15,dtype=int)
    chosen=np.concatenate([cal,ev]);ids=np.concatenate([inp['obj_id'][chosen],gate['obj_id'][gg]])
    for look,h in enumerate([5,10,20,100]):
        folder=ROOT/f'study-v1/fold{fold}';meta=json.loads((folder/f'h{h}_metadata.json').read_text())
        frame=pd.read_parquet(ROOT/f'cache/X_h{h}.parquet').set_index('obj_id').loc[ids]
        X=frame[meta['features']].astype(float)
        top=lgb.Booster(model_file=str(folder/f'h{h}_top.txt'));branch=lgb.Booster(model_file=str(folder/f'h{h}_branch.txt'))
        p=top.predict(X,num_threads=1);b=branch.predict(X,num_threads=1)
        out=np.column_stack([p[:,0],p[:,1]*(1-b),p[:,1]*b,p[:,2],p[:,3],p[:,4]])
        expected=np.concatenate([inp['probabilities'][chosen,look],gate['probabilities'][gg,look]])
        counts=np.concatenate([inp['nobs'][chosen,look],gate['nobs'][gg,look]])
        checks.append(dict(fold=fold,horizon=h,n_objects=len(ids),max_probability_difference=float(np.max(np.abs(expected-out))),
                           probability_match=bool(np.allclose(expected,out,rtol=0,atol=1e-12)),
                           nobs_match=bool(np.array_equal(counts,frame.n_obs_total.to_numpy()))))
report=dict(checks=checks,n_predictions=sum(c['n_objects'] for c in checks),
            n_failures=sum(not(c['probability_match'] and c['nobs_match']) for c in checks))
Path(__file__).with_name('saved_prediction_replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
