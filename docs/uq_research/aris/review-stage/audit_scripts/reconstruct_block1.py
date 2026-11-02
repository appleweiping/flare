"""Independent audit reconstruction; imports no experiment implementation."""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/workspace/.research/flare-uq')
OUT = ROOT / 'rare-budget-v1'
REPORT = Path(__file__).with_name('block1_reconstruction.json')
CLASSES = ['SN_Ia','SN_CC','SLSN','AGN','TDE','CV']
H = np.array([5,10,20,100])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
checks, failures, hashes = [], [], {}
def check(name, good, detail=None):
    item = dict(check=name, passed=bool(good), detail=detail)
    checks.append(item)
    if not good: failures.append(item)
for directory in ['study-v1', 'learned-v1', 'rare-budget-v1']:
    d = ROOT / directory
    check(directory+'_complete_hash', sha(d/'results.json') == (d/'COMPLETE').read_text().strip())
results = read(OUT/'results.json')
for p, digest in results['artifacts_sha256'].items():
    check('artifact:'+p, (OUT/p).is_file() and sha(OUT/p)==digest)
for p, digest in read(OUT/'summary_manifest.json').items():
    check('summary_artifact:'+p, (OUT/p).is_file() and sha(OUT/p)==digest)
for p,digest in results['provenance']['code_sha256'].items():
    check('executed_code:'+p, sha(OUT/'code'/p)==digest)
membership = pd.read_csv(ROOT/'study-v1/membership.csv').set_index('obj_id')
source = pd.concat([pd.read_csv(ROOT/'paper_data/data/splits'/f'{s}.csv') for s in ['train','val']]).set_index('obj_id')
check('dataset_manifest_exact_ids', set(source.index)==set(membership.index))
check('dataset_manifest_exact_truth', source.loc[membership.index,'cls'].equals(membership.cls))
source_prov=read(ROOT/'study-v1/provenance.json')
for horizon in H:
    p=ROOT/'cache'/f'X_h{horizon}.parquet'
    check(f'cache_hash_h{horizon}',sha(p)==source_prov['cache_sha256'][f'h{horizon}'])
    frame=pd.read_parquet(p,columns=['obj_id','split','cls','n_obs_total']).set_index('obj_id')
    check(f'cache_dataset_truth_h{horizon}',frame.loc[membership.index,'cls'].equals(membership.cls))
    check(f'development_roles_h{horizon}',frame.loc[membership.index,'split'].isin(['train','val']).all())

sample_keys = [('envelope_mondrian',a,100) for a in [.01,.05,.1,.2]]
sample_keys += [('bonferroni_mondrian',a,20) for a in [.05,.2]]
sample_keys += [('fixedlook_mondrian',.1,int(h)) for h in H]
sample_keys += [('envelope_pooled',.1,100),('terminal_mondrian',.2,100),('bonferroni_pooled',.1,20)]
sample_records=[]
pooled_arrays={key:[] for key in sample_keys}
seen=[]
calibration_counts=[]
def metrics(sets,y,pred,stop,nobs,mask):
    selected=pred>=0; wrong=selected&(pred!=y); rows=np.arange(len(y))
    truth=sets[rows[:,None],np.arange(4)[None,:],y[:,None]]
    lifetime=~truth.all(axis=1)
    consumed=((~truth)&(np.arange(4)[None,:]<=stop[:,None])).any(axis=1)
    retained=(~selected)&truth[rows,stop]
    size=sets[rows,stop].sum(axis=1)
    n=int(mask.sum()); ns=int((mask&selected).sum()); nw=int((mask&wrong).sum()); nu=n-ns
    total=float(nobs[rows,stop][mask].sum())
    return dict(n_objects=n,n_selected=ns,n_unresolved=nu,n_wrong_automatic=nw,n_correct_automatic=ns-nw,
      n_true_class_retained_unresolved=int((mask&retained).sum()),n_true_class_lifetime_excluded=int((mask&lifetime).sum()),
      n_true_class_consumed_prefix_excluded=int((mask&consumed).sum()),
      wrong_automatic_per_all=nw/n if n else None,correct_automatic_per_all=(ns-nw)/n if n else None,
      selected_error=nw/ns if ns else None,selection_coverage=ns/n if n else None,
      true_class_lifetime_exclusion=float(lifetime[mask].mean()) if n else None,
      true_class_consumed_prefix_exclusion=float(consumed[mask].mean()) if n else None,
      true_class_retained_unresolved_per_all=float(retained[mask].mean()) if n else None,
      true_class_retained_given_unresolved=int((mask&retained).sum())/nu if nu else None,
      mean_set_size_at_consumed=float(size[mask].mean()) if n else None,
      mean_set_size_unresolved=float(size[mask&~selected].mean()) if nu else None,
      mean_day_selected=float(H[stop][mask&selected].mean()) if ns else None,
      mean_day_consumed_all=float(H[stop][mask].mean()) if n else None,
      total_nobs_consumed=total,mean_nobs_consumed_all=total/n if n else None)
def equal_metrics(expected,actual):
    return all((actual[k] is None if v is None else actual[k] is not None and np.isclose(v,actual[k],rtol=1e-12,atol=1e-12)) for k,v in expected.items())
for fold in range(5):
    inp=np.load(OUT/f'fold{fold}/inputs.npz',allow_pickle=False)
    orig=np.load(ROOT/f'study-v1/fold{fold}/predictions.npz',allow_pickle=False)
    saved=np.load(OUT/f'fold{fold}/sets_decisions.npz',allow_pickle=False)
    gate=np.load(ROOT/f'learned-v1/fold{fold}/gate_training.npz',allow_pickle=False)
    nc=len(inp['calibration_ids']); ne=len(inp['evaluation_ids']); seen.extend(inp['evaluation_ids'].tolist())
    calibration_counts.append(np.bincount(inp['calibration_y'],minlength=6).tolist())
    for key in inp.files:
        origkey='labels' if key=='hierarchy_labels' else key
        val=orig[origkey]
        if key in ['obj_id','probabilities','nobs','hierarchy_labels']: val=val[:nc+ne]
        check(f'fold{fold}:source_copy:{key}',np.array_equal(inp[key],val))
    roles=[]
    for horizon in H:
        meta=read(ROOT/f'study-v1/fold{fold}/h{horizon}_metadata.json')
        roles.append((set(meta['training_ids']),set(meta['early_stop_ids'])))
        for part in ['top','branch']:
            check(f'fold{fold}:model_h{horizon}_{part}',sha(ROOT/f'study-v1/fold{fold}/h{horizon}_{part}.txt')==meta[f'{part}_sha256'])
    cal=set(inp['calibration_ids']); ev=set(inp['evaluation_ids']); train,es=roles[0]
    check(f'fold{fold}:roles_fixed',all(x==roles[0] for x in roles))
    check(f'fold{fold}:four_way_disjoint',len(train|es|cal|ev)==len(train)+len(es)+len(cal)+len(ev))
    check(f'fold{fold}:roles_cover_development',train|es|cal|ev==set(membership.index))
    check(f'fold{fold}:gate_is_earlystop',set(gate['obj_id'])==es)
    check(f'fold{fold}:gate_truth',np.array_equal(gate['y'],[CLASSES.index(c) for c in membership.loc[gate['obj_id'],'cls']]))
    for role,fold_expected in [('calibration',(fold+1)%5),('evaluation',fold)]:
        ids=inp[role+'_ids']; yy=inp[role+'_y']
        check(f'fold{fold}:{role}_membership',set(ids)==set(membership.index[membership.fold==fold_expected]))
        check(f'fold{fold}:{role}_truth',np.array_equal(yy,[CLASSES.index(c) for c in membership.loc[ids,'cls']]))
    recs={(r['arm'],r['alpha'],r['budget_day']):r for r in results['folds'][fold]['records']}
    check(f'fold{fold}:96_unique_cells',len(recs)==96)
    cp,qp=inp['probabilities'][:nc],inp['probabilities'][nc:]; cy=inp['calibration_y']; y=inp['evaluation_y']; nobs=inp['nobs'][nc:]
    for key in sample_keys:
        arm,alpha,day=key; cap=list(H).index(day); r=recs[key]; savedkey=r['calibration_key']
        method=arm.split('_')[0]; mondrian=arm.endswith('mondrian'); q=np.full((4,6),np.inf)
        true_scores=1-cp[np.arange(nc)[:,None],np.arange(4)[None,:],cy[:,None]]
        for c in range(6):
            ss=true_scores[cy==c] if mondrian else true_scores
            levels=[cap] if method=='fixedlook' else [3] if method=='terminal' else range(4)
            for t in levels:
                vals=ss.max(axis=1) if method=='envelope' else ss[:,t]
                aa=alpha/4 if method=='bonferroni' else alpha
                k=math.ceil((len(vals)+1)*(1-aa))
                q[t,c]=np.sort(vals)[k-1] if k<=len(vals) else np.inf
        sets=np.empty(qp.shape,bool)
        for t in range(4): sets[:,t]=np.all(1-qp[:,:t+1]<=q[None,:t+1],axis=1)
        check(f'fold{fold}:{key}:thresholds',np.array_equal(q,saved[savedkey+'_thresholds']))
        check(f'fold{fold}:{key}:sets',np.array_equal(sets,saved[savedkey+'_candidate_sets']))
        pred=np.full(ne,-1); stop=np.full(ne,cap)
        for i in range(ne):
            for t in range(cap+1):
                if sets[i,t].sum()==1:
                    pred[i]=np.flatnonzero(sets[i,t])[0];stop[i]=t;break
        check(f'fold{fold}:{key}:prediction',np.array_equal(pred,saved[f'{savedkey}_b{day}_prediction']))
        check(f'fold{fold}:{key}:stop_index',np.array_equal(stop,saved[f'{savedkey}_b{day}_stop_index']))
        reconstructed={'overall':metrics(sets,y,pred,stop,nobs,np.ones(ne,bool)),
                       'by_true_class':{c:metrics(sets,y,pred,stop,nobs,y==ci) for ci,c in enumerate(CLASSES)}}
        check(f'fold{fold}:{key}:overall_metrics',equal_metrics(reconstructed['overall'],r['metrics']['overall']))
        for c in CLASSES: check(f'fold{fold}:{key}:{c}_metrics',equal_metrics(reconstructed['by_true_class'][c],r['metrics']['by_true_class'][c]))
        pooled_arrays[key].append((sets,y,pred,stop,nobs))
        sample_records.append(dict(fold=fold,arm=arm,alpha=alpha,day=day,metrics=reconstructed))
check('evaluation_objects_exactly_once',len(seen)==len(set(seen))==8922 and set(seen)==set(membership.index))
raw_index={(r['arm'],r['alpha'],r['budget_day']):r for r in results['pooled']}
diagnostics=read(OUT/'prediction_diagnostics.json')
diag_index={(r['arm'],r['alpha'],r['budget_day']):r for r in diagnostics['pooled']}
check('summary_pooled_matches_results',read(OUT/'summary.json')['pooled']==results['pooled'])
for csv_name,which in [('compact_frontier.csv','overall'),('classwise_frontier.csv','by_true_class'),('emitted_label_precision.csv','by_predicted_class')]:
    rows=list(csv.DictReader((OUT/csv_name).open()))
    for i,row in enumerate(rows):
        key=(row['arm'],float(row['alpha']),int(row['budget_day']))
        if which=='by_predicted_class': target=diag_index[key][which][row['predicted_class']]
        elif which=='by_true_class': target=raw_index[key]['metrics'][which][row['true_class']]
        else: target=raw_index[key]['metrics'][which]
        check(f'{csv_name}:row{i}',all(row[k]=='' if v is None else np.isclose(float(row[k]),v,rtol=1e-12,atol=1e-12) for k,v in target.items()))
pooled_reconstructed=[]
for key,arrays in pooled_arrays.items():
    sets,y,pred,stop,nobs=[np.concatenate([a[i] for a in arrays]) for i in range(5)]
    rr=next(r for r in results['pooled'] if (r['arm'],r['alpha'],r['budget_day'])==key)
    m=metrics(sets,y,pred,stop,nobs,np.ones(len(y),bool))
    check(f'pooled:{key}:overall_metrics',equal_metrics(m,rr['metrics']['overall']))
    by_class={c:metrics(sets,y,pred,stop,nobs,y==ci) for ci,c in enumerate(CLASSES)}
    for c in CLASSES: check(f'pooled:{key}:{c}_metrics',equal_metrics(by_class[c],rr['metrics']['by_true_class'][c]))
    confusion=np.zeros((6,6),int);sel=pred>=0;np.add.at(confusion,(y[sel],pred[sel]),1)
    check(f'pooled:{key}:emitted_confusion',np.array_equal(confusion,diag_index[key]['confusion_true_rows_predicted_columns']))
    pooled_reconstructed.append(dict(arm=key[0],alpha=key[1],day=key[2],metrics=m,by_true_class=by_class,confusion=confusion.tolist()))
for p in [OUT/'results.json',OUT/'summary_manifest.json',ROOT/'study-v1/membership.csv',ROOT/'study-v1/provenance.json']:
    hashes[str(p)]=sha(p)
report=dict(audit='independent reconstruction, no experiment implementation imported',n_checks=len(checks),n_failures=len(failures),
            failures=failures,calibration_counts=calibration_counts,n_fold_cells_reconstructed=len(sample_records),
            pooled_reconstructed=pooled_reconstructed,checks=checks,audited_input_hashes=hashes)
REPORT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','pooled_reconstructed']},indent=2))
for r in pooled_reconstructed:
    print(r['arm'],r['alpha'],r['day'],{k:r['metrics'][k] for k in ['n_selected','n_correct_automatic','n_wrong_automatic','selected_error','mean_nobs_consumed_all']})
