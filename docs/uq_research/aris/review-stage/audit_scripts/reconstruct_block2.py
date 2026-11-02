"""Independent integrity audit of saved allocation outputs; no runner imports."""
import ast, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/.research/flare-uq'); OUT=Path(sys.argv[1]); HERE=Path(__file__).parent
H=np.array([5,10,20,100]); CLASSES=['SN_Ia','SN_CC','SLSN','AGN','TDE','CV']
# Reuse only the auditor's independently written metric definitions, never the experiment code.
tree=ast.parse((HERE/'reconstruct_block1.py').read_text())
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['metrics','equal_metrics']:
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<auditor metrics>','exec'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
checks=[]; failures=[]; sampled=[]; model_checks=[]
def check(name,good,detail=None):
    record=dict(check=name,passed=bool(good),detail=detail);checks.append(record)
    if not good:failures.append(record)
def features(probs,nobs):
    # Every feature at look t accesses probabilities/counts through t only.
    label=probs.argmax(axis=2); sortedp=np.sort(probs,axis=2)
    d={f'probability_{c}':probs[:,:,i] for i,c in enumerate(CLASSES)}
    d['predicted_label_probability']=probs.max(axis=2)
    d['normalized_entropy']=-(probs*np.log(np.maximum(probs,np.finfo(float).tiny))).sum(axis=2)/np.log(6)
    d['top_two_margin']=sortedp[:,:,-1]-sortedp[:,:,-2]
    d.update({f'predicted_label_{c}':(label==i).astype(float) for i,c in enumerate(CLASSES)})
    d['log1p_nobs']=np.log1p(nobs)
    d['log1p_horizon']=np.broadcast_to(np.log1p(H),label.shape)
    change=np.zeros_like(nobs);change[:,1:]=np.abs(probs[:,1:]-probs[:,:-1]).sum(axis=2)
    switch=np.zeros_like(nobs);switch[:,1:]=label[:,1:]!=label[:,:-1]
    d['previous_look_probability_l1']=change;d['previous_look_label_switch']=switch
    d['current_look_index']=np.broadcast_to(np.arange(4),label.shape)
    return d
def design(probs,nobs,names):
    d=features(probs,nobs);return np.stack([d[name] for name in names],axis=2)
def predict(coeff,X):
    return ((X-np.asarray(coeff['scaler_mean']))/np.asarray(coeff['scaler_scale']))@np.asarray(coeff['coefficient'])+coeff['intercept']
results=read(OUT/'results.json')
check('completion_hash',sha(OUT/'results.json')==(OUT/'COMPLETE').read_text().strip())
for path,digest in results['artifacts_sha256'].items():check('artifact:'+path,(OUT/path).is_file() and sha(OUT/path)==digest)
for name,digest in results['provenance']['code_sha256'].items():check('executed_code:'+name,sha(OUT/'code'/name)==digest)
check('copied_protocol_hash',sha(OUT/'code/EXPERIMENT_PLAN.md')==results['provenance']['protocol_sha256'])
keys={(r['regime'],r['alpha'],r['policy'],r['budget_multiplier']) for r in results['pooled']}
policies={'round_robin','random_seed0','random_seed1','random_seed2','confidence','entropy','candidate_count','rare_mass','aggregate_output','macro_output','macro_quality'}
expected={(reg,a,p,b) for reg in ['pooled','mondrian'] for a in [.1,.2] for p in policies for b in [0,.25,.5,1,2,3]}
check('complete_264_pooled_grid',keys==expected and len(results['pooled'])==264)
aggregate={k:[] for k in keys}
for fold in range(5):
    directory=OUT/f'fold{fold}'; inp=np.load(directory/'inputs.npz',allow_pickle=False)
    source=np.load(ROOT/f'rare-budget-v1/fold{fold}/inputs.npz',allow_pickle=False)
    gate=np.load(directory/'gate_training.npz',allow_pickle=False)
    oldgate=np.load(ROOT/f'learned-v1/fold{fold}/gate_training.npz',allow_pickle=False)
    for name in inp.files:check(f'fold{fold}:input_copy:{name}',np.array_equal(inp[name],source[name]))
    for name in gate.files:check(f'fold{fold}:gate_copy:{name}',np.array_equal(gate[name],oldgate[name]))
    nc=len(inp['calibration_ids']); y=inp['evaluation_y']; n=len(y)
    p=inp['probabilities'][nc:]; nobs=inp['nobs'][nc:]; ids=inp['evaluation_ids']
    saved_paths=np.load(directory/'candidate_paths.npz',allow_pickle=False)
    metadata=read(directory/'model_metadata.json')
    for regime in ['pooled','mondrian']:
        for ai,alpha in enumerate([.1,.2]):
            base=f'envelope_{regime}_a{ai}'; cy=inp['calibration_y'];cp=inp['probabilities'][:nc]
            scores=(1-cp[np.arange(nc)[:,None],np.arange(4)[None,:],cy[:,None]]).max(axis=1)
            q=[]
            for c in range(6):
                ss=scores[cy==c] if regime=='mondrian' else scores;k=math.ceil((len(ss)+1)*(1-alpha))
                q.append(np.sort(ss)[k-1] if k<=len(ss) else np.inf)
            q=np.broadcast_to(q,(4,6))
            check(f'fold{fold}:{base}:thresholds',np.array_equal(q,saved_paths[base+'_thresholds']))
            for role,prob in [('evaluation',p),('training',gate['probabilities'])]:
                sets=np.stack([np.all(1-prob[:,:t+1]<=q[None,:t+1],axis=1) for t in range(4)],axis=1)
                check(f'fold{fold}:{base}:{role}_sets',np.array_equal(sets,saved_paths[base+f'_{role}_candidate_sets']))
            sets=saved_paths[base+'_training_candidate_sets']; gy=gate['y'];cc=np.bincount(gy,minlength=6)
            obj=[];look=[];rew=[];macro=[];cost=[];weight=[]
            for i in range(len(sets)):
                sizes=sets[i].sum(axis=1); closed=np.flatnonzero(sizes<=1)
                stop=int(closed[0]) if len(closed) else 3
                eligible=list(range(min(stop,3)))
                for t in eligible:
                    obj.append(i);look.append(t);correct=int(sizes[stop]==1 and sets[i,stop,gy[i]])
                    rew.append(correct);macro.append(correct*len(sets)/(6*cc[gy[i]]));cost.append(stop-t);weight.append(1/len(eligible))
            targets=dict(object_index=obj,look_index=look,reward_aggregate=rew,reward_macro=macro,cost=cost,sample_weight=weight)
            target_saved=np.load(directory/f'{base}_training_targets.npz',allow_pickle=False)
            for name,vals in targets.items():check(f'fold{fold}:{base}:target:{name}',np.allclose(vals,target_saved[name],rtol=1e-12,atol=1e-12))
            if (fold,regime,alpha) in [(0,'pooled',.1),(3,'mondrian',.2)]:
                for variant,record in metadata[base]['models'].items():
                    X=design(gate['probabilities'],gate['nobs'],record['feature_names'])[obj,look]
                    w=np.asarray(weight);mean=np.average(X,axis=0,weights=w);var=np.average((X-mean)**2,axis=0,weights=w)
                    scale=np.where(var>1e-25,np.sqrt(var),1);Z=(X-mean)/scale
                    for target in ['reward','cost']:
                        coeff=record[target+'_coefficients']
                        yy=np.asarray(targets['cost' if target=='cost' else 'reward_aggregate' if variant=='aggregate_output' else 'reward_macro'])
                        if coeff is None:
                            check(f'fold{fold}:{base}:{variant}:{target}_zero_fallback',not np.any(yy>0));continue
                        ym=np.average(yy,weights=w)
                        beta=np.linalg.solve(Z.T@(w[:,None]*Z)+np.eye(Z.shape[1]),Z.T@(w*(yy-ym)))
                        good=(np.allclose(mean,coeff['scaler_mean'],rtol=1e-9,atol=1e-10) and np.allclose(scale,coeff['scaler_scale'],rtol=1e-8,atol=1e-9)
                              and np.allclose(beta,coeff['coefficient'],rtol=1e-6,atol=1e-7) and np.isclose(ym,coeff['intercept'],rtol=1e-8,atol=1e-9))
                        check(f'fold{fold}:{base}:{variant}:{target}_weighted_ridge',good)
                        model_checks.append(dict(fold=fold,base=base,variant=variant,target=target,passed=bool(good)))
    records=results['folds'][fold]['records']
    check(f'fold{fold}:complete_grid',{(r['regime'],r['alpha'],r['policy'],r['budget_multiplier']) for r in records}==expected and len(records)==264)
    # Deep numerical reconstruction spans every policy, both protections and both alpha levels.
    for r in records:
        key=(r['regime'],r['alpha'],r['policy'],r['budget_multiplier']);aggregate[key].append(r['metrics'])
        base=f"envelope_{r['regime']}_a{0 if r['alpha']==.1 else 1}"
        sets=saved_paths[base+'_evaluation_candidate_sets'];path=directory/r['decision_artifact']
        saved=np.load(path,allow_pickle=False);final=saved['final_look'];pred=saved['prediction'];selected=pred>=0
        N=len(saved['action_token']);budget=math.floor(r['budget_multiplier']*n)
        prefix=f'fold{fold}:{key}'
        check(prefix+':budget',N<=budget and int(final.sum())==N and np.array_equal(saved['action_token'],np.arange(1,N+1)))
        check(prefix+':ids',np.array_equal(saved['evaluation_ids'],ids) and np.array_equal(saved['action_obj_id'],ids[saved['action_index']]))
        resource=r['metrics']['resources']
        check(prefix+':resources',resource==dict(initial_checkpoint_queries=n,allocated_extra_tokens=budget,consumed_extra_tokens=N,unused_extra_tokens=budget-N,total_checkpoint_queries=n+N))
        check(prefix+':count_and_truth',r['metrics']['overall']['n_selected']==int(selected.sum()) and r['metrics']['overall']['n_wrong_automatic']==int((selected&(pred!=y)).sum()))
        deep=(fold in [0,3] and r['budget_multiplier']==1 and (r['regime'],r['alpha']) in [('pooled',.1),('mondrian',.2)])
        if not deep:continue
        policy=r['policy']; current=np.zeros(n,int);status=np.full(n,'budget',dtype='U9');prediction=np.full(n,-1)
        def close(i):
            ss=sets[i,current[i]];size=int(ss.sum())
            if size==1:status[i]='automatic';prediction[i]=np.flatnonzero(ss)[0]
            elif size==0:status[i]='empty'
            elif current[i]==3:status[i]='exhausted'
        for i in range(n):close(i)
        active=sorted(np.flatnonzero(status=='budget').tolist(),key=lambda i:(ids[i],i))
        rng=np.random.default_rng(int(policy[-1])) if policy.startswith('random_seed') else None
        if policy in ['aggregate_output','macro_output','macro_quality']:
            model=metadata[base]['models'][policy]
            X=design(p,nobs,model['feature_names'])
            scores=np.zeros((n,4)) if model['constant_reward']==0 else np.maximum(predict(model['reward_coefficients'],X),0)/np.clip(predict(model['cost_coefficients'],X),1,3)
        elif policy=='round_robin':scores=np.broadcast_to(-np.arange(4),(n,4))
        elif policy=='confidence':scores=p.max(axis=2)
        elif policy=='entropy':scores=-(p*np.log(np.maximum(p,np.finfo(float).tiny))).sum(axis=2)
        elif policy=='candidate_count':scores=sets.sum(axis=2)
        elif policy=='rare_mass':scores=p[:,:,[2,4]].sum(axis=2)
        ledger_ok=True;ranking_ok=True;score_ok=True
        for j,i in enumerate(saved['action_index']):
            ledger_ok &= status[i]=='budget' and saved['action_from_look'][j]==current[i] and saved['action_to_look'][j]==current[i]+1
            if rng is not None:
                pos=int(rng.integers(len(active)));ranking_ok &= active[pos]==i
            elif j<50 or j%37==0 or j>=N-50:
                aa=np.flatnonzero(status=='budget');values=scores[aa,current[aa]];maximum=values.max()
                tied=aa[values==maximum];chosen=min(tied,key=lambda k:(ids[k],k))
                ranking_ok &= chosen==i
                score_ok &= np.isclose(saved['action_score'][j],scores[i,current[i]],rtol=1e-9,atol=1e-11)
            current[i]+=1
            ledger_ok &= saved['action_observed_nobs'][j]==nobs[i,current[i]]
            close(i)
            if rng is not None and status[i]!='budget':active[pos]=active[-1];active.pop()
        check(prefix+':full_ledger',ledger_ok)
        check(prefix+':sampled_priority_or_all_random_draws',ranking_ok)
        check(prefix+':score_from_saved_coefficients',score_ok)
        check(prefix+':final_state',np.array_equal(current,final) and np.array_equal(status,saved['status']) and np.array_equal(prediction,pred))
        m=metrics(sets,y,pred,final,nobs,np.ones(n,bool));check(prefix+':all_overall_metrics',equal_metrics(m,r['metrics']['overall']))
        for ci,c in enumerate(CLASSES):check(prefix+':class_metrics:'+c,equal_metrics(metrics(sets,y,pred,final,nobs,y==ci),r['metrics']['by_true_class'][c]))
        sampled.append(dict(fold=fold,regime=r['regime'],alpha=r['alpha'],policy=policy,budget_multiplier=1,n_actions=N))
for r in results['pooled']:
    key=(r['regime'],r['alpha'],r['policy'],r['budget_multiplier']);fs=aggregate[key]
    for group in ['overall']+CLASSES:
        groups=[m['overall'] if group=='overall' else m['by_true_class'][group] for m in fs]
        target=r['metrics']['overall'] if group=='overall' else r['metrics']['by_true_class'][group]
        names=[name for name in target if name.startswith('n_')]+['total_nobs_consumed']
        check(f'pooled:{key}:{group}:counts',all(np.isclose(sum(m[name] for m in groups),target[name]) for name in names))
    check(f'pooled:{key}:macro',np.isclose(r['metrics']['macro_true_class_correct_automatic_rate'],np.mean([r['metrics']['by_true_class'][c]['n_correct_automatic']/r['metrics']['by_true_class'][c]['n_objects'] for c in CLASSES])))
report=dict(audit='independent raw-artifact reconstruction; no experiment implementation imports',output_directory=str(OUT),n_checks=len(checks),n_failures=len(failures),failures=failures,
            sampled_cells=sampled,n_deep_cells=len(sampled),weighted_ridge_reconstructions=model_checks,checks=checks,
            audited_input_hashes={str(OUT/'results.json'):sha(OUT/'results.json'),str(OUT/'provenance.json'):sha(OUT/'provenance.json')})
(HERE/'block2_reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','sampled_cells','weighted_ridge_reconstructions']},indent=2))
