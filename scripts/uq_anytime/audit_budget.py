from __future__ import annotations
import hashlib
import numpy as np
from scripts.uq_anytime.budget import allocate_query_budget

POLICIES = ('round_robin','confidence','entropy','candidate_count','rare_mass','custom')

def custom(obs):
    return float(obs.probabilities[-1,0] + .001*obs.nobs[-1] - .01*obs.look)

def kwargs(policy):
    return {'policy':policy, 'rare_indices':(0,1), **({'scorer':custom} if policy=='custom' else {})}

def score(policy, index, look, sets, probs, nobs):
    if policy=='round_robin': return -float(look)
    if policy=='candidate_count':return float(sets[index,look].sum())
    if policy=='confidence':return float(probs[index,look].max())
    if policy=='entropy':
        p=probs[index,look]; positive=p>0
        return float(-np.sum(p[positive]*np.log(p[positive])))
    if policy=='rare_mass':return float(probs[index,look,:2].sum())
    return float(probs[index,look,0]+.001*nobs[index,look]-.01*look)

def reference(sets, probs, nobs, ids, budget, policy):
    n,L,_=sets.shape;looks=np.zeros(n,dtype=int);prediction=np.full(n,-1);status=np.full(n,'budget',dtype='U9');actions=[]
    def classify(i):
        possible=np.flatnonzero(sets[i,looks[i]])
        if len(possible)==0:status[i]='empty'
        elif len(possible)==1:status[i]='automatic';prediction[i]=possible[0]
        elif looks[i]==L-1:status[i]='exhausted'
    for i in range(n):classify(i)
    for token in range(budget):
        unresolved=[i for i in range(n) if status[i]=='budget']
        if not unresolved:break
        chosen=min(unresolved,key=lambda i:(-score(policy,i,looks[i],sets,probs,nobs),ids[i],i))
        before=looks[chosen];looks[chosen]+=1
        actions.append((ids[chosen],int(before),int(looks[chosen])))
        classify(chosen)
    return looks,prediction,status,actions

rng=np.random.default_rng(543802);comparisons=0;causal=0;random_checks=0
for case in range(80):
    n=int(rng.integers(1,7));L=int(rng.integers(1,7));K=int(rng.integers(2,7))
    eliminations=rng.integers(0,L+1,size=(n,K));sets=np.arange(L)[None,:,None]<eliminations[:,None,:]
    probs=rng.dirichlet(np.ones(K),size=(n,L));nobs=np.cumsum(rng.integers(0,20,size=(n,L)),axis=1)
    ids=[f'obj-{int(i):02d}' for i in rng.permutation(n)]
    for policy in POLICIES:
        previous=()
        for budget in range(n*(L-1)+3):
            actual=allocate_query_budget(sets,probs,nobs,ids,budget,**kwargs(policy))
            expected=reference(sets,probs,nobs,ids,budget,policy)
            for got,want in zip((actual.final_look,actual.predictions,actual.status),expected[:3]):np.testing.assert_array_equal(got,want)
            assert [(a.obj_id,a.from_look,a.to_look) for a in actual.actions]==expected[3]
            assert actual.actions[:len(previous)]==previous
            assert actual.consumed_tokens==len(actual.actions)==int(actual.final_look.sum())<=budget
            assert all(np.isfinite(a.score) for a in actual.actions)
            previous=actual.actions;comparisons+=1
        baseline=allocate_query_budget(sets,probs,nobs,ids,n*(L-1)+1,**kwargs(policy))
        revealed=np.zeros(n,dtype=int)
        for t,action in enumerate(baseline.actions):
            changed_sets=sets.copy();changed_probs=probs.copy();changed_nobs=nobs.copy()
            for i,look in enumerate(revealed):
                changed_sets[i,look+1:]=False
                changed_probs[i,look+1:]=0;changed_probs[i,look+1:,0]=1
                changed_nobs[i,look+1:]=10**12
            changed=allocate_query_budget(changed_sets,changed_probs,changed_nobs,ids,t+1,**kwargs(policy))
            assert [(a.obj_id,a.from_look,a.to_look) for a in changed.actions]==[(a.obj_id,a.from_look,a.to_look) for a in baseline.actions[:t+1]]
            revealed[action.index]+=1;causal+=1
    first=allocate_query_budget(sets,probs,nobs,ids,n*(L-1)+2,policy='random',seed=738)
    repeat=allocate_query_budget(sets,probs,nobs,ids,n*(L-1)+2,policy='random',seed=738)
    assert first.actions==repeat.actions
    for budget in range(first.consumed_tokens+2):
        actual=allocate_query_budget(sets,probs,nobs,ids,budget,policy='random',seed=738)
        assert actual.actions==first.actions[:budget]
        assert actual.consumed_tokens==len(actual.actions)==int(actual.final_look.sum())<=budget
        random_checks+=1
# Exact probability ties never force an argmax label on unresolved/empty sets.
sets=np.array([[[1,1],[1,1]],[[0,0],[0,0]],[[1,0],[0,0]]],dtype=bool)
result=allocate_query_budget(sets,np.full(sets.shape,.5),np.ones((3,2)),['tied','empty','single'],100)
assert result.predictions.tolist()==[-1,-1,0]
assert result.status.tolist()==['exhausted','empty','automatic']
print({'deterministic_full_recompute_comparisons':comparisons,'future_perturbation_checks':causal,'seeded_random_prefix_checks':random_checks,'probability_tie_and_empty_case':'passed','numpy':np.__version__,'script_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest()})
