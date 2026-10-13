"""Actual deployed MAP paths optimized by a shared frozen neural byte critic.

Segmentation observes the whole formula. Conditional neural reconstruction code
is a proxy, not a proper unconditional LM likelihood. Costs remain in deployment.
"""
import hashlib,time
import numpy as np
from .learned_scope_stage2 import LearnedScopeTokenizer
from .signal_stage2 import paper_token_rate


def neural_path_code(model,rows,teacher):
    groups={};enc=[]
    for row in rows:
        ids=model.encode(row['latex']);enc.append(ids);raw=row['latex'].encode();cuts=np.zeros(len(raw)+1,dtype=bool);cuts[0]=cuts[-1]=True
        pos=0
        for id in ids:pos+=len(model.vocab[id]);cuts[pos]=True
        bits,_=teacher.score(raw,cuts);b,n=groups.get(row['split_group'],(0.,0));groups[row['split_group']]=(b+float(bits.sum()),n+len(raw))
    return float(np.mean([b/n for b,n in groups.values()])),paper_token_rate(enc,rows)


def optimize(base,reward,teacher,seed,algorithm,rounds=32,samples=4,sigma=.25):
    if algorithm not in ('parameter_rl','random_search','cem_search','no_update'):raise ValueError('unknown optimizer')
    v=len(base.vocab);initial=np.zeros(v);mean=initial.copy();rng=np.random.default_rng(seed);first=np.zeros(v);second=np.zeros(v)
    started=time.perf_counter();initialcode,reference=neural_path_code(base,reward,teacher);pool=[(initialcode,reference,initial)]
    lagrange=0.;trace=[]
    def model_for(theta):return LearnedScopeTokenizer(base.vocab,base.atoms,base.pairs,base.scope_cost,base.costs,base.token_costs+np.clip(theta,-.8,.8))
    for iteration in range(rounds):
        center=initial if algorithm in ('random_search','no_update') else mean
        theta=[center+sigma*rng.normal(size=v) for _ in range(samples)] if algorithm!='no_update' else [center.copy() for _ in range(samples)]
        codes=[];rates=[]
        for t in theta:
            code,rate=neural_path_code(model_for(t),reward,teacher);codes.append(code);rates.append(rate);pool.append((code,rate,t.copy()))
        utility=np.array(codes)+lagrange*(np.array(rates)-reference);r=-utility;adv=r-(r.sum()-r)/(samples-1)
        gradient=np.mean(np.array([(t-center)/sigma**2 for t in theta])*adv[:,None],axis=0)
        gradient/=max(1.,np.linalg.norm(gradient))
        if algorithm=='parameter_rl':
            first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
            mean+=.08*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8)
        elif algorithm=='cem_search':mean=.5*mean+.5*np.mean([theta[k] for k in np.argsort(utility)[:samples//2]],axis=0)
        lagrange=float(np.clip(lagrange+2*(np.mean(rates)-reference),0,20))
        trace.append({'round':iteration,'mean_conditional_neural_bpb':float(np.mean(codes)),'mean_tokens_per_byte':float(np.mean(rates)),
                      'gradient_norm':float(np.linalg.norm(gradient)),'lagrange':lagrange})
    chosen=min(pool,key=lambda x:(x[1]>reference*1.01,max(0,x[1]/(reference*1.01)-1),x[0]))
    model=model_for(chosen[2]);model.train_seconds=time.perf_counter()-started
    model.training_log=[{'algorithm':algorithm,'seed':seed,'trace':trace,'oracle_evaluations':rounds*samples+1,
          'selected_neural_code_bpb':chosen[0],'initial_code_bpb':initialcode,'selected_rate':chosen[1],'initial_rate':reference,
          'zero_parameter_selected':bool(np.count_nonzero(chosen[2])==0),'same_training_and_deployment_MAP':True,
          'reward_ids':[r['id'] for r in reward],'teacher_parameters':teacher.memory_summary(),
          'objective':'conditional frozen neural patch code; no mathematical structure metrics'}]
    return model
