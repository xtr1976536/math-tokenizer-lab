"""Deployable cost learning with a frozen verified-operation conditional critic."""
import time
import numpy as np
from .learned_scope_stage2 import LearnedScopeTokenizer
from .operation_critic_stage2 import score_model


def source_rate(model,rows):
    # Equal source weighting: four operations do not create four independent inputs.
    groups={}
    for row in rows:
        groups[row['group']]=len(model.encode(row['source_latex']))/len(row['source_latex'].encode())
    return float(np.mean(list(groups.values())))


def optimize(base,reward,teacher,seed,algorithm,rounds=16,samples=4):
    if algorithm not in ('parameter_rl','random_search','cem_search','no_update'):raise ValueError('unknown optimizer')
    v=len(base.vocab);initial=np.zeros(v);mean=initial.copy();rng=np.random.default_rng(seed)
    first=np.zeros(v);second=np.zeros(v);sigma=.25;lagrange=0.;trace=[];started=time.perf_counter()
    reference=source_rate(base,reward);initialcode=score_model(base,reward,teacher)['group_macro_target_byte_bpb']
    pool=[(initialcode,reference,initial)];oracle=1
    def factory(theta):return LearnedScopeTokenizer(base.vocab,base.atoms,base.pairs,base.scope_cost,base.costs,base.token_costs+np.clip(theta,-.8,.8))
    for iteration in range(rounds):
        center=initial if algorithm in ('random_search','no_update') else mean
        proposals=[center+sigma*rng.normal(size=v) for _ in range(samples)] if algorithm!='no_update' else [center.copy() for _ in range(samples)]
        codes=[];rates=[]
        for theta in proposals:
            model=factory(theta);code=score_model(model,reward,teacher)['group_macro_target_byte_bpb'];rate=source_rate(model,reward)
            codes.append(code);rates.append(rate);pool.append((code,rate,theta.copy()));oracle+=1
        rewardvalues=-(np.array(codes)+lagrange*(np.array(rates)-reference))
        advantage=rewardvalues-(rewardvalues.sum()-rewardvalues)/(samples-1)
        gradient=np.mean(np.array([(t-center)/sigma**2 for t in proposals])*advantage[:,None],axis=0)
        gradient/=max(1.,np.linalg.norm(gradient))
        if algorithm=='parameter_rl':
            first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
            mean+=.08*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8)
        elif algorithm=='cem_search':
            indices=np.argsort(-rewardvalues)[:samples//2];mean=.5*mean+.5*np.mean([proposals[k] for k in indices],axis=0)
        lagrange=float(np.clip(lagrange+2*(np.mean(rates)-reference),0,20))
        trace.append({'round':iteration,'mean_operation_target_bpb':float(np.mean(codes)),
                      'mean_source_token_byte':float(np.mean(rates)),'gradient_norm':float(np.linalg.norm(gradient)),'lagrange':lagrange})
    chosen=min(pool,key=lambda x:(x[1]>reference*1.01,max(0,x[1]/(reference*1.01)-1),x[0]))
    model=factory(chosen[2]);model.train_seconds=time.perf_counter()-started
    model.training_log=[{'algorithm':algorithm,'seed':seed,'trace':trace,'oracle_evaluations':oracle,
        'reward_pair_ids':[r['id'] for r in reward],'selected_operation_bpb':chosen[0],'initial_operation_bpb':initialcode,
        'source_rate':chosen[1],'initial_source_rate':reference,'zero_selected':bool(np.count_nonzero(chosen[2])==0),
        'objective':'conditional byte prediction of verified polynomial operations, no math segmentation quality reward',
        'same_training_deployment_encoder':True}]
    return model
