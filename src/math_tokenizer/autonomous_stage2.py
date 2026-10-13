"""Causal, annotation-free boundary RL with raw-byte prediction feedback.

This is a statistical pilot, not Dauncey's neural U-net reproduction. The
fit-only predictor is frozen across optimizers so the oracle is identical.
Policies contain one parameter per raw-byte context, not mathematical features.
Inference projection onto a dictionary is evaluated separately.
"""
from collections import Counter,defaultdict
from dataclasses import asdict,dataclass
import time
import numpy as np
from .signal_stage2 import CONTEXTS,contexts


@dataclass(frozen=True)
class PolicyConfig:
    rounds:int=16
    samples:int=4
    reward_formulas:int=48
    max_patch:int=32
    prefix_bytes:int=8
    previous_bytes:int=4
    smoothing:float=8.
    learning_rate:float=.12
    discount:float=1.
    proposal_scale:float=.25


def probabilities(logits):return 1/(1+np.exp(-np.clip(np.asarray(logits),-12,12)))


def sample_boundaries(raw,logits,rng,max_patch=32):
    if max_patch<1:raise ValueError('positive patch limit required')
    n=len(raw);cs=contexts(raw);p=probabilities(logits[cs]);u=rng.random(n+1)
    cuts=np.zeros(n+1,dtype=bool);active=np.zeros(n+1,dtype=bool);cuts[0]=cuts[n]=True;last=0
    for i in range(1,n):
        forced=i-last>=max_patch
        active[i]=not forced;cuts[i]=forced or u[i]<p[i]
        if cuts[i]:last=i
    return cuts,active,cs,p


class PatchPredictor:
    """Predict x[i] from completed-patch suffix + current prefix, all < i.

    Smoothed conditional counts back off to a raw first-order byte predictor.
    Lookup state length is bounded and shared across optimizer comparisons.
    """
    def __init__(self,base,config):
        self.base=base;self.config=config;self.counts=defaultdict(Counter);self.totals=Counter()
    def state(self,raw,i,start,previous):
        return (previous,raw[max(start,i-self.config.prefix_bytes):i])
    def fit(self,rows,logits,seed):
        rng=np.random.default_rng(seed)
        for r in rows:
            raw=r['latex'].encode();cuts,_,_,_=sample_boundaries(raw,logits,rng,self.config.max_patch)
            start=0;previous=b''
            for i,b in enumerate(raw):
                if i and cuts[i]:previous=raw[max(start,i-self.config.previous_bytes):i];start=i
                key=self.state(raw,i,start,previous);self.counts[key][b]+=1;self.totals[key]+=1
        return self
    def score(self,raw,cuts):
        bits=np.empty(len(raw));gain=np.empty(len(raw));start=0;previous=b'';beta=self.config.smoothing
        for i,b in enumerate(raw):
            if i and cuts[i]:previous=raw[max(start,i-self.config.previous_bytes):i];start=i
            key=self.state(raw,i,start,previous);total=self.totals.get(key,0)
            back=float(self.base[raw[i-1] if i else 256,b])
            count=self.counts.get(key,{}).get(b,0)
            p=(count+beta*back)/(total+beta)
            bits[i]=-np.log2(p);gain[i]=np.log2(p/back)
        return bits,gain
    def memory_summary(self):
        return {'states':len(self.counts),'observed_state_byte_counts':sum(map(len,self.counts.values())),
                'context':'completed patch suffix + current prefix; no future byte or current target'}


def reward_to_go(reward,discount=1.):
    if not 0<discount<=1:raise ValueError('discount must be in (0,1]')
    out=np.empty(len(reward));value=0.
    for i in range(len(reward)-1,-1,-1):value=reward[i]+discount*value;out[i]=value
    return out


def boundary_gradient(samples,discount=1.):
    """Samples of the SAME formula; LOO baseline excludes each own action path."""
    if len(samples)<2:raise ValueError('independent rollouts required')
    returns=np.array([reward_to_go(s['reward'],discount) for s in samples]);n=returns.shape[1]
    advantages=returns-(returns.sum(axis=0)-returns)/(len(samples)-1)
    gradient=np.zeros(CONTEXTS)
    for s,a in zip(samples,advantages):
        mask=s['active'][:n]
        score=(s['cuts'][:n].astype(float)-s['p'][:n])*mask
        np.add.at(gradient,s['contexts'][:n],score*a/max(1,n))
    return gradient/len(samples)


def rollout(row,logits,predictor,seed,lagrange=0.):
    raw=row['latex'].encode();cuts,active,cs,p=sample_boundaries(raw,logits,np.random.default_rng(seed),predictor.config.max_patch)
    bits,gain=predictor.score(raw,cuts)
    return {'id':row['id'],'group':row['split_group'],'contexts':cs,'cuts':cuts,'active':active,'p':p,
            'reward':gain-lagrange*cuts[:len(raw)],'bits':bits,'bytes':len(raw),
            'boundary_rate':float(cuts[1:-1].sum()/max(1,len(raw)-1))}


def evaluate_policy(rows,logits,predictor,seed,draws=4):
    per_formula=[];rates=[]
    for j,r in enumerate(rows):
        runs=[rollout(r,logits,predictor,seed+100003*j+7919*k) for k in range(draws)]
        per_formula.append({'id':r['id'],'group':r['split_group'],'bytes':runs[0]['bytes'],
                            'bits':float(np.mean([s['bits'].sum() for s in runs])),
                            'boundary_rate':float(np.mean([s['boundary_rate'] for s in runs]))})
    groups=defaultdict(lambda:[0.,0])
    for r in per_formula:groups[r['group']][0]+=r['bits'];groups[r['group']][1]+=r['bytes'];rates.append(r['boundary_rate'])
    return {'causal_byte_prediction_bpb':sum(r['bits'] for r in per_formula)/max(1,sum(r['bytes'] for r in per_formula)),
            'paper_macro_causal_byte_prediction_bpb':float(np.mean([b/n for b,n in groups.values()])),
            'boundary_rate':float(np.mean(rates)),'draws':draws,'rows':per_formula,
            'interpretation':'frozen causal statistical patch predictor, prior-byte boundary decisions; not deployed LM'}


def train_policy(fit,reward,signals,target_rate,seed,algorithm,config=PolicyConfig()):
    if algorithm not in ('boundary_rl','random_search','cem_search','no_update'):raise ValueError('unknown optimizer')
    if config.samples<2 or not 0<target_rate<1:raise ValueError('invalid rollout count or rate')
    started=time.perf_counter();initial=np.full(CONTEXTS,np.log(target_rate/(1-target_rate)))
    predictor=PatchPredictor(signals.base,config).fit(fit,initial,seed+17011)
    logits=initial.copy();first=np.zeros(CONTEXTS);second=np.zeros(CONTEXTS);lagrange=0.
    rng=np.random.default_rng(seed);trace=[];observed=signals.mass>0
    # Every method uses every reward formula and the same number of evaluations.
    # A full oracle evaluation = one raw-byte prediction trajectory per formula.
    oracle_trajectories=0
    for iteration in range(config.rounds):
        if algorithm in ('random_search','cem_search'):
            center=initial if algorithm=='random_search' else logits
            proposals=[np.clip(center+config.proposal_scale*rng.normal(size=CONTEXTS)*observed,-6,6) for _ in range(config.samples)]
        else:proposals=[logits]*config.samples
        grad=np.zeros(CONTEXTS);values=np.zeros(config.samples);rates=np.zeros(config.samples)
        for j,row in enumerate(reward):
            batch=[rollout(row,theta,predictor,seed+iteration*10000019+j*100003+k*7919,lagrange)
                   for k,theta in enumerate(proposals)]
            oracle_trajectories+=len(batch)
            for k,s in enumerate(batch):
                values[k]+=float(s['bits'].mean());rates[k]+=s['boundary_rate']
            if algorithm=='boundary_rl':grad+=boundary_gradient(batch,config.discount)
        values/=len(reward);rates/=len(reward);grad/=len(reward);grad*=observed
        if algorithm=='boundary_rl':
            first=.9*first+.1*grad;second=.999*second+.001*grad**2
            step=config.learning_rate*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8)
            logits=np.clip(logits+step,-6,6)
        elif algorithm=='cem_search':
            utility=values+lagrange*(rates-target_rate)
            chosen=np.argsort(utility)[:max(1,config.samples//2)]
            logits=np.clip(.5*logits+.5*np.mean([proposals[k] for k in chosen],axis=0),-6,6)
        elif algorithm=='random_search':
            # Keep the best training-only candidate; do not inspect dev/eval.
            # A fixed key avoids comparing utility under changing multipliers.
            keys=[(abs(rate-target_rate)>.02,max(0.,abs(rate-target_rate)-.02),float(value))
                  for rate,value in zip(rates,values)]
            k=min(range(len(keys)),key=lambda k:keys[k])
            if not trace or keys[k]<best_key:logits=proposals[k].copy();best_key=keys[k]
        # Equality rate target is an architecture budget, not a mathematical label.
        lagrange=float(np.clip(lagrange+.5*(rates.mean()-target_rate),-4,4))
        trace.append({'round':iteration,'mean_raw_byte_bpb':float(values.mean()),'mean_boundary_rate':float(rates.mean()),
                      'lagrange':lagrange,'gradient_norm':float(np.linalg.norm(grad)),
                      'oracle_trajectories':oracle_trajectories,'oracle_byte_targets':sum(len(r['latex'].encode()) for r in reward)*config.samples*(iteration+1)})
    return {'algorithm':algorithm,'seed':seed,'logits':logits,'initial_logits':initial,'trace':trace,
            'config':asdict(config),'target_boundary_rate':target_rate,'oracle_trajectories':oracle_trajectories,
            'training_seconds':time.perf_counter()-started,'predictor_summary':predictor.memory_summary(),
            'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],
            'selection':'final learned policy; random search uses training-only incumbent; no dev selection'},predictor
