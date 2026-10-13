"""Causal neural prediction feedback for autonomous raw-byte boundaries.

Small local-window MLP, not the upstream Gemma U-net or a world model.
The early head sees a fixed preceding-byte window; the patch head sees only
completed-patch suffix and current-patch prefix. Outputs predict current bytes.
"""
from dataclasses import asdict,dataclass
from collections import defaultdict
import time
import numpy as np
import torch
from torch import nn
from .autonomous_stage2 import (PolicyConfig,probabilities,sample_boundaries,boundary_gradient)
from .signal_stage2 import CONTEXTS,contexts


@dataclass(frozen=True)
class NeuralConfig:
    prefix:int=8
    previous:int=4
    embedding:int=16
    hidden:int=96
    steps:int=600
    batch_targets:int=512
    learning_rate:float=.003


def patch_features(raw,cuts,prefix=8,previous=4):
    n=len(raw);features=np.full((n,prefix+previous),256,dtype=np.int64)
    early=np.full((n,prefix),256,dtype=np.int64);start=0;old=b''
    for i in range(n):
        if i and cuts[i]:old=raw[max(start,i-previous):i];start=i
        a=raw[max(start,i-prefix):i];b=raw[max(0,i-prefix):i]
        if old:features[i,previous-len(old):previous]=list(old)
        if a:features[i,previous+prefix-len(a):]=list(a)
        if b:early[i,prefix-len(b):]=list(b)
    return features,early


class ByteHead(nn.Module):
    def __init__(self,width,config):
        super().__init__();self.embed=nn.Embedding(257,config.embedding,padding_idx=256)
        self.layers=nn.Sequential(nn.Linear(width*config.embedding,config.hidden),nn.GELU(),nn.Linear(config.hidden,256))
    def forward(self,x):return self.layers(self.embed(x).flatten(1))


class NeuralPatchPredictor:
    def __init__(self,config=NeuralConfig(),policy_config=PolicyConfig()):
        self.neural_config=config;self.config=policy_config
        self.early=ByteHead(config.prefix,config);self.late=ByteHead(config.prefix+config.previous,config)
        self.fit_log=[];self.cache={}
    @classmethod
    def fit(cls,rows,initial,seed,config=NeuralConfig(),policy_config=PolicyConfig()):
        torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        obj=cls(config,policy_config);rng=np.random.default_rng(seed);xs=[];es=[];ys=[]
        started=time.perf_counter()
        # Multiple boundary rates expose the teacher to alternatives, fit only.
        for rate in (.2,.4,.6):
            logit=np.full(CONTEXTS,np.log(rate/(1-rate)))
            for row in rows:
                raw=row['latex'].encode();cuts,*_=sample_boundaries(raw,logit,rng,policy_config.max_patch)
                x,e=patch_features(raw,cuts,config.prefix,config.previous)
                xs.append(x);es.append(e);ys.extend(raw)
        x=torch.from_numpy(np.concatenate(xs));e=torch.from_numpy(np.concatenate(es));y=torch.tensor(ys,dtype=torch.long)
        optimizer=torch.optim.AdamW(list(obj.early.parameters())+list(obj.late.parameters()),lr=config.learning_rate)
        for step in range(config.steps):
            indices=torch.from_numpy(rng.integers(0,len(y),config.batch_targets))
            early=nn.functional.cross_entropy(obj.early(e[indices]),y[indices])
            late=nn.functional.cross_entropy(obj.late(x[indices]),y[indices]);loss=early+late
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%100==0 or step==config.steps-1:
                obj.fit_log.append({'step':step,'early_nats':float(early.detach()),'patch_nats':float(late.detach())})
        obj.early.eval();obj.late.eval();obj.fit_ids=[r['id'] for r in rows]
        obj.train_seconds=time.perf_counter()-started;obj.teacher_byte_targets=config.steps*config.batch_targets
        return obj
    def score(self,raw,cuts):
        x,e=patch_features(raw,cuts,self.neural_config.prefix,self.neural_config.previous)
        if not len(raw):return np.array([]),np.array([])
        target=torch.tensor(list(raw),dtype=torch.long)
        with torch.no_grad():
            lp=torch.log_softmax(self.late(torch.from_numpy(x)),dim=-1)
            ep=torch.log_softmax(self.early(torch.from_numpy(e)),dim=-1)
            indices=torch.arange(len(raw));late=lp[indices,target].numpy()/np.log(2);early=ep[indices,target].numpy()/np.log(2)
        return -late,late-early
    def memory_summary(self):
        return {'neural_parameters':sum(p.numel() for m in (self.early,self.late) for p in m.parameters()),
                'parameter_bytes':sum(p.numel()*p.element_size() for m in (self.early,self.late) for p in m.parameters()),
                'teacher_byte_targets':getattr(self,'teacher_byte_targets',0),
                'context':'causal local byte windows, fixed early head and patch-conditioned head'}
    def save(self,path):
        torch.save({'early':self.early.state_dict(),'late':self.late.state_dict(),'config':asdict(self.neural_config),
                    'policy_config':asdict(self.config),'fit_ids':self.fit_ids,'fit_log':self.fit_log,
                    'train_seconds':self.train_seconds,'teacher_byte_targets':self.teacher_byte_targets},path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True);obj=cls(NeuralConfig(**d['config']),PolicyConfig(**d['policy_config']))
        obj.early.load_state_dict(d['early']);obj.late.load_state_dict(d['late']);obj.early.eval();obj.late.eval()
        for k in ('fit_ids','fit_log','train_seconds','teacher_byte_targets'):setattr(obj,k,d[k])
        return obj


def optimize_neural_policy(fit,reward,signals,target_rate,seed,algorithm,predictor,config=PolicyConfig()):
    """Same frozen neural oracle and same byte-target count for every optimizer."""
    from .autonomous_stage2 import rollout
    if algorithm not in ('boundary_rl','random_search','cem_search','no_update'):raise ValueError('unknown algorithm')
    initial=np.full(CONTEXTS,np.log(target_rate/(1-target_rate)));logits=initial.copy()
    rng=np.random.default_rng(seed);first=np.zeros(CONTEXTS);second=np.zeros(CONTEXTS);observed=signals.mass>0
    trace=[];lagrange=0.;started=time.perf_counter();budget=0
    for iteration in range(config.rounds):
        if algorithm in ('random_search','cem_search'):
            center=initial if algorithm=='random_search' else logits
            proposals=[np.clip(center+config.proposal_scale*rng.normal(size=CONTEXTS)*observed,-6,6) for _ in range(config.samples)]
        else:proposals=[logits]*config.samples
        values=np.zeros(config.samples);rates=np.zeros(config.samples);grad=np.zeros(CONTEXTS)
        for j,row in enumerate(reward):
            batch=[rollout(row,theta,predictor,seed+iteration*10000019+j*100003+k*7919,lagrange) for k,theta in enumerate(proposals)]
            budget+=config.samples
            for k,s in enumerate(batch):values[k]+=float(s['bits'].mean());rates[k]+=s['boundary_rate']
            if algorithm=='boundary_rl':grad+=boundary_gradient(batch,config.discount)
        values/=len(reward);rates/=len(reward);grad/=len(reward);grad*=observed
        if algorithm=='boundary_rl':
            first=.9*first+.1*grad;second=.999*second+.001*grad**2
            logits=np.clip(logits+config.learning_rate*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8),-6,6)
        elif algorithm=='cem_search':
            chosen=np.argsort(values+lagrange*(rates-target_rate))[:config.samples//2]
            logits=.5*logits+.5*np.mean([proposals[k] for k in chosen],axis=0)
        elif algorithm=='random_search':
            keys=[(abs(r-target_rate)>.02,max(0.,abs(r-target_rate)-.02),float(v)) for r,v in zip(rates,values)]
            k=min(range(len(keys)),key=keys.__getitem__)
            if not trace or keys[k]<best_key:logits=proposals[k].copy();best_key=keys[k]
        lagrange=float(np.clip(lagrange+.5*(rates.mean()-target_rate),-4,4))
        trace.append({'round':iteration,'mean_byte_bpb':float(values.mean()),'mean_boundary_rate':float(rates.mean()),
                      'lagrange':lagrange,'gradient_norm':float(np.linalg.norm(grad)),'oracle_trajectories':budget})
    return {'algorithm':algorithm,'seed':seed,'logits':logits,'initial_logits':initial,'trace':trace,
            'config':asdict(config),'target_boundary_rate':target_rate,'oracle_trajectories':budget,
            'oracle_byte_targets':sum(len(r['latex'].encode()) for r in reward)*config.samples*config.rounds,
            'training_seconds':time.perf_counter()-started,'predictor_summary':predictor.memory_summary(),
            'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],
            'selection':'final learned policy; random training incumbent; frozen shared neural predictor'}
