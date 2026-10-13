"""Longer causal actor and fit-only alternating critic learning.

Actor is additive byte-position logits, not recurrent semantic understanding.
No mathematical features; co-adaptation is evaluated against shared critics too.
"""
from dataclasses import asdict
import copy,time
import numpy as np
import torch
from torch import nn
from .neural_stage2 import patch_features
from .autonomous_stage2 import reward_to_go


def actor_features(raw,width=16):
    x=np.full((len(raw)+1,width),256,dtype=np.int32)
    for i in range(len(raw)+1):
        context=raw[max(0,i-width):i]
        if context:x[i,width-len(context):]=list(context)
    return x


class ContextActor:
    def __init__(self,width=16,weights=None):
        self.width=width;self.weights=np.zeros((width,257)) if weights is None else np.asarray(weights,dtype=float).copy()
        self.bias=np.log(.4/.6)
    def logits(self,raw):
        x=actor_features(raw,self.width)
        return self.bias+self.weights[np.arange(self.width)[None,:],x].sum(axis=1)/np.sqrt(self.width)
    def sample(self,raw,rng,max_patch=32):
        logit=self.logits(raw);p=1/(1+np.exp(-np.clip(logit,-12,12)));u=rng.random(len(raw)+1)
        cut=np.zeros(len(raw)+1,dtype=bool);active=np.zeros(len(raw)+1,dtype=bool);cut[0]=cut[-1]=True;last=0
        for i in range(1,len(raw)):
            forced=i-last>=max_patch;active[i]=not forced;cut[i]=forced or u[i]<p[i]
            if cut[i]:last=i
        return cut,active,p
    def deterministic(self,raw,max_patch=32):
        p=self.logits(raw)>=0.;cut=np.zeros(len(raw)+1,dtype=bool);cut[0]=cut[-1]=True;last=0
        for i in range(1,len(raw)):
            cut[i]=p[i] or i-last>=max_patch
            if cut[i]:last=i
        return cut
    def save(self,path):np.savez_compressed(path,weights=self.weights,bias=self.bias,width=self.width)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            actor=cls(int(d['width']),d['weights']);actor.bias=float(d['bias']);return actor


def actor_gradient(raw,samples,actor):
    if len(samples)<2:raise ValueError('LOO requires independent samples')
    returns=np.array([reward_to_go(s['reward']) for s in samples]);adv=returns-(returns.sum(axis=0)-returns)/(len(samples)-1)
    gradient=np.zeros_like(actor.weights);x=actor_features(raw,actor.width)[:len(raw)]
    for sample,a in zip(samples,adv):
        score=(sample['cut'][:len(raw)].astype(float)-sample['p'][:len(raw)])*sample['active'][:len(raw)]
        value=score*a/max(1,len(raw))/np.sqrt(actor.width)
        for position in range(actor.width):np.add.at(gradient[position],x[:,position],value)
    return gradient/len(samples)


def expected_rate(rows,actor,max_patch=32):
    values=[]
    for row in rows:
        raw=row['latex'].encode();p=1/(1+np.exp(-np.clip(actor.logits(raw),-12,12)));ages=np.zeros(max_patch+1);ages[1]=1.;total=0.
        for i in range(1,len(raw)):
            cut=float(ages[:-1].sum()*p[i]+ages[-1]);total+=cut;new=np.zeros(max_patch+1);new[1]=cut
            new[2:]=(1-p[i])*ages[1:-1];ages=new
        values.append(total/max(1,len(raw)-1))
    return float(np.mean(values))


def calibrate(rows,actor,target=.4):
    result=copy.deepcopy(actor);lo=-8.;hi=8.
    for _ in range(24):
        mid=(lo+hi)/2;result.bias=actor.bias+mid
        if expected_rate(rows,result)>target:hi=mid
        else:lo=mid
    result.bias=actor.bias+(lo+hi)/2;return result


def adapt_teacher(teacher,fit,actor,seed,steps=50,batch_targets=512):
    rng=np.random.default_rng(seed);xs=[];es=[];ys=[]
    for row in fit:
        raw=row['latex'].encode();cuts,*_=actor.sample(raw,rng,teacher.config.max_patch)
        x,e=patch_features(raw,cuts,teacher.neural_config.prefix,teacher.neural_config.previous)
        xs.append(x);es.append(e);ys.extend(raw)
    x=torch.from_numpy(np.concatenate(xs));e=torch.from_numpy(np.concatenate(es));y=torch.tensor(ys,dtype=torch.long)
    optimizer=torch.optim.AdamW(list(teacher.early.parameters())+list(teacher.late.parameters()),lr=.001)
    for _ in range(steps):
        indices=torch.from_numpy(rng.integers(0,len(y),batch_targets))
        loss=nn.functional.cross_entropy(teacher.early(e[indices]),y[indices])+nn.functional.cross_entropy(teacher.late(x[indices]),y[indices])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    teacher.early.eval();teacher.late.eval();teacher.teacher_byte_targets+=steps*batch_targets


def optimize(fit,reward,initial_teacher,seed,algorithm,joint=False,rounds=32,samples=4,width=16):
    if algorithm not in ('boundary_rl','random_search','cem_search','no_update'):raise ValueError('unknown optimizer')
    actor=ContextActor(width);initial=actor.weights.copy();teacher=copy.deepcopy(initial_teacher)
    rng=np.random.default_rng(seed);first=np.zeros_like(initial);second=np.zeros_like(initial);trace=[];lagrange=0.
    started=time.perf_counter();oracle=0;byte_targets=0;best_key=None
    for iteration in range(rounds):
        batchrows=[reward[(iteration*8+j)%len(reward)] for j in range(8)]
        if algorithm in ('random_search','cem_search'):
            center=initial if algorithm=='random_search' else actor.weights
            proposals=[ContextActor(width,center+.15*rng.normal(size=center.shape)) for _ in range(samples)]
        else:proposals=[actor]*samples
        grad=np.zeros_like(initial);values=np.zeros(samples);rates=np.zeros(samples)
        for j,row in enumerate(batchrows):
            raw=row['latex'].encode();runs=[]
            for k,policy in enumerate(proposals):
                cut,active,p=policy.sample(raw,np.random.default_rng(seed+iteration*10000019+j*100003+k*7919))
                bits,gain=teacher.score(raw,cut);rate=float(cut[1:-1].sum()/max(1,len(raw)-1))
                values[k]+=float(bits.mean());rates[k]+=rate
                runs.append({'cut':cut,'active':active,'p':p,'reward':gain-lagrange*cut[:len(raw)]})
                oracle+=1;byte_targets+=len(raw)
            if algorithm=='boundary_rl':grad+=actor_gradient(raw,runs,actor)
        values/=len(batchrows);rates/=len(batchrows);grad/=len(batchrows)
        if algorithm=='boundary_rl':
            first=.9*first+.1*grad;second=.999*second+.001*grad**2
            actor.weights=np.clip(actor.weights+.03*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8),-3,3)
        elif algorithm=='cem_search':
            chosen=np.argsort(values+lagrange*(rates-.4))[:samples//2]
            actor.weights=.5*actor.weights+.5*np.mean([proposals[k].weights for k in chosen],axis=0)
        elif algorithm=='random_search':
            # Online best among proposals for THIS common minibatch. Historical
            # minibatch losses cannot be compared as if they were one oracle.
            chosen=int(np.argmin(values+lagrange*(rates-.4)));actor.weights=proposals[chosen].weights.copy()
        lagrange=float(np.clip(lagrange+.5*(rates.mean()-.4),-4,4))
        trace.append({'round':iteration,'mean_byte_bpb':float(values.mean()),'mean_boundary_rate':float(rates.mean()),
            'gradient_norm':float(np.linalg.norm(grad)),'lagrange':lagrange,'oracle_trajectories':oracle,'oracle_byte_targets':byte_targets})
        if joint and (iteration+1)%8==0:adapt_teacher(teacher,fit,actor,373+iteration,steps=50)
    actor=calibrate(reward,actor)
    return actor,teacher,{'algorithm':algorithm,'seed':seed,'joint':joint,'width':width,'trace':trace,
        'training_seconds':time.perf_counter()-started,'oracle_trajectories':oracle,'oracle_byte_targets':byte_targets,
        'teacher_byte_targets':teacher.teacher_byte_targets,'calibrated_reward_rate':expected_rate(reward,actor),
        'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],
        'random_control':'independent random candidates each round; common current minibatch best, no dev selection',
        'selection':'final policy after32rounds; reward-only expected rate calibration'}


def evaluate(rows,actor,teacher,seed,draws=4,deterministic=False):
    per=[];groups={};rates=[]
    for j,row in enumerate(rows):
        raw=row['latex'].encode();values=[];rr=[]
        for k in range(1 if deterministic else draws):
            cut=actor.deterministic(raw) if deterministic else actor.sample(raw,np.random.default_rng(seed+100003*j+7919*k))[0]
            bits,_=teacher.score(raw,cut);values.append(float(bits.sum()));rr.append(float(cut[1:-1].sum()/max(1,len(raw)-1)))
        bits=float(np.mean(values));rate=float(np.mean(rr));b,n=groups.get(row['split_group'],(0.,0));groups[row['split_group']]=(b+bits,n+len(raw));rates.append(rate)
        per.append({'id':row['id'],'group':row['split_group'],'bytes':len(raw),'bits':bits,'rate':rate})
    return {'paper_macro_causal_byte_bpb':float(np.mean([b/n for b,n in groups.values()])),
            'byte_bpb':sum(r['bits'] for r in per)/sum(r['bytes'] for r in per),'boundary_rate':float(np.mean(rates)),
            'rows':per,'draws':1 if deterministic else draws,'deterministic':deterministic}
