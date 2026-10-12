"""v5-family terminal policy RL with fit-only action features and transfer.

Policy parameters are shared across fixed training subsets, and future credit
comes from the actual exported encoder's composite terminal score. Full-corpus
vocabulary construction uses only fit statistics; reward is never a feature.
Original Rust BPE and lexical codec serve online IDs, no policy inference.
"""
from collections import Counter
from dataclasses import dataclass,asdict
import hashlib,math,time
import numpy as np
from .rl_route_v6 import AtomicVocab,AtomCodec
from .rl_vocab_v5 import Indexed,VocabRLConfig
from .trajectory_v5 import probabilities_and_score,rloo_gradient,TrajectoryConfig
from .rl_route_v6 import ReferenceReward
from .fit_index_v7 import NativeFitIndexed


@dataclass(frozen=True)
class TransferConfig:
    epochs:int=4
    samples:int=4
    learning_rate:float=.15
    temperature:float=.8
    exploration:float=.08
    fit_task_limit:int=1600
    reward_task_limit:int=128
    training_capacity:int=1024
    tree_weight:float=.5
    candidates:int=16


def features(index,pairs,vocab,step,steps):
    """Fit-only generic payload/statistical features, phase and interactions."""
    phase=step/max(1,steps-1);total=max(1,sum(index.gains[p] for p in pairs));x=[]
    for a,b in pairs:
        payload=vocab[a]+vocab[b];gain=index.gains[(a,b)]
        base=[math.log1p(gain)/10,math.log1p(len(payload))/4,float(payload.startswith(b'\\')),
              float(payload.isdigit()),float(any(c>=128 for c in payload)),
              sum(c in b'{}_^' for c in payload)/len(payload),
              len(vocab[a])/len(payload),gain/total]
        x.append(base+[value*phase for value in base]+[value*value for value in base])
    return np.asarray(x,dtype=float)


def rollout(fit,atoms,capacity,theta,config,rng=None,method='policy',index_class=NativeFitIndexed):
    started=time.perf_counter()
    model=AtomicVocab(VocabRLConfig(candidates=config.candidates),atoms)
    sequences=model.codec.indexed([r['latex'] for r in fit]).seqs
    index=index_class([r['latex'] for r in fit],sequences=sequences);size={p:i for i,p in enumerate(model.vocab)}
    score=np.zeros(24);steps=max(0,capacity-len(atoms));entropy=0.;step=0
    while len(model.vocab)<capacity:
        pairs=index.top(config.candidates)
        if not pairs:break
        x=features(index,pairs,model.vocab,step,steps)
        if method=='frequency':action=0
        elif rng is None:action=int(np.argmax(x@theta))
        else:
            p=probabilities_and_score(x,theta,temperature=config.temperature,exploration=config.exploration)
            action=int(rng.choice(len(p),p=p));_,gradient=probabilities_and_score(x,theta,action,config.temperature,config.exploration)
            score+=gradient;entropy-=float(p@np.log(np.maximum(p,1e-300)))
        pair=pairs[action];payload=model.vocab[pair[0]]+model.vocab[pair[1]];new=size.get(payload)
        if new is None:new=len(model.vocab);size[payload]=new;model.vocab.append(payload)
        model.merges.append((*pair,new));index.merge(pair,new);step+=1
    model.theta=np.asarray(theta).copy();model.capacity=len(model.vocab);model.seed=0
    model.method='v5_fit_only_transfer_policy' if method=='policy' else 'v5_same_atoms_fit_frequency'
    model.logs=[{'policy_action_features':'fit_only_24_statistics','reward_used_in_action_features':False,
                 'future_terminal_credit':True,'steps':step,'training_config':asdict(config)}]
    model.fit_ids=[r['id'] for r in fit];model.reward_ids=[];model.build_official()
    model.train_seconds=time.perf_counter()-started
    return model,score,entropy/max(1,step)


def task_rows(rows,limit,task):
    # Different per-paper resamples, preserving domain/source exposure. Fit and
    # reward pools remain disjoint throughout, no dev/test observations here.
    buckets={}
    for r in rows:buckets.setdefault(r['split_group'],[]).append(r)
    ordered=[]
    for group,rs in sorted(buckets.items()):
        ordered.append(sorted(rs,key=lambda r:hashlib.sha256(f'v7-task:{task}:{r["id"]}'.encode()).digest()))
    output=[];i=0
    while len(output)<limit and any(len(rs)>i for rs in ordered):
        for rs in ordered:
            if len(rs)>i:output.append(rs[i])
            if len(output)>=limit:break
        i+=1
    return output


def train_policy(fit,reward,atoms,config,seed=17,update=True,progress=None):
    started=time.perf_counter();rng=np.random.default_rng(seed)
    theta=np.zeros(24);theta[0]=2. # Explicit shared frequency warm start.
    initial=theta.copy();history=[];best=None;best_value=math.inf;first=np.zeros(24);second=np.zeros(24)
    for epoch in range(config.epochs):
        task=epoch%2;f=task_rows(fit,config.fit_task_limit,task);r=task_rows(reward,config.reward_task_limit,task)
        objective=ReferenceReward(f,r,TrajectoryConfig(tree_weight=config.tree_weight,code_weight=.04,command_weight=.15,number_weight=.03,robustness_weight=.04,rare_weight=.02))
        capacity=max(len(atoms)+16,config.training_capacity)
        scores=[];returns=[];statistics=[]
        for sample in range(config.samples):
            model,score,entropy=rollout(f,atoms,capacity,theta,config,rng)
            stats=objective.score(model);scores.append(score);returns.append(-stats['utility']);statistics.append(stats)
            if stats['utility']<best_value:best,best_value=model,stats['utility']
            if progress:progress(epoch,sample,stats)
        gradient,advantages=rloo_gradient(scores,returns)
        # Reward centering preserves on-policy score-function direction.
        norm=float(np.linalg.norm(gradient));gradient/=max(1.,norm)
        if update:
            first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
            theta+=config.learning_rate*(first/(1-.9**(epoch+1)))/(np.sqrt(second/(1-.999**(epoch+1)))+1e-8)
        history.append({'epoch':epoch,'task':task,'fit_ids':[v['id'] for v in f],
            'reward_ids':[v['id'] for v in r],'terminal_statistics':statistics,'gradient_norm':norm,
            'advantages':advantages.tolist(),'policy':theta.tolist(),'updated':update})
    elapsed=time.perf_counter()-started
    return {'policy':theta.tolist(),'initial_policy':initial.tolist(),'history':history,
            'config':asdict(config),'seed':seed,'updates_enabled':update,'training_seconds':elapsed,
            'policy_change_norm':float(np.linalg.norm(theta-initial)),
            'scope':'fit-only feature policy trained by full trajectory RLOO terminal scores; shared across source-balanced task subsets',
            'best_sample_on_training_task':best_value},best
