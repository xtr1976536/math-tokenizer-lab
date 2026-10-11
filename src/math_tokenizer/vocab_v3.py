"""BPE-seeded minimum-path vocabulary refinement plus RL vocabulary swaps.

All selection rewards use a train-internal document holdout; dev only chooses
predeclared configurations. Byte coverage mandatory; sampled vocab actions obey
capacity. The RL part is explicitly a contextual vocabulary-swap bandit, not a
Dauncey end-to-end LM. No-RL pruned vocab and zero-update controls are preserved.
"""
from dataclasses import dataclass,asdict
from collections import Counter,defaultdict
import hashlib,json,math,time
import numpy as np
from .ast_v3 import AtomicASTBPE
from .merge_v2 import FastBPE
from .native_v3 import NativeTokenizer
from .structure_v2 import structure

@dataclass(frozen=True)
class VocabConfig:
    structural: bool = False
    atom_budget: int = 96
    seed_multiplier: int = 3
    pruning_rounds: int = 4
    symbol_reward: float = 0.
    policy_rounds: int = 12
    policy_samples: int = 6
    policy_lr: float = .08
    reward_formulas: int = 256
    seed: int = 17


def train_holdout(records,max_formulas=256):
    groups=sorted({r['split_group'] for r in records},key=lambda g:hashlib.sha256(('v3-reward:'+g).encode()).digest())
    chosen=set(groups[:max(1,len(groups)//5)])
    fit=[r for r in records if r['split_group'] not in chosen]
    reward=[r for r in records if r['split_group'] in chosen]
    reward=sorted(reward,key=lambda r:hashlib.sha256(('reward-row:'+r['id']).encode()).digest())[:max_formulas]
    return fit,reward


def fit_seed(texts,capacity,config):
    if config.structural:
        seed=AtomicASTBPE(True).train(texts,capacity*config.seed_multiplier,command_budget=config.atom_budget)
        mandatory={token for token in seed.command_atoms}
    else:
        seed=FastBPE().train(texts,capacity*config.seed_multiplier);mandatory=set()
    return seed,mandatory


def shortest_model(vocab,config,penalty=0.):
    return NativeTokenizer(vocab,config.structural,'shortest',splitfee=penalty,crossfee=0.)


def estimate_usage(model,texts):
    counts=np.zeros(len(model.vocab),dtype=float)
    for text in texts:
        for id in model.encode(text):counts[id]+=1
    return counts


def refine(records,capacity,config,seed_base=None):
    started=time.perf_counter();fit,reward=train_holdout(records,config.reward_formulas)
    texts=[r['latex'] for r in fit]
    seed,mandatory=seed_base or fit_seed(texts,capacity,config)
    original=list(seed.vocab);vocab=list(original)
    mandatory|={bytes([i]) for i in range(256)}
    history=[]
    # Staged hard minimum-path usage pruning; tokens learn their utility from
    # full formula segmentations, not substring frequency alone.
    for round in range(config.pruning_rounds):
        model=shortest_model(vocab,config);counts=estimate_usage(model,texts)
        target=capacity if round==config.pruning_rounds-1 else max(capacity,int(len(vocab)*.72))
        scores=counts*np.array([max(0,len(t)-1) for t in vocab])
        keep=[i for i,t in enumerate(vocab) if t in mandatory]
        remaining=[i for i,t in enumerate(vocab) if t not in mandatory]
        remaining.sort(key=lambda i:(-scores[i],-counts[i],vocab[i]))
        keep+=remaining[:max(0,target-len(keep))]
        vocab=[vocab[i] for i in sorted(keep)]
        history.append({'round':round,'target':target,'actual_vocab':len(vocab),'used_types':int(np.sum(counts>0))})
    no_rl=shortest_model(vocab,config)
    no_rl.capacity=capacity;no_rl.train_seconds=time.perf_counter()-started;no_rl.training_log=[{'stage':'vocabulary_pruning','config':asdict(config),'history':history,'fit_documents':len({r['split_group'] for r in fit}),'reward_documents':len({r['split_group'] for r in reward}),'reward_ids':[r['id'] for r in reward],'vocabulary_seed':'train_fit_only'}]
    return no_rl,original,mandatory,fit,reward


def boundary_cost(model,records,symbol_weight):
    total=0.;bytes_total=0;tokens_total=0;broken_total=0
    for r in records:
        text=r['latex'];ids=model.encode(text);raw=text.encode();bytes_total+=len(raw);tokens_total+=len(ids)
        if symbol_weight:
            pos=0;boundaries={0}
            for id in ids:pos+=len(model.vocab[id]);boundaries.add(pos)
            broken=0
            for a,b,t in structure(text).lexemes:
                if t.startswith('\\') or t[0:1].isdigit():broken+=any(a<x<b for x in boundaries)
            broken_total+=broken
    return (tokens_total+symbol_weight*broken_total)/max(1,bytes_total),tokens_total,broken_total


def optimize_swaps(no_rl,original,mandatory,fit,reward,config):
    started=time.perf_counter();rng=np.random.default_rng(config.seed)
    vocab=list(no_rl.vocab);texts=[r['latex'] for r in fit];current=boundary_cost(no_rl,reward,config.symbol_reward)[0]
    # Features preserve shared policy meaning across changing action sets.
    theta=np.zeros(5);logs=[]
    counts_seed=estimate_usage(shortest_model(original,config),texts)
    seedscore={t:counts_seed[i]*max(1,len(t)-1) for i,t in enumerate(original)}
    for round in range(config.policy_rounds):
        model=shortest_model(vocab,config);usage=estimate_usage(model,[r['latex'] for r in reward])
        removable=[i for i,t in enumerate(vocab) if t not in mandatory]
        removable.sort(key=lambda i:(usage[i]*max(1,len(vocab[i])-1),vocab[i]))
        excluded=[t for t in original if t not in set(vocab)]
        excluded.sort(key=lambda t:(-seedscore[t],t))
        # Diversify proposals beyond high seed scores using reproducible sampling.
        additions=excluded[:16]
        if len(excluded)>16:
            additions+= [excluded[i] for i in rng.choice(np.arange(16,len(excluded)),min(8,len(excluded)-16),replace=False)]
        drops=removable[:8]
        if not drops or not additions:break
        actions=[(drop,new) for drop in drops for new in additions]
        feature=[]
        for drop,new in actions:
            feature.append([math.log1p(seedscore[new])/10.,-math.log1p(usage[drop])/8.,math.log(len(new))/4.,float(new.startswith(b'\\')),float(b'{' in new or b'}' in new)])
        x=np.asarray(feature);logits=x@theta;logits-=logits.max();p=np.exp(logits);p/=p.sum()
        sampled=rng.choice(len(actions),config.policy_samples,p=p,replace=True)
        outcomes=[];models=[]
        for index in sampled:
            drop,new=actions[int(index)];proposal=vocab.copy();proposal[drop]=new
            trial=shortest_model(proposal,config);cost,tokens,broken=boundary_cost(trial,reward,config.symbol_reward)
            outcomes.append(current-cost);models.append((proposal,trial,cost,tokens,broken))
        r=np.asarray(outcomes);grad=np.zeros_like(theta)
        for k,index in enumerate(sampled):
            baseline=(r.sum()-r[k])/max(1,len(r)-1)
            grad+=(r[k]-baseline)*(x[index]-p@x)/len(r)
        theta+=config.policy_lr*1000*grad
        # Conservative incumbent selection explicitly part of search, not hidden.
        # RLOO samples guide proposal policy; keep only actual train-reward wins.
        best=int(np.argmax(r));accepted=bool(r[best]>1e-12)
        if accepted:vocab,model,current,tokens,broken=models[best]
        logs.append({'round':round,'rewards':r.tolist(),'accepted':accepted,'incumbent_train_reward_cost':current,
                     'policy':theta.tolist(),'estimator':'RLOO independent vocabulary swap samples','sampled_actions':[[actions[int(i)][0],actions[int(i)][1].hex()] for i in sampled],
                     'reward_scope':'train_internal_document_holdout'})
    result=shortest_model(vocab,config);result.capacity=no_rl.capacity;result.train_seconds=no_rl.train_seconds+time.perf_counter()-started
    result.training_log=no_rl.training_log+[{'stage':'rl_vocabulary_swaps','config':asdict(config),'logs':logs,'theta':theta.tolist(),'reward_ids':[r['id'] for r in reward]}]
    return result
