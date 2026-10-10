"""Exact v1 training semantics with incremental indices; official v1 preserved."""
from collections import Counter,defaultdict
import math,time
import numpy as np
from .bpe import MathBPE,Sequence

def gains(sequence):
    result=Counter();last_pair=None;last_position=-2
    for i in range(len(sequence.ids)-1):
        if not sequence.legal(i):continue
        pair=(sequence.ids[i],sequence.ids[i+1])
        if pair==last_pair and i==last_position+1:continue
        result[pair]+=1;last_pair=pair;last_position=i
    return result

class LegacyFastBPE(MathBPE):
    """Same merges/policy/RNG order as MathBPE, fewer full-corpus scans."""
    def train(self,texts,vocab_size=512,seed=17,episodes=48):
        if vocab_size<256:raise ValueError('complete byte fallback required')
        started=time.perf_counter();rng=np.random.default_rng(seed);indices=np.arange(len(texts));rng.shuffle(indices)
        rewardidx=set(indices[:max(1,len(texts)//5)]) if self.rl else set()
        fit=[Sequence.make(texts[i],self.structural) for i in range(len(texts)) if i not in rewardidx]
        reward=[Sequence.make(texts[i],self.structural) for i in sorted(rewardidx)]
        seqs=fit+reward;nf=len(fit)
        local=[s.counts() for s in seqs];lg=[gains(s) for s in seqs]
        globalcounts=Counter();rewardgains=Counter();index=defaultdict(set)
        for i,counts in enumerate(local):
            if i<nf:globalcounts.update(counts)
            else:rewardgains.update(lg[i])
            for pair in counts:index[pair].add(i)
        rewardtokens=sum(len(s.ids) for s in reward)
        for step in range(vocab_size-256):
            candidates=sorted((p for p,c in globalcounts.items() if c>0),key=lambda p:(-globalcounts[p],p))[:24 if self.rl else 1]
            if not candidates:break
            if self.rl and reward:
                features=[];rewards=[]
                for pair in candidates:
                    chunk=self.vocab[pair[0]]+self.vocab[pair[1]]
                    features.append([math.log1p(globalcounts[pair]),math.log1p(len(chunk)),float(b'\\' in chunk),float(any(c in chunk for c in (b'{',b'}',b'^',b'_')))])
                    rewards.append(rewardgains[pair]/max(1,rewardtokens))
                x=np.asarray(features)/np.array([8.,4.,1.,1.]);r=np.asarray(rewards)
                for _ in range(episodes):
                    logits=x@self.policy;logits-=logits.max();p=np.exp(logits);p/=p.sum();action=int(rng.choice(len(p),p=p));baseline=float(p@r)
                    self.policy+=.08*((r[action]-baseline)*100)*(x[action]-p@x)
                choice=int(np.argmax(x@self.policy));pair=candidates[choice]
                self.training_log.append({'step':step,'selected_pair':list(pair),'reward':rewards[choice],'best_available_reward':max(rewards),'policy':self.policy.tolist(),'episodes':episodes,'reward_scope':'train_internal_holdout'})
            else:pair=candidates[0]
            chunk=self.vocab[pair[0]]+self.vocab[pair[1]]
            new=self.vocab.index(chunk) if chunk in self.vocab else len(self.vocab)
            if new==len(self.vocab):self.vocab.append(chunk)
            self.merges.append((*pair,new))
            for i in sorted(index[pair].copy()):
                old=local[i];oldgain=lg[i];oldtokens=len(seqs[i].ids)
                seqs[i].merge(pair,new);updated=seqs[i].counts();newgain=gains(seqs[i]);local[i]=updated;lg[i]=newgain
                if i<nf:
                    for p in old.keys()|updated.keys():globalcounts[p]+=updated[p]-old[p]
                else:
                    rewardtokens+=len(seqs[i].ids)-oldtokens
                    for p in oldgain.keys()|newgain.keys():rewardgains[p]+=newgain[p]-oldgain[p]
                for p in old.keys()|updated.keys():
                    if p not in updated:index[p].discard(i)
                    elif p not in old:index[p].add(i)
        self.train_seconds=time.perf_counter()-started;self.capacity=vocab_size;self.seed=seed;return self
