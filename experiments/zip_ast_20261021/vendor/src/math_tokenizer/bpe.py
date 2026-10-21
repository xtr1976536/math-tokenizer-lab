"""Byte BPE and source-span AST-constrained BPE; exact-byte decode.

RL learns merge selection, not dynamic byte gating inside a language model.
Training and reward are confined to training documents. Gamma=1 for independent
one-step merge decisions; no unsupported transfer of Dauncey's causal discount.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
import time
import heapq
import numpy as np
from .ast import byte_spans, compatible, ParseError

@dataclass
class Sequence:
    ids: list
    starts: list
    ends: list
    spans: tuple

    @classmethod
    def make(cls,text,structural):
        spans = ()
        if structural:
            try:
                spans = byte_spans(text)
            except (ParseError,RecursionError):
                # Explicit byte fallback, separately measured during evaluation.
                # No merges are permitted for an unsupported formula.
                spans = None
        data = text.encode('utf-8')
        return cls(list(data),list(range(len(data))),list(range(1,len(data)+1)),spans)

    def legal(self,i):
        return self.spans is not None and (not self.spans or compatible(self.starts[i],self.ends[i+1],self.spans))
    def counts(self):
        return Counter((self.ids[i],self.ids[i+1]) for i in range(len(self.ids)-1) if self.legal(i))
    def gain(self,pair):
        # Exact non-overlapping replacements, unlike raw adjacent-pair frequency.
        n,i=0,0
        while i < len(self.ids)-1:
            if (self.ids[i],self.ids[i+1])==pair and self.legal(i):
                n+=1;i+=2
            else:
                i+=1
        return n
    def merge(self,pair,new):
        ids,starts,ends=[],[],[]
        i=0
        while i<len(self.ids):
            if i+1<len(self.ids) and (self.ids[i],self.ids[i+1])==pair and self.legal(i):
                ids.append(new);starts.append(self.starts[i]);ends.append(self.ends[i+1]);i+=2
            else:
                ids.append(self.ids[i]);starts.append(self.starts[i]);ends.append(self.ends[i]);i+=1
        self.ids,self.starts,self.ends=ids,starts,ends

class MathBPE:
    def __init__(self,structural=False,rl=False):
        self.structural=structural;self.rl=rl
        self.vocab=[bytes([i]) for i in range(256)]
        self.merges=[];self.training_log=[]
        self.features_mean=np.zeros(4);self.features_scale=np.ones(4)
        self.policy=np.zeros(4)

    def train(self,texts,vocab_size=384,seed=17,episodes=48):
        if vocab_size<256:
            raise ValueError('complete byte fallback requires capacity >=256')
        rng=np.random.default_rng(seed)
        # Holdout is drawn INSIDE training data, never the benchmark dev/test.
        indices=np.arange(len(texts));rng.shuffle(indices)
        nreward=max(1,len(texts)//5) if self.rl else 0
        reward_idx=set(indices[:nreward]);fit_idx=[i for i in range(len(texts)) if i not in reward_idx]
        fitting=[Sequence.make(texts[i],self.structural) for i in fit_idx]
        reward=[Sequence.make(texts[i],self.structural) for i in sorted(reward_idx)]
        started=time.perf_counter()
        for step in range(vocab_size-256):
            counter=Counter()
            for seq in fitting:
                counter.update(seq.counts())
            candidates=sorted(counter,key=lambda p:(-counter[p],p))[:24 if self.rl else 1]
            if not candidates:
                break
            if self.rl and reward:
                # Learning action-independent baseline from all candidate rewards
                # is valid for this finite contextual one-step policy.
                features=[];rewards=[]
                denominator=max(1,sum(len(s.ids) for s in reward))
                for pair in candidates:
                    chunk=self.vocab[pair[0]]+self.vocab[pair[1]]
                    gain=sum(s.gain(pair) for s in reward)
                    features.append([math.log1p(counter[pair]),math.log1p(len(chunk)),
                                     float(b'\\' in chunk),float(any(c in chunk for c in (b'{',b'}',b'^',b'_')))])
                    rewards.append(gain/denominator)
                x=np.asarray(features);r=np.asarray(rewards)
                mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1
                # Stable feature interpretation across steps: fixed transforms.
                x=x/np.array([8.,4.,1.,1.])
                for _ in range(episodes):
                    logits=x@self.policy;logits-=logits.max();p=np.exp(logits);p/=p.sum()
                    action=int(rng.choice(len(p),p=p))
                    baseline=float(p@r)
                    advantage=(r[action]-baseline)*100
                    self.policy+=.08*advantage*(x[action]-p@x)
                logits=x@self.policy
                choice=int(np.argmax(logits))
                pair=candidates[choice]
                self.training_log.append({'step':step,'selected_pair':list(pair),'reward':rewards[choice],
                                          'best_available_reward':max(rewards),'policy':self.policy.tolist(),
                                          'episodes':episodes,'reward_scope':'train_internal_holdout'})
            else:
                pair=candidates[0]
            # Avoid duplicate byte tokens and preserve actual vocabulary accounting.
            chunk=self.vocab[pair[0]]+self.vocab[pair[1]]
            if chunk in self.vocab:
                # Reuse same byte token ID with an additional derivation rule.
                new=self.vocab.index(chunk)
            else:
                new=len(self.vocab);self.vocab.append(chunk)
            self.merges.append((pair[0],pair[1],new))
            for seq in fitting+reward:
                seq.merge(pair,new)
        self.train_seconds=time.perf_counter()-started
        self.capacity=vocab_size;self.seed=seed
        return self

    def encode_reference(self,text):
        s=Sequence.make(text,self.structural)
        for a,b,c in self.merges:
            s.merge((a,b),c)
        return s.ids

    def encode(self,text):
        s=Sequence.make(text,self.structural)
        if s.spans is None or not self.merges:
            return s.ids
        n=len(s.ids)
        if n<2:return s.ids
        # Ranked merge heap plus linked arrays. Only neighboring candidates
        # are updated, instead of rescanning all bytes once per vocabulary entry.
        rules={}
        for rank,(a,b,c) in enumerate(self.merges):
            rules.setdefault((a,b),(rank,c))
        left=list(range(-1,n-1));right=list(range(1,n+1));right[-1]=-1
        alive=[True]*n;versions=[0]*n;heap=[]
        def push(i):
            if i<0 or not alive[i]:return
            j=right[i]
            if j<0:return
            rule=rules.get((s.ids[i],s.ids[j]))
            if rule is not None and (not s.spans or compatible(s.starts[i],s.ends[j],s.spans)):
                heapq.heappush(heap,(rule[0],i,j,versions[i],versions[j],rule[1]))
        for i in range(n-1):push(i)
        while heap:
            rank,i,j,vi,vj,new=heapq.heappop(heap)
            if not alive[i] or not alive[j] or right[i]!=j or versions[i]!=vi or versions[j]!=vj:continue
            s.ids[i]=new;s.ends[i]=s.ends[j];versions[i]+=1;alive[j]=False
            k=right[j];right[i]=k
            if k>=0:left[k]=i
            push(left[i]);push(i)
        return [s.ids[i] for i in range(n) if alive[i]]
    def decode_bytes(self,ids):
        return b''.join(self.vocab[i] for i in ids)
    def decode(self,ids):
        return self.decode_bytes(ids).decode('utf-8')
    def save(self,path):
        payload={'format':'mathbpe-v1','structural':self.structural,'rl':self.rl,
                 'vocab_hex':[x.hex() for x in self.vocab],'merges':self.merges,
                 'policy':self.policy.tolist(),'training_log':self.training_log,
                 'capacity':self.capacity,'seed':self.seed,'train_seconds':self.train_seconds}
        path.write_text(json.dumps(payload,indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text());obj=cls(d['structural'],d['rl'])
        obj.vocab=[bytes.fromhex(x) for x in d['vocab_hex']]
        obj.merges=[tuple(x) for x in d['merges']];obj.policy=np.array(d['policy'])
        obj.training_log=d['training_log'];obj.capacity=d['capacity'];obj.seed=d['seed'];obj.train_seconds=d['train_seconds']
        return obj
