"""Standalone sequential RL segmentation of a BPE-derived fixed vocabulary.

No downstream LM. A DAG enumerates legal token edges over original UTF8 bytes.
REINFORCE learns contextual boundary/action probabilities from complete paths;
multiple independent trajectories provide byte-position leave-one-out baselines.
This is an adaptation of variance-reduction ideas, not Dauncey's architecture.
"""
from dataclasses import dataclass,asdict
from collections import Counter
import hashlib,json,math,time
import numpy as np
from .structure_v2 import structure
from .ast import compatible
from .merge_v2 import FastBPE

@dataclass(frozen=True)
class RLConfig:
    estimator: str = 'rloo'
    gamma: float = 1.0
    samples: int = 4
    epochs: int = 6
    lr: float = .025
    entropy: float = .005
    structure_weight: float = 0.0
    max_train_formulas: int = 512
    initial_length_prior: float = 1.5
    gradient_clip: float = 1.0
    batch_size: int = 16

class LatticeTokenizer:
    """Fixed BPE vocabulary with DP-shortest or learned policy segmentation."""
    def __init__(self,base,structural=False,mode='shortest',config=None):
        self.vocab=list(base.vocab);self.capacity=base.capacity
        self.structural=structural;self.mode=mode;self.config=config or RLConfig()
        self.theta=np.array([self.config.initial_length_prior,0.,0.,0.,0.,0.,0.],dtype=float)
        self.training_log=[];self.train_seconds=0.;self.seed=0
        self._trie={}
        for id,payload in enumerate(self.vocab):
            trie=self._trie
            for byte in payload:trie=trie.setdefault(byte,{})
            trie[None]=id
        self.frequency=np.ones(len(self.vocab))

    def graph(self,text):
        data=text.encode();n=len(data);s=structure(text)
        spans=s.spans if self.structural and s.kind!='byte_fallback' else ()
        # Hard structures preserve AST/CST source spans. Unknown parse uses
        # lexical-safe fallback edges, not a method-specific corpus exclusion.
        command_spans=[(a,b) for a,b,t in s.lexemes if t.startswith('\\')]
        safe={0,n}|{a for a,b,t in s.lexemes}|{b for a,b,t in s.lexemes}
        atom_interior=set()
        for a,b,t in s.lexemes:
            if t.startswith('\\') or t[0:1].isdigit():atom_interior.update(range(a+1,b))
        longest=[1]*(n+1)
        for begin in range(n):
            trie=self._trie;end=begin
            while end<n and data[end] in trie:
                trie=trie[data[end]];end+=1
                if None in trie:longest[begin]=end-begin
        graph=[]
        for start in range(n):
            options=[];trie=self._trie;end=start
            while end<n and data[end] in trie:
                trie=trie[data[end]];end+=1
                if None not in trie:continue
                id=trie[None]
                if spans and not compatible(start,end,spans):continue
                # Single-byte fallback guarantees graph completeness even when
                # a command/number crosses a byte boundary.
                length=end-start
                split=float(end in atom_interior)
                crossing=float(bool(s.spans) and not compatible(start,end,s.spans))
                feature=[math.log(length),math.log(self.frequency[id])/8,
                         float(end in safe),-split,-crossing,length/max(1,n),
                         math.log(longest[end])]
                cost=1+self.config.structure_weight*(split+crossing)
                options.append((end,id,feature[:len(self.theta)],cost))
            assert options,'complete byte fallback invariant violated'
            ends=np.array([o[0] for o in options],dtype=np.int32)
            ids=np.array([o[1] for o in options],dtype=np.int32)
            x=np.array([o[2] for o in options],dtype=float)
            costs=np.array([o[3] for o in options],dtype=float)
            graph.append((ends,ids,x,costs))
        return graph

    def policy(self,x):
        logits=x@self.theta;logits-=np.max(logits);p=np.exp(logits);p/=p.sum();return p

    def sample(self,graph,rng):
        n=len(graph);pos=0;path=[]
        while pos<n:
            ends,ids,x,cost=graph[pos];p=self.policy(x);action=int(rng.choice(len(p),p=p))
            centered=x[action]-p@x
            logp=np.log(np.maximum(p,1e-30))
            entropy_grad=-(p*(logp+1))@(x-p@x)
            path.append((pos,int(ends[action]),int(ids[action]),float(cost[action]),centered,entropy_grad))
            pos=int(ends[action])
        return path

    def train_policy(self,texts,seed=17):
        self.seed=seed;rng=np.random.default_rng(seed);started=time.perf_counter()
        # Text selection is fixed hash ordering, identical across ablations.
        ordered=sorted(texts,key=lambda t:hashlib.sha256(t.encode()).digest())
        selected=ordered[:self.config.max_train_formulas]
        # Corpus frequencies only from train data, not dev/test.
        # Initial BPE base exposure is separately recorded by experiment runner.
        for text in texts:
            for byte in text.encode():self.frequency[byte]+=1
        # Byte frequencies do not define merged-token priors; estimate substring
        # counts by enumerating reachable vocabulary edges over selected training.
        graphs=[self.graph(t) for t in selected]
        counts=np.ones(len(self.vocab))
        for g in graphs:
            for ends,ids,x,cost in g:
                for id in ids:counts[id]+=1
        self.frequency=counts
        graphs=[self.graph(t) for t in selected]
        adam_m=np.zeros_like(self.theta);adam_v=np.zeros_like(self.theta);update=0
        for epoch in range(self.config.epochs):
            order=rng.permutation(len(graphs));returns=[];gradient_norms=[];grad_variance=[];acc=np.zeros_like(self.theta);batch=0
            for index in order:
                graph=graphs[index];n=len(graph)
                paths=[self.sample(graph,rng) for _ in range(self.config.samples)]
                # Dense cost-to-go at each RAW BYTE index, matching paper's
                # byte-index reward alignment despite variable token count.
                dense=[];terminal=[]
                for path in paths:
                    ret=0.;array=np.zeros(n)
                    for pos,end,id,cost,gradient,entropygrad in reversed(path):
                        ret=-cost+(self.config.gamma**(end-pos))*ret;array[pos:end]=ret
                    dense.append(array);terminal.append(-sum(x[3] for x in path))
                dense=np.asarray(dense);returns.extend([-x for x in terminal]);sample_grads=[]
                for k,path in enumerate(paths):
                    if self.config.estimator=='rloo':
                        baseline=(dense.sum(axis=0)-dense[k])/max(1,len(paths)-1)
                    elif self.config.estimator=='relative':
                        # Fixed byte-boundary baseline: one token per remaining
                        # byte; action-independent, unlike a learned LM head.
                        remaining=np.arange(n,0,-1,dtype=float)
                        baseline=-remaining if self.config.gamma==1 else -(1-self.config.gamma**remaining)/(1-self.config.gamma)
                    else:baseline=np.zeros(n)
                    grad=np.zeros_like(self.theta)
                    for pos,end,id,cost,score,entropygrad in path:
                        advantage=dense[k,pos]-baseline[pos]
                        grad+=advantage*score+self.config.entropy*entropygrad
                    grad/=max(1,n)  # Document-normalized objective.
                    sample_grads.append(grad)
                grad=np.mean(sample_grads,axis=0)
                grad_variance.append(float(np.mean(np.var(sample_grads,axis=0))))
                acc+=grad;batch+=1
                if batch==self.config.batch_size or index==order[-1]:
                    grad=acc/batch;norm=float(np.linalg.norm(grad));gradient_norms.append(norm)
                    if norm>self.config.gradient_clip:grad*=self.config.gradient_clip/norm
                    update+=1;adam_m=.9*adam_m+.1*grad;adam_v=.999*adam_v+.001*grad*grad
                    self.theta+=self.config.lr*(adam_m/(1-.9**update))/(np.sqrt(adam_v/(1-.999**update))+1e-8)
                    acc[:]=0;batch=0
            self.training_log.append({'epoch':epoch,'train_sampled_cost_mean':float(np.mean(returns)),
                'gradient_variance':float(np.mean(grad_variance)),'gradient_norm_mean':float(np.mean(gradient_norms)),
                'policy_parameters':self.theta.tolist(),'training_formulas':len(selected),'reward_scope':'train_only'})
        self.train_seconds=time.perf_counter()-started;return self

    def encode(self,text):
        graph=self.graph(text);n=len(graph)
        if not n:return []
        # Shortest is the fixed-vocabulary non-RL lower bound on token count;
        # policy-map is MAP of normalized learned local action probabilities.
        best=np.full(n+1,np.inf);best[n]=0;chosen=[None]*n
        for pos in range(n-1,-1,-1):
            ends,ids,x,cost=graph[pos]
            if self.mode=='shortest':local=cost
            elif self.mode=='greedy':
                chosen[pos]=int(np.argmax(self.policy(x)));continue
            else:local=-np.log(np.maximum(self.policy(x),1e-30))
            score=local+best[ends];choice=int(np.argmin(score));chosen[pos]=choice;best[pos]=score[choice]
        result=[];pos=0
        while pos<n:
            ends,ids,x,cost=graph[pos];a=chosen[pos];result.append(int(ids[a]));pos=int(ends[a])
        return result
    def decode_bytes(self,ids):return b''.join(self.vocab[i] for i in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()
    def save(self,path):
        d={'format':'rl-bpe-lattice-v2','feature_version':'context7-lookahead-v2','vocab_hex':[x.hex() for x in self.vocab],'capacity':self.capacity,
            'structural':self.structural,'mode':self.mode,'config':asdict(self.config),'theta':self.theta.tolist(),
            'frequency':self.frequency.tolist(),'training_log':self.training_log,'seed':self.seed,'train_seconds':self.train_seconds}
        path.write_text(json.dumps(d,indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text());base=FastBPE();base.vocab=[bytes.fromhex(x) for x in d['vocab_hex']];base.capacity=d['capacity']
        obj=cls(base,d['structural'],d['mode'],RLConfig(**d['config']));obj.theta=np.array(d['theta']);obj.frequency=np.array(d['frequency']);obj.training_log=d['training_log'];obj.seed=d['seed'];obj.train_seconds=d['train_seconds'];return obj
