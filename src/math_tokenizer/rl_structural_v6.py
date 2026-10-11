"""v5 exact expected-policy-gradient branch with independent tree credit.

Learn merge preferences from a training-internal contextual bandit. Reward is
the exact immediate change in document-heldout compression/code/lexical costs,
plus independent KaTeX crossing change. This is not an optimal long-horizon
claim. Whole-trajectory RLOO remains a separately measured v5 branch.
"""
from collections import Counter
import math
import time
import numpy as np
from .rl_vocab_v5 import Indexed,MergeSession,internal_split
from .rl_route_v6 import AtomicVocab,AtomCodec,derive_atoms
from .bpe import Sequence


class TreeIndexed(Indexed):
    def __init__(self,texts,cuts,sequences,references):
        self.masks={};self.tree_local={}
        for text,seq in zip(texts,sequences):
            spans=references.get(text)
            if spans is None:continue
            n=len(text.encode());end=[0]*(n+1);start=[n+1]*(n+1)
            for a,b in spans:end[a]=max(end[a],b);start[b]=min(start[b],a)
            self.masks[id(seq)]=(end,start)
        super().__init__(texts,cuts,sequences)
        self.tree_delta=Counter();self.tree_bad=0
        for seq,(raw,gain,boundary) in zip(self.seqs,self.local):
            delta,bad=self.tree_local.get(id(raw),(Counter(),0));self.tree_delta.update(delta);self.tree_bad+=bad

    def scan(self,seq,cuts=None):
        raw,gain,boundary=Indexed.scan(seq,cuts);mask=self.masks.get(id(seq))
        if mask is not None:
            maximum,minimum=mask
            def crosses(a,b):return max(maximum[a+1:b],default=0)>b or min(minimum[a+1:b],default=b)<a
            existing=[int(crosses(a,b)) for a,b in zip(seq.starts,seq.ends)];delta=Counter();last={}
            for i,pair in enumerate(zip(seq.ids,seq.ids[1:])):
                if pair[0]==pair[1]:
                    if last.get(pair,-2)==i-1:continue
                    last[pair]=i
                delta[pair]+=int(crosses(seq.starts[i],seq.ends[i+1]))-existing[i]-existing[i+1]
            self.tree_local[id(raw)]=(delta,sum(existing))
        return raw,gain,boundary

    def merge(self,pair,new):
        affected=list(self.index[pair]);old={i:(id(self.local[i][0]),self.tree_local.get(id(self.local[i][0]))) for i in affected}
        super().merge(pair,new)
        for i,(key,previous) in old.items():
            current=self.tree_local.get(id(self.local[i][0]))
            if previous is not None:
                self.tree_delta.subtract(previous[0]);self.tree_bad-=previous[1];self.tree_local.pop(key,None)
            if current is not None:self.tree_delta.update(current[0]);self.tree_bad+=current[1]

    def clone(self):
        obj=super().clone();obj.masks={id(target):self.masks[id(source)] for source,target in zip(self.seqs,obj.seqs) if id(source) in self.masks}
        obj.tree_local=self.tree_local.copy();obj.tree_delta=self.tree_delta.copy();obj.tree_bad=self.tree_bad;return obj


class StructuralCodec(AtomCodec):
    def __init__(self,atoms,reward):
        super().__init__(atoms);self.references={r['latex']:r['reference']['spans'] for r in reward}
    def indexed(self,texts,cuts=None):
        key=(tuple(texts),None if cuts is None else tuple(frozenset(c) for c in cuts))
        if key in self.training_indices:return self.training_indices[key].clone()
        seqs=[]
        for text in texts:
            ids=[self.symbol_reverse[c] for c in self.symbols(text)];starts=[];ends=[];position=0
            for id in ids:starts.append(position);position+=len(self.atoms[id]);ends.append(position)
            seqs.append(Sequence(ids,starts,ends,()))
        template=TreeIndexed(texts,cuts,seqs,self.references);self.training_indices[key]=template;return template.clone()


class StructuralSession(MergeSession):
    def __init__(self,model,records,tree_weight=.5):super().__init__(model,records);self.tree_weight=tree_weight
    def candidates(self,frequency_only=False):
        pairs,x,rewards,parts=super().candidates(frequency_only)
        if not pairs:return pairs,x,rewards,parts
        tree=[]
        for pair,component in zip(pairs,parts):
            gain=self.h.gains[pair];delta=self.h.tree_delta[pair]
            before=self.h.tree_bad/max(1,self.totalheld);after=(self.h.tree_bad+delta)/max(1,self.totalheld-gain)
            improvement=before-after;tree.append(improvement*1000)
            component['independent_tree_rate_improvement']=improvement;component['total']+=self.tree_weight*improvement
        return pairs,np.column_stack((x,tree)),rewards+self.tree_weight*np.array(tree)/1000,parts


def train_structural(records,capacity,config,seed=17,tree_weight=.5,method='rl',atom_budget=384):
    started=time.perf_counter();fit,reward=internal_split(records,config);atoms=derive_atoms(fit,capacity,atom_budget)
    model=AtomicVocab(config,atoms,StructuralCodec(atoms,reward));model.theta=np.zeros(12);session=StructuralSession(model,records,tree_weight)
    while len(model.vocab)<capacity:
        pairs,x,r,parts=session.candidates(method=='frequency')
        if not pairs:break
        if method=='rl':
            # Exact conditional policy-gradient expectation eliminates action
            # sampling variance for this finite, fully scored bandit.
            for _ in range(config.updates):
                logits=x@model.theta;logits-=logits.max();p=np.exp(logits);p/=p.sum()
                gradient=(1-config.exploration)*(p*(r-float(p@r)))@x
                norm=float(np.linalg.norm(gradient))
                if norm>1:gradient/=norm
                model.theta+=config.learning_rate*1000*gradient
            action=int(np.argmax(x@model.theta))
        elif method=='reward_greedy':action=int(np.argmax(r))
        elif method=='frequency':action=0
        else:raise ValueError('unknown structural method')
        session.commit(pairs[action],parts[action],model.theta,float(np.max(r)),method)
    session.finish(capacity,seed,'structural_expected_policy_gradient' if method=='rl' else 'structural_'+method,time.perf_counter()-started)
    model.logs.insert(0,{'reward_reference':'independent KaTeX syntax AST','tree_weight':tree_weight,'exact_one_step_credit':True,'long_horizon_optimality_claim':False,'atom_budget':atom_budget})
    return model
