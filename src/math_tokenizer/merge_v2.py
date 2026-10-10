"""Incremental pair-index training with hybrid AST/syntax spans."""
from collections import Counter,defaultdict
import heapq,time,json
from .bpe import MathBPE,Sequence
from .structure_v2 import structure
from .ast import compatible

class HybridSequence(Sequence):
    @classmethod
    def make(cls,text,structural):
        data=text.encode();s=structure(text) if structural else None
        spans=None if s and s.kind=='byte_fallback' else s.spans if s else ()
        return cls(list(data),list(range(len(data))),list(range(1,len(data)+1)),spans)

class FastBPE(MathBPE):
    def __init__(self,structural=False):
        super().__init__(structural,False);self.sequence_class=HybridSequence
    def train(self,texts,vocab_size=512,seed=17,episodes=0):
        if vocab_size<256:raise ValueError('byte fallback requires256')
        if hasattr(self,'_rules'):del self._rules
        started=time.perf_counter();seqs=[HybridSequence.make(t,self.structural) for t in texts]
        local=[s.counts() for s in seqs];global_counts=Counter();index=defaultdict(set);heap=[]
        for i,counts in enumerate(local):
            global_counts.update(counts)
            for pair in counts:index[pair].add(i)
        for pair,count in global_counts.items():heapq.heappush(heap,(-count,pair))
        while len(self.vocab)<vocab_size and heap:
            neg,pair=heapq.heappop(heap)
            if global_counts[pair]!=-neg or -neg<=0:continue
            chunk=self.vocab[pair[0]]+self.vocab[pair[1]]
            new=len(self.vocab)
            if chunk in self.vocab:new=self.vocab.index(chunk)
            else:self.vocab.append(chunk)
            self.merges.append((*pair,new));changed=set()
            for i in sorted(index[pair].copy()):
                old=local[i];seqs[i].merge(pair,new);updated=seqs[i].counts();local[i]=updated
                for p in old.keys()|updated.keys():
                    delta=updated[p]-old[p]
                    if delta:global_counts[p]+=delta;changed.add(p)
                    if p not in updated:index[p].discard(i)
                    elif p not in old:index[p].add(i)
            for p in changed:
                if global_counts[p]>0:heapq.heappush(heap,(-global_counts[p],p))
        self.train_seconds=time.perf_counter()-started;self.capacity=vocab_size;self.seed=seed
        return self
    def encode_reference(self,text):
        s=HybridSequence.make(text,self.structural)
        for a,b,c in self.merges:s.merge((a,b),c)
        return s.ids
    def encode(self,text):
        # Same reference semantics, hybrid front-end. Inference reused through
        # local method dispatch; fast ranked heap extracted below.
        return heap_encode(self,text)
    def save(self,path):
        super().save(path)
        d=json.loads(path.read_text());d['format']='mathbpe-v2-hybrid';path.write_text(json.dumps(d,indent=2))
    @classmethod
    def load(cls,path):
        old=MathBPE.load(path);obj=cls(old.structural);obj.__dict__.update(old.__dict__);return obj


def heap_encode(model,text):
    s=HybridSequence.make(text,model.structural);n=len(s.ids)
    if s.spans is None or n<2 or not model.merges:return s.ids
    # Merge dictionary is immutable once trained; avoid rebuilding per formula.
    if not hasattr(model,'_rules'):
        model._rules={}
        for rank,(a,b,c) in enumerate(model.merges):model._rules.setdefault((a,b),(rank,c))
    left=list(range(-1,n-1));right=list(range(1,n+1));right[-1]=-1
    alive=[True]*n;versions=[0]*n;heap=[]
    def push(i):
        if i<0 or not alive[i]:return
        j=right[i]
        if j<0:return
        rule=model._rules.get((s.ids[i],s.ids[j]))
        if rule and (not s.spans or compatible(s.starts[i],s.ends[j],s.spans)):
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
