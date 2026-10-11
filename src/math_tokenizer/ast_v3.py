"""Atomic-command BPE training; train-only seed IDs count toward full capacity."""
from collections import Counter,defaultdict
import heapq,time,re,json
from .merge_v2 import FastBPE,HybridSequence
from .structure_v2 import structure

class AtomicASTBPE(FastBPE):
    def train(self,texts,vocab_size=512,seed=17,command_budget=96,min_command_freq=2):
        started=time.perf_counter();commands=Counter()
        for text in texts:
            for a,b,t in structure(text).lexemes:
                if t.startswith('\\') and len(t)>1:commands[t.encode()]+=1
        atoms=[token for token,count in sorted(commands.items(),key=lambda x:(-x[1],x[0])) if count>=min_command_freq][:min(command_budget,vocab_size-256)]
        self.vocab=[bytes([i]) for i in range(256)]+atoms;self.merges=[]
        atomids={v:i for i,v in enumerate(self.vocab)};seqs=[]
        for text in texts:
            seq=HybridSequence.make(text,self.structural);raw=text.encode();ids=[];starts=[];ends=[];pos=0
            for a,b,t in structure(text).lexemes:
                if t.encode() not in atomids or not t.startswith('\\'):continue
                for i in range(pos,a):ids.append(raw[i]);starts.append(i);ends.append(i+1)
                ids.append(atomids[t.encode()]);starts.append(a);ends.append(b);pos=b
            for i in range(pos,len(raw)):ids.append(raw[i]);starts.append(i);ends.append(i+1)
            seq.ids,seq.starts,seq.ends=ids,starts,ends;seqs.append(seq)
        local=[s.counts() for s in seqs];counts=Counter();index=defaultdict(set);heap=[]
        for i,c in enumerate(local):
            counts.update(c)
            for pair in c:index[pair].add(i)
        for pair,count in counts.items():heapq.heappush(heap,(-count,pair))
        while len(self.vocab)<vocab_size and heap:
            neg,pair=heapq.heappop(heap)
            if counts[pair]!=-neg or -neg<=0:continue
            payload=self.vocab[pair[0]]+self.vocab[pair[1]]
            new=self.vocab.index(payload) if payload in self.vocab else len(self.vocab)
            if new==len(self.vocab):self.vocab.append(payload)
            self.merges.append((*pair,new));changed=set()
            for i in sorted(index[pair].copy()):
                old=local[i];seqs[i].merge(pair,new);current=seqs[i].counts();local[i]=current
                for p in old.keys()|current.keys():
                    delta=current[p]-old[p]
                    if delta:counts[p]+=delta;changed.add(p)
                    if p not in current:index[p].discard(i)
                    elif p not in old:index[p].add(i)
            for p in changed:
                if counts[p]>0:heapq.heappush(heap,(-counts[p],p))
        self.capacity=vocab_size;self.seed=seed;self.command_atoms=atoms;self.train_seconds=time.perf_counter()-started
        self.training_log=[{'method':'atomic_command_ast_bpe','command_budget':command_budget,'selected_commands':[a.decode() for a in atoms],'min_frequency':min_command_freq,'scope':'train_only'}]
        return self

    def decoder(self,splitfee=1.,crossfee=0.):
        from .native_v3 import NativeTokenizer
        model=NativeTokenizer(self.vocab,self.structural,'shortest',splitfee=splitfee,crossfee=crossfee)
        model.capacity=self.capacity;model.train_seconds=self.train_seconds;model.training_log=self.training_log
        return model
