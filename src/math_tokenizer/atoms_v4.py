"""Corpus-derived lexical atoms; every reserved ID counts against capacity.

No manually ranked command inventory. All observed train control sequences and
non-ASCII codepoints can be reserved; arbitrary whole formulas are not IDs.
"""
from collections import Counter,defaultdict
import heapq,time
from .ast_v3 import AtomicASTBPE
from .merge_v2 import HybridSequence
from .structure_v3 import structure_fast

class GrammarBPE(AtomicASTBPE):
 def train(self,texts,vocab_size=512,seed=17,atom_budget=256,include_numbers=False):
  start=time.perf_counter();atoms=Counter()
  for text in texts:
   for a,b,t in structure_fast(text).lexemes:
    if t.startswith('\\') or (include_numbers and t[0:1].isdigit()):atoms[t.encode()]+=1
   for char in text:
    if not char.isascii():atoms[char.encode()]+=1
  selected=[token for token,count in sorted(atoms.items(),key=lambda x:(-x[1],x[0])) if len(token)>1][:min(atom_budget,vocab_size-256)]
  self.vocab=[bytes([i]) for i in range(256)]+selected;self.merges=[];idsmap={t:i for i,t in enumerate(self.vocab)};seqs=[]
  for text in texts:
   seq=HybridSequence.make(text,self.structural);raw=text.encode();ranges=[]
   for a,b,t in structure_fast(text).lexemes:
    if t.encode() in idsmap and len(t.encode())>1:ranges.append((a,b,idsmap[t.encode()]))
   offset=0
   for c in text:
    b=c.encode();end=offset+len(b)
    if not c.isascii() and b in idsmap:ranges.append((offset,end,idsmap[b]))
    offset=end
   ranges=sorted(set(ranges));tokenids=[];starts=[];ends=[];pos=0
   for a,b,id in ranges:
    if a<pos:continue
    for j in range(pos,a):tokenids.append(raw[j]);starts.append(j);ends.append(j+1)
    tokenids.append(id);starts.append(a);ends.append(b);pos=b
   for j in range(pos,len(raw)):tokenids.append(raw[j]);starts.append(j);ends.append(j+1)
   seq.ids,seq.starts,seq.ends=tokenids,starts,ends;seqs.append(seq)
  local=[s.counts() for s in seqs];counts=Counter();index=defaultdict(set);heap=[]
  for i,c in enumerate(local):
   counts.update(c)
   for pair in c:index[pair].add(i)
  for pair,count in counts.items():heapq.heappush(heap,(-count,pair))
  while len(self.vocab)<vocab_size and heap:
   neg,pair=heapq.heappop(heap)
   if counts[pair]!=-neg or -neg<=0:continue
   payload=self.vocab[pair[0]]+self.vocab[pair[1]];new=self.vocab.index(payload) if payload in self.vocab else len(self.vocab)
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
  self.capacity=vocab_size;self.seed=seed;self.command_atoms=selected;self.train_seconds=time.perf_counter()-start
  self.training_log=[{'stage':'train_derived_grammar_atoms','all_atom_types':len(atoms),'reserved_types':len(selected),'atom_budget':atom_budget,'include_numbers':include_numbers,'selected_atom_hex':[a.hex() for a in selected],'scope':'train_only'}];return self
