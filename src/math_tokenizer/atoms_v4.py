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
 def train(self,texts,vocab_size=512,seed=17,atom_budget=256,include_numbers=False,subtree_budget=0,max_subtree_bytes=48,atom_score='frequency'):
  if vocab_size<256 or atom_budget<0 or subtree_budget<0:raise ValueError('invalid vocabulary budget')
  if atom_score not in ('frequency','gain'):raise ValueError('invalid atom selection criterion')
  start=time.perf_counter();atoms=Counter();subtrees=Counter();structures=[structure_fast(t) for t in texts]
  for text,s in zip(texts,structures):
   for a,b,t in s.lexemes:
    if t.startswith('\\') or (include_numbers and t[0:1].isdigit()):atoms[t.encode()]+=1
   for char in text:
    if not char.isascii():atoms[char.encode()]+=1
   if subtree_budget and s.kind!='byte_fallback':
    raw=text.encode()
    for a,b in s.spans:
     if 3<=b-a<=max_subtree_bytes:subtrees[raw[a:b]]+=1
  key=lambda x:(-x[1]*(len(x[0])-1 if atom_score=='gain' else 1),x[0])
  selected=[token for token,count in sorted(atoms.items(),key=key) if len(token)>1][:min(atom_budget,vocab_size-256)]
  selected_set=set(selected)
  fragments=[t for t,count in sorted(subtrees.items(),key=lambda x:(-x[1]*(len(x[0])-1),x[0])) if count>=2 and t not in selected_set][:min(subtree_budget,vocab_size-256-len(selected))]
  fragment_set=set(fragments)
  self.vocab=[bytes([i]) for i in range(256)]+selected+fragments;self.merges=[];idsmap={t:i for i,t in enumerate(self.vocab)};seqs=[]
  for text,s in zip(texts,structures):
   raw=text.encode();spans=None if self.structural and s.kind=='byte_fallback' else s.spans if self.structural else ()
   seq=HybridSequence(list(raw),list(range(len(raw))),list(range(1,len(raw)+1)),spans);ranges=[]
   for a,b,t in s.lexemes:
    if t.encode() in idsmap and len(t.encode())>1:ranges.append((a,b,idsmap[t.encode()]))
   offset=0
   for c in text:
    b=c.encode();end=offset+len(b)
    if not c.isascii() and b in idsmap:ranges.append((offset,end,idsmap[b]))
    offset=end
   if fragments:
    for a,b in s.spans:
     payload=raw[a:b]
     if payload in fragment_set:ranges.append((a,b,idsmap[payload]))
   ranges=sorted(set(ranges),key=lambda x:(x[0],-x[1],x[2])) if fragments else sorted(set(ranges));tokenids=[];starts=[];ends=[];pos=0
   for a,b,id in ranges:
    if a<pos:continue
    for j in range(pos,a):tokenids.append(raw[j]);starts.append(j);ends.append(j+1)
    tokenids.append(id);starts.append(a);ends.append(b);pos=b
   for j in range(pos,len(raw)):tokenids.append(raw[j]);starts.append(j);ends.append(j+1)
   seq.ids,seq.starts,seq.ends=tokenids,starts,ends;seqs.append(seq)
  local=[s.counts() for s in seqs];counts=Counter();index=defaultdict(set);heap=[];payload_ids=dict(idsmap)
  for i,c in enumerate(local):
   counts.update(c)
   for pair in c:index[pair].add(i)
  for pair,count in counts.items():heapq.heappush(heap,(-count,pair))
  while len(self.vocab)<vocab_size and heap:
   neg,pair=heapq.heappop(heap)
   if counts[pair]!=-neg or -neg<=0:continue
   payload=self.vocab[pair[0]]+self.vocab[pair[1]];new=payload_ids.get(payload)
   if new is None:new=len(self.vocab);payload_ids[payload]=new;self.vocab.append(payload)
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
  self.training_log=[{'stage':'train_derived_grammar_atoms','all_atom_types':len(atoms),'reserved_types':len(selected),'atom_budget':atom_budget,'include_numbers':include_numbers,'selected_atom_hex':[a.hex() for a in selected],'subtree_budget':subtree_budget,'reserved_subtrees':len(fragments),'max_subtree_bytes':max_subtree_bytes,'subtree_hex':[t.hex() for t in fragments],'atom_score':atom_score,'scope':'train_only'}];return self
