"""Same primitives, structural merge vocabulary with original frequency BPE.

This is a closest-prior AST BPE control, not a new algorithm contribution.
Primitive IDs are shared with every atomized baseline in the comparison.
"""
from collections import Counter,defaultdict
import heapq,time
from .atoms_v4 import FastSpanSequence
from .rl_route_v6 import AtomCodec
from .frontend_v6 import structure_native
from .ast_route_v6 import NativeASTV4


def train_structural_atoms(rows,atoms,capacity,tree=True):
    start=time.perf_counter();codec=AtomCodec(atoms);vocab=list(atoms);seqs=[]
    for r in rows:
        ids=[codec.symbol_reverse[c] for c in codec.symbols(r['latex'])];ends=[];pos=0
        for id in ids:pos+=len(atoms[id]);ends.append(pos)
        spans=structure_native(r['latex']).spans if tree else ()
        seqs.append(FastSpanSequence(ids,[0]+ends[:-1],ends,spans))
    counts=Counter();index=defaultdict(set);local=[];payload_ids={p:i for i,p in enumerate(vocab)}
    for i,s in enumerate(seqs):
        c=s.counts();local.append(c);counts.update(c)
        for pair in c:index[pair].add(i)
    heap=[(-n,p) for p,n in counts.items() if n>0];heapq.heapify(heap)
    while len(vocab)<capacity and heap:
        negative,pair=heapq.heappop(heap)
        if counts[pair]!=-negative:continue
        if -negative<2:break
        payload=vocab[pair[0]]+vocab[pair[1]];new=payload_ids.get(payload)
        if new is None:new=len(vocab);payload_ids[payload]=new;vocab.append(payload)
        changed=set()
        for i in sorted(index[pair].copy()):
            old=local[i];seqs[i].merge(pair,new);current=seqs[i].counts();local[i]=current
            for p in old.keys()|current.keys():
                delta=current[p]-old[p]
                if delta:counts[p]+=delta;changed.add(p)
                if p not in current:index[p].discard(i)
                elif p not in old:index[p].add(i)
        for p in changed:
            if counts[p]>0:heapq.heappush(heap,(-counts[p],p))
    model=NativeASTV4(vocab,tree);model.capacity=len(vocab);model.train_seconds=time.perf_counter()-start
    model.training_log=[{'same_atom_hex':[p.hex() for p in atoms],'fit_ids':[r['id'] for r in rows],
                        'tree_legal_training':tree,'scope':'frequency_BPE_closest_prior_control'}]
    return model
