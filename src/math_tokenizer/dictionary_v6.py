"""Native AST-v4 fit-subtree enrichment, train-reward document pruning."""
from collections import Counter,defaultdict
import time
from .ast_route_v6 import NativeASTV4
from .frontend_v6 import structure_native


def augment_native(base,fit,reward,capacity,extra=512,max_bytes=64):
    started=time.perf_counter();counts=Counter();units=set()
    for row in fit:
        text=row['latex'];raw=text.encode();s=structure_native(text)
        units.update(t.encode() for a,b,t in s.lexemes if t.startswith('\\') or not t.isascii())
        for a,b in s.spans:
            if 3<=b-a<=max_bytes:counts[raw[a:b]]+=1
    present=set(base.vocab)
    added=[p for p,c in sorted(counts.items(),key=lambda x:(-x[1]*(len(x[0])-1),x[0])) if c>=2 and p not in present][:extra]
    vocab=list(base.vocab)+added;model=NativeASTV4(vocab);protected={i for i,p in enumerate(vocab) if i<256 or p in units}
    if len(protected)>capacity:raise ValueError('protected budget exceeded')
    group_bytes=Counter();usage=defaultdict(float)
    for r in reward:group_bytes[r['split_group']]+=len(r['latex'].encode())
    for r in reward:
        for id in model.encode(r['latex']):usage[id]+=1/group_bytes[r['split_group']]
    score={}
    for id in range(256,len(vocab)):
        if id in protected:continue
        if not usage[id]:score[id]=0.;continue
        try:text=vocab[id].decode()
        except UnicodeDecodeError:alternate=len(vocab[id])
        else:
            # Remove exactly this trie terminal. Bytes guarantee an alternative;
            # lexical/tree constraints and the actual DP stay enabled.
            nodes=(model.terminal==id).nonzero()[0];model.terminal[nodes]=-1
            try:alternate=len(model.encode(text))
            finally:model.terminal[nodes]=id
        score[id]=usage[id]*max(0,alternate-1)
    order=sorted(score,key=lambda id:(-score[id],id>=len(base.vocab),id));kept=protected|set(order[:capacity-len(protected)])
    final=NativeASTV4([p for id,p in enumerate(vocab) if id in kept]);final.capacity=capacity;final.train_seconds=base.train_seconds+time.perf_counter()-started
    final.training_log=list(base.training_log)+[{'stage':'native_fit_subtree_reward_pruning','extra_proposed':len(added),'extra_retained':sum(id>=len(base.vocab) for id in kept),'source_tree':'native_binding_cst','reward_ids':[r['id'] for r in reward],'fit_ids':[r['id'] for r in fit],'proxy':'docmacro actual isolated alternative DP benefit, not global exact marginal gain'}]
    return final
