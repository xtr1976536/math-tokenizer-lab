"""Whole-formula marginal dictionary pruning with explicit structural scope.

Vocabulary candidates come from fit. Deletion loss is actual constrained
shortest-path token increase on reward documents, never isolated token spelling.
Inverse indexes invalidate every affected cached loss after each removal.
This is exact current deletion loss, not a submodularity/global-optimality claim.
"""
from collections import Counter,defaultdict
import heapq,time
import numpy as np
from .ast_route_v6 import NativeASTV4
from .frontend_v6 import structure_native
from .rl_route_v6 import derive_atoms


def candidates(fit, atoms, base_vocab, extra=2048, max_bytes=48):
    counts=Counter();paper_counts=defaultdict(set)
    for row in fit:
        raw=row['latex'].encode();structure=structure_native(row['latex'])
        for a,b in structure.spans:
            if 2<=b-a<=max_bytes:
                piece=raw[a:b];counts[piece]+=1;paper_counts[piece].add(row['split_group'])
        # Adjacent complete lexical units: candidates remain strings, and the
        # actual tree-constrained path decides their legal occurrence contexts.
        lex=structure.lexemes
        for i in range(len(lex)):
            for width in range(2,min(5,len(lex)-i)+1):
                a,b=lex[i][0],lex[i+width-1][1]
                if b-a>max_bytes:break
                piece=raw[a:b];counts[piece]+=1;paper_counts[piece].add(row['split_group'])
    reserved=set(atoms)|set(base_vocab)
    ranked=sorted((p for p,n in counts.items() if n>=3 and len(paper_counts[p])>=2 and p not in reserved),
                  key=lambda p:(-len(paper_counts[p])*np.log1p(counts[p])*(len(p)-1),p))
    return list(dict.fromkeys(list(atoms)+list(base_vocab)+ranked[:extra]))


def reward_weights(rows, mode='paper'):
    group_bytes=Counter();domain_groups=defaultdict(set)
    for row in rows:
        group_bytes[row['split_group']]+=len(row['latex'].encode())
        domain_groups[row['paper_domain']].add(row['split_group'])
    weights=[]
    for row in rows:
        weight=1/group_bytes[row['split_group']]
        if mode=='domain':weight/=len(domain_groups[row['paper_domain']])
        weights.append(weight)
    return np.array(weights)/sum(weights)


def prune_whole_formula(vocab, atoms, reward, capacity, mode='paper', hard_tree=True,
                        progress=None, protected_extra=(), penalty=None):
    from .ast_balanced_v7 import BalancedAST
    factory=(lambda v:BalancedAST(v,penalty)) if penalty is not None else (lambda v:NativeASTV4(v,hard_tree))
    started=time.perf_counter();model=factory(vocab)
    protected_payload=set(atoms)|set(protected_extra)
    # Removing a command/codepoint terminal would also relax the lexical mask,
    # changing the admissible graph. Protect every terminal that determines
    # that mask so monotone deletion-loss claims hold on a fixed graph.
    from .frontend_v6 import lexical_ranges
    mask_units=set()
    for piece in vocab:
        if len(piece)<=1 or not (piece.startswith(b'\\') or piece[0]>=128):continue
        try:text=piece.decode()
        except UnicodeDecodeError:continue
        if lexical_ranges(text).tolist()==[[0,len(piece)]]:mask_units.add(piece)
    protected_payload.update(mask_units)
    protected={id for id,p in enumerate(vocab) if p in protected_payload}
    if len(protected)>capacity:raise ValueError('protected atoms exceed capacity')
    weights=reward_weights(reward,mode);texts=[r['latex'] for r in reward]
    if penalty is not None:
        pairs=[model.encode_with_cost(t) for t in texts];encodings=[p[0] for p in pairs];values=[p[1] for p in pairs]
    else:encodings=[model.encode(t) for t in texts];values=list(map(len,encodings))
    uses=defaultdict(set)
    for i,ids in enumerate(encodings):
        for id in set(ids):uses[id].add(i)
    terminal_nodes={int(id):int(node) for node,id in enumerate(model.terminal) if id>=0}
    kept=set(range(len(vocab)));generation=Counter();heap=[];calls=0;removed=[]

    def evaluate(id):
        nonlocal calls
        if not uses[id]:return 0.
        node=terminal_nodes[id];saved=int(model.terminal[node]);model.terminal[node]=-1
        loss=0.
        try:
            for i in sorted(uses[id]):
                if penalty is not None:alternative,value=model.encode_with_cost(texts[i])
                else:alternative=model.encode(texts[i]);value=len(alternative)
                calls+=1;delta=value-values[i]
                if delta< -1e-10:raise AssertionError('removing a terminal shortened exact shortest path')
                loss+=weights[i]*delta
        finally:model.terminal[node]=saved
        return float(loss)

    def update(id):
        generation[id]+=1
        heapq.heappush(heap,(evaluate(id),id,generation[id]))

    for id in sorted(kept-protected):update(id)
    initial_tokens=sum(weights[i]*len(ids) for i,ids in enumerate(encodings))
    while len(kept)>capacity:
        while True:
            loss,id,version=heapq.heappop(heap)
            if id in kept and generation[id]==version:break
        affected=sorted(uses[id]);model.terminal[terminal_nodes[id]]=-1;kept.remove(id)
        changed=set()
        for i in affected:
            old=set(encodings[i])
            if penalty is not None:new,value=model.encode_with_cost(texts[i])
            else:new=model.encode(texts[i]);value=len(new)
            calls+=1
            assert value>=values[i]-1e-10;values[i]=value;encodings[i]=new;changed.update(old);changed.update(new)
            for token in old:uses[token].discard(i)
            for token in set(new):uses[token].add(i)
        uses[id].clear();removed.append({'id':id,'marginal_loss':loss,'affected_formulas':len(affected)})
        # A removed token can be an alternative even in a formula whose best
        # path did not use it. Invalidate all co-occurring payload candidates,
        # not only selected IDs; every possibly affected counterfactual loss.
        removed_piece=vocab[id]
        potential={i for i,text in enumerate(texts) if removed_piece in text.encode()}
        for token in kept-protected:
            if uses[token]&potential:changed.add(token)
        for token in sorted(changed & (kept-protected)):update(token)
        if progress and len(removed)%128==0:progress(len(kept),calls)
    final=factory([p for id,p in enumerate(vocab) if id in kept])
    final.capacity=len(final.vocab);final.train_seconds=time.perf_counter()-started
    final.training_log=[{'stage':'whole_formula_counterfactual_pruning','mode':mode,
        'initial_vocab':len(vocab),'final_vocab':len(kept),'protected_atoms':len(protected),
        'protected_lexical_mask_terminals':len(mask_units),'lexical_mask_unchanged_under_pruning':True,
        'reward_ids':[r['id'] for r in reward],'counterfactual_encode_calls':calls,
        'initial_reward_weighted_tokens':float(initial_tokens),
        'final_reward_weighted_tokens':float(sum(weights[i]*len(ids) for i,ids in enumerate(encodings))),
        'same_actual_inference_decoder':True,'penalty':penalty,'global_optimality_claim':False,'deleted':removed}]
    return final
