"""Distill fit-derived statistical merge-tree signals into per-ID path costs.

No mathematical AST labels. Exact merge trees are a training-only teacher; their
approximate student uses the existing fast native primitive/scope decoder.
"""
from collections import Counter
import numpy as np
from .learned_scope_stage2 import LearnedScopeTokenizer
from .signal_stage2 import TokenCodePredictor,paper_token_rate


def distill_crossing(teacher,fit):
    total=np.zeros(len(teacher.vocab));bad=np.zeros(len(teacher.vocab))
    for r in fit:
        raw,allowed,marked,spans,_=teacher.reference_spans(r['latex'])
        for a in range(len(raw)):
            if not allowed[a]:continue
            node=0
            for b in range(a,min(len(raw),a+teacher.maxlen)):
                node=int(teacher.children[node,raw[b]])
                if node<0:break
                id=int(teacher.terminal[node]);e=b+1
                if id>=0 and allowed[e]:
                    total[id]+=1;bad[id]+=any(a<s<e<t or s<a<t<e for s,t in spans)
    rate=bad/np.maximum(1,total)
    return rate,{'fit_ids':[r['id'] for r in fit],'candidate_edge_occurrences':int(total.sum()),
                 'partial_crossing_occurrences':int(bad.sum()),'teacher':'fit-only BPE merge trees; not mathematical AST',
                 'total':total.tolist(),'crossing':bad.tolist(),'probability':rate.tolist()}


def select_distilled(teacher,pairs,fit,reward,reference_rate):
    probability,teacherlog=distill_crossing(teacher,fit);best=None;logs=[]
    for scale in (0.,.1,.3,.6):
        for scope in (0.,.1,.3,.6):
            model=LearnedScopeTokenizer(teacher.vocab,teacher.atoms,pairs,scope,token_costs=scale*probability)
            fitids=[model.encode(r['latex']) for r in fit];rr=[model.encode(r['latex']) for r in reward]
            code=TokenCodePredictor(fitids,len(model.vocab)).score(rr,reward)['paper_macro_token_bigram_code_bits_per_byte'];rate=paper_token_rate(rr,reward)
            violation=max(0.,rate/(reference_rate*1.01)-1.);key=(violation>0,violation,code,scale,scope)
            logs.append({'scale':scale,'scope':scope,'code_bpb':code,'rate':rate,'violation':violation})
            if best is None or key<best[0]:best=(key,model,scale,scope)
    best[1].training_log=[{'selection':'training-internal token code subject to compression guard',
                          'candidate_settings':logs,'selected_scale':best[2],'selected_scope':best[3],
                          'teacher':teacherlog,'reward_ids':[r['id'] for r in reward],
                          'runtime_teacher':False,'approximation':'average per-vocabulary candidate crossing, ignores occurrence context'}]
    return best[1],logs
