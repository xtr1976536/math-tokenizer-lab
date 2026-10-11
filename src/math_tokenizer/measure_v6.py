"""One intrinsic protocol; independent reference tree, exact raw input.

Redundant compression units are descriptive, entropy is not a quality proof.
Timing includes every model's public string-to-ID pipeline and excludes metrics.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import math
import re
import time
import numpy as np
from .frontend_v6 import reference_lex
from .measure_v5 import cross_count
from .metrics_v2 import token_bytes
from .trajectory_v5 import safe_layout_variant


def clear(model):
    if hasattr(model,'clear_cache'):model.clear_cache()
    elif hasattr(model,'tokenizer') and hasattr(model.tokenizer,'model') and hasattr(model.tokenizer.model,'clear_cache'):model.tokenizer.model.clear_cache()
    from .structure_v3 import structure_fast
    if hasattr(structure_fast,'cache_clear'):structure_fast.cache_clear()
    from .structure_v2 import structure
    structure.cache_clear()


def fit_usage(model,records):return Counter(id for r in records for id in model.encode(r['latex']))


def measure(model,records,name,seed=0,repeats=3,training_usage=None):
    times=[];encodings=None
    for repeat in range(repeats):
        clear(model);started=time.perf_counter();ids=[model.encode(r['latex']) for r in records];times.append(time.perf_counter()-started)
        if encodings is None:encodings=ids
        else:assert encodings==ids,'nondeterministic inference'
    vocab=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size
    counts=Counter();rows=[];decode_times=[];rawtotal=0;totalcmd=goodcmd=totalnum=goodnum=lexcount=lexok=opcount=opok=0
    crossing=tree_tokens=utfok=allpieces=0;boundary_tp=boundary_pred=boundary_ref=0;codebits=rare=0.;valid_payload=True;robust=[]
    denominator=sum(training_usage.values())+.5*vocab if training_usage is not None else 1
    for record,ids in zip(records,encodings):
        text=record['latex'];raw=text.encode();counts.update(ids);rawtotal+=len(raw)
        payload=[token_bytes(model,id) for id in ids]
        exact=model.decode(ids)==text;payload_exact=b''.join(payload)==raw;valid_payload &= payload_exact
        bounds=[0]
        for p in payload:
            bounds.append(bounds[-1]+len(p));allpieces+=1
            try:p.decode();utfok+=1
            except UnicodeDecodeError:pass
        row={'id':record['id'],'group':record['split_group'],'domain':record['domain'],'paper_domain':record['paper_domain'],
             'origin':record['origin'],'complexity':record['complexity']['class'],'method':name,'seed':seed,
             'bytes':len(raw),'tokens':len(ids),'roundtrip':exact,'payload_exact':payload_exact,
             'commands':0,'commands_intact':0,'numbers':0,'numbers_intact':0,'lexemes':0,'lexemes_intact':0,'operators':0,'operators_isolated':0,
             'crossing_tokens':None,'boundary_tp':0,'boundary_pred':0,'boundary_ref':0,'predictive_bits':None}
        if payload_exact:
            lexical=reference_lex(text);expected=set();internal=set(bounds[1:-1])
            for a,b in lexical:
                token=raw[a:b].decode()
                if not token.strip():continue
                row['lexemes']+=1;row['lexemes_intact']+=not bool(internal & set(range(a+1,b)))
                intact=bisect_right(bounds,a)>=bisect_left(bounds,b)
                if token.startswith('\\'):
                    row['commands']+=1;row['commands_intact']+=intact
                if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',token):row['numbers']+=1;row['numbers_intact']+=intact
                if token in ('+','-','*','/','=','<','>',r'\cdot',r'\times',r'\leq',r'\geq',r'\neq'):
                    row['operators']+=1;row['operators_isolated']+=a in bounds and b in bounds
                expected.update((a,b))
            expected.discard(0);expected.discard(len(raw))
            row['boundary_tp']=len(internal&expected);row['boundary_pred']=len(internal);row['boundary_ref']=len(expected)
            row['crossing_tokens']=cross_count(bounds,record['reference']['spans']);crossing+=row['crossing_tokens'];tree_tokens+=len(ids)
        if training_usage is not None:
            row['predictive_bits']=-sum(math.log2((training_usage[id]+.5)/denominator) for id in ids)
            codebits+=row['predictive_bits'];rare+=sum(training_usage[id]<=5 for id in ids)
        totalcmd+=row['commands'];goodcmd+=row['commands_intact'];totalnum+=row['numbers'];goodnum+=row['numbers_intact']
        lexcount+=row['lexemes'];lexok+=row['lexemes_intact'];opcount+=row['operators'];opok+=row['operators_isolated']
        boundary_tp+=row['boundary_tp'];boundary_pred+=row['boundary_pred'];boundary_ref+=row['boundary_ref']
        variant=safe_layout_variant(text)
        original_content=sum(bool(p.strip()) for p in payload)
        variant_content=sum(bool(token_bytes(model,id).strip()) for id in model.encode(variant))
        row['layout_token_variation']=abs(original_content-variant_content)/max(1,original_content) if variant!=text else None
        if variant!=text:robust.append(row['layout_token_variation'])
        rows.append(row)
    for repeat in range(repeats):
        started=time.perf_counter();decoded=[model.decode(ids) for ids in encodings];decode_times.append(time.perf_counter()-started)
        assert len(decoded)==len(records)
    total=sum(counts.values());p=np.array(list(counts.values()))/max(1,total);entropy=-float(np.sum(p*np.log2(p)))
    groups=defaultdict(lambda:[0,0])
    for row in rows:groups[row['group']][0]+=row['tokens'];groups[row['group']][1]+=row['bytes']
    summary={'method':name,'seed':seed,'formulas':len(rows),'tokens':total,'bytes':rawtotal,'vocab_actual':vocab,
       'test_id_sha256':hashlib.sha256('\n'.join(sorted(r['id'] for r in rows)).encode()).hexdigest(),
       'tokens_per_formula':total/max(1,len(rows)),'tokens_per_byte':total/max(1,rawtotal),'bytes_per_token':rawtotal/max(1,total),
       'document_macro_tokens_per_byte':float(np.mean([t/b for t,b in groups.values()])),
       'length_p50_p95_p99':np.quantile([r['tokens'] for r in rows],[.5,.95,.99]).tolist(),
       'roundtrip_rate':sum(r['roundtrip'] for r in rows)/max(1,len(rows)),'payload_fidelity_rate':sum(r['payload_exact'] for r in rows)/max(1,len(rows)),
       'command_integrity':goodcmd/totalcmd if totalcmd else None,'number_integrity':goodnum/totalnum if totalnum else None,
       'lexeme_preservation':lexok/max(1,lexcount),'operator_isolation':opok/opcount if opcount else None,
       'independent_tree_crossing':crossing/max(1,tree_tokens),'tree_measurable_fraction':tree_tokens/max(1,total),
       'lexical_boundary_precision':boundary_tp/max(1,boundary_pred),'lexical_boundary_recall':boundary_tp/max(1,boundary_ref),
       'lexical_boundary_f1':2*boundary_tp/max(1,boundary_pred+boundary_ref),
       'utf8_complete_token_rate':utfok/max(1,allpieces),'vocab_used':len(counts),'vocab_utilization':len(counts)/vocab,
       'renyi_efficiency':{str(alpha):(entropy if alpha==1 else float(np.log2(np.sum(p**alpha))/(1-alpha)))/math.log2(vocab) for alpha in (1.,2.,2.5,3.)},
       'empirical_unigram_bits_per_byte':entropy*total/max(1,rawtotal),
       'fit_predictive_unigram_bits_per_byte':codebits/max(1,rawtotal) if training_usage is not None else None,
       'fit_rare_token_rate':rare/max(1,total) if training_usage is not None else None,
       'layout_token_variation':float(np.mean(robust)) if robust else None,'layout_probe_formulas':len(robust),
       'encode_seconds_repeats':times,'encode_mib_s':rawtotal/(1024**2*max(float(np.median(times)),1e-12)),
       'decode_seconds_repeats':decode_times,'decode_mib_s':rawtotal/(1024**2*max(float(np.median(decode_times)),1e-12)),
       'train_seconds':getattr(model,'train_seconds',0), 'metric_version':'v6-independent-katex-and-raw-byte-protocol',
       'rank_eligible_lossless':all(r['roundtrip'] and r['payload_exact'] for r in rows),'usage':{str(k):v for k,v in counts.items()}}
    return summary,rows


def selection_utility(metric):
    command=metric['command_integrity'] if metric['command_integrity'] is not None else 1.
    number=metric['number_integrity'] if metric['number_integrity'] is not None else 1.
    return metric['document_macro_tokens_per_byte']+.15*(1-command)+.03*(1-number)+.10*metric['independent_tree_crossing']
