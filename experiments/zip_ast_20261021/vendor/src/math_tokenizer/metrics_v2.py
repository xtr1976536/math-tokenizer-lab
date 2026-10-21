"""Tokenizer-only intrinsic metrics; all adapters share exact raw byte spans."""
from collections import Counter,defaultdict
import json,math,re,time
import numpy as np
from functools import lru_cache
from .structure_v2 import structure
from .ast import compatible


@lru_cache(maxsize=100000)
def token_bytes(model,id):
    if hasattr(model,'decode_bytes'):return model.decode_bytes([id])
    tok=model.tokenizer
    if hasattr(tok,'decode_single_token_bytes'):return tok.decode_single_token_bytes(id)
    # HF ByteLevel mapping is invertible, even for incomplete UTF8 codepoints.
    if hasattr(tok,'id_to_token'):
        from tokenizers import pre_tokenizers
        # Canonical GPT2 byte-unicode bijection (official ByteLevel convention).
        bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1));cs=bs[:];extra=0
        for b in range(256):
            if b not in bs:bs.append(b);cs.append(256+extra);extra+=1
        reverse={chr(c):b for b,c in zip(bs,cs)}
        piece=tok.id_to_token(id)
        return bytes(reverse[c] for c in piece)
    if hasattr(tok,'id_to_piece'):
        piece=tok.id_to_piece(id)
        if re.fullmatch(r'<0x[0-9A-Fa-f]{2}>',piece):return bytes([int(piece[3:5],16)])
        return piece.replace('▁',' ').encode()
    raise TypeError('adapter has no independently verified byte payload mapping')


def boundaries(payloads):
    result=[0]
    for b in payloads:result.append(result[-1]+len(b))
    return result


def upstream_information(name,encodings,texts,vocabsize):
    # Published upstream functions, unchanged. Tokenized provider exposes only
    # common raw corpus and declared vocab size; no per-method special corpus.
    from tokenizer_analysis.metrics.information_theoretic import InformationTheoreticMetrics
    from tokenizer_analysis.core.input_types import TokenizedData
    class Provider:
        def get_tokenizer_names(self):return [name]
        def get_vocab_size(self,key):return vocabsize
        def get_languages(self,key=None):return ['math']
    metric=InformationTheoreticMetrics(Provider())
    data={name:[TokenizedData(name,'math',list(map(int,ids)),text=text) for ids,text in zip(encodings,texts)]}
    output={}
    for function,key in [('compute_renyi_efficiency_analysis','renyi_efficiency'),('compute_bigram_entropy','bigram_entropy'),('compute_trigram_entropy','trigram_entropy')]:
        output[key]=getattr(metric,function)(data)
    return output


def measure(model,records,name,seed,repeats=3,train_frequency=None,upstream=True):
    # Clear structural memoization for each model/repeat; include true parsing
    # cost in timings and publish cold protocol. No unequal cross-model cache.
    times=[];encodings=None
    for repeat in range(repeats):
        structure.cache_clear();st=time.perf_counter();ids=[model.encode(r['latex']) for r in records];times.append(time.perf_counter()-st)
        if encodings is None:encodings=ids
        else:assert ids==encodings,'deterministic inference required'
    rows=[];usage=Counter();utf8_ok=0;utf8_total=0;total_commands=0;commands_intact=0;total_numeric=0;numeric_intact=0;operators=0;operator_isolated=0
    all_payload_lengths=[];lexeme_totals=0;lexeme_intact_totals=0;total_crossing=0;structural_tokens=0;astcount=0;cstcount=0
    number_expected=set();digit_tp=0;digit_fp=0;digit_fn=0;digit_spans=0
    for record,ids in zip(records,encodings):
        raw=record['latex'].encode();payloads=[token_bytes(model,id) for id in ids]
        assert b''.join(payloads)==raw,'payload map must reconstruct original bytes'
        bds=boundaries(payloads);internal=set(bds[1:-1]);usage.update(ids)
        s=structure(record['latex']);astcount+=s.kind=='semantic_ast';cstcount+=s.kind=='latex_syntax_tree'
        crossings=sum(not compatible(a,b,s.spans) for a,b in zip(bds,bds[1:])) if s.kind!='byte_fallback' else None
        if crossings is not None:total_crossing+=crossings;structural_tokens+=len(ids)
        cmd_n=cmd_i=num_n=num_i=lex_i=op_n=op_i=0
        for start,end,t in s.lexemes:
            intact=not any(start<x<end for x in internal);lex_i+=intact
            if t.startswith('\\') and len(t)>1 and t[1:].isalpha():cmd_n+=1;cmd_i+=intact
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',t):num_n+=1;num_i+=intact
            if t in ('+','-','*','/','=','<','>') or t in (r'\cdot',r'\times',r'\leq',r'\geq',r'\neq'):
                op_n+=1;op_i+=start in bds and end in bds
            if t.isdigit() and len(t)>3:
                expected={end-k for k in range(3,len(t),3)};predicted={x for x in internal if start<x<end}
                digit_tp+=len(expected&predicted);digit_fp+=len(predicted-expected);digit_fn+=len(expected-predicted);digit_spans+=1
        total_commands+=cmd_n;commands_intact+=cmd_i;total_numeric+=num_n;numeric_intact+=num_i;operators+=op_n;operator_isolated+=op_i
        lexeme_totals+=len(s.lexemes);lexeme_intact_totals+=lex_i
        for payload in payloads:
            try:payload.decode();utf8_ok+=1
            except UnicodeDecodeError:pass
        utf8_total+=len(ids);all_payload_lengths.extend(map(len,payloads))
        rows.append({'id':record['id'],'group':record['split_group'],'domain':record['domain'],'origin':record['origin'],
            'complexity':record['complexity']['class'],'method':name,'seed':seed,'bytes':len(raw),'chars':len(record['latex']),
            'tokens':len(ids),'roundtrip':model.decode(ids)==record['latex'],'structure_kind':s.kind,
            'crossing_tokens':crossings,'lexemes':len(s.lexemes),'lexemes_intact':lex_i,'commands':cmd_n,'commands_intact':cmd_i,
            'numbers':num_n,'numbers_intact':num_i,'operators':op_n,'operators_isolated':op_i})
    totalbytes=sum(r['bytes'] for r in rows);totaltokens=sum(r['tokens'] for r in rows);vocab=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size
    p=np.array(list(usage.values()),dtype=float);p/=max(1,p.sum());entropy=-float(np.sum(p*np.log2(p))) if len(p) else 0
    efficiency={}
    for alpha in (1.,2.,2.5,3.):
        h=entropy if alpha==1 else float(np.log2(np.sum(p**alpha))/(1-alpha))
        efficiency[str(alpha)]=h/math.log2(vocab) if vocab>1 else None
    quantiles=np.quantile([r['tokens'] for r in rows],[.5,.9,.95,.99]).tolist()
    summary={'method':name,'seed':seed,'n_formulas':len(rows),'bytes':totalbytes,'tokens':totaltokens,'tokens_per_formula':totaltokens/len(rows),
        'bytes_per_token':totalbytes/totaltokens,'chars_per_token':sum(r['chars'] for r in rows)/totaltokens,'tokens_per_latex_lexeme':totaltokens/max(1,lexeme_totals),
        'tokens_length_quantiles_p50_p90_p95_p99':quantiles,'byte_compression_rate':1-totaltokens/totalbytes,
        'roundtrip_rate':sum(r['roundtrip'] for r in rows)/len(rows),'vocab_actual':vocab,'vocab_used':len(usage),'vocab_utilization':len(usage)/vocab,
        'type_token_ratio':len(usage)/totaltokens,'shannon_entropy_bits_per_token':entropy,'renyi_efficiency':efficiency,
        'empirical_unigram_bits_per_byte':entropy*totaltokens/totalbytes,
        'fixed_width_index_bits_per_byte':math.ceil(math.log2(vocab))*totaltokens/totalbytes,
        'self_vocab_payload_bytes':sum(map(len,model.vocab)) if hasattr(model,'vocab') else None,
        'singleton_type_fraction':sum(v==1 for v in usage.values())/max(1,len(usage)),
        'byte_token_usage_rate':sum(count for id,count in usage.items() if len(token_bytes(model,id))==1)/totaltokens,
        'mean_token_bytes':float(np.mean(all_payload_lengths)),'token_bytes_p95':float(np.quantile(all_payload_lengths,.95)),
        'utf8_complete_token_rate':utf8_ok/max(1,utf8_total),'lexeme_preservation_rate':lexeme_intact_totals/max(1,lexeme_totals),
        'command_integrity_rate':commands_intact/total_commands if total_commands else None,
        'number_integrity_rate':numeric_intact/total_numeric if total_numeric else None,
        'operator_isolation_rate':operator_isolated/operators if operators else None,
        'three_digit_boundary_f1':2*digit_tp/(2*digit_tp+digit_fp+digit_fn) if 2*digit_tp+digit_fp+digit_fn else None,
        'three_digit_eligible_spans':digit_spans,'semantic_ast_count':astcount,'latex_syntax_tree_count':cstcount,'byte_fallback_count':len(rows)-astcount-cstcount,
        'hybrid_tree_crossing_rate':total_crossing/max(1,structural_tokens),'hybrid_tree_measurable_tokens':structural_tokens,
        'train_seconds':getattr(model,'train_seconds',0),'encode_seconds_repeats':times,'encode_seconds_median':float(np.median(times)),
        'encode_mib_s':totalbytes/(1024**2*max(np.median(times),1e-12)),'cache_protocol':'structure_cache_cleared_before_each_repeat',
        'metrics_version':'mathlab-intrinsic-v2','token_usage':{str(k):v for k,v in sorted(usage.items())}}
    if train_frequency is not None:
        summary['unseen_in_training_token_rate']=sum(count for id,count in usage.items() if train_frequency.get(id,0)==0)/totaltokens
        summary['rare_train_token_rate']=sum(count for id,count in usage.items() if train_frequency.get(id,0)<=5)/totaltokens
    if upstream:
        summary['tokeval_official']=upstream_information(name,encodings,[r['latex'] for r in records],vocab)
        official=summary['tokeval_official']['renyi_efficiency']['per_tokenizer'][name]
        for alpha,value in efficiency.items():assert abs(official['renyi_'+alpha]['overall']-value)<1e-10
    return summary,rows
