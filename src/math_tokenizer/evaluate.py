"""One evaluation routine for all adapters, with paired group bootstrap."""
import hashlib,json,platform,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .ast import byte_spans,compatible,ParseError

METRICS_VERSION='mathlab-metrics-v1'

def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x]

def measure(model,records,name,seed,repeats=5):
    rows=[]
    # Identical no-special-token, full-string, no truncation, no per-model filtering.
    # Time contains encode only, including AST parsing, excludes decode/metrics.
    durations=[];encodings=[]
    for _ in range(repeats):
        start=time.perf_counter()
        current=[model.encode(r['latex']) for r in records]
        durations.append(time.perf_counter()-start)
        if not encodings:encodings=current
        elif current!=encodings:raise AssertionError('nondeterministic evaluation')
    for r,ids in zip(records,encodings):
        text=r['latex'];raw=text.encode();decoded=model.decode(ids)
        exact=decoded==text
        violations=None;coverage=False
        try:
            spans=byte_spans(text);coverage=True
            # Recover byte spans by decoding individual ID payloads. Byte tokens
            # can be incomplete UTF8, so adapters exposing bytes are preferred.
            if hasattr(model,'decode_bytes'):
                payloads=[model.decode_bytes([i]) for i in ids]
            elif hasattr(model,'tokenizer') and hasattr(model.tokenizer,'decode_single_token_bytes'):
                payloads=[model.tokenizer.decode_single_token_bytes(i) for i in ids]
            else:
                # String-only decoders may insert U+FFFD per token. Count only
                # when concatenation independently equals original bytes.
                payloads=[model.decode([i]).encode() for i in ids]
            if b''.join(payloads)==raw:
                offset=0;violations=0
                for chunk in payloads:
                    end=offset+len(chunk)
                    violations+=not compatible(offset,end,spans);offset=end
        except (ParseError,RecursionError):pass
        rows.append({'id':r['id'],'group':r['split_group'],'domain':r['domain'],'origin':r['origin'],
                     'bytes':len(raw),'tokens':len(ids),'roundtrip':exact,'ast_supported':coverage,
                     'ast_crossing_tokens':violations,'method':name,'seed':seed})
    totalbytes=sum(r['bytes'] for r in rows);totaltokens=sum(r['tokens'] for r in rows)
    median=float(np.median(durations))
    supported=[r for r in rows if r['ast_supported']]
    comparable=[r for r in supported if r['ast_crossing_tokens'] is not None]
    summary={'method':name,'seed':seed,'n_formulas':len(rows),'bytes':totalbytes,'tokens':totaltokens,
             'tokens_per_formula':totaltokens/max(1,len(rows)),'bytes_per_token':totalbytes/max(1,totaltokens),
             'roundtrip_rate':sum(r['roundtrip'] for r in rows)/max(1,len(rows)),
             'vocab_actual':len(model.vocab) if hasattr(model,'vocab') else model.vocab_size,
             'train_seconds':getattr(model,'train_seconds',0),'encode_seconds_median':median,
             'encode_seconds_repeats':durations,'encode_mib_s':totalbytes/(1024**2*max(median,1e-12)),
             'ast_supported_count':len(supported),'ast_boundary_measurable_count':len(comparable),
             'ast_crossing_token_rate':sum(r['ast_crossing_tokens'] for r in comparable)/max(1,sum(r['tokens'] for r in comparable)) if comparable else None,
             'metrics_version':METRICS_VERSION}
    return summary,rows

def bootstrap_difference(a,b,seed=20261010,n=1000):
    # Paired document/group bootstrap, not iid formula resampling.
    amap={r['id']:r for r in a};bmap={r['id']:r for r in b}
    assert amap.keys()==bmap.keys()
    groups=defaultdict(list)
    for i,r in amap.items():groups[r['group']].append(i)
    keys=sorted(groups);rng=np.random.default_rng(seed);d=[]
    for _ in range(n):
        sampled=rng.choice(keys,len(keys),replace=True)
        ids=[i for g in sampled for i in groups[g]]
        d.append(sum(amap[i]['tokens']-bmap[i]['tokens'] for i in ids)/max(1,len(ids)))
    return {'delta_tokens_per_formula':sum(amap[i]['tokens']-bmap[i]['tokens'] for i in amap)/len(amap),
            'ci95_group_bootstrap':np.quantile(d,[.025,.975]).tolist(),'groups':len(keys),'resamples':n,
            'interpretation':'A minus B; negative favors A; conditional on this training seed/corpus'}
