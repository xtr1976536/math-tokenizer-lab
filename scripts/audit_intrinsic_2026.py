"""Independent archive audit, separate from the measurement implementation."""
import gzip,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.frontend_v6 import reference_lex
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.final_models import load_final
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from tokenizers import Tokenizer
import sentencepiece as spm
import re
from tokenizer_analysis.metrics.information_theoretic import InformationTheoreticMetrics

OUT=ROOT/'experiments/intrinsic_20261015'
def main():
    catalog={r['id']:r for r in read_catalog()};protocol=json.loads((OUT/'protocol.json').read_text())
    details=[];tool=InformationTheoreticMetrics.__new__(InformationTheoreticMetrics)
    fixed={'qwen25':load_final('Qwen2.5-0.5B'),'llama3':load_final('meta-llama/Meta-Llama-3-8B'),'o200k':load_final('tiktoken_o200k_base')}
    for scope in protocol['splits']:
        folder=OUT/scope['name'];summary=json.loads((folder/'summary.json').read_text())
        with gzip.open(folder/'per_formula.jsonl.gz','rt') as stream:rows=list(map(json.loads,stream))
        groups=defaultdict(list)
        for row in rows:groups[row['method']].append(row)
        for s in summary:
            rr=groups[s['method']];assert len(rr)==len(scope['test'])
            assert {r['id'] for r in rr}==set(scope['test']) and s['roundtrip']==1
            counts=Counter(i for r in rr for i in r['ids']);N=sum(counts.values())
            assert abs(tool.compute_renyi_entropy(counts,2.5)/math.log2(s['vocab'])-s['renyi'])<1e-12
            assert abs(len(counts)/s['vocab']-s['used'])<1e-12
            assert sum(r['tokens'] for r in rr)==N
            # Cmd/Num and AST metrics reconstructed independently for all rows
            # using stored token length endpoints from published checkpoints.
            if s['method'] in ('hard_ast_bpe','ast_v4','boundary_ast','math_atom_bpe'):
                d=json.loads((folder/'models'/f"{s['method']}.json").read_text())
                if 'vocab_hex' in d:vocab=[bytes.fromhex(p) for p in d['vocab_hex']]
                else:
                    from math_tokenizer.atomic_bpe_v7 import AtomicBPE
                    vocab=AtomicBPE.load(folder/'models/math_atom_bpe.json').vocab
            else:vocab=None
            if vocab is None:
                model=fixed.get(s['method'])
                if s['method']=='byte_bpe':model=HFTokenizer(Tokenizer.from_file(str(folder/'models/byte_bpe.json')))
                if s['method']=='sp_unigram':model=SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(folder/'models/sp_unigram.model')))
            totals=Counter();papers=defaultdict(lambda:[0,0])
            for row in rr:
                record=catalog[row['id']];raw=record['latex'].encode();assert len(raw)==row['bytes'];assert row['tokens']==len(row['ids'])
                papers[row['group']][0]+=row['tokens'];papers[row['group']][1]+=row['bytes']
                payload=[vocab[i] for i in row['ids']] if vocab is not None else [token_bytes(model,i) for i in row['ids']]
                assert b''.join(payload)==raw
                lengths=np.cumsum([len(p) for p in payload]);cuts={0,*map(int,lengths)};inside=sorted(cuts-{0,len(raw)})
                lex=reference_lex(record['latex']);lexset=set(lex)
                ast=[(a,b) for a,b in set(map(tuple,record['reference']['spans'])) if (a,b) not in lexset and (a,b)!=(0,len(raw))]
                aligned=sum(a in cuts and b in cuts for a,b in ast);assert aligned==row['ast_good'] and len(ast)==row['ast_nodes']
                from bisect import bisect_right
                command_count=command_good=number_count=number_good=0
                for a,b in lex:
                    value=raw[a:b].decode();j=bisect_right(inside,a);ok=j==len(inside) or inside[j]>=b
                    if value.startswith('\\'):command_count+=1;command_good+=ok
                    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',value):number_count+=1;number_good+=ok
                assert (command_count,command_good,number_count,number_good)==(row['commands'],row['command_good'],row['numbers'],row['number_good'])
                # Direct interval predicate, independent of sweep cross_count.
                if int(record['id'][:8],16)%29==0:
                    bounds=[0,*map(int,lengths)];spans=record['reference']['spans']
                    cross=sum(any(a<u<b<v or u<a<v<b for u,v in spans) for a,b in zip(bounds,bounds[1:]))
                    assert cross==row['crossings']
            assert abs(np.mean([t/b for t,b in papers.values()])-s['tb'])<1e-12
            assert abs(s['bytes']/1048576/np.median(s['timing_seconds'])-s['speed'])<1e-12 and len(s['timing_seconds'])==7
            details.append({'protocol':scope['name'],'method':s['method'],'rows':len(rr),'passed':True})
    for rel,sha in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((ROOT/'data/benchmark/metadata/formulas.jsonl.gz').read_bytes()).hexdigest()==protocol['catalog_sha256']
    out={'round':5,'passed':True,'scope':'independent frozen archive replay;all declared-vocab Renyi and Used;all custom AST node boundaries;sample direct crossing predicate;all paper means and timing;code/data hashes','checks':details}
    (OUT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('INDEPENDENT AUDIT PASS',len(details),'records',sum(d['rows'] for d in details))
if __name__=='__main__':main()
