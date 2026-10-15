"""Exploratory development-only vocabulary sweep of the two stage-1 routes.

No test scoring, model promotion, Cross/BPB composite, or production overwrite.
The primitive alphabet and boundary weights are fixed across every budget.
"""
import argparse,csv,gzip,hashlib,json,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.final_models import load_final
from math_tokenizer.measure_v6 import measure,fit_usage,clear
from math_tokenizer.metrics_v2 import token_bytes


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def pareto(rows):
    # Compare measured compression/speed/byte-trie memory only. Structural and
    # prediction proxies remain descriptive and cannot determine a best budget.
    names=[]
    for a in rows:
        def dominates(b):
            pairs=[(b['document_macro_tokens_per_byte'],a['document_macro_tokens_per_byte']),
                   (-b['encode_mib_s'],-a['encode_mib_s']),
                   (b['persistent_numeric_bytes_lower_bound'],a['persistent_numeric_bytes_lower_bound'])]
            return all(x<=y for x,y in pairs) and any(x<y for x,y in pairs)
        if not any(dominates(b) for b in rows if b is not a):names.append(a['capacity'])
    return names


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'experiments/diagnostics/vocab_sweep_20261015')
    parser.add_argument('--capacities',default='768,1024,1536,2048,3072,4096,8192');args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    if ROOT/'experiments/diagnostics' not in out.parents:raise ValueError('Use isolated diagnostic directory')
    capacities=list(map(int,args.capacities.split(',')));out.mkdir(parents=True)
    catalog=read_catalog();byid={r['id']:r for r in catalog};splits=json.loads((ROOT/'data/benchmark/splits.json').read_text())
    testids=set().union(*(set(v) for k,v in splits.items() if 'test' in k));manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
        'capacities':capacities,'boundary_weights':[.25,.30],'status':'exploratory development-only;not downstream optimum',
        'catalog_sha256':sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),'source_sha256':sha(Path(__file__)),
        'test_scores_read':False,'selection_metrics':['compression','encode speed','memory'],
        'descriptive_only':['Cross%','fit unigram BPB','usage','Renyi'],
        'human_decisions':'fixed primitive alphabet,budget grid,BoundaryASTweights;no new boundary learning',
        'scenarios':{}}
    # Write protocol before any new development scoring.
    write(out/'manifest.json',manifest);summaries=[];perrows=[];audits=[]
    for scenario in ('broad','specialized'):
        scope=json.loads((ROOT/f'experiments/current/{scenario}_scope.json').read_text())
        fit=[byid[i] for i in scope['fit_ids']];aux=[byid[i] for i in scope['reward_ids']];dev=[byid[i] for i in scope['dev_ids']]
        idsets=[{r['id'] for r in rr} for rr in (fit,aux,dev)];groups=[{r['split_group'] for r in rr} for rr in (fit,aux,dev)]
        assert all(not idsets[i]&idsets[j] and not groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2)))
        assert not set.union(*idsets)&testids
        atoms=derive_atoms(fit,max(capacities),384)
        assert len(atoms)<=min(capacities) and [p.hex() for p in atoms]==scope['shared_atom_hex']
        manifest['scenarios'][scenario]={'fit_ids':scope['fit_ids'],'auxiliary_ids':scope['reward_ids'],'dev_ids':scope['dev_ids'],
            'fit':len(fit),'aux':len(aux),'dev':len(dev),'dev_papers':len(groups[2]),'atom_count':len(atoms),
            'atom_sha256':hashlib.sha256(b''.join(atoms)).hexdigest()}
        write(out/'manifest.json',manifest)
        for capacity in capacities:
            folder=out/scenario/str(capacity);folder.mkdir(parents=True)
            started=time.perf_counter();bpe=AtomicBPE.train(fit+aux,atoms,capacity);ast=BoundaryAST(bpe.vocab,.25,.30)
            ast.train_seconds=bpe.train_seconds;assert len(bpe.vocab)<=capacity
            assert ast.vocab==bpe.vocab
            bpe.save(folder/'math_atom_bpe.json');ast.save(folder/'boundary_ast.json')
            frozen_match=None
            if capacity in (1024,2048):
                frozen=load_final('official_atomic_bpe',scenario,capacity);assert frozen.vocab==bpe.vocab;frozen_match=True
                del frozen
            models={'official_atomic_bpe':bpe,'ast_boundary':ast};times=defaultdict(list);nbytes=sum(len(r['latex'].encode()) for r in dev)
            references={}
            for repeat in range(7):
                order=list(models) if repeat%2==0 else list(reversed(models))
                for name in order:
                    model=models[name];clear(model);t=time.perf_counter();ids=[model.encode(r['latex']) for r in dev];times[name].append(time.perf_counter()-t)
                    if name not in references:references[name]=ids
                    else:assert references[name]==ids
            for name,model in models.items():
                usage=fit_usage(model,fit);summary,rr=measure(model,dev,name,repeats=1,training_usage=usage)
                bits={str(alpha):0. for alpha in (.01,.1,.5,1.)};den={str(alpha):sum(usage.values())+alpha*len(model.vocab) for alpha in (.01,.1,.5,1.)}
                import math
                for ids in references[name]:
                    for alpha in (.01,.1,.5,1.):bits[str(alpha)]+=-sum(math.log2((usage[id]+alpha)/den[str(alpha)]) for id in ids)
                summary.update(capacity=capacity,scenario=scenario,suite='development_only',primitive_count=len(atoms),
                    encode_seconds_repeats=times[name],encode_mib_s=nbytes/(1048576*np.median(times[name])),
                    bpb_smoothing_sensitivity={a:v/nbytes for a,v in bits.items()},
                    persistent_numeric_bytes_lower_bound=sum(v.nbytes for v in vars(model).values() if isinstance(v,np.ndarray)),
                    frozen_vocab_reproduced=frozen_match,training_and_build_seconds=time.perf_counter()-started)
                # Include codec trie numeric arrays in BPE's footprint lower bound.
                if hasattr(model,'codec'):summary['persistent_numeric_bytes_lower_bound']+=sum(v.nbytes for v in vars(model.codec).values() if isinstance(v,np.ndarray))
                summary.pop('usage');summaries.append(summary)
                for row in rr:row.update(capacity=capacity,scenario=scenario)
                perrows.extend(rr)
                assert all(r['roundtrip'] and r['payload_exact'] for r in rr)
            token_bytes.cache_clear();del models,bpe,ast,references
            print('Development sweep completed',scenario,capacity,flush=True)
    audits.append({'round':1,'passed':True,'scope':'source/ID split isolation;zero test scoring;fixed primitive hashes and decoder weights'})
    audits.append({'round':2,'passed':True,'scope':'same dictionary per pair;all raw payload/roundtrip checks;deterministic inference;1024/2048 original dictionaries exactly reproduced'})
    grouped=defaultdict(list)
    for r in perrows:grouped[(r['scenario'],r['capacity'],r['method'])].append(r)
    for summary in summaries:
        rr=grouped[(summary['scenario'],summary['capacity'],summary['method'])];assert len(rr)==summary['formulas']
        assert sum(r['tokens'] for r in rr)==summary['tokens'] and sum(r['bytes'] for r in rr)==summary['bytes']
        papers=defaultdict(lambda:[0,0])
        for r in rr:papers[r['group']][0]+=r['tokens'];papers[r['group']][1]+=r['bytes']
        assert abs(np.mean([t/b for t,b in papers.values()])-summary['document_macro_tokens_per_byte'])<1e-12
        assert abs(sum(r['commands_intact'] for r in rr)/sum(r['commands'] for r in rr)-summary['command_integrity'])<1e-12
    assert manifest['catalog_sha256']==sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')
    audits.append({'round':3,'passed':True,'scope':'all per-formula and paper-macro count aggregates;common7pass timings;unchanged catalog','rows':len(perrows)})
    write(out/'metrics.json',summaries);write(out/'audits.json',audits)
    with gzip.open(out/'per_formula.jsonl.gz','wt') as f:
        for r in perrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    fields=['scenario','method','capacity','vocab_actual','formulas','document_macro_tokens_per_byte','encode_mib_s','decode_mib_s',
            'command_integrity','number_integrity','fit_predictive_unigram_bits_per_byte','vocab_utilization','independent_tree_crossing','persistent_numeric_bytes_lower_bound']
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    front={}
    for scenario in ('broad','specialized'):
        for method in ('ast_boundary','official_atomic_bpe'):
            selected=[m for m in summaries if m['scenario']==scenario and m['method']==method]
            front[scenario+'/'+method]=pareto(selected)
    write(out/'pareto.json',{'metrics':'compression/speed/numeric-memory lower bound;no Cross/BPB weight','frontiers':front,
        'global_best':None,'reason':'best vocabulary depends on objective and downstream model/compute budget;development data already exposed'})
    print('Sweep complete',out,flush=True)


if __name__=='__main__':main()
