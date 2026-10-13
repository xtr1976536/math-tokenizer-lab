"""Two-capacity deployed signal/parameter-RL experiment; no AST training labels."""
import argparse,gzip,json,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.signal_stage2 import SignalTokenizer,paper_token_rate,TokenCodePredictor
from math_tokenizer.learned_scope_stage2 import LearnedScopeTokenizer,discover_pairs,select_scope,optimize_deployed_costs
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.independent_v7 import additional,source_scopes
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.final_models import load_final
from run_stage2 import balanced,provenance_audit,timing,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/stage2/pilot06')
    p.add_argument('--capacities',default='1024,2048');p.add_argument('--rounds',type=int,default=16)
    args=p.parse_args();out=args.output.resolve()
    if (out/'manifest.json').exists():raise ValueError('use new output directory')
    if ROOT/'experiments/stage2' not in out.parents:raise ValueError('invalid output directory')
    out.mkdir(parents=True,exist_ok=True);by={r['id']:r for r in read_catalog()}
    scopeids=json.loads((ROOT/'experiments/stage2/pilot04/run_manifest.json').read_text())['scopes']
    scopes={k:[by[i] for i in ids] for k,ids in scopeids.items()};fit=scopes['fit'];reward=scopes['reward']
    sourcefiles=['scripts/run_scope_stage2.py','src/math_tokenizer/learned_scope_stage2.py','native/learned_scope_stage2.c',
                 'docs/stage2_next_protocol.md','scripts/run_stage2.py','src/math_tokenizer/signal_stage2.py']
    for f in sourcefiles:
        dest=out/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scopes':scopeids,
        'source_sha256':{s:sha(ROOT/s) for s in sourcefiles},'catalog_sha256':sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),
        'capacities':list(map(int,args.capacities.split(','))),'seeds':[17,29,43],
        'status':'adaptive internal two-capacity deployment experiment, not confirmatory',
        'prior_accounting':{'raw_byte_route':'generic primitive, pair-balance and DP priors only',
                            'math_atom_route':'one inherited mathematical command/codepoint alphabet prior plus generic primitive/pair mechanisms',
                            'training_AST_labels':False,'reward_math_metrics':False,'manual_pair_identities':False}}
    write(out/'manifest.json',manifest);audits=[provenance_audit(scopes)];write(out/'audits.json',audits)
    pairs,pairlog=discover_pairs(fit);write(out/'learned_pairs.json',pairlog)
    print('Fit-only paired bytes learned:',pairs,flush=True)
    allmetrics=[];allrows=[];oraclelog=[];nativechecks=0
    for capacity in manifest['capacities']:
        folder=out/str(capacity);modelsdir=folder/'models';modelsdir.mkdir(parents=True,exist_ok=True);models={}
        byte=HFTokenizer.train([r['latex'] for r in fit],capacity,modelsdir/'byte_bpe.json');models['byte_bpe']=byte
        atoms=derive_atoms(fit,capacity,384);atom=AtomicBPE.train(fit,atoms,capacity);atom.save(modelsdir/'math_atom_bpe.json')
        ast=BoundaryAST(atom.vocab,.25,.3);ast.save(modelsdir/'boundary_ast.json');models['math_atom_bpe']=atom;models['boundary_ast']=ast
        models['unigram']=SentencePieceTokenizer.train([r['latex'] for r in fit],capacity,'unigram',modelsdir/'unigram')
        for method,name in [('Qwen2.5-0.5B','qwen25'),('meta-llama/Meta-Llama-3-8B','llama3'),('tiktoken_o200k_base','o200k')]:models[name]=load_final(method)
        families={'raw':([token_bytes(byte,i) for i in range(byte.vocab_size)],[bytes([i]) for i in range(256)]),
                  'minimal_prior':(atom.vocab,atoms)}
        for family,(vocab,primitives) in families.items():
            plain=LearnedScopeTokenizer(vocab,primitives,[],0.);models[family+'_no_scope']=plain
            models[family+'_unprotected']=SignalTokenizer(vocab)
            reference=paper_token_rate([plain.encode(r['latex']) for r in reward],reward)
            selected,logs=select_scope(vocab,primitives,pairs,fit,reward,reference*1.01)
            models[family+'_learned_scope']=selected;write(folder/f'{family}_selection.json',logs)
            for seed in (17,29,43):
                for algorithm in ('parameter_rl','random_search','cem_search','no_update'):
                    model=optimize_deployed_costs(selected,balanced(fit,768),balanced(reward,128),seed,algorithm,args.rounds)
                    name=f'{family}_{algorithm}_{seed}';models[name]=model;oraclelog.append({'capacity':capacity,'family':family,'name':name,**model.training_log[0]})
        print('Trained deployed cost models',capacity,len(models),'systems',flush=True)
        probes=balanced(fit,32)+[{'latex':s} for s in ('','α😊',r'\Novel{a{b}}','\x00')]
        for name,model in models.items():
            if isinstance(model,LearnedScopeTokenizer):
                for row in probes:
                    ids,cost=model.encode_with_cost(row['latex']);ref,value=model.encode_reference(row['latex'])
                    assert ids==ref and abs(cost-value)<1e-8 and model.decode(ids)==row['latex'];nativechecks+=1
                path=modelsdir/f'{name}.npz';model.save(path);loaded=LearnedScopeTokenizer.load(path)
                assert all(loaded.encode(r['latex'])==model.encode(r['latex']) for r in probes)
            elif isinstance(model,SignalTokenizer):model.save(modelsdir/f'{name}.npz')
        write(folder/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
             'model_sha256':{str(p.relative_to(folder)):sha(p) for p in modelsdir.iterdir()},'selection':'reward only before evaluation'})
        for suite in ('dev','internal_eval','domain_ood'):
            rows=scopes[suite];cache={r['id']:source_scopes(r['latex']) for r in rows};times=timing(models,rows)
            for name,model in models.items():
                usage=fit_usage(model,fit);m,rr=measure(model,rows,name,repeats=1,training_usage=usage)
                extra,sr=additional(model,rows,usage,scope_cache=cache,probe_limit=128);m.update(extra);m.update(times[name])
                m.update(suite=suite,capacity=capacity,train_information='same fit; new cost learners use reward subset, no evaluation quality feedback')
                # Numeric arrays are a lower bound, not total resident memory.
                arrays=[v for v in vars(model).values() if isinstance(v,np.ndarray)]
                if hasattr(model,'primitive'):arrays += [v for v in vars(model.primitive).values() if isinstance(v,np.ndarray)]
                m['persistent_numeric_array_bytes_lower_bound']=sum(a.nbytes for a in arrays)
                if m['vocab_actual']<=2048:
                    encfit=[model.encode(r['latex']) for r in fit];enc=[model.encode(r['latex']) for r in rows]
                    code=TokenCodePredictor(encfit,m['vocab_actual']).score(enc,rows)
                    m.update({k:v for k,v in code.items() if k!='bits'})
                    for r,bits in zip(rr,code['bits']):r['token_bigram_code_bits']=bits
                m.pop('usage');scopeby={r['id']:r for r in sr}
                for r in rr:r.update(capacity=capacity,suite=suite,**{k:v for k,v in scopeby.get(r['id'],{}).items() if k!='id'})
                allmetrics.append(m);allrows.extend(rr)
            print('Evaluated',capacity,suite,len(rows),'formulas',flush=True)
    audits.append({'round':2,'passed':True,'native_reference_serialization_checks':nativechecks,
                   'scope':'generic primitive protection and exact DP; source-hashed save/load; seeded models unchanged'})
    write(out/'oracle_log.json',oraclelog);write(out/'metrics.json',allmetrics)
    with gzip.open(out/'per_formula.jsonl.gz','wt') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    grouped=defaultdict(list)
    for r in allrows:grouped[(r['capacity'],r['method'],r['suite'])].append(r)
    for m in allmetrics:
        rr=grouped[(m['capacity'],m['method'],m['suite'])];assert len(rr)==m['formulas']
        assert sum(r['tokens'] for r in rr)==m['tokens'] and all(r['roundtrip'] and r['payload_exact'] for r in rr)
        groups=defaultdict(lambda:[0,0]);tp=total=0
        for r in rr:
            groups[r['group']][0]+=r['tokens'];groups[r['group']][1]+=r['bytes']
            tp+=r.get('scope_boundaries_preserved',0);total+=r.get('scope_boundaries',0)
        assert abs(np.mean([t/b for t,b in groups.values()])-m['document_macro_tokens_per_byte'])<1e-12
        if total:assert abs(tp/total-m['independent_scope_boundary_recall'])<1e-12
    for capacity in manifest['capacities']:
        for family in families:
            for seed in (17,29,43):
                logs=[r for r in oraclelog if r['capacity']==capacity and r['family']==family and r['seed']==seed]
                assert len({r['oracle_evaluations'] for r in logs})==1 and len({r['teacher_sha256'] for r in logs})==1
        freeze=json.loads((out/str(capacity)/'freeze.json').read_text())
        assert all(sha(out/str(capacity)/p)==h for p,h in freeze['model_sha256'].items())
    assert sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')==manifest['catalog_sha256']
    audits.append({'round':3,'passed':True,'scope':'full row/paper/scope/fidelity aggregation; equal oracle budgets and teacher; frozen model/catalog hashes',
                   'result_rows':len(allrows),'metric_records':len(allmetrics)})
    write(out/'audits.json',audits);write(out/'completion.json',{'status':'exploratory completed; pending success gate analysis',
        'metric_records':len(allmetrics),'result_rows':len(allrows),'capacities':manifest['capacities']})
    print('Deployment-consistent pilot completed.',flush=True)


if __name__=='__main__':main()
