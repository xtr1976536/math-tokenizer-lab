"""Run an isolated, low-intervention stage-2 pilot with explicit controls."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,platform,time
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.final_models import load_final
from math_tokenizer.signal_stage2 import (ByteSignals,SignalTokenizer,TokenCodePredictor,
    select_strength,paper_token_rate)
from math_tokenizer.autonomous_stage2 import PolicyConfig,train_policy,evaluate_policy
from math_tokenizer.predictive_stage2 import PredictiveTokenizer,train_predictive
from math_tokenizer.learned_units_stage2 import LearnedUnitBPE,select_units,select_context_units
from math_tokenizer.measure_v6 import measure,fit_usage,clear
from math_tokenizer.independent_v7 import additional,source_scopes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def key(s):return hashlib.sha256(('stage2-pilot01:'+s).encode()).hexdigest()


def balanced(rows,limit):
    groups=defaultdict(list)
    for r in sorted(rows,key=lambda r:key(r['id'])):groups[r['split_group']].append(r)
    selected=[];depth=0;names=sorted(groups,key=key)
    while len(selected)<min(limit,len(rows)):
        for name in names:
            if depth<len(groups[name]):selected.append(groups[name][depth])
            if len(selected)>=limit:break
        depth+=1
    return selected


def make_scopes(rows,args):
    previous=json.loads((ROOT/'experiments/current/broad_scope.json').read_text())
    allowed=set(previous['fit_ids']);pool=[r for r in rows if r['id'] in allowed]
    oodgroups={r['split_group'] for r in pool if r['paper_domain']=='number_theory'}
    domain=[r for r in pool if r['split_group'] in oodgroups]
    other=[r for r in pool if r['split_group'] not in oodgroups]
    bins={}
    for r in other:
        h=int(key(r['split_group'])[:8],16)%10
        bucket='fit' if h<6 else 'reward' if h<8 else 'dev' if h==8 else 'internal_eval'
        bins.setdefault(bucket,[]).append(r)
    return {'fit':balanced(bins['fit'],args.fit),'reward':balanced(bins['reward'],args.reward),
            'dev':balanced(bins['dev'],args.dev),'internal_eval':balanced(bins['internal_eval'],args.test),
            'domain_ood':balanced(domain,args.ood)}


def provenance_audit(scopes):
    allids=[];groups={k:{r['split_group'] for r in v} for k,v in scopes.items()}
    keys=list(groups)
    for i,k in enumerate(keys):
        assert scopes[k]
        assert all(not groups[k]&groups[j] for j in keys[i+1:])
        allids.extend(r['id'] for r in scopes[k])
    assert len(allids)==len(set(allids))
    old=json.loads((ROOT/'data/benchmark/splits.json').read_text())
    public=set().union(*(set(v) for k,v in old.items() if 'test' in k))
    assert not set(allids)&public
    assert all(r['paper_domain']!='number_theory' for k in ('fit','reward','dev') for r in scopes[k])
    return {'round':1,'passed':True,'scope':'ID/source-paper isolation; old public tests excluded; number-theory source exclusion',
            'counts':{k:len(v) for k,v in scopes.items()},'papers':{k:len(v) for k,v in groups.items()},
            'freshness':'new source-disjoint internal partition of previously available training data, not fresh external evidence'}


def native_audit(models,scopes,out):
    checked=0
    probes=balanced(scopes['fit']+scopes['reward'],64)+[{'latex':s} for s in ('','\x00','α😊',r'\UnseenCommand{a{b}}')]
    for name,m in models.items():
        if isinstance(m,LearnedUnitBPE):
            path=out/'models'/f'{name}.json';m.save(path);loaded=LearnedUnitBPE.load(path)
            for r in probes:
                assert m.decode(m.encode(r['latex']))==r['latex'] and loaded.encode(r['latex'])==m.encode(r['latex']);checked+=1
        if not isinstance(m,SignalTokenizer):continue
        for r in probes:
            ids,cost=m.encode_with_cost(r['latex']);ref,value=m.encode_reference(r['latex'])
            assert ids==ref and abs(cost-value)<1e-8 and m.decode(ids)==r['latex'];checked+=1
        path=out/'models'/f'{name}.npz';m.save(path);loaded=type(m).load(path)
        assert all(loaded.encode(r['latex'])==m.encode(r['latex']) for r in probes)
    return {'round':2,'passed':True,'native_reference_and_serialization_probes':checked,
            'scope':'native/python exact path and roundtrip; source-hashed serialization; pytest covers causality and exact expected gradient'}


def timing(models,rows,repeats=7):
    names=list(models);runs=defaultdict(list);first={};nbytes=sum(len(r['latex'].encode()) for r in rows)
    # Prebuild code outside measurement; public encode includes all input handling.
    for i in range(repeats):
        order=names[i:]+names[:i]
        if i%2:order=order[::-1]
        for name in order:
            m=models[name];clear(m);start=time.perf_counter();ids=[m.encode(r['latex']) for r in rows]
            runs[name].append(time.perf_counter()-start)
            if name in first:assert ids==first[name]
            else:first[name]=ids
    return {k:{'seconds':v,'encode_mib_s':nbytes/(1024**2*np.median(v)),
               'mib_s_p10_p90':(nbytes/(1024**2*np.quantile(v,[.9,.1]))).tolist(),
               'protocol':'7 interleaved full public-encoder passes, clear upstream cache; all signal work included'} for k,v in runs.items()}


def paired_interval(left,right,seed=811,draws=5000):
    a={r['id']:r for r in left};b={r['id']:r for r in right};assert a.keys()==b.keys()
    papers=defaultdict(lambda:[0.,0.,0])
    for id,r in a.items():papers[r['group']][0]+=r['bits'];papers[r['group']][1]+=b[id]['bits'];papers[r['group']][2]+=r['bytes']
    diff=np.array([(x-y)/n for x,y,n in papers.values()]);rng=np.random.default_rng(seed)
    samples=np.mean(diff[rng.integers(0,len(diff),(draws,len(diff)))],axis=1)
    return {'left_minus_right_paper_bpb':float(diff.mean()),'paper_cluster_ci95':np.quantile(samples,[.025,.975]).tolist(),
            'papers':len(diff),'multiplicity':'exploratory unadjusted interval; not confirmatory significance'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/stage2/pilot01')
    p.add_argument('--fit',type=int,default=4096);p.add_argument('--reward',type=int,default=512)
    p.add_argument('--dev',type=int,default=512);p.add_argument('--test',type=int,default=1024);p.add_argument('--ood',type=int,default=512)
    p.add_argument('--capacity',type=int,default=1024);p.add_argument('--rounds',type=int,default=16)
    p.add_argument('--seeds',default='17,29,43');p.add_argument('--predictive',action='store_true');p.add_argument('--learned-units',action='store_true');p.add_argument('--context-units',action='store_true');args=p.parse_args();out=args.output.resolve()
    if (out/'run_manifest.json').exists():raise ValueError('Use a new output directory; do not overwrite a recorded pilot')
    if ROOT/'experiments/stage2' not in out.parents:raise ValueError('Output must be under experiments/stage2')
    out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    start=time.perf_counter();cataloghash=sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')
    scopes=make_scopes(read_catalog(),args);audits=[provenance_audit(scopes)]
    sources=[ROOT/'src/math_tokenizer/signal_stage2.py',ROOT/'src/math_tokenizer/autonomous_stage2.py',
             ROOT/'native/signal_stage2.c',Path(__file__).resolve(),ROOT/'docs/stage2_protocol.md']
    if args.predictive:sources += [ROOT/'src/math_tokenizer/predictive_stage2.py',ROOT/'native/predictive_stage2.c']
    if args.learned_units or args.context_units:sources += [ROOT/'src/math_tokenizer/learned_units_stage2.py']
    for source in sources:
        dest=out/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'args':{**vars(args),'output':str(out)},
              'catalog_sha256':cataloghash,'source_sha256':{str(s.relative_to(ROOT)):sha(s) for s in sources},
              'platform':platform.platform(),'numpy':np.__version__,'training_reads':'fit/reward only; no reference syntax/quality labels',
              'scopes':{k:[r['id'] for r in v] for k,v in scopes.items()},
              'supervision':{'human_boundary_labels':0,'command_table':False,'AST_training_labels':False,
                   'math_metric_reward':False,'human_choices':['model family and smoothing','patch cap','dictionary capacity',
                   'compute/compression guard','optimizer settings','projection strength grid','domain split']}}
    write(out/'run_manifest.json',manifest);write(out/'audits.json',audits)
    fit=scopes['fit'];reward=scopes['reward'];models={};training={}
    hf=HFTokenizer.train([r['latex'] for r in fit],args.capacity,out/'models/byte_bpe.json');models['byte_bpe']=hf
    from math_tokenizer.metrics_v2 import token_bytes
    vocab=[token_bytes(hf,i) for i in range(hf.vocab_size)]
    signals=ByteSignals.fit(fit);signals.save(out/'models/byte_signals.npz')
    zero=SignalTokenizer(vocab);models['zero_signal']=zero
    reference=paper_token_rate([zero.encode(r['latex']) for r in reward],reward)
    for index,name in enumerate(signals.names):
        m,log=select_strength(vocab,signals.features[:,index],fit,reward,reference);models[name+'_signal']=m;training[name]=log
    shuffled=signals.features[:,2].copy();seen=np.flatnonzero(signals.mass>0)
    shuffled[seen]=np.random.default_rng(911).permutation(shuffled[seen])
    m,log=select_strength(vocab,shuffled,fit,reward,reference);models['shuffled_signal']=m;training['shuffled_signal']=log
    if args.predictive:
        model=train_predictive(vocab,fit,reward,reference);models['predictive_path']=model
        training['predictive_path']=model.training_log
        print('Exact predictive-signal route trained.',flush=True)
    if args.learned_units:
        for index,name in enumerate(signals.names):
            model,logs=select_units(fit,reward,signals.features[:,index],signals.mass,args.capacity,reference)
            models[name+'_learned_units']=model;training[name+'_learned_units']=logs
    if args.context_units:
        unitmodel,logs=select_context_units(fit,reward,args.capacity,reference)
        models['context_units_bpe']=unitmodel;training['context_units']=logs
        plain=SignalTokenizer(unitmodel.vocab);models['context_units_fast']=plain
        if args.predictive:models['context_units_predictive']=train_predictive(unitmodel.vocab,fit,reward,
            paper_token_rate([plain.encode(r['latex']) for r in reward],reward))
    print('Learned signal routes and shuffled controls trained.',flush=True)
    atoms=derive_atoms(fit,args.capacity,384);atomic=AtomicBPE.train(fit,atoms,args.capacity);atomic.save(out/'models/math_atom_bpe.json')
    ast=BoundaryAST(atomic.vocab,.25,.3);ast.save(out/'models/boundary_ast.json')
    models['math_atom_bpe']=atomic;models['boundary_ast']=ast
    sp=SentencePieceTokenizer.train([r['latex'] for r in fit],args.capacity,'unigram',out/'models/unigram');models['unigram']=sp
    for method,name in [('Qwen2.5-0.5B','qwen25'),('meta-llama/Meta-Llama-3-8B','llama3'),('tiktoken_o200k_base','o200k')]:
        models[name]=load_final(method)
    config=PolicyConfig(rounds=args.rounds);policyfit=balanced(fit,768);policyreward=balanced(reward,config.reward_formulas)
    target=paper_token_rate([hf.encode(r['latex']) for r in policyfit],policyfit);target=float(np.clip(target,.05,.95))
    seeds=list(map(int,args.seeds.split(',')));policy_records=[]
    for seed in seeds:
        for algorithm in ('boundary_rl','random_search','cem_search','no_update'):
            result,predictor=train_policy(policyfit,policyreward,signals,target,seed,algorithm,config)
            name=f'{algorithm}_{seed}';np.savez_compressed(out/'models'/f'{name}_router.npz',logits=result['logits'],initial=result['initial_logits'])
            table=result['logits']-result['initial_logits'];weights=signals.mass/max(1,signals.mass.sum())
            mean=float(weights@table);scale=float(np.sqrt(weights@((table-mean)**2)))
            table=(table-mean)/max(scale,1e-8);table[signals.mass==0]=0.
            model,logs=select_strength(vocab,table,fit,reward,reference);models[name]=model
            if args.learned_units:
                unitmodel,unitlogs=select_units(fit,reward,table,signals.mass,args.capacity,reference)
                models[name+'_learned_units']=unitmodel;training[name+'_learned_units']=unitlogs
            metadata={k:v for k,v in result.items() if k not in ('logits','initial_logits')}
            metadata.update(name=name,projection_selection=logs,projection_weighted_mean=mean,projection_weighted_scale=scale)
            write(out/f'{name}_training.json',metadata);policy_records.append((metadata,result,predictor))
            print(name,'trained',round(result['training_seconds'],2),'s',flush=True)
    write(out/'signal_training.json',training)
    audits.append(native_audit(models,scopes,out));write(out/'audits.json',audits)
    freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':{str(x.relative_to(out)):sha(x) for x in sorted((out/'models').iterdir())},
            'selection_scope':'fit/reward only; dev/eval/OOD not scored before this freeze'}
    write(out/'freeze.json',freeze)
    # Raw-router probes are separate from deployed-ID scores.
    raw=[];devprobe=balanced(scopes['dev'],128)
    for metadata,result,predictor in policy_records:
        score=evaluate_policy(devprobe,result['logits'],predictor,99173+metadata['seed'],draws=4)
        raw.append({**metadata,'development':score})
    write(out/'router_results.json',raw)
    comparisons=[]
    for seed in seeds:
        rl=next(r for r in raw if r['algorithm']=='boundary_rl' and r['seed']==seed)
        for control in ('random_search','cem_search','no_update'):
            other=next(r for r in raw if r['algorithm']==control and r['seed']==seed)
            a=rl['development'];b=other['development']
            comparisons.append({'seed':seed,'control':control,'relative_byte_loss_improvement':1-a['paper_macro_causal_byte_prediction_bpb']/b['paper_macro_causal_byte_prediction_bpb'],
                                **paired_interval(a['rows'],b['rows'])})
    write(out/'router_comparisons.json',comparisons)
    suites={k:scopes[k] for k in ('dev','internal_eval','domain_ood')};summaries=[];all_rows=[]
    for suite,rows in suites.items():
        cache={r['id']:source_scopes(r['latex']) for r in rows};interleaved=timing(models,rows)
        for name,model in models.items():
            usage=fit_usage(model,fit);summary,rr=measure(model,rows,name,repeats=1,training_usage=usage)
            extra,sr=additional(model,rows,usage,scope_cache=cache,probe_limit=128);summary.update(extra);summary.update(interleaved[name])
            summary.update(suite=suite,vocab_family='learned_units' if isinstance(model,LearnedUnitBPE) else 'byte_bpe_shared' if isinstance(model,SignalTokenizer) or name=='byte_bpe' else
                'math_atom_shared' if name in ('boundary_ast','math_atom_bpe') else 'pretrained' if name in ('qwen25','llama3','o200k') else 'unigram')
            if summary['vocab_actual']<=2048:
                fitenc=[model.encode(r['latex']) for r in fit];enc=[model.encode(r['latex']) for r in rows]
                code=TokenCodePredictor(fitenc,summary['vocab_actual']).score(enc,rows)
                summary.update({k:v for k,v in code.items() if k!='bits'})
                for r,bits in zip(rr,code['bits']):r['token_bigram_code_bits']=bits
            byscope={r['id']:r for r in sr}
            for r in rr:r.update(suite=suite,**{k:v for k,v in byscope.get(r['id'],{}).items() if k!='id'})
            summary.pop('usage');summaries.append(summary);all_rows.extend(rr)
        print('Evaluated',suite,len(rows),'formulas /',len(models),'systems',flush=True)
    write(out/'metrics.json',summaries)
    with gzip.open(out/'per_formula.jsonl.gz','wt') as f:
        for r in all_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Actual independent reaggregation over every reported row and budget log.
    for m in summaries:
        rr=[r for r in all_rows if r['method']==m['method'] and r['suite']==m['suite']]
        assert len(rr)==m['formulas'] and sum(r['tokens'] for r in rr)==m['tokens'] and sum(r['bytes'] for r in rr)==m['bytes']
        assert all(r['roundtrip'] and r['payload_exact'] for r in rr)
        groups=defaultdict(lambda:[0,0]);scopeok=scopetotal=0
        for r in rr:
            groups[r['group']][0]+=r['tokens'];groups[r['group']][1]+=r['bytes']
            scopeok+=r.get('scope_boundaries_preserved',0);scopetotal+=r.get('scope_boundaries',0)
        assert abs(np.mean([t/b for t,b in groups.values()])-m['document_macro_tokens_per_byte'])<1e-12
        if scopetotal:assert abs(scopeok/scopetotal-m['independent_scope_boundary_recall'])<1e-12
    for seed in seeds:
        budgets=[r['oracle_trajectories'] for r in raw if r['seed']==seed]
        assert len(set(budgets))==1
        predictor_states=[r['predictor_summary'] for r in raw if r['seed']==seed];assert all(x==predictor_states[0] for x in predictor_states)
    assert cataloghash==sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')
    assert all(sha(out/path)==digest for path,digest in freeze['model_sha256'].items())
    audits.append({'round':3,'passed':True,'scope':'all per-formula count and paper-macro aggregates; scope recall; raw fidelity; equal oracle budgets; identical predictor; interleaved public timings; frozen hashes',
                   'result_rows':len(all_rows),'metric_records':len(summaries),'catalog_unchanged':True})
    write(out/'audits.json',audits);write(out/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
              'seconds':time.perf_counter()-start,'models':len(models),'metric_records':len(summaries),
              'status':'exploratory completed; success requires separate gate evaluation and confirmatory evidence'})
    print('Completed pilot',out,flush=True)


if __name__=='__main__':main()
