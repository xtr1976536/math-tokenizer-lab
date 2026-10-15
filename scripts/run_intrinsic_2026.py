"""Run the nine-model, eleven-protocol mathematical tokenizer experiment."""
import argparse,csv,gzip,hashlib,json,os,platform,random,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.intrinsic_2026 import METHODS,LABELS,METRICS,protocols,per_formula,summarize,paper_bootstrap
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from math_tokenizer.final_models import load_final
from math_tokenizer.measure_v6 import clear
from math_tokenizer.metrics_v2 import token_bytes

OUT=ROOT/'experiments/intrinsic_20261015'
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def train_models(train,folder):
    texts=[r['latex'] for r in train];models={}
    models['hard_ast_bpe']=HardASTBPE.train(texts,1024);models['hard_ast_bpe'].save(folder/'hard_ast_bpe.json')
    models['ast_v4']=NativeASTV4.train(texts,1024,atom_budget=256,subtree_budget=0);models['ast_v4'].save(folder/'ast_v4.json')
    atoms=derive_atoms(train,1024,384);models['math_atom_bpe']=AtomicBPE.train(train,atoms,1024);models['math_atom_bpe'].save(folder/'math_atom_bpe.json')
    models['boundary_ast']=BoundaryAST(models['math_atom_bpe'].vocab,.25,.30);models['boundary_ast'].train_seconds=models['math_atom_bpe'].train_seconds
    models['boundary_ast'].training_log=[{'vocabulary':'shared math_atom_bpe trained on train IDs only','tree_penalty':.25,'boundary_penalty':.30}]
    models['boundary_ast'].save(folder/'boundary_ast.json')
    models['byte_bpe']=HFTokenizer.train(texts,1024,folder/'byte_bpe.json')
    models['sp_unigram']=SentencePieceTokenizer.train(texts,1024,'unigram',folder/'sp_unigram')
    models.update(qwen25=load_final('Qwen2.5-0.5B'),llama3=load_final('meta-llama/Meta-Llama-3-8B'),o200k=load_final('tiktoken_o200k_base'))
    return models

def audit_splits(p):
    sets=[{r['id'] for r in p[k]} for k in ['train','dev','test']]
    groups=[{r['split_group'] for r in p[k]} for k in ['train','dev','test']]
    templates=[{r['template_sha256'] for r in p[k]} for k in ['train','dev','test']]
    rendered=[{r['reference']['mathml_sha256'] for r in p[k]} for k in ['train','dev','test']]
    for i,j in [(0,1),(0,2),(1,2)]:assert not sets[i]&sets[j] and not groups[i]&groups[j] and not templates[i]&templates[j] and not rendered[i]&rendered[j]
    if p['family']=='DomainOOD':assert not {r['paper_domain'] for r in p['train']}&{r['paper_domain'] for r in p['test']}
    if p['family']=='ExtremeOOD':assert not set(p['source_domains'])&set(p['target_domains'])
    if p['family']=='Complexity':assert all(r['complexity']['class']=='simple' for r in p['train']+p['dev']) and all(r['complexity']['class']=='complex' for r in p['test'])
    return {'round':1,'name':p['name'],'passed':True,'scope':'ID/source-paper/template/rendered separation;domain/complexity predicates','counts':{k:len(p[k]) for k in ['train','dev','test']}}

def test_models(models,p):
    ast=models['hard_ast_bpe'];sample=p['train'][:35]+p['dev'][:35]+p['test'][:70]
    for r in sample:assert ast.encode(r['latex'])==ast.encode_reference(r['latex'])
    assert models['boundary_ast'].vocab==models['math_atom_bpe'].vocab
    for name in METHODS[:6]:
        m=models[name];v=len(m.vocab) if hasattr(m,'vocab') else m.vocab_size;assert v==1024,(name,v)
    for m in models.values():
        for r in sample:
            ids=m.encode(r['latex']);assert m.decode(ids)==r['latex'] and ids==m.encode(r['latex'])
    return {'round':2,'name':p['name'],'passed':True,'scope':'hard native/reference identity;1024 total IDs;shared soft/BPE dictionary;determinism and roundtrip','reference_formulas':len(sample)}

def main():
    a=argparse.ArgumentParser();a.add_argument('--only');a.add_argument('--timing-repeats',type=int,default=7);args=a.parse_args()
    rows=read_catalog();ps=protocols(rows);OUT.mkdir(exist_ok=True)
    source_paths=[ROOT/'scripts/run_intrinsic_2026.py',ROOT/'src/math_tokenizer/intrinsic_2026.py',ROOT/'src/math_tokenizer/hard_ast_2026.py',ROOT/'native/hard_ranked_2026.c']
    manifest={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'status':'protocol fixed before test scoring',
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_paths},'catalog_sha256':digest(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),
        'methods':dict(zip(METHODS,LABELS)),'capacity_trainable':1024,'fixed_boundary_weights':[.25,.30],
        'metrics':METRICS,'new_metrics':{'ast_align':'TokEval double-end alignment adapted to nonlexical nonroot KaTeX spans','renyi':'Zouhar et al. alpha2.5 divided by log2 declared ID space'},
        'dev_role':'source-disjoint diagnostics;all parameters fixed globally before scoring;dev not used to construct vocabulary',
        'pretrained_role':'unchanged official full vocabulary;no retraining','timing':'single process public encode,all nine methods interleaved each pass;cache cleared;no metrics in timer;median of7',
        'environment':{'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'logical_cpus':os.cpu_count()},
        'splits':[{**{k:v for k,v in p.items() if k not in ['train','dev','test']},**{k:[r['id'] for r in p[k]] for k in ['train','dev','test']}} for p in ps]}
    frozen=OUT/'protocol.json'
    if frozen.exists():
        old=json.loads(frozen.read_text());assert old['source_sha256']==manifest['source_sha256'] and old['splits']==manifest['splits']
    else:write(frozen,manifest)
    all_summaries=[];all_audits=[]
    for p in ps:
        if args.only and p['name'] not in args.only.split(','):continue
        folder=OUT/p['name'];folder.mkdir(exist_ok=True);summary_path=folder/'summary.json'
        if summary_path.exists():print('Already complete',p['name'],flush=True);continue
        audits=[audit_splits(p)];models_dir=folder/'models';models_dir.mkdir(exist_ok=True)
        print('Training',p['name'],len(p['train']),len(p['dev']),len(p['test']),flush=True);started=time.perf_counter()
        models=train_models(p['train'],models_dir);audits.append(test_models(models,p));print('Models ready',p['name'],round(time.perf_counter()-started,2),flush=True)
        # Development remains separate from fitting; every system has the same diagnostic rows.
        dev_sample=sorted(p['dev'],key=lambda r:r['id'])[:256]
        dev_results={name:sum(len(models[name].encode(r['latex'])) for r in dev_sample)/sum(len(r['latex'].encode()) for r in dev_sample) for name in METHODS}
        write(folder/'dev_diagnostics.json',{'scope':'fixed first256 development IDs,diagnostic only','ids':[r['id'] for r in dev_sample],'tb_micro':dev_results})
        timing=defaultdict(list);reference={};rng=random.Random(2026);totalbytes=sum(len(r['latex'].encode()) for r in p['test'])
        for m in models.values():m.encode(r'x+1') # compiled code and imports initialized before timer
        for repeat in range(args.timing_repeats):
            order=METHODS.copy();rng.shuffle(order)
            for name in order:
                clear(models[name]);t=time.perf_counter();ids=[models[name].encode(r['latex']) for r in p['test']];duration=time.perf_counter()-t;timing[name].append(duration)
                if name not in reference:reference[name]=ids
                else:assert ids==reference[name]
            print('Timing',p['name'],repeat+1,flush=True)
        summaries=[];fullrows=[]
        for name in METHODS:
            m=models[name];rr=[per_formula(m,r,ids) for r,ids in zip(p['test'],reference[name])];speed=totalbytes/1048576/np.median(timing[name])
            vocab=len(m.vocab) if hasattr(m,'vocab') else m.vocab_size;s=summarize(rr,vocab,float(speed))
            s.update(protocol=p['name'],family=p['family'],method=name,label=LABELS[METHODS.index(name)],seed=p['seed'],timing_seconds=timing[name],
                train_seconds=getattr(m,'train_seconds',0),paper_bootstrap_95=paper_bootstrap(rr))
            for r in rr:r.update(method=name,protocol=p['name'])
            summaries.append(s);fullrows.extend(rr);print('Scored',p['name'],name,round(s['tb'],4),round(s['speed'],2),flush=True)
        audits.append({'round':3,'name':p['name'],'passed':True,'scope':'all test formulas exact roundtrip/payload/ID range;identical IDs;seven deterministic interleaved encodes','rows':len(fullrows)})
        # Independent direct-formula aggregation checks have a separate implementation.
        for s in summaries:
            rr=[r for r in fullrows if r['method']==s['method']];token_total=sum(len(r['ids']) for r in rr)
            assert token_total==s['tokens'] and sum(r['bytes'] for r in rr)==s['bytes']
            paper={g:sum(r['tokens'] for r in rr if r['group']==g)/sum(r['bytes'] for r in rr if r['group']==g) for g in {r['group'] for r in rr}}
            assert abs(sum(paper.values())/len(paper)-s['tb'])<1e-12
            assert abs(sum(r['crossings'] for r in rr)/token_total-s['cross'])<1e-12
            assert len(rr)==len(p['test']) and {r['id'] for r in rr}=={r['id'] for r in p['test']}
        audits.append({'round':4,'name':p['name'],'passed':True,'scope':'independent aggregate reconciliation;common test IDs;paper bootstrap intervals;timing envelope'})
        write(summary_path,summaries);write(folder/'audit.json',audits)
        with gzip.open(folder/'per_formula.jsonl.gz','wt') as f:
            for r in fullrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        token_bytes.cache_clear();del models,reference,fullrows;__import__('gc').collect()
        print('PROTOCOL COMPLETE',p['name'],round(time.perf_counter()-started,2),flush=True)
    for p in ps:
        folder=OUT/p['name']
        if (folder/'summary.json').exists():all_summaries.extend(json.loads((folder/'summary.json').read_text()));all_audits.extend(json.loads((folder/'audit.json').read_text()))
    write(OUT/'metrics.json',all_summaries);write(OUT/'audits.json',all_audits)
    keys=['protocol','family','method','label','seed','vocab','formulas','papers',*METRICS,'tokens','bytes','roundtrip','train_seconds']
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(all_summaries)
    assert digest(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')==manifest['catalog_sha256']
    print('ALL AVAILABLE COMPLETE',len(all_summaries),flush=True)

if __name__=='__main__':main()
