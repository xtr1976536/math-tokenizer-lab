"""Frozen-corpus optimization; no new baselines. Dev selection then test once."""
import argparse,json,time,hashlib,platform
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import asdict
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash,bootstrap_difference
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.rl_v2 import LatticeTokenizer,RLConfig
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.metrics_v2 import measure
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.vocab_v3 import VocabConfig,refine,optimize_swaps
from math_tokenizer.structure_v2 import structure
from math_tokenizer.ast import compatible

POLICIES={
 'rloo6':RLConfig(epochs=6,max_train_formulas=768,structure_weight=0),
 'rloo24':RLConfig(epochs=24,lr=.05,entropy=.001,max_train_formulas=768),
 'relative12':RLConfig(estimator='relative',epochs=12,max_train_formulas=768),
 'rloo_symbols':RLConfig(epochs=12,structure_weight=.5,max_train_formulas=768),
}


def dev_score(model,records):
    counts=defaultdict(lambda:[0,0]);cmd=totalcmd=0;cross=totaltokens=0;numtotal=numintact=0
    for r in records:
        text=r['latex'];ids=model.encode(text);raw=text.encode();group=r['split_group'];counts[group][0]+=len(ids);counts[group][1]+=len(raw)
        offsets={0};pos=0
        s=structure(text)
        for id in ids:
            end=pos+len(model.vocab[id]);offsets.add(end)
            if s.kind!='byte_fallback':cross+=not compatible(pos,end,s.spans);totaltokens+=1
            pos=end
        for a,b,t in structure(text).lexemes:
            if t.startswith('\\') and t[1:].isalpha():totalcmd+=1;cmd+=not any(a<x<b for x in offsets)
            if t[0:1].isdigit():numtotal+=1;numintact+=not any(a<x<b for x in offsets)
    return {'document_macro_tokens_per_byte':float(np.mean([a/b for a,b in counts.values()])),
            'tokens_per_formula':sum(a for a,b in counts.values())/len(records),'command_integrity':cmd/max(1,totalcmd),'number_integrity':numintact/max(1,numtotal),'tree_crossing':cross/max(1,totaltokens)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capacity',type=int,default=512);parser.add_argument('--output',required=True);a=parser.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('immutable output')
    data=ROOT/'data/releases/v2/curated';train=read_jsonl(data/'splits/train.jsonl');dev=read_jsonl(data/'splits/dev.jsonl');test=read_jsonl(data/'splits/test.jsonl')
    digest=file_hash(data/'corpus.jsonl');texts=[r['latex'] for r in train]
    dev_groups=defaultdict(list)
    for r in dev:dev_groups[r['split_group']].append(r)
    selection=[]
    for group,rows in sorted(dev_groups.items()):selection.extend(sorted(rows,key=lambda r:r['id'])[:32])
    selection=sorted(selection,key=lambda r:r['id'])[:1200]
    protocol={'stage':'v3_no_corpus_or_baseline_change','capacity':a.capacity,'corpus_sha256':digest,'train_ids':[r['id'] for r in train],'dev_ids':[r['id'] for r in dev],
        'test_ids':[r['id'] for r in test],'selection_dev_ids':[r['id'] for r in selection],
        'ast_atom_budgets':[64,96,128,160,192],'rl_atom_budgets':[64,128,192],'policy_cross_penalties':[0.,.5,1.5],'policy_configs':{k:asdict(v) for k,v in POLICIES.items()},
        'rl_selection_seeds':[17,29],'formal_seeds':[17,29,43],
        'selection_rule':'lowest dev document macro tokens/byte, command and number integrity>=old-v2, tree crossing<=old-v2; deterministic name tiebreak',
        'test_usage':'same existing test as v2, previously inspected; no claim of pristine holdout','new_baselines_run':False,
        'timing':'all selected old-new pairs retimed on same shared input using identical cold structure cache protocol',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'platform':platform.platform()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));models=out/'models';models.mkdir(exist_ok=True)
    cache_source=ROOT/f'results/v3_{a.capacity}/models'
    if cache_source.exists() and cache_source!=models:
        import shutil
        for file in cache_source.glob('*.json'):
            if file.name.startswith(('ast-atom','rl-atom')):shutil.copy2(file,models/file.name)
    ablations=[];ast_models={};byte_bases={};trained={}
    def record(model,arm,name,seed=0):
        score=dev_score(model,selection);row=dict(arm=arm,name=name,seed=seed,score=score,train_seconds=model.train_seconds)
        ablations.append(row);(out/'dev_ablations.json').write_text(json.dumps(ablations,indent=2))
        print('dev',arm,name,seed,score,flush=True);return row
    # Entire training corpus exposure shared, command seeds charged to capacity.
    for atoms in protocol['ast_atom_budgets']:
        name=f'atom{atoms}';cached=out/'models'/f'ast-{name}.json'
        if cached.exists():model=NativeTokenizer.load(cached)
        else:
            base=AtomicASTBPE(True).train(texts,a.capacity,command_budget=atoms);model=base.decoder()
        model.save(models/f'ast-{name}.json');ast_models[name]=model;record(model,'ast',name)
    for atoms in protocol['rl_atom_budgets']:
        base=AtomicASTBPE(False).train(texts,a.capacity,command_budget=atoms);byte_bases[atoms]=base
        model=base.decoder(0.);name=f'atom{atoms}-shortest';model.save(models/f'byte-{name}.json');record(model,'byte_nonrl',name)
        for seed in (17,29):
            for policy,config in POLICIES.items():
                cached=models/f'rl-atom{atoms}-{policy}-{seed}.json'
                if cached.exists():model=NativeTokenizer.load(cached)
                else:
                    old=LatticeTokenizer(base,False,'policy_map',config).train_policy(texts,seed)
                    model=NativeTokenizer.from_v2(old);model.capacity=a.capacity;model.training_log=old.training_log
                name=f'atom{atoms}-{policy}';model.save(models/f'rl-{name}-{seed}.json');trained[(atoms,policy,seed)]=model;record(model,'rl_byte',name+'-cross0.0',seed)
                for penalty in (.5,1.5):
                    soft=NativeTokenizer(model.vocab,False,'policy_structure',model.theta,model.frequency,crossfee=penalty)
                    soft.train_seconds=model.train_seconds;soft.training_log=model.training_log
                    record(soft,'rl_byte',name+f'-cross{penalty}',seed)
    old_ast=FastBPE.load(ROOT/f'results/v2_release{a.capacity}/models/base-ast.json')
    old_rl=LatticeTokenizer.load(ROOT/f'results/v2_release{a.capacity}/models/rl_byte_selected-17.json')
    old_astscore=dev_score(old_ast,selection);old_rlscore=dev_score(NativeTokenizer.from_v2(old_rl),selection)
    ast_options={name:r['score'] for r in ablations if r['arm']=='ast' for name in [r['name']] if r['score']['command_integrity']>=old_astscore['command_integrity'] and r['score']['number_integrity']>=old_astscore['number_integrity']-1e-12}
    selected_ast=min(ast_options,key=lambda name:(ast_options[name]['document_macro_tokens_per_byte'],name))
    rl_options=defaultdict(list)
    for r in ablations:
        if r['arm']=='rl_byte':rl_options[r['name']].append(r['score'])
    eligible={name:float(np.mean([r['document_macro_tokens_per_byte'] for r in scores])) for name,scores in rl_options.items()
              if np.mean([r['command_integrity'] for r in scores])>=old_rlscore['command_integrity']
              and np.mean([r['tree_crossing'] for r in scores])<=old_rlscore['tree_crossing']+1e-12
              and np.mean([r['number_integrity'] for r in scores])>=old_rlscore['number_integrity']-1e-12}
    selected_rl=min(eligible,key=lambda name:(eligible[name],name))
    selected={'ast':selected_ast,'rl_byte':selected_rl,'scores':eligible,'old_dev_scores':{'ast':old_astscore,'rl':old_rlscore},'test_used_for_selection':False}
    (out/'selection.json').write_text(json.dumps(selected,indent=2));print('selected',selected_ast,selected_rl,flush=True)
    # RL+AST trains selected atom basis, preserve sampled policy and no-RL control.
    atom_ast=int(selected_ast.removeprefix('atom'));astbase=AtomicASTBPE(True).train(texts,a.capacity,command_budget=atom_ast)
    summaries=[];rows_all=[];by={};paired={}
    def evaluate(model,name,seed=0):
        freq=Counter()
        for t in texts:freq.update(model.encode(t))
        summary,rows=measure(model,test,name,seed,repeats=5,train_frequency=freq)
        summary['capacity_budget']=a.capacity;summary['comparison_group']='same_corpus_trained';summaries.append(summary);rows_all.extend(rows);by[(name,seed)]=rows
        (out/'metrics.json').write_text(json.dumps(summaries,indent=2));(out/'per_formula.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows_all))
        print('test',name,seed,round(summary['tokens_per_formula'],4),'cmd',round(summary['command_integrity_rate'],4),'MiBs',round(summary['encode_mib_s'],3),flush=True)
    # Only our existing methods are retested; mature baseline checkpoints unused.
    evaluate(old_ast,'ast_bpe_v2')
    evaluate(ast_models[selected_ast],'ast_bpe_v3')
    atoms=int(selected_rl.split('-')[0][4:]);policy_and_penalty=selected_rl.split('-',1)[1];policy,penaltytext=policy_and_penalty.rsplit('-cross',1);penalty=float(penaltytext)
    bytebase=byte_bases[atoms];shortest=bytebase.decoder(0.);shortest.save(models/'new_byte_no_rl_shortest.json');evaluate(shortest,'atomic_byte_shortest_control')
    for seed in (17,29,43):
        prior=LatticeTokenizer.load(ROOT/f'results/v2_release{a.capacity}/models/rl_byte_selected-{seed}.json');evaluate(prior,'rl_bpe_v2',seed)
        exact=NativeTokenizer.from_v2(prior);exact.save(models/f'old_rl_native-{seed}.json');evaluate(exact,'rl_bpe_v2_native_exact',seed)
        candidate=trained.get((atoms,policy,seed))
        if candidate is None:
            old=LatticeTokenizer(bytebase,False,'policy_map',POLICIES[policy]).train_policy(texts,seed);candidate=NativeTokenizer.from_v2(old);candidate.training_log=old.training_log
        if penalty:
            core=candidate
            candidate=NativeTokenizer(core.vocab,False,'policy_structure',core.theta,core.frequency,crossfee=penalty)
            candidate.train_seconds=core.train_seconds;candidate.training_log=core.training_log
        candidate.save(models/f'rl_bpe_v3-{seed}.json');evaluate(candidate,'rl_bpe_v3',seed)
        priorast=LatticeTokenizer.load(ROOT/f'results/v2_release{a.capacity}/models/rl_ast_selected-{seed}.json');evaluate(priorast,'rl_ast_bpe_v2',seed)
        newast=LatticeTokenizer(astbase,True,'policy_map',POLICIES[policy]).train_policy(texts,seed);compiled=NativeTokenizer.from_v2(newast);compiled.training_log=newast.training_log;compiled.save(models/f'rl_ast_bpe_v3-{seed}.json');evaluate(compiled,'rl_ast_bpe_v3',seed)
        for current,previous in [('rl_bpe_v3','rl_bpe_v2'),('rl_ast_bpe_v3','rl_ast_bpe_v2')]:
            paired[f'{current}-vs-{previous}/{seed}']=bootstrap_difference(by[(current,seed)],by[(previous,seed)])
    paired['ast_bpe_v3-vs-v2']=bootstrap_difference(by[('ast_bpe_v3',0)],by[('ast_bpe_v2',0)])
    (out/'pairwise.json').write_text(json.dumps(paired,indent=2))
    assert file_hash(data/'corpus.jsonl')==digest
    (out/'manifest.json').write_text(json.dumps({'corpus_sha256':digest,'protocol_sha256':file_hash(out/'protocol.json'),'selection_sha256':file_hash(out/'selection.json'),
        'models':{p.name:file_hash(p) for p in models.glob('*.json')},'method_seed_runs':len(summaries),'records':len(rows_all)},indent=2))
    from math_tokenizer.report_v3 import report
    report(out)
if __name__=='__main__':main()
