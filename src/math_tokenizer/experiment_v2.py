"""Frozen v2 protocol: dev-only RL ablations, test unlocked after selection."""
import argparse,hashlib,json,platform,time
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter,defaultdict
import numpy as np
from tokenizers import Tokenizer
from .corpus import ROOT
from .evaluate import read_jsonl,file_hash,bootstrap_difference
from .metrics_v2 import measure
from .merge_v2 import FastBPE
from .bpe import MathBPE
from .legacy_fast import LegacyFastBPE
from .rl_v2 import LatticeTokenizer,RLConfig
from .baselines import HFTokenizer,SentencePieceTokenizer,TikToken,ByteTokenizer

CONFIGS={
 'naive':RLConfig(estimator='naive'),
 'relative':RLConfig(estimator='relative'),
 'rloo':RLConfig(estimator='rloo'),
 'rloo_discount':RLConfig(estimator='rloo',gamma=.99),
 'rloo_entropy':RLConfig(estimator='rloo',entropy=.02),
 'rloo_more_samples':RLConfig(estimator='rloo',samples=8),
 'rloo_zero_prior':RLConfig(estimator='rloo',initial_length_prior=0),
 'rloo_longer':RLConfig(estimator='rloo',epochs=12),
 'rloo_more_updates':RLConfig(estimator='rloo',epochs=24,lr=.05,entropy=.001),
 'rloo_soft_structure':RLConfig(estimator='rloo',structure_weight=.2),
}

def macro_score(model,records):
    # Dev document-balanced tokens per byte; no test labels or results.
    sums=defaultdict(lambda:[0,0])
    for r in records:
        sums[r['split_group']][0]+=len(model.encode(r['latex']));sums[r['split_group']][1]+=len(r['latex'].encode())
    return float(np.mean([a/b for a,b in sums.values()]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/v2_512');p.add_argument('--capacity',type=int,default=512);p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);p.add_argument('--limit-train',type=int,default=0);p.add_argument('--limit-eval',type=int,default=0);a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('immutable run: choose new output')
    data=ROOT/'data/releases/v2/curated'
    train=read_jsonl(data/'splits/train.jsonl');dev=read_jsonl(data/'splits/dev.jsonl');test=read_jsonl(data/'splits/test.jsonl')
    if a.limit_train:train=train[:a.limit_train]
    if a.limit_eval:dev=dev[:a.limit_eval];test=test[:a.limit_eval]
    # Paper-balanced dev selection sample fixed by hashes, max24/doc and <=768.
    groups=defaultdict(list)
    for r in dev:groups[r['split_group']].append(r)
    selection=[]
    for group,rs in sorted(groups.items()):selection.extend(sorted(rs,key=lambda r:r['id'])[:24])
    selection=sorted(selection,key=lambda r:r['id'])[:768]
    protocol={'version':'mathlab-v2','capacity':a.capacity,'seeds':a.seeds,'train_ids':[r['id'] for r in train],'dev_ids':[r['id'] for r in dev],'test_ids':[r['id'] for r in test],
        'dev_selection_ids':[r['id'] for r in selection],'selection_metric':'document_macro_tokens_per_byte','selection_seeds':[17,29],
        'configs':{k:asdict(v) for k,v in CONFIGS.items()},'data_sha256':file_hash(data/'corpus.jsonl'),
        'raw_string':True,'truncation':False,'special_tokens':False,'encode_repeats':3,'structure_cache':'cleared each encode repeat',
        'vocab_exposure':'all trained methods see identical external train formulas; RL segmentation shares fitted BPE vocabulary and uses hash-selected <=512 train formulas for policy updates',
        'pretrained_references':'same raw test metrics; different vocab/data budgets; separate group',
        'selection_rule':'lowest mean dev document-macro tokens/byte across seeds17,29, separately byte/AST; deterministic config-name tie-break',
        'python':platform.python_version(),'platform':platform.platform(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    models=out/'models';models.mkdir(exist_ok=True)
    print('train',len(train),'dev',len(dev),'test',len(test),'selection',len(selection),'capacity',a.capacity,flush=True)
    texts=[r['latex'] for r in train]
    bases={False:FastBPE().train(texts,a.capacity),True:FastBPE(True).train(texts,a.capacity)}
    for st,base in bases.items():base.save(models/f'base-{"ast" if st else "byte"}.json');print('base trained',st,base.train_seconds,flush=True)
    ablations=[];trained={};selection_rows=[]
    for st in (False,True):
        for seed in (17,29):
            for name,config in CONFIGS.items():
                if st and config.structure_weight:continue # hard AST has separate constraint, avoid redundant soft penalty
                model=LatticeTokenizer(bases[st],st,'policy_map',config).train_policy(texts,seed)
                path=models/f'rl-dev-{"ast" if st else "byte"}-{name}-{seed}.json';model.save(path)
                score=macro_score(model,selection)
                row={'arm':'ast' if st else 'byte','config':name,'seed':seed,'dev_macro_tokens_per_byte':score,'train_seconds':model.train_seconds,
                     'final_gradient_variance':model.training_log[-1]['gradient_variance'],'training_log':model.training_log,'checkpoint':path.name}
                ablations.append(row);trained[(st,name,seed)]=model
                (out/'dev_ablations.json').write_text(json.dumps(ablations,indent=2))
                print('dev',row['arm'],name,seed,round(score,6),'sec',round(model.train_seconds,2),flush=True)
    selected={}
    for st in (False,True):
        options={}
        for row in ablations:
            if row['arm']==('ast' if st else 'byte'):options.setdefault(row['config'],[]).append(row['dev_macro_tokens_per_byte'])
        winner=min(options,key=lambda key:(np.mean(options[key]),key));selected['ast' if st else 'byte']=winner
    # Freeze selection before calling common test evaluator for any model.
    selection_record={'selected':selected,'dev_config_mean_scores':{arm:{name:float(np.mean([r['dev_macro_tokens_per_byte'] for r in ablations if r['arm']==arm and r['config']==name])) for name in CONFIGS if any(r['arm']==arm and r['config']==name for r in ablations)} for arm in ('byte','ast')},
                      'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_used_for_selection':False}
    (out/'selection.json').write_text(json.dumps(selection_record,indent=2))
    print('selected',selected,flush=True)
    summaries=[];allrows=[];paired={};testrows={}
    def evaluate(model,name,seed,group='trained'):
        freq=Counter()
        for text in texts:freq.update(model.encode(text))
        summary,rows=measure(model,test,name,seed,train_frequency=freq)
        summary['comparison_group']=group;summary['capacity_budget']=a.capacity if group=='trained' else None
        summaries.append(summary);allrows.extend(rows);testrows[(name,seed)]=rows
        (out/'metrics.json').write_text(json.dumps(summaries,indent=2))
        (out/'per_formula.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
        print('test',name,seed,'tpf',round(summary['tokens_per_formula'],4),'lossless',summary['roundtrip_rate'],flush=True)
    for st in (False,True):
        suffix='ast' if st else 'byte';base=bases[st]
        evaluate(base,'ast_bpe_v2' if st else 'byte_bpe',0)
        shortest=LatticeTokenizer(base,st,'shortest');shortest.save(models/f'{suffix}-shortest.json');evaluate(shortest,suffix+'_shortest_path',0)
        prior=LatticeTokenizer(base,st,'policy_map',CONFIGS[selected[suffix]]);prior.save(models/f'{suffix}-no_rl-prior.json');evaluate(prior,suffix+'_policy_no_rl',0)
        for seed in a.seeds:
            name=selected[suffix]
            model=trained.get((st,name,seed))
            if model is None:model=LatticeTokenizer(base,st,'policy_map',CONFIGS[name]).train_policy(texts,seed)
            model.save(models/f'rl_{suffix}_selected-{seed}.json')
            evaluate(model,'rl_ast_bpe_v2' if st else 'rl_bpe_v2',seed)
    # Old methods comparison on same new test data; keep v1 implementation.
    for name,st in [('ast_bpe_v1',True),('rl_bpe_v1',False),('rl_ast_bpe_v1',True)]:
        m=LegacyFastBPE(st,'rl_' in name).train(texts,a.capacity,17);m.save(models/f'{name}-17.json');evaluate(m,name,17)
    hf=HFTokenizer.train(texts,a.capacity,models/'hf-bpe.json');evaluate(hf,'hf_byte_bpe',0)
    for kind in ('bpe','unigram'):
        m=SentencePieceTokenizer.train(texts,a.capacity,kind,models/f'sentencepiece-{kind}');evaluate(m,'sentencepiece_'+kind,0)
    evaluate(ByteTokenizer(),'utf8_bytes',0,'reference')
    for name in ('gpt2','cl100k_base','o200k_base'):evaluate(TikToken(name),'tiktoken_'+name,0,'pretrained_reference')
    registry=json.loads((ROOT/'data/baseline_registry.json').read_text());path=ROOT/registry['qwen']['local'];assert file_hash(path)==registry['qwen']['sha256']
    evaluate(HFTokenizer(Tokenizer.from_file(str(path))),'qwen2.5_official',0,'pretrained_reference')
    for seed in a.seeds:
        for name,base in [('rl_bpe_v2','byte_bpe'),('rl_ast_bpe_v2','ast_bpe_v2')]:
            paired[f'{name}-vs-{base}/{seed}']=bootstrap_difference(testrows[(name,seed)],testrows[(base,0)])
    (out/'pairwise.json').write_text(json.dumps(paired,indent=2))
    (out/'manifest.json').write_text(json.dumps({'protocol_sha256':file_hash(out/'protocol.json'),'selection_sha256':file_hash(out/'selection.json'),
        'models':{p.name:file_hash(p) for p in models.iterdir() if p.is_file()},'test_records':len(allrows),'method_seed_runs':len(summaries)},indent=2))
    from .report_v2 import report
    report(out)
if __name__=='__main__':main()
