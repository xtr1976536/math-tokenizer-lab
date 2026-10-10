import argparse,hashlib,json,os,platform,random,time
from pathlib import Path
from tokenizers import Tokenizer
from .bpe import MathBPE
from .baselines import ByteTokenizer,HFTokenizer,SentencePieceTokenizer,TikToken
from .evaluate import measure,read_jsonl,file_hash,bootstrap_difference
from .corpus import ROOT,build

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--output',default='results/pilot');p.add_argument('--capacity',type=int,default=384);p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);p.add_argument('--limit-train',type=int,default=0);p.add_argument('--limit-test',type=int,default=0);a=p.parse_args()
    if a.build:print(build())
    out=ROOT/a.output
    # Refuse accidental overwrite or mixing results from different protocols.
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('choose a new output directory; completed runs are immutable')
    train=read_jsonl(ROOT/'data/curated/splits/train.jsonl');test=read_jsonl(ROOT/'data/curated/splits/test.jsonl')
    if a.limit_train:train=train[:a.limit_train]
    if a.limit_test:test=test[:a.limit_test]
    texts=[r['latex'] for r in train]
    manifest={'protocol':'mathlab-v1','capacity':a.capacity,'seeds':a.seeds,'train_ids':[r['id'] for r in train],'test_ids':[r['id'] for r in test],
        'corpus_sha256':file_hash(ROOT/'data/curated/corpus.jsonl'),'train_sha256':file_hash(ROOT/'data/curated/splits/train.jsonl'),'test_sha256':file_hash(ROOT/'data/curated/splits/test.jsonl'),
        'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
        'train_count':len(train),'test_count':len(test),'encode_repeats':5,'special_tokens':False,'truncation':False,'normalization':'none at evaluator',
        'rl_reward':'nonoverlapping token reduction on internal 20% training holdout; REINFORCE contextual merge policy; gamma=1',
        'rl_episodes_per_merge':48,'rl_candidates':24,'rl_lr':0.08,'bootstrap_seed':20261010,
        'fairness':'custom methods share capacity and external train/test; RL reserves internal holdout, reference BPE includes all training; no equal-compute claim',
        'notes':'capacity is total budget; actual vocab differs for upstream SentencePiece byte/reserved/character overhead; fixed tokenizers are contextual references only',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    modeldir=out/'models';modeldir.mkdir(exist_ok=True)
    summaries=[];allrows=[];pairwise={}
    def evaluate(model,name,seed,group):
        summary,rows=measure(model,test,name,seed)
        summary['comparison_group']=group;summary['capacity_budget']=a.capacity if group=='trained' else None
        summaries.append(summary);allrows.extend(rows)
        print(name,seed,'tokens',summary['tokens'],'roundtrip',summary['roundtrip_rate'],'time',round(summary['encode_seconds_median'],4),flush=True)
        return rows
    for seed in a.seeds:
        seedrows={}
        for name,structural,rl in [('byte_bpe',False,False),('ast_bpe',True,False),('rl_bpe',False,True),('rl_ast_bpe',True,True)]:
            model=MathBPE(structural,rl).train(texts,a.capacity,seed)
            model.save(modeldir/f'{name}-{seed}.json')
            seedrows[name]=evaluate(model,name,seed,'trained')
        # RL control: frequency BPE given EXACT same fit subset as RL.
        import numpy as np
        indices=np.arange(len(texts));np.random.default_rng(seed).shuffle(indices)
        reward_idx=set(indices[:max(1,len(texts)//5)])
        fit=[t for i,t in enumerate(texts) if i not in reward_idx]
        for name,structural in [('byte_bpe_rl_fit_control',False),('ast_bpe_rl_fit_control',True)]:
            model=MathBPE(structural,False).train(fit,a.capacity,seed);model.save(modeldir/f'{name}-{seed}.json')
            seedrows[name]=evaluate(model,name,seed,'trained')
        for name in ('ast_bpe','rl_bpe','rl_ast_bpe'):
            pairwise[f'{name}-vs-byte_bpe/seed-{seed}']=bootstrap_difference(seedrows[name],seedrows['byte_bpe'])
        pairwise[f'rl_bpe-vs-fit-control/seed-{seed}']=bootstrap_difference(seedrows['rl_bpe'],seedrows['byte_bpe_rl_fit_control'])
        pairwise[f'rl_ast_bpe-vs-fit-control/seed-{seed}']=bootstrap_difference(seedrows['rl_ast_bpe'],seedrows['ast_bpe_rl_fit_control'])
    evaluate(ByteTokenizer(),'utf8_bytes',0,'reference')
    model=HFTokenizer.train(texts,a.capacity,modeldir/'hf-bpe.json');evaluate(model,'hf_byte_bpe',0,'trained')
    for kind in ('bpe','unigram'):
        model=SentencePieceTokenizer.train(texts,a.capacity,kind,modeldir/f'sentencepiece-{kind}')
        evaluate(model,'sentencepiece_'+kind,0,'trained')
    for encoding in ('gpt2','cl100k_base','o200k_base'):
        evaluate(TikToken(encoding),'tiktoken_'+encoding,0,'pretrained_reference')
    registry=json.loads((ROOT/'data/baseline_registry.json').read_text())
    path=ROOT/registry['qwen']['local']
    assert file_hash(path)==registry['qwen']['sha256']
    evaluate(HFTokenizer(Tokenizer.from_file(str(path))),'qwen2.5_official',0,'pretrained_reference')
    manifest['baselines']=registry
    manifest['model_hashes']={path.name:file_hash(path) for path in modeldir.iterdir() if path.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'metrics.json').write_text(json.dumps(summaries,indent=2))
    (out/'per_formula.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
    (out/'pairwise.json').write_text(json.dumps(pairwise,indent=2))
    from .report import report
    report(out)
if __name__=='__main__':main()
