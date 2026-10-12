"""Sequential, cold-structure-cache timing replay, preserving original timings."""
import json,sys,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import sentencepiece as spm
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.structure_v2 import structure
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.bpe import MathBPE
from math_tokenizer.rl_v2 import LatticeTokenizer
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer,TikToken,ByteTokenizer

def main(out):
    manifest=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
    ids=set(manifest['test_ids']);records=[r for r in read_jsonl(ROOT/'data/releases/v2/curated/corpus.jsonl') if r['id'] in ids];models=out/'models';evidence=[]
    for m in metrics:
        name=m['method'];seed=m['seed']
        if name in ('byte_bpe','ast_bpe_v2'):model=FastBPE.load(models/f'base-{"byte" if name=="byte_bpe" else "ast"}.json')
        elif name in ('byte_shortest_path','ast_shortest_path'):model=LatticeTokenizer.load(models/f'{name.split("_")[0]}-shortest.json')
        elif name.endswith('_policy_no_rl'):model=LatticeTokenizer.load(models/f'{name.split("_")[0]}-no_rl-prior.json')
        elif name in ('rl_bpe_v2','rl_ast_bpe_v2'):model=LatticeTokenizer.load(models/f'rl_{"byte" if name=="rl_bpe_v2" else "ast"}_selected-{seed}.json')
        elif name in ('ast_bpe_v1','rl_bpe_v1','rl_ast_bpe_v1'):model=MathBPE.load(models/f'{name}-{seed}.json')
        elif name=='hf_byte_bpe':model=HFTokenizer(Tokenizer.from_file(str(models/'hf-bpe.json')))
        elif name.startswith('sentencepiece_'):model=SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(models/f'sentencepiece-{name.split("_")[1]}.model')))
        elif name=='utf8_bytes':model=ByteTokenizer()
        elif name.startswith('tiktoken_'):model=TikToken(name[len('tiktoken_'):])
        else:reg=json.loads((ROOT/'data/baseline_registry.json').read_text());model=HFTokenizer(Tokenizer.from_file(str(ROOT/reg['qwen']['local'])))
        timings=[];total=None
        for _ in range(3):
            structure.cache_clear();start=time.perf_counter();current=[model.encode(r['latex']) for r in records];timings.append(time.perf_counter()-start)
            n=sum(map(len,current));assert n==m['tokens']
            if total is None:total=current
            else:assert total==current
        m['original_encode_seconds_repeats']=m['encode_seconds_repeats'];m['encode_seconds_repeats']=timings;m['encode_seconds_median']=float(np.median(timings));m['encode_mib_s']=m['bytes']/(1024**2*np.median(timings));m['timing_replay']='sequential isolated replay after all training completed'
        evidence.append({'method':name,'seed':seed,'tokens_unchanged':True,'seconds':timings})
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'timing_replay.json').write_text(json.dumps(evidence,indent=2))
    from math_tokenizer.report_v2 import report
    report(out);print('Timing replay complete',len(evidence))
if __name__=='__main__':main(ROOT/sys.argv[1])
