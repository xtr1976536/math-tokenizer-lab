"""Isolated public-interface speed replay after all fitting; counts fixed."""
import json,sys,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import sentencepiece as spm
from math_tokenizer.corpus import ROOT
from math_tokenizer.rl_vocab_v5 import RLVocab
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer,TikToken
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.evaluate import read_jsonl
out=ROOT/sys.argv[1];p=json.loads((out/'protocol.json').read_text());s=json.loads((out/'selection.json').read_text());metrics=json.loads((out/'metrics.json').read_text());models=out/'models';loaded={};timings=[]
for m in metrics:
 scenario=m['scenario'];name=m['method'];seed=m['seed'];key=(scenario,name,seed)
 if key not in loaded:
  if name=='rl_composite':model=RLVocab.load(models/f'{scenario}-rl-selected-{seed}.json')
  elif name in ('frequency','random','reward_greedy'):model=RLVocab.load(models/f'{scenario}-{name}-{seed}.json')
  elif name=='rl_length_only':model=RLVocab.load(models/f'{scenario}-length_only-{seed}.json')
  elif name=='hf_bpe_fulltrain':model=HFTokenizer(Tokenizer.from_file(str(models/f'{scenario}-hf-bpe.json')))
  elif name=='hf_bpe_matched_fit':model=HFTokenizer(Tokenizer.from_file(str(models/f'{scenario}-hf-bpe-fit.json')))
  elif name.startswith('sp_'):model=SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(models/f'{scenario}-sp-{name[3:]}.model')))
  elif name.startswith('tiktoken_'):model=TikToken(name[len('tiktoken_'):])
  elif name=='legacy_v3_rl':model=NativeTokenizer.load(ROOT/f'results/v3_{p["capacity"]}_release/models/rl_bpe_v3-17.json')
  else:model=ConstrainedTokenizer.load(ROOT/f'results/v4_{p["capacity"]}/models/rl_v4-17.json')
  loaded[key]=model
 model=loaded[key];records=read_jsonl(ROOT/f'data/releases/v5/curated/splits/{m["suite"]}.jsonl');seconds=[]
 for _ in range(5):
  if hasattr(model,'clear_cache'):model.clear_cache()
  elif hasattr(model,'tokenizer') and hasattr(model.tokenizer,'model') and hasattr(model.tokenizer.model,'clear_cache'):model.tokenizer.model.clear_cache()
  start=time.perf_counter();ids=[model.encode(r['latex']) for r in records];seconds.append(time.perf_counter()-start);assert sum(map(len,ids))==m['tokens']
 m['original_encode_seconds_repeats']=m['encode_seconds_repeats'];m['encode_seconds_repeats']=seconds;m['encode_seconds_median']=float(np.median(seconds));m['encode_mib_s']=m['bytes']/(1024**2*np.median(seconds));m['timing_replay']='isolated after fitting, same public per-formula API/cache-clear'
 timings.append({'scenario':scenario,'method':name,'seed':seed,'suite':m['suite'],'seconds':seconds,'counts_unchanged':True})
 print('timed',scenario,m['suite'],name,seed,round(m['encode_mib_s'],3),flush=True)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'timing_replay.json').write_text(json.dumps(timings,indent=2))
