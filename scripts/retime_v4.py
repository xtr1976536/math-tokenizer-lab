"""After all training, replay cold-cache speed for identical selected checkpoints."""
import json,sys,time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.structure_v2 import structure

out=ROOT/sys.argv[1];p=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());metrics=json.loads((out/'metrics.json').read_text());records=read_jsonl(ROOT/'data/releases/v2/curated/splits/test.jsonl');models=out/'models';evidence=[]
for m in metrics:
 name=m['method'];seed=m['seed'];cap=p['capacity']
 if name=='ast_v4':model=ConstrainedTokenizer.load(models/f'ast-atom{selection["ast_atoms"]}.json')
 elif name=='ast_v3':old=json.loads((ROOT/f'results/v3_{cap}_release/selection.json').read_text());model=NativeTokenizer.load(ROOT/f'results/v3_{cap}_release/models/ast-{old["ast"]}.json')
 elif name=='masked_zero_policy':model=ConstrainedTokenizer.load(models/'no_rl_zero.json')
 elif name=='masked_matched_initial_policy':model=ConstrainedTokenizer.load(models/'no_rl_matched_initial.json')
 elif name=='masked_shortest_control':model=ConstrainedTokenizer.load(models/'no_rl_shortest.json')
 elif name=='rl_v3':model=NativeTokenizer.load(ROOT/f'results/v3_{cap}_release/models/rl_bpe_v3-{seed}.json')
 elif name=='v3_mask_only_control':model=ConstrainedTokenizer.load(models/f'mask_only-{seed}.json')
 else:model=ConstrainedTokenizer.load(models/f'rl_v4-{seed}.json')
 times=[];prior=None
 for _ in range(5):
  structure.cache_clear();start=time.perf_counter();ids=[model.encode(r['latex']) for r in records];times.append(time.perf_counter()-start);assert sum(map(len,ids))==m['tokens']
  if prior is not None:assert prior==ids
  prior=ids
 m['original_encode_seconds_repeats']=m['encode_seconds_repeats'];m['encode_seconds_repeats']=times;m['encode_seconds_median']=float(np.median(times));m['encode_mib_s']=m['bytes']/(1024**2*np.median(times));m['timing_replay']='cold structure, same frozen input, after all training'
 evidence.append({'method':name,'seed':seed,'seconds':times,'tokens_unchanged':True});print('timed',name,seed,round(m['encode_mib_s'],3),flush=True)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'timing_replay.json').write_text(json.dumps(evidence,indent=2))
