"""Isolated final timings after experiment training, optimized parser verified."""
import json,sys,time
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.rl_v2 import LatticeTokenizer
from math_tokenizer.structure_v2 import structure

out=ROOT/sys.argv[1];protocol=json.loads((out/'protocol.json').read_text());records=read_jsonl(ROOT/'data/releases/v2/curated/splits/test.jsonl');models=out/'models';metrics=json.loads((out/'metrics.json').read_text());selection=json.loads((out/'selection.json').read_text());evidence=[]
for m in metrics:
 name=m['method'];seed=m['seed'];cap=protocol['capacity']
 if name=='ast_bpe_v2':model=FastBPE.load(ROOT/f'results/v2_release{cap}/models/base-ast.json')
 elif name=='ast_bpe_v3':model=NativeTokenizer.load(models/f'ast-{selection["ast"]}.json')
 elif name=='atomic_byte_shortest_control':model=NativeTokenizer.load(models/'new_byte_no_rl_shortest.json')
 elif name=='rl_bpe_v2':model=LatticeTokenizer.load(ROOT/f'results/v2_release{cap}/models/rl_byte_selected-{seed}.json')
 elif name=='rl_bpe_v2_native_exact':model=NativeTokenizer.load(models/f'old_rl_native-{seed}.json')
 elif name=='rl_bpe_v3':model=NativeTokenizer.load(models/f'rl_bpe_v3-{seed}.json')
 elif name=='rl_ast_bpe_v2':model=LatticeTokenizer.load(ROOT/f'results/v2_release{cap}/models/rl_ast_selected-{seed}.json')
 else:model=NativeTokenizer.load(models/f'rl_ast_bpe_v3-{seed}.json')
 timings=[];old=None
 for _ in range(5):
  structure.cache_clear();start=time.perf_counter();encoded=[model.encode(r['latex']) for r in records];timings.append(time.perf_counter()-start)
  assert sum(map(len,encoded))==m['tokens']
  if old is not None:assert old==encoded
  old=encoded
 m['original_encode_seconds_repeats']=m['encode_seconds_repeats'];m['encode_seconds_repeats']=timings;m['encode_seconds_median']=float(np.median(timings));m['encode_mib_s']=m['bytes']/(1024**2*np.median(timings));m['timing_replay']='isolated after all training, same cold-structure protocol'
 evidence.append({'method':name,'seed':seed,'counts_unchanged':True,'seconds':timings})
 print('timed',name,seed,round(m['encode_mib_s'],3),flush=True)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'timing_replay.json').write_text(json.dumps(evidence,indent=2))
from math_tokenizer.report_v3 import report
report(out)
