from pathlib import Path
import time,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.rl_v2 import LatticeTokenizer
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.structure_v2 import structure
r=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
results=[]
for cap in (512,1024):
 for kind in ('byte','ast'):
  slow=LatticeTokenizer.load(ROOT/f'results/v2_release{cap}/models/rl_{kind}_selected-17.json');fast=NativeTokenizer.from_v2(slow)
  timings={};previous=None
  for name,m in [('python',slow),('native',fast)]:
   structure.cache_clear();start=time.perf_counter();current=[m.encode(x['latex']) for x in r];timings[name]=time.perf_counter()-start
   if previous is not None:assert current==previous
   previous=current
  result={'capacity':cap,'kind':kind,'timings':timings,'speedup':timings['python']/timings['native'],'exact_tokens_equal':True,'scope':'all_dev_formulas'};results.append(result);print(result,flush=True)
(ROOT/'results/v3_native_probe.json').write_text(json.dumps(results,indent=2))
