import json,sys,subprocess
from pathlib import Path
from collections import defaultdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import file_hash,read_jsonl
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.ast import compatible

out=ROOT/sys.argv[1];p=json.loads((out/'protocol.json').read_text());s=json.loads((out/'selection.json').read_text());manifest=json.loads((out/'manifest.json').read_text());rows=read_jsonl(out/'per_formula.jsonl');metrics=json.loads((out/'metrics.json').read_text());by=defaultdict(list)
assert p['corpus_sha256']==file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')==manifest['corpus_sha256'];assert file_hash(out/'protocol.json')==manifest['protocol_sha256'];assert file_hash(out/'selection.json')==manifest['selection_sha256'];assert manifest['native_source_sha256']==file_hash(ROOT/'native/decoder_v4.c');assert not p['new_baselines_run'] and not s['test_used_for_selection']
ablations=json.loads((out/'dev_ablations.json').read_text())
ast=[r for r in ablations if r['arm']=='ast'];eligible=[r for r in ast if r['stats']['command_integrity']>=.999] or ast
best=min(eligible,key=lambda r:(r['stats']['macro_tokens_per_byte'],r['atoms']));assert s['ast_atoms']==best['atoms']
options=defaultdict(list)
for r in ablations:
 if r['arm']=='rl':options[(r['atoms'],r['config'],r['mode'])].append(r)
import numpy as np
means={key:float(np.mean([r['stats']['macro_tokens_per_byte'] for r in rr])) for key,rr in options.items()}
eligible={key:score for key,score in means.items() if np.mean([r['stats']['command_integrity'] for r in options[key]])>=.999} or means
assert (s['rl_atoms'],s['rl_config'],s['rl_mode'])==min(eligible,key=lambda key:(eligible[key],key))
assert set(p['train_ids']).isdisjoint(p['test_ids']);assert set(p['dev_ids']).isdisjoint(p['test_ids'])
for r in rows:by[(r['method'],r['seed'])].append(r)
for key,rr in by.items():assert len(rr)==len(p['test_ids']) and {r['id'] for r in rr}==set(p['test_ids']) and all(r['roundtrip'] for r in rr)
for m in metrics:
 rr=by[(m['method'],m['seed'])];assert sum(x['tokens'] for x in rr)==m['tokens'];assert m['vocab_actual']<=p['capacity']
 if m['method'] in ('rl_v4','ast_v4'):assert m['hybrid_tree_crossing_rate']==0
for filename,digest in manifest['models'].items():assert file_hash(out/'models'/filename)==digest
records={r['id']:r for r in read_jsonl(ROOT/'data/releases/v2/curated/splits/test.jsonl')};count=0
for path in (out/'models').glob('*.json'):
 model=ConstrainedTokenizer.load(path);sample=sorted(records)[::max(1,len(records)//100)]
 for id in sample:
  t=records[id]['latex'];encoded=model.encode(t);assert model.decode(encoded)==t
  raw,spans,safe,interior,allowed,exceptions,st=model.prepare(t);offset=0
  for token in encoded:
   end=offset+len(model.vocab[token]);assert allowed[offset] and allowed[end]
   if model.structural and st.kind!='byte_fallback':assert compatible(offset,end,st.spans)
   offset=end
 count+=1
result={'corpus_unchanged':True,'fixed_budget_and_same_test':True,'all_lossless':True,'new_core_crossing_zero':True,'checkpoints_replayed':count,'source_and_model_hashes':True,'no_new_baselines':True,'rl_gradient_finite_difference_test':'tests/test_v4.py','test_formulas':len(records),'runs':len(metrics)}
tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,capture_output=True,text=True);assert tests.returncode==0,tests.stdout+tests.stderr;result['tests']=tests.stdout
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
