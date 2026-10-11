import json,sys
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.constrained_v4 import ConstrainedTokenizer

out=ROOT/sys.argv[1];selection=json.loads((out/'selection.json').read_text());records={r['id']:r for r in read_jsonl(ROOT/'data/releases/v2/curated/splits/test.jsonl')};rows=read_jsonl(out/'per_formula.jsonl');replays=[]
for name,seed,path in [('ast_v4',0,out/'models'/f'ast-atom{selection["ast_atoms"]}.json'),('masked_zero_policy',0,out/'models'/'no_rl_zero.json'),('masked_shortest_control',0,out/'models'/'no_rl_shortest.json'),('masked_matched_initial_policy',0,out/'models'/'no_rl_matched_initial.json')]+[(name,seed,out/'models'/f'{prefix}-{seed}.json') for seed in (17,29,43) for name,prefix in [('rl_v4','rl_v4'),('v3_mask_only_control','mask_only')]]:
 model=ConstrainedTokenizer.load(path);rr=[r for r in rows if r['method']==name and r['seed']==seed];assert len(rr)==len(records)
 for r in rr:
  text=records[r['id']]['latex'];ids=model.encode(text);assert len(ids)==r['tokens'];assert model.decode(ids)==text
 replays.append({'method':name,'seed':seed,'formulas':len(rr),'counts_lossless_equal':True})
(out/'checkpoint_replay.json').write_text(json.dumps({'native_source_sha256':file_hash(ROOT/'native/decoder_v4.c'),'corpus_sha256':file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl'),'replays':replays},indent=2));print('v4 full checkpoint replay passed',len(replays))
