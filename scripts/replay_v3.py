"""Full frozen test replay of all new final checkpoint counts, no baselines."""
import json,sys
from collections import defaultdict
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.native_v3 import NativeTokenizer

out=ROOT/sys.argv[1];protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());records={r['id']:r for r in read_jsonl(ROOT/'data/releases/v2/curated/splits/test.jsonl')};rows=read_jsonl(out/'per_formula.jsonl');by=defaultdict(list)
for r in rows:by[(r['method'],r['seed'])].append(r)
replays=[]
for name,seed in by:
 if name=='ast_bpe_v3':path=out/'models'/f'ast-{selection["ast"]}.json'
 elif name=='atomic_byte_shortest_control':path=out/'models'/'new_byte_no_rl_shortest.json'
 elif name=='rl_bpe_v2_native_exact':path=out/'models'/f'old_rl_native-{seed}.json'
 elif name in ('rl_bpe_v3','rl_ast_bpe_v3'):path=out/'models'/f'{name}-{seed}.json'
 else:continue
 model=NativeTokenizer.load(path)
 for row in by[(name,seed)]:
  text=records[row['id']]['latex'];encoded=model.encode(text)
  assert len(encoded)==row['tokens'];assert model.decode(encoded)==text
 replays.append({'method':name,'seed':seed,'test_records':len(by[(name,seed)]),'counts_and_losslessness_match':True})
result={'corpus_sha256':file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl'),'capacity':protocol['capacity'],'native_source_sha256':file_hash(ROOT/'native/decoder.c'),'replays':replays}
(out/'full_checkpoint_replay.json').write_text(json.dumps(result,indent=2));print('Full native checkpoint replay passed',len(replays),'runs')
