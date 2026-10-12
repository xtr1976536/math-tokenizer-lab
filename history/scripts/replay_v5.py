"""Replay selected learned merge tables on every sealed suite; no retraining."""
import json,sys
from collections import defaultdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.corpus_archive_v5 import ensure_master
ensure_master()
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.rl_vocab_v5 import RLVocab
out=ROOT/sys.argv[1];rows=read_jsonl(out/'per_formula.jsonl');by=defaultdict(list)
for r in rows:by[(r['scenario'],r['method'],r['seed'],r['suite'])].append(r)
replays=[]
for scenario in ('broad','specialized'):
 for seed in (17,29,43):
  m=RLVocab.load(out/'models'/f'{scenario}-rl-selected-{seed}.json')
  suites=[s for sc,name,se,s in by if sc==scenario and name=='rl_composite' and se==seed]
  for suite in suites:
   corpus={r['id']:r for r in read_jsonl(ROOT/f'data/releases/v5/curated/splits/{suite}.jsonl')}
   rr=by[(scenario,'rl_composite',seed,suite)]
   for r in rr:
    text=corpus[r['id']]['latex'];ids=m.encode(text);assert len(ids)==r['tokens'];assert m.decode(ids)==text
   replays.append({'scenario':scenario,'seed':seed,'suite':suite,'formulas':len(rr),'same_counts_losslessness':True})
(out/'checkpoint_replay.json').write_text(json.dumps({'replays':replays,'all_equal':True,'corpus_sha256':file_hash(ROOT/'data/releases/v5/curated/corpus.jsonl')},indent=2));print('v5 selected checkpoint replay',len(replays))
