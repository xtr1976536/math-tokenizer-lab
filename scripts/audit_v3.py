"""Frozen corpus, selection, exact native equivalence and shared test auditing."""
import argparse,json,subprocess,sys
from pathlib import Path
from collections import defaultdict,Counter
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.rl_v2 import LatticeTokenizer
from math_tokenizer.structure_v2 import structure


def audit(out,replay=True):
 protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());manifest=json.loads((out/'manifest.json').read_text());data=ROOT/'data/releases/v2/curated'
 assert file_hash(data/'corpus.jsonl')==protocol['corpus_sha256']==manifest['corpus_sha256']
 assert file_hash(out/'protocol.json')==manifest['protocol_sha256'];assert file_hash(out/'selection.json')==manifest['selection_sha256']
 assert file_hash(ROOT/'native/decoder.c')==manifest['native_source_sha256']
 assert protocol['new_baselines_run'] is False;assert selection['test_used_for_selection'] is False
 ablations=json.loads((out/'dev_ablations.json').read_text())
 oldast=selection['old_dev_scores']['ast'];oldrl=selection['old_dev_scores']['rl']
 astoptions={r['name']:r['score']['document_macro_tokens_per_byte'] for r in ablations if r['arm']=='ast' and r['score']['command_integrity']>=oldast['command_integrity'] and r['score']['number_integrity']>=oldast['number_integrity']-1e-12}
 assert selection['ast']==min(astoptions,key=lambda k:(astoptions[k],k))
 import numpy as np
 rloptions=defaultdict(list)
 for r in ablations:
  if r['arm']=='rl_byte':rloptions[r['name']].append(r['score'])
 eligible={name:float(np.mean([r['document_macro_tokens_per_byte'] for r in scores])) for name,scores in rloptions.items()
  if np.mean([r['command_integrity'] for r in scores])>=oldrl['command_integrity'] and np.mean([r['tree_crossing'] for r in scores])<=oldrl['tree_crossing']+1e-12 and np.mean([r['number_integrity'] for r in scores])>=oldrl['number_integrity']-1e-12}
 assert selection['rl_byte']==min(eligible,key=lambda k:(eligible[k],k))
 assert set(protocol['selection_dev_ids'])<=set(protocol['dev_ids']);assert set(protocol['train_ids']).isdisjoint(protocol['test_ids'])
 rows=read_jsonl(out/'per_formula.jsonl');by=defaultdict(list)
 for r in rows:by[(r['method'],r['seed'])].append(r)
 assert len(by)==manifest['method_seed_runs'];assert len(rows)==manifest['records']
 ids=set(protocol['test_ids']);lengths=defaultdict(set)
 for key,rr in by.items():
  assert len(rr)==len(ids);assert {r['id'] for r in rr}==ids;assert all(r['roundtrip'] for r in rr)
  for r in rr:lengths[r['id']].add(r['bytes'])
 assert all(len(v)==1 for v in lengths.values())
 metrics=json.loads((out/'metrics.json').read_text())
 for m in metrics:
  rr=by[(m['method'],m['seed'])];assert sum(r['tokens'] for r in rr)==m['tokens'];assert sum(r['bytes'] for r in rr)==m['bytes']
  assert m['vocab_actual']<=protocol['capacity'];assert m['capacity_budget']==protocol['capacity'];assert len(m['encode_seconds_repeats'])==5
  assert m['cache_protocol']=='structure_cache_cleared_before_each_repeat'
 for filename,digest in manifest['models'].items():assert file_hash(out/'models'/filename)==digest
 result={'round1_frozen_corpus':{'passed':True,'hash':protocol['corpus_sha256'],'formulas':len(read_jsonl(data/'corpus.jsonl'))},
  'round2_selection_and_budget':{'passed':True,'dev_only_selection':True,'new_baselines_run':False,'capacity':protocol['capacity']},
  'round3_raw_test_metrics':{'passed':True,'test_formulas':len(ids),'method_seed_runs':len(by),'records':len(rows),'all_lossless':True}}
 if replay:
  records={r['id']:r for r in read_jsonl(data/'splits/test.jsonl')};samples=sorted(ids)[::max(1,len(ids)//128)]
  count=0
  for path in sorted((out/'models').glob('*.json')):
   model=NativeTokenizer.load(path)
   for id in samples:
    text=records[id]['latex'];assert model.decode(model.encode(text))==text
   count+=1
  # Full test exact mode must match all recorded Python-v2 counts, plus actual
  # token IDs on common stratified replay sample and multiple saved seeds.
  for seed in (17,29,43):
   native=NativeTokenizer.load(out/'models'/f'old_rl_native-{seed}.json');old=LatticeTokenizer.load(ROOT/f'results/v2_release{protocol["capacity"]}/models/rl_byte_selected-{seed}.json')
   for id in samples:assert native.encode(records[id]['latex'])==old.encode(records[id]['latex'])
   aa={r['id']:r['tokens'] for r in by[('rl_bpe_v2',seed)]};bb={r['id']:r['tokens'] for r in by[('rl_bpe_v2_native_exact',seed)]};assert aa==bb
  result['round4_checkpoint_and_native_differential']={'passed':True,'checkpoints':count,'sample_formulas':len(samples),'exact_native_python_seed_count':3}
 tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,capture_output=True,text=True);assert tests.returncode==0,tests.stdout+tests.stderr
 result['round5_tests_and_error_cases']={'passed':True,'output':tests.stdout}
 # Frozen corpus is independently validated after all encoding experiments.
 assert file_hash(data/'corpus.jsonl')==manifest['corpus_sha256']
 result['round6_post_experiment_data_integrity']={'passed':True,'unchanged':True,'model_hashes':len(manifest['models'])}
 (out/'audit.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();print(json.dumps(audit(ROOT/a.output),indent=2))
