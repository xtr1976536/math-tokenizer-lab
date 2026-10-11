"""RL vocabulary learning with sealed fresh/held-domain/length stress suites.

Only dev is loaded during candidate fitting/selection. Fresh tests are read
AFTER selection.json is written and hashed. No algorithm changes after unlock.
"""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter,defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.corpus_archive_v5 import ensure_master
ensure_master()
from math_tokenizer.evaluate import read_jsonl,file_hash,bootstrap_difference
from math_tokenizer.rl_vocab_v5 import RLVocab,VocabRLConfig
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer,TikToken
from math_tokenizer.measure_v5 import measure

CONFIGS={
 'composite_balanced':VocabRLConfig(mdl_weight=.4,boundary_weight=.4,rare_weight=.15,robustness_weight=.25),
 'composite_structure':VocabRLConfig(mdl_weight=.2,boundary_weight=1.,rare_weight=.15,robustness_weight=.3),
 'composite_code':VocabRLConfig(mdl_weight=.8,boundary_weight=.2,rare_weight=.3,robustness_weight=.2),
 'composite_robust':VocabRLConfig(mdl_weight=.3,boundary_weight=.3,rare_weight=.2,robustness_weight=.8),
 'length_only':VocabRLConfig(objective='length'),
}

def macrocost(model,records):
 groups=defaultdict(lambda:[0,0]);cmd=cmdok=0
 import re
 for r in records:
  text=r['latex'];ids=model.encode(text);groups[r['split_group']][0]+=len(ids);groups[r['split_group']][1]+=len(text.encode())
  offset=0;bounds={0}
  for id in ids:offset+=len(model.vocab[id]);bounds.add(offset)
  bs=[0]
  for c in text:bs.append(bs[-1]+len(c.encode()))
  for match in re.finditer(r'\\[A-Za-z]+',text):cmd+=1;cmdok+=not any(bs[match.start()]<b<bs[match.end()] for b in bounds)
 cost=float(np.mean([t/b for t,b in groups.values()]));cmdrate=cmdok/max(1,cmd)
 return {'document_macro_tokens_per_byte':cost,'command_integrity':cmdrate,'dev_utility':cost+.15*(1-cmdrate)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,default=1024);p.add_argument('--output',required=True);a=p.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists():raise RuntimeError('immutable run')
 data=ROOT/'data/releases/v5/curated';meta=json.loads((data/'summary.json').read_text());frozen=json.loads((ROOT/'data/releases/v5/test_freeze.json').read_text());models=out/'models';models.mkdir(exist_ok=True)
 # Seal all natural/stress test snapshots by hash without loading their content.
 suite_names=['fresh_paper_test','extreme_domain_test','synthetic_stress_test'];suite_hashes={s:file_hash(data/f'splits/{s}.jsonl') for s in suite_names}
 assert suite_hashes==frozen['suites']
 protocol={'version':'v5_rl_vocab_fresh_generalization','capacity':a.capacity,'corpus_sha256':meta['corpus_sha256'],'suite_hashes':suite_hashes,
  'training_scenarios':{'broad':'train/dev; all domains except fresh papers','specialized':'transfer_train/transfer_dev; PDE/ODE/analysis only; target-domain document exclusion'},
  'held_domain_targets':['algebra','geometry_topology','number_theory','combinatorics','logic'],
  'configs':{k:asdict(v) for k,v in CONFIGS.items()},'dev_seeds':[17,29],'formal_seeds':[17,29,43],
  'selection_rule':'min mean(dev docmacro tokens/byte+.15*(1-command integrity)) across seeds17,29 among composite configs only; length reward ablation never selected main',
  'reward_scope':'fit/reward paper groups split inside train; neither dev nor fresh/held-domain/stress test in reward',
  'online_backend':'official HuggingFace Rust byte BPE, unchanged implementation, learned merge table exported; no AST parser at inference',
  'human_intervention':'no curated command list or formula-level annotations; raw lexer and reward weights/selection/seed budget still explicit design choices',
  'first_test_load_after_selection':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));modelpool={};selections={};trainrecords={};ablations=[]
 for scenario in ('broad','specialized'):
  train=read_jsonl(data/f'splits/{"train" if scenario=="broad" else "transfer_train"}.jsonl');dev=read_jsonl(data/f'splits/{"dev" if scenario=="broad" else "transfer_dev"}.jsonl');trainrecords[scenario]=train
  # Fixed document-balanced dev measurement subset; not model-dependent.
  grouped=defaultdict(list)
  for r in dev:grouped[r['split_group']].append(r)
  selection=[]
  for group,rr in sorted(grouped.items()):selection+=sorted(rr,key=lambda r:r['id'])[:32]
  selection=sorted(selection,key=lambda r:r['id'])[:1600]
  (out/f'{scenario}_training_ids.json').write_text(json.dumps({'train_ids':[r['id'] for r in train],'dev_ids':[r['id'] for r in dev],'selection_ids':[r['id'] for r in selection]},indent=2))
  for name,config in CONFIGS.items():
   for seed in (17,29):
    model=RLVocab(config).train(train,a.capacity,seed,'rl');model.save(models/f'{scenario}-{name}-{seed}.json');modelpool[(scenario,name,seed)]=model
    stats=macrocost(model,selection);row={'scenario':scenario,'config':name,'seed':seed,'stats':stats,'train_seconds':model.train_seconds,'vocab_actual':len(model.vocab),'fit_ids':model.fit_ids,'reward_ids':model.reward_ids};ablations.append(row)
    (out/'dev_ablations.json').write_text(json.dumps(ablations,indent=2));print('dev',scenario,name,seed,stats,flush=True)
  options={name:float(np.mean([r['stats']['dev_utility'] for r in ablations if r['scenario']==scenario and r['config']==name])) for name in CONFIGS if name!='length_only'}
  winner=min(options,key=lambda name:(options[name],name));selections[scenario]={'config':winner,'dev_scores':options}
 (out/'selection.json').write_text(json.dumps({'selections':selections,'test_used':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2));print('selection frozen',selections,flush=True)
 # Test unlock occurs exactly here, after both scenario choices fixed.
 suites={s:read_jsonl(data/f'splits/{s}.jsonl') for s in suite_names}
 assert all(file_hash(data/f'splits/{s}.jsonl')==suite_hashes[s] for s in suites)
 summaries=[];allrows=[];pairwise={};buckets={}
 def evaluate(model,name,seed,scenario,suite):
  summary,rows=measure(model,suites[suite],name,seed,repeats=5);summary['scenario']=scenario;summary['suite']=suite;summary['capacity_budget']=a.capacity if not name.startswith('tiktoken_') else None
  for row in rows:row['scenario']=scenario;row['suite']=suite
  summaries.append(summary)
  allrows.extend(rows);buckets[(scenario,suite,name,seed)]=rows
  (out/'metrics.json').write_text(json.dumps(summaries,indent=2));(out/'per_formula.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
  print('test',scenario,suite,name,seed,'tpf',round(summary['tokens_per_formula'],4),'cmd',summary['command_integrity_rate'],'MiBs',round(summary['encode_mib_s'],3),flush=True)
 for scenario in ('broad','specialized'):
  train=trainrecords[scenario];config=CONFIGS[selections[scenario]['config']];texts=[r['latex'] for r in train];active=['fresh_paper_test','synthetic_stress_test'] if scenario=='broad' else ['extreme_domain_test','synthetic_stress_test']
  for method in ('frequency','random','reward_greedy'):
   model=RLVocab(config).train(train,a.capacity,17,method);model.save(models/f'{scenario}-{method}-17.json')
   for suite in active:evaluate(model,method,17,scenario,suite)
  for seed in (17,29,43):
   model=modelpool.get((scenario,selections[scenario]['config'],seed))
   if model is None:model=RLVocab(config).train(train,a.capacity,seed,'rl')
   model.save(models/f'{scenario}-rl-selected-{seed}.json')
   for suite in active:evaluate(model,'rl_composite',seed,scenario,suite)
  length=modelpool[(scenario,'length_only',17)]
  for suite in active:evaluate(length,'rl_length_only',17,scenario,suite)
  # Fair same-budget standard BPE sees identical external train; its additional
  # full-train exposure vs RL fit subset explicitly reported.
  hf=HFTokenizer.train(texts,a.capacity,models/f'{scenario}-hf-bpe.json')
  for suite in active:evaluate(hf,'hf_bpe_fulltrain',0,scenario,suite)
  fit,_=__import__('math_tokenizer.rl_vocab_v5',fromlist=['internal_split']).internal_split(train,config);fittexts=[r['latex'] for r in fit]
  hf_fit=HFTokenizer.train(fittexts,a.capacity,models/f'{scenario}-hf-bpe-fit.json')
  for suite in active:evaluate(hf_fit,'hf_bpe_matched_fit',0,scenario,suite)
  for kind in ('bpe','unigram'):
   sp=SentencePieceTokenizer.train(texts,a.capacity,kind,models/f'{scenario}-sp-{kind}')
   for suite in active:evaluate(sp,'sp_'+kind,0,scenario,suite)
  for suite in active:
   for seed in (17,29,43):
    for control,cseed in [('frequency',17),('rl_length_only',17),('reward_greedy',17),('hf_bpe_fulltrain',0),('hf_bpe_matched_fit',0)]:
     pairwise[f'{scenario}/{suite}/rl-vs-{control}/{seed}']=bootstrap_difference(buckets[(scenario,suite,'rl_composite',seed)],buckets[(scenario,suite,control,cseed)])
 # Frozen historical models are references only: trained on prior v2 data,
 # not reselected using the fresh benchmark. They cannot be fair-trained peers.
 from math_tokenizer.native_v3 import NativeTokenizer
 from math_tokenizer.constrained_v4 import ConstrainedTokenizer
 if a.capacity in (512,1024):
  legacy=[('legacy_v3_rl',NativeTokenizer.load(ROOT/f'results/v3_{a.capacity}_release/models/rl_bpe_v3-17.json')),
          ('legacy_v4_masked_rl',ConstrainedTokenizer.load(ROOT/f'results/v4_{a.capacity}/models/rl_v4-17.json'))]
  for name,model in legacy:
   for suite in ('fresh_paper_test','extreme_domain_test'):evaluate(model,name,17,'prior_data_reference',suite)
 # Pretrained byte reference same test strings, never fair vocabulary comparison.
 for name in ('cl100k_base','o200k_base'):
  model=TikToken(name)
  for suite in suite_names:evaluate(model,'tiktoken_'+name,0,'reference',suite)
 (out/'pairwise.json').write_text(json.dumps(pairwise,indent=2))
 (out/'manifest.json').write_text(json.dumps({'protocol_sha256':file_hash(out/'protocol.json'),'selection_sha256':file_hash(out/'selection.json'),'suite_hashes':suite_hashes,
   'models':{p.name:file_hash(p) for p in models.iterdir() if p.is_file()},'runs':len(summaries),'records':len(allrows)},indent=2))
if __name__=='__main__':main()
