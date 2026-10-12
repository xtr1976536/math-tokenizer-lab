"""Supplemental matched initialization control, no test-based configuration tuning."""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash,bootstrap_difference
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.metrics_v2 import measure

out=ROOT/sys.argv[1];selection=json.loads((out/'selection.json').read_text());protocol=json.loads((out/'protocol.json').read_text());sample=ConstrainedTokenizer.load(out/'models'/'rl_v4-17.json');config=protocol['configs'][selection['rl_config']]
theta=np.zeros(7);theta[0]=config['initial_length_prior'];control=ConstrainedTokenizer(sample.vocab,True,'commands',selection['rl_mode'],theta,sample.frequency);control.save(out/'models'/'no_rl_matched_initial.json')
train=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');test=read_jsonl(ROOT/'data/releases/v2/curated/splits/test.jsonl');frequency=Counter()
for r in train:frequency.update(control.encode(r['latex']))
summary,rows=measure(control,test,'masked_matched_initial_policy',0,repeats=5,train_frequency=frequency);summary['capacity_budget']=protocol['capacity'];summary['comparison_group']='same_frozen_train_test';summary['supplemental_control']='same selected config initialization and post-train frequency features; zero policy updates'
metrics=json.loads((out/'metrics.json').read_text());allrows=read_jsonl(out/'per_formula.jsonl');assert not any(r['method']==summary['method'] for r in metrics)
metrics.append(summary);allrows+=rows;(out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'per_formula.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
paired=json.loads((out/'pairwise.json').read_text())
for seed in (17,29,43):
 a=[r for r in allrows if r['method']=='rl_v4' and r['seed']==seed];paired[f'rl_v4-vs-matched_initial/{seed}']=bootstrap_difference(a,rows)
(out/'pairwise.json').write_text(json.dumps(paired,indent=2));manifest=json.loads((out/'manifest.json').read_text());manifest['models']={p.name:file_hash(p) for p in (out/'models').glob('*.json')};manifest['runs']=len(metrics);manifest['records']=len(allrows);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'supplemental_control.json').write_text(json.dumps({'method':'masked_matched_initial_policy','added_after_selection':True,'affects_selection':False,'purpose':'separate RL update gains from selected initialization prior and frequency table','initial_length_prior':config['initial_length_prior'],'test_tuning':False},indent=2));print('Matched initial control',summary['tokens_per_formula'])
