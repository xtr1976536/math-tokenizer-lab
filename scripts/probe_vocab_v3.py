import json
from pathlib import Path
from dataclasses import replace
from math_tokenizer.corpus import ROOT
from math_tokenizer.vocab_v3 import VocabConfig,refine,optimize_swaps
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.metrics_v2 import measure
train=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
for structural in (False,True):
 config=VocabConfig(structural=structural,seed_multiplier=3,pruning_rounds=4,policy_rounds=8,reward_formulas=192)
 m,original,mandatory,fit,reward=refine(train,512,config)
 s,_=measure(m,dev,'pruned',0,repeats=1,upstream=False)
 print('pruned',structural,s['tokens_per_formula'],s['command_integrity_rate'],s['encode_mib_s'],flush=True)
 rl=optimize_swaps(m,original,mandatory,fit,reward,config)
 s,_=measure(rl,dev,'rl',17,repeats=1,upstream=False)
 print('rl',structural,s['tokens_per_formula'],s['command_integrity_rate'],s['encode_mib_s'],'accepted',sum(x['accepted'] for x in rl.training_log[-1]['logs']),flush=True)
