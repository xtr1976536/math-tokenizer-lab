from pathlib import Path
from dataclasses import replace
from math_tokenizer.corpus import ROOT
from math_tokenizer.rl_vocab_v5 import RLVocab,VocabRLConfig
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.metrics_v2 import measure
records=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
for method,objective in [('frequency','composite'),('reward_greedy','composite'),('rl','length'),('rl','composite'),('random','composite')]:
 m=RLVocab(VocabRLConfig(objective=objective,fit_limit=4000,reward_limit=256,updates=24)).train(records,512,17,method)
 s,_=measure(m,dev,method+'-'+objective,17,repeats=2,upstream=False)
 print(method,objective,s['tokens_per_formula'],s['command_integrity_rate'],s['hybrid_tree_crossing_rate'],s['encode_mib_s'],'train',m.train_seconds,flush=True)
