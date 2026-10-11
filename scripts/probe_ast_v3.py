import json,time
from pathlib import Path
from collections import defaultdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.structure_v2 import structure
from math_tokenizer.metrics_v2 import measure
train=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
for cap in (512,1024):
 for atom_budget in (48,96,160):
  base=AtomicASTBPE(True).train([r['latex'] for r in train],cap,command_budget=atom_budget)
  for penalty in (0.,.5,2.):
   m=base.decoder(penalty);s,rows=measure(m,dev,f'ast_atomic{atom_budget}_penalty{penalty}',0,repeats=1,upstream=False)
   print(cap,atom_budget,penalty,round(s['tokens_per_formula'],4),round(s['command_integrity_rate'],4),round(s['encode_mib_s'],3),'train',round(base.train_seconds,2),flush=True)
