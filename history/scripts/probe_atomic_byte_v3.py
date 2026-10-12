from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.metrics_v2 import measure
train=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
for cap in (512,1024):
 for atom in (64,128,192):
  base=AtomicASTBPE(False).train([r['latex'] for r in train],cap,command_budget=atom)
  for penalty in (0.,.5,1.):
   model=base.decoder(penalty);s,_=measure(model,dev,'atomic_byte',0,repeats=1,upstream=False)
   print(cap,atom,penalty,round(s['tokens_per_formula'],4),round(s['command_integrity_rate'],4),round(s['hybrid_tree_crossing_rate'],4),flush=True)
