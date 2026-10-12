from dataclasses import replace
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.rl_v4 import MaskedPolicy,MaskedConfig
from math_tokenizer.metrics_v2 import measure
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
train=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl');texts=[r['latex'] for r in train]
for cap in (512,1024):
 for atoms in (128,256):
  base=AtomicASTBPE(True).train(texts,cap,command_budget=atoms)
  for mode in ('shortest','policy_map','greedy'):
   if mode=='shortest':m=ConstrainedTokenizer(base.vocab,True,'commands',mode)
   else:m=MaskedPolicy(base.vocab,MaskedConfig(epochs=12,max_train_formulas=384,infer=mode)).train(texts,17)
   s,_=measure(m,dev,f'v4_{mode}',17,repeats=1,upstream=False)
   print(cap,atoms,mode,'tokens',round(s['tokens_per_formula'],4),'command',s['command_integrity_rate'],'cross',s['hybrid_tree_crossing_rate'],'speed',round(s['encode_mib_s'],3),'train',round(m.train_seconds,3),flush=True)
