from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.metrics_v2 import measure
from math_tokenizer.vocab_v3 import estimate_usage
import numpy as np
train=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl');dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl');texts=[r['latex'] for r in train]
for cap in (512,1024):
 for ratio in (2,4):
  base=AtomicASTBPE(True).train(texts,cap*ratio,command_budget=128);vocab=base.vocab[:];mandatory=set(base.command_atoms)|{bytes([b]) for b in range(256)}
  for round in range(5):
   model=NativeTokenizer(vocab,True);counts=estimate_usage(model,texts);target=cap if round==4 else max(cap,int(len(vocab)*.75));scores=counts*np.array([max(0,len(t)-1) for t in vocab]);keep=[i for i,t in enumerate(vocab) if t in mandatory];remaining=[i for i,t in enumerate(vocab) if t not in mandatory];remaining.sort(key=lambda i:(-scores[i],-counts[i],vocab[i]));keep+=remaining[:target-len(keep)];vocab=[vocab[i] for i in sorted(keep)]
  model=NativeTokenizer(vocab,True);s,_=measure(model,dev,'refinefull',0,repeats=1,upstream=False);print('refinefull',cap,ratio,s['tokens_per_formula'],s['command_integrity_rate'],flush=True)
