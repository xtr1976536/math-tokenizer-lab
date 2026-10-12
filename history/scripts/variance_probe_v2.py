"""Compare score-estimator variance at ONE fixed policy with paired trajectories.

Train-only sample and no model/hyperparameter selection. Monte Carlo diagnostic,
not a proof of general convergence. Both estimators share sampled paths.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.rl_v2 import LatticeTokenizer,RLConfig

base=FastBPE.load(ROOT/'results/v2_release512/models/base-byte.json')
records=read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl')
import hashlib
records=sorted(records,key=lambda r:hashlib.sha256(('variance:'+r['id']).encode()).digest())[:32]
model=LatticeTokenizer(base,mode='policy_map',config=RLConfig(samples=4,epochs=0))
rng=np.random.default_rng(20261011);gradients={'naive':[],'rloo':[]};document_results=[]
for r in records:
 graph=model.graph(r['latex']);n=len(graph);local={'naive':[],'rloo':[]}
 for repeat in range(128):
  paths=[model.sample(graph,rng) for _ in range(4)];dense=[]
  for path in paths:
   ret=0.;g=np.zeros(n)
   for pos,end,id,cost,score,ent in reversed(path):ret=-cost+ret;g[pos:end]=ret
   dense.append(g)
  dense=np.asarray(dense)
  estimates={name:np.zeros_like(model.theta) for name in local}
  for k,path in enumerate(paths):
   baseline=(dense.sum(axis=0)-dense[k])/3
   for pos,end,id,cost,score,ent in path:
    estimates['naive']+=dense[k,pos]*score/(4*n)
    estimates['rloo']+=(dense[k,pos]-baseline[pos])*score/(4*n)
  for name,g in estimates.items():local[name].append(g)
 variance={name:float(np.mean(np.var(gs,axis=0,ddof=1))) for name,gs in local.items()}
 means={name:np.mean(gs,axis=0).tolist() for name,gs in local.items()}
 document_results.append({'id':r['id'],'domain':r['domain'],'bytes':n,'variance':variance,'gradient_means':means})
summary={'scope':'32 fixed train formulas; fixed initial policy and vocabulary;128 independent batches each;4 trajectories; paired naive/RLOO samples','seed':20261011,'samples':4,'repeats':128,
 'policy_parameters':model.theta.tolist(),'document_ids':[r['id'] for r in records],
 'mean_coordinate_variance':{name:float(np.mean([d['variance'][name] for d in document_results])) for name in local},'documents':document_results,
 'used_for_selection':False,'limitation':'Monte Carlo diagnostic at this fixed policy; no claim of unbiased empirical mean equality at finite samples or universal variance guarantee.'}
summary['rloo_variance_ratio']=summary['mean_coordinate_variance']['rloo']/summary['mean_coordinate_variance']['naive']
(ROOT/'results/v2_variance_probe.json').write_text(json.dumps(summary,indent=2));print('fixed-policy RLOO variance ratio',summary['rloo_variance_ratio'])
