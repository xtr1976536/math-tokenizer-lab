"""Same paper resamples for paired four-metric comparisons; no extra metric."""
import csv,gzip,importlib.util,json
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('study',HERE/'run.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
COLUMNS=['cuts','possible_cuts','commands_intact','commands','crossing_tokens','tokens','numbers_intact','numbers']
def metric(v):
 with np.errstate(divide='ignore',invalid='ignore'):
  return np.array([100*v[...,0]/v[...,1],100*v[...,2]/v[...,3],100*(1-v[...,4]/v[...,5]),100*v[...,6]/v[...,7]]).T

def main():
 protocol=json.loads((s.OUT/'protocol.json').read_text());output=[];rng=np.random.default_rng(20261021)
 for p in protocol['splits']:
  folder=s.OUT/p['name'];models={}
  for name in ['ast_bpe','soft_ast_bpe']+s.BASELINES:
   file=folder/f'{name.replace("/","--")}_{1024 if name in ["ast_bpe","soft_ast_bpe"] else "official"}.jsonl.gz'
   with gzip.open(file,'rt') as f:rows=[json.loads(line) for line in f]
   groups=defaultdict(list)
   for r in rows:groups[r['group']].append(r)
   models[name]=groups
  groups=sorted(models['ast_bpe']);assert all(sorted(m)==groups for m in models.values())
  w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=1000)
  reps={};point={}
  for name,gg in models.items():
   a=np.array([[sum(r[c] for r in gg[g]) for c in COLUMNS] for g in groups],float)
   reps[name]=metric(w@a);point[name]=metric(a.sum(axis=0))
  for name in ['ast_bpe','soft_ast_bpe']:
   for base in s.BASELINES+(['soft_ast_bpe'] if name=='ast_bpe' else []):
    # Positive is improvement for all four predeclared directions.
    sign=np.array([-1,1,1,1]);delta=(point[name]-point[base])*sign;draws=(reps[name]-reps[base])*sign
    for i,k in enumerate(s.METRICS):
     valid=draws[:,i][np.isfinite(draws[:,i])];lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [None,None]
     output.append(dict(protocol=p['name'],model=name,reference=base,metric=k,beneficial_difference_pp=float(delta[i]),ci95_lower=lo,ci95_upper=hi,source_groups=len(groups),valid_bootstrap_replicates=len(valid)))
 s.write(s.OUT/'paired_comparisons_1024.json',dict(description='Paired source-cluster bootstrap,1000 draws; positive delta favors model; point delta uses ratio of aggregated counts; unadjusted exploratory95% intervals,conditional on this source split; no pooling overlapping splits',rows=output))
 with (s.OUT/'paired_comparisons_1024.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator='\n');w.writeheader();w.writerows(output)
 print('Paired comparisons:',len(output))
if __name__=='__main__':main()
