"""Formula-macro fragmentation dispersion; no retraining or threshold fitting."""
from pathlib import Path
import csv,gzip,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/zip_ast_20261021'
def main():
 protocol=json.loads((OUT/'protocol.json').read_text());results=[]
 for p in protocol['splits']:
  for file in sorted((OUT/p['name']).glob('*.jsonl.gz')):
   with gzip.open(file,'rt') as f:rows=[json.loads(line) for line in f]
   valid=[r for r in rows if r['bytes']>1]
   x=np.array([100*(r['tokens']-1)/(r['bytes']-1) for r in valid])
   summary=json.loads(file.with_suffix('').with_suffix('.json').read_text())
   results.append(dict(protocol=p['name'],method=summary['method'],requested_total_ids=summary['requested_total_ids'],formulas=len(valid),excluded_bytes_le1=len(rows)-len(valid),formula_macro_mean_pct=float(x.mean()),fragmentation_sd_pp=float(x.std(ddof=0)),p10_pct=float(np.quantile(x,.1)),p50_pct=float(np.quantile(x,.5)),p90_pct=float(np.quantile(x,.9))))
 assert len(results)==228
 payload=dict(metric='Formula-level fragmentation standard deviation',definition='Population SD of f_i=100*(T_i-1)/(B_i-1), each formula equal weight; B_i>1',unit='percentage points',interpretation='Larger means less uniform formula-level segmentation density. Not a standalone objective: all-byte and all-whole-formula tokenizers both have SD0. Interpret with mean and structural metrics; formula difficulty/length/domain also cause variation. Does not establish token reusability.',rows=results)
 (OUT/'granularity_dispersion.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'granularity_dispersion.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
 # Mathematical check of the exact distinction the user requested.
 assert np.std([.5,.5])==0 and np.std([0.,1.])==.5
 print('Computed dispersion for',len(results),'evaluations')
if __name__=='__main__':main()
