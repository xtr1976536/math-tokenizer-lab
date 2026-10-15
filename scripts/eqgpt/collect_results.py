"""Collect complete paired runs; smoke remains explicitly separated."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from math_tokenizer.eqgpt.evaluation import summarize_run
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--include-smoke',action='store_true');a=p.parse_args()
rows=[];detailed=[];seen=set()
for path in sorted(a.root.rglob('manifest.json')):
 data=json.loads(path.read_text())
 if 'model_initialization' not in data:continue
 if data['smoke'] and not a.include_smoke:continue
 key=(data['equation'],data['shared']['observations'],data['shared']['noise_percent'],data['seed'],data['smoke'])
 if key in seen:raise ValueError('Duplicate experimental setting: '+str(key))
 seen.add(key);summary=summarize_run(path.parent);summary.update(observations=key[1],noise_percent=key[2]);detailed.append(summary)
 for model,value in summary['models'].items():
  rows.append(dict(equation=summary['equation'],observations=key[1],noise_percent=key[2],seed=summary['seed'],smoke=summary['smoke'],model=model,
      top1=value['top1'],top10_correct=value['top10_correct'],reward=value['reward'],best_equation=value['equation'],
      pinn_status=value['pinn_status'],coefficient_relative_error=value['coefficient_relative_error']))
(a.root/'results.json').write_text(json.dumps(detailed,ensure_ascii=False,indent=2)+'\n')
if rows:
 with (a.root/'results.csv').open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 groups=defaultdict(list)
 for r in rows:groups[r['model']].append(r)
 comparison={}
 for model,rr in groups.items():
  valid=[r['coefficient_relative_error'] for r in rr if r['coefficient_relative_error'] is not None]
  comparison[model]=dict(runs=len(rr),correct=sum(r['top1']=='correct' for r in rr),redundant=sum(r['top1']=='redundant' for r in rr),
    incorrect=sum(r['top1']=='incorrect' for r in rr),top1_recovery_rate=np.mean([r['top1']=='correct' for r in rr]).item(),
    top10_recovery_rate=np.mean([r['top10_correct'] for r in rr]).item(),pinn_failures=sum(r['pinn_status']!='completed' for r in rr),
    coefficient_evaluable=len(valid),mean_coefficient_relative_error=float(np.mean(valid)) if valid else None)
 order=sorted(comparison,key=lambda m:(-comparison[m]['top1_recovery_rate'],-comparison[m]['top10_recovery_rate'],comparison[m]['pinn_failures']))
 (a.root/'comparison.json').write_text(json.dumps(dict(smoke=any(r['smoke'] for r in rows),ranking_order='Top-1 recovery, Top-10 recovery, PINN failure count; exact ties are retained in statistics',ranking=order,models=comparison),indent=2)+'\n')
print('Collected runs',len(detailed),'model rows',len(rows),'include smoke',a.include_smoke)
