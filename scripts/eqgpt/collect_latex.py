"""Aggregate round-one outcomes, retaining all failures and paired runs."""
import argparse,csv,json
from pathlib import Path
from collections import Counter,defaultdict
from math_tokenizer.eqgpt.evaluation import classify
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--include-smoke',action='store_true');a=p.parse_args()
rows=[];cycles=[];seen=set();fingerprints=set()
for path in sorted(a.root.rglob('manifest.json')):
 m=json.loads(path.read_text())
 if m.get('version')!='eqgpt-round1-main-v1':continue
 if m['smoke'] and not a.include_smoke:continue
 setting=(m['equation'],m['seed'],m['mode'],m['shared']['observations'],m['shared']['noise_percent'])
 if setting in seen:raise ValueError('Duplicate paired setting: '+str(setting))
 seen.add(setting);fingerprints.add((m['runtime_sha256'],m['frozen_tokenizers_sha256']))
 assert set(m['results'])=={'bpe','ast_bpe','soft_ast_bpe','eqgpt_reference'}
 timing=json.loads((path.parent/'timing.json').read_text())
 for name,result in m['results'].items():
  s=json.loads((path.parent/name/'search.json').read_text());attempts=s['attempts'];states=Counter(r['status'] for r in attempts)
  assert len(attempts)==400*(2 if m['smoke'] else 5)
  t=result['training'];seconds=timing[name]
  row=dict(equation=m['equation'],seed=m['seed'],mode=m['mode'],smoke=m['smoke'],method=name,
      sat=(name=='eqgpt_reference'),reference_gre_length_limit=50 if name=='eqgpt_reference' else None,
      actual_ids=m['model_initialization'][name]['input_shape'][0],parameters=m['model_initialization'][name]['parameters'],
      attempts=len(attempts),evaluated=states.get('evaluated',0),evaluable_rate=states.get('evaluated',0)/len(attempts),
      top1=result['top1'],top10_correct=result['top10_correct'],best_reward=result['best_reward'],
      pinn_status=result['pinn_status'],coefficient_relative_error=result['coefficient_relative_error'],
      training_epochs=t['epochs'],optimizer_steps=t['optimizer_steps'],training_target_tokens=t['training_targets_per_epoch']*t['epochs'],
      train_seconds=seconds['train_seconds'],search_seconds=seconds['search_seconds'],pinn_seconds=seconds['pinn_seconds'],statuses=dict(states),
      generation_stop_reasons=dict(Counter(r['stop_reason'] for r in attempts)))
  row['attempt_recovery_count']=sum(r['status']=='evaluated' and classify(r['gre'],m['equation'])['status']=='correct' for r in attempts)
  row['attempt_recovery_rate']=row['attempt_recovery_count']/len(attempts);rows.append(row)
  for cycle in s['history']:
   cycles.append(dict(equation=m['equation'],seed=m['seed'],method=name,cycle=cycle['cycle'],attempts=cycle['attempts'],
      evaluated=cycle['evaluated'],evaluable_rate=cycle['evaluated']/cycle['attempts'],
      best_reward=cycle['rewards'][0] if cycle['rewards'] else None,candidates=len(cycle['gre']),statuses=cycle['statuses']))
if len(fingerprints)>1:raise ValueError('Different runtime or frozen tokenizer releases must not be pooled')
def save(stem,data):
 (a.root/(stem+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 if data:
  with (a.root/(stem+'.csv')).open('w',newline='') as out:
   writer=csv.DictWriter(out,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
save('latex_results',rows);save('cycle_results',cycles)
groups=defaultdict(list)
for r in rows:groups[r['method']].append(r)
summary={}
for name,items in groups.items():
 n=len(items);errors=[r['coefficient_relative_error'] for r in items if r['coefficient_relative_error'] is not None]
 summary[name]=dict(runs=n,top1_exact_count=sum(r['top1']=='correct' for r in items),
    top1_exact_rate=sum(r['top1']=='correct' for r in items)/n,
    top1_exact_or_redundant_rate=sum(r['top1'] in ('correct','redundant') for r in items)/n,
    top10_exact_rate=sum(r['top10_correct'] for r in items)/n,
    no_evaluable_candidate_count=sum(r['top1']=='no_evaluable_candidate' for r in items),
    evaluable_rate=sum(r['evaluated'] for r in items)/sum(r['attempts'] for r in items),
    pinn_statuses=dict(Counter(r['pinn_status'] for r in items)),coefficient_error_n=len(errors),
    coefficient_error_mean=sum(errors)/len(errors) if errors else None,
    train_hours=sum(r['train_seconds'] for r in items)/3600,search_hours=sum(r['search_seconds'] for r in items)/3600,
    pinn_hours=sum(r['pinn_seconds'] for r in items)/3600)
paired=[]
bysetting=defaultdict(dict)
for r in rows:bysetting[(r['equation'],r['seed'])][r['method']]=r
for name in ('ast_bpe','soft_ast_bpe','eqgpt_reference'):
 differences=[int(v[name]['top1']=='correct')-int(v['bpe']['top1']=='correct') for v in bysetting.values()]
 if differences:paired.append(dict(method=name,reference='bpe',pairs=len(differences),
     exact_success_difference=sum(differences)/len(differences),wins=sum(d>0 for d in differences),losses=sum(d<0 for d in differences),ties=sum(d==0 for d in differences)))
plan=a.root/'plan.json';expected=len(json.loads(plan.read_text())['jobs']) if plan.exists() else None
costfile=a.root/'scheduler_walltimes.json';cost=json.loads(costfile.read_text()) if costfile.exists() else []
(a.root/'summary.json').write_text(json.dumps(dict(completed_paired_jobs=len(seen),expected_paired_jobs=expected,
    complete=(len(seen)==expected) if expected is not None else None,methods=summary,paired_differences=paired,
    recorded_scheduler_wall_hours=sum(c['seconds'] for c in cost)/3600,
    cost_note='Scheduler wall time includes shared surrogate and completed invocations; platform opening/installation/idle time is additional. Per-model stage time excludes the one shared surrogate.'),indent=2)+'\n')
print('Reported model runs:',len(rows),'completed paired jobs:',len(seen),'expected:',expected,'smoke included:',a.include_smoke)
