"""Read-only reaggregation and route inventory; never trains/modifies methods/data."""
import json,hashlib,subprocess
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash

RUNS=['pilot01','sensitivity512','v2_release512','v2_release1024','v3_512_release','v3_1024_release','v4_512','v4_1024']

def audit_run(name):
 out=ROOT/'results'/name;metrics=json.loads((out/'metrics.json').read_text());rows=read_jsonl(out/'per_formula.jsonl');by=defaultdict(list)
 for r in rows:by[(r['method'],r['seed'])].append(r)
 records=0;byte_lengths=defaultdict(set)
 for row in rows:byte_lengths[row['id']].add(row['bytes'])
 assert all(len(values)==1 for values in byte_lengths.values())
 for metric in metrics:
  rr=by[(metric['method'],metric['seed'])];records+=len(rr)
  assert len(rr)==metric['n_formulas']
  assert len({r['id'] for r in rr})==len(rr)
  assert sum(r['tokens'] for r in rr)==metric['tokens']
  assert sum(r['bytes'] for r in rr)==metric['bytes']
  assert abs(metric['tokens_per_formula']-metric['tokens']/len(rr))<1e-12
  assert abs(metric['roundtrip_rate']-sum(r['roundtrip'] for r in rr)/len(rr))<1e-12
 assert records==len(rows)
 ids={key:frozenset(r['id'] for r in rr) for key,rr in by.items()};first=next(iter(ids.values()));assert all(i==first for i in ids.values())
 grouped=defaultdict(list)
 for m in metrics:grouped[m['method']].append(m)
 output={}
 for method,rr in grouped.items():
  summary={'seeds':[r['seed'] for r in rr],'formulas':rr[0]['n_formulas']}
  for k in ('tokens_per_formula','bytes_per_token','roundtrip_rate','command_integrity_rate','number_integrity_rate','lexeme_preservation_rate','hybrid_tree_crossing_rate','encode_mib_s','vocab_actual','train_seconds'):
   if all(k in r and r[k] is not None for r in rr):summary[k]=float(np.mean([r[k] for r in rr]))
  output[method]=summary
 return {'passed':True,'runs':len(metrics),'records':len(rows),'test_id_count':len(first),'test_ids_sha256':hashlib.sha256('\n'.join(sorted(first)).encode()).hexdigest(),
         'artifact_hashes':{p.name:file_hash(p) for p in [out/'metrics.json',out/'per_formula.jsonl',out/'manifest.json']},'methods':output}


def policy_profile(cap):
 path=ROOT/f'results/v4_{cap}/models/rl_v4-17.json';d=json.loads(path.read_text())
 from math_tokenizer.rl_v4 import MaskedPolicy,MaskedConfig
 from math_tokenizer.constrained_v4 import ConstrainedTokenizer
 log=d['training_log'][-1];config=MaskedConfig(**log['config']);trainer=MaskedPolicy([bytes.fromhex(x) for x in d['vocab_hex']],config)
 trainer.model.frequency=np.array(d['frequency']);trainer.model.logfreq=np.log(trainer.model.frequency)
 trained=np.array(d['theta']);initial=np.zeros(7);initial[0]=config.initial_length_prior
 texts=[r['latex'] for r in read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl')];selected=sorted(set(texts),key=lambda t:hashlib.sha256(t.encode()).digest())[:config.max_train_formulas]
 graphs=[trainer.graph(t) for t in selected];results={};branchstates=0;totalstates=0
 for graph in graphs:
  for g in graph:
   if g is not None:totalstates+=1;branchstates+=len(g[0])>1
 for name,theta in [('matched_initial',initial),('trained',trained)]:
  trainer.model.theta=theta
  values=[trainer.expected_gradient(g)[0] for g in graphs]
  results[name]={'mean_expected_tokens_on_identical_training_subset':float(np.mean(values)),
                 'document_normalized_expected_cost':float(np.mean([v/max(1,len(g)) for v,g in zip(values,graphs)]))}
 actual=ConstrainedTokenizer.load(path);initmodel=ConstrainedTokenizer(actual.vocab,True,actual.atomic,actual.inference,initial,actual.frequency)
 dev=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
 changes=sum(actual.encode(r['latex'])!=initmodel.encode(r['latex']) for r in dev)
 return {'capacity':cap,'train_subset_count':len(selected),'policy_parameter_count':len(trained),'policy_kind':'shared global linear action scorer, not neural context encoder',
         'learned_vocab_by_RL':False,'teacher_supervision':False,'training_config':log['config'],'deployment_mode':d['inference'],
         'branching_states':branchstates,'all_candidate_states':totalstates,'profile':results,'dev_segmentations_changed_vs_matched_initial':changes,'dev_formulas':len(dev),
         'important':'fixed before/after policies measured on same frozen train subset; training log first epoch is not exact initial checkpoint objective'}


def main():
 current=file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl');run_audits={name:audit_run(name) for name in RUNS}
 for cap in (512,1024):
  runs=[run_audits[f'v2_release{cap}'],run_audits[f'v3_{cap}_release'],run_audits[f'v4_{cap}']]
  assert len({r['test_ids_sha256'] for r in runs})==1
  for name in [f'v2_release{cap}',f'v3_{cap}_release',f'v4_{cap}']:
   p=json.loads((ROOT/'results'/name/'protocol.json').read_text());assert p.get('data_sha256',p.get('corpus_sha256'))==current
 result={'review_type':'retrospective_only_no_model_or_corpus_changes','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'frozen_v2_corpus_sha256':current,'formal_runs':run_audits,'v4_learning_vs_initial_profile':{str(cap):policy_profile(cap) for cap in (512,1024)},
         'invalidated_or_incomplete_excluded':['v2_512','v2_512_final','v2_1024_final','v3_512'],
         'artifact_count_corrections':{str(cap):{'v4_dev_rows':len(json.loads((ROOT/f'results/v4_{cap}/dev_ablations.json').read_text())),
             'v4_unique_policy_train_runs':30,'v3_dev_rows':len(json.loads((ROOT/f'results/v3_{cap}_release/dev_ablations.json').read_text())),
             'v3_unique_policy_train_runs':24} for cap in (512,1024)},
         'verification_scope':'aggregate results independently from stored per_formula records, inspect saved model policies and fixed initial/final train expectations; not full training rerun'}
 dest=ROOT/'results/retrospective/evidence.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Audited',len(run_audits),'formal run directories,',sum(r['records'] for r in run_audits.values()),'records')
 print(json.dumps(result['v4_learning_vs_initial_profile'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
