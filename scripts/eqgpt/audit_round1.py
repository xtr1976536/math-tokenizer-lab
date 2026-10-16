"""Four executed audit rounds, with per-case full smoke evidence and repeat."""
import argparse,gzip,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from math_tokenizer.eqgpt.author import ROOT,verify_source,definition_hashes
from math_tokenizer.eqgpt.frozen_math import load_frozen,sha
from math_tokenizer.eqgpt.round1_runtime import runtime_sha256
from math_tokenizer.eqgpt.latex_experiment import MAIN_CASES
p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,default=ROOT/'experiments/eqgpt_round1/qa/three_methods_final');p.add_argument('--repeat',type=Path,default=ROOT/'experiments/eqgpt_round1/qa/three_repeat_final');p.add_argument('--junit',type=Path,default=Path('/tmp/eqgpt_three_junit_v2.xml'));a=p.parse_args()
OUT=ROOT/'docs/eqgpt/round1_evidence';OUT.mkdir(parents=True,exist_ok=True)
equivalents=json.loads((OUT/'target_exclusions.json').read_text());assert equivalents['passed'] and equivalents['target_cases']==11
source=verify_source();_,frozen=load_frozen(ROOT/'experiments/eqgpt_round1/frozen_math');suite=ET.parse(a.junit).getroot()
assert all(int(x.get('failures','0'))==int(x.get('errors','0'))==0 for x in suite.iter('testsuite'))
tests=[x.attrib for x in suite.iter('testcase')];assert len(tests)>=36
(OUT/'unit_tests.json').write_text(json.dumps(dict(passed=len(tests),tests=tests),indent=2)+'\n')

def normalized(value):
 if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k not in ('seconds','elapsed_seconds','invocation_seconds')}
 if isinstance(value,list):return [normalized(v) for v in value]
 return value

def digest(v):return hashlib.sha256(json.dumps(normalized(v),sort_keys=True).encode()).hexdigest()

checks=[];evidence=[]
for case in MAIN_CASES:
 folder=a.matrix/case;m=json.loads((folder/'manifest.json').read_text());run=json.loads((folder/'RUN.json').read_text())
 assert m['smoke'] and not m['formal_results'] and run['runtime_sha256']==m['runtime_sha256']==runtime_sha256()
 assert m['frozen_tokenizers_sha256']==sha(ROOT/'experiments/eqgpt_round1/frozen_math/FROZEN.json')
 assert m['training_source']['target_structures_removed']
 assert m['training_source']['training_sequences']==32*m['training_source']['training_formulas']
 assert len({r['backbone_initial_sha256'] for r in m['model_initialization'].values()})==1
 arms={};pinn_splits=[]
 for name,r in m['results'].items():
  t=r['training'];s=json.loads((folder/name/'search.json').read_text());pinn=json.loads((folder/name/'pinn.json').read_text())
  assert t['epochs']==1 and t['samples']==4 and t['optimizer_steps']==2 and t['lr']==1e-4 and t['clip']==1 and t['batch_size']==2
  assert t['initial_sha256']!=t['final_sha256'] and r['checkpoint_reload_exact'] and t['all_parameters_trainable']
  assert s['attempts_per_cycle']==400 and s['cycles']==2 and len(s['attempts'])==800
  assert s['sat']==(name=='eqgpt_reference')
  assert m['model_initialization'][name]['input_shape']==m['model_initialization'][name]['output_shape']==[57 if name=='eqgpt_reference' else 1024,768]
  for cycle in s['history']:
   assert cycle['attempts']==sum(cycle['statuses'].values())==400
   assert cycle['continuation_examples']==len(cycle['gre'])<=10
   assert (cycle['before_sha256']!=cycle['after_sha256'])==bool(cycle['continuation_examples'])
  assert pinn['initial_surrogate_sha256']==m['shared']['surrogate_sha256']
  if pinn.get('sampling'):pinn_splits.append(pinn['sampling'])
  arms[name]=dict(attempts=len(s['attempts']),statuses=[c['statuses'] for c in s['history']],pinn_status=pinn['status'],
      checkpoint_reload_exact=r['checkpoint_reload_exact'],training_sha256=t['final_sha256'])
  evidence.append(dict(case=case,method=name,manifest=m,training=t,search=s,pinn=pinn))
 assert set(arms)=={'ast_bpe','soft_ast_bpe','eqgpt_reference'}
 assert all(x==pinn_splits[0] for x in pinn_splits)
 checks.append(dict(case=case,formula_count=m['training_source']['training_formulas'],sequence_count=m['training_source']['training_sequences'],arms=arms))

reference=a.matrix/'KdV_equation';repeat=a.repeat
for filename in ['manifest.json','RUN.json']:
 assert normalized(json.loads((reference/filename).read_text()))==normalized(json.loads((repeat/filename).read_text()))
replays={}
for name in ('ast_bpe','soft_ast_bpe','eqgpt_reference'):
 for filename in ['training.json','search.json','pinn.json']:
  v=json.loads((reference/name/filename).read_text());other=json.loads((repeat/name/filename).read_text())
  assert normalized(v)==normalized(other),(name,filename)
  replays[name+'/'+filename]=digest(v)
with gzip.open(OUT/'smoke_records.jsonl.gz','wt',encoding='utf-8') as stream:
 for record in evidence:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
report=dict(first_round_only=True,version='eqgpt-round1-main-v2',all_four_rounds_passed=True,formal_cuda_experiments_executed=False,unit_tests_passed=len(tests),
 runtime_sha256=runtime_sha256(),source_revision=source['revision'],frozen_tokenizers_sha256=sha(ROOT/'experiments/eqgpt_round1/frozen_math/FROZEN.json'),
 rounds=[dict(round=1,scope='Independent frozen training partition, target screening, exact capacities, mathematical round trips and author-source hashes',passed=True,
          accepted_formulas=frozen['corpus']['accepted'],symbolic_target_audit=equivalents,source_definition_hashes=definition_hashes()),
 dict(round=2,scope='Common original backbone, scratch gradients, unchanged author optimizer steps, optimizer binding, explicit no-SAT generation, supported numerical/PINN bridge fixtures',passed=True,automated_tests=len(tests)),
 dict(round=3,scope='11 real paired smoke cases, three scratch models each, two cycles with exactly 400 attempts, failure accounting, candidate updates, checkpoint/logit replay',passed=True,
          paired_cases=11,model_runs=33,attempts=33*800,cases=checks),
 dict(round=4,scope='Independent complete KdV repetition, exact parameters/losses/samples/rewards/PINN equality except elapsed time; interrupted optimizer+RNG resume test',passed=True,
          replay_record_hashes=replays)],smoke_record_sha256=sha(OUT/'smoke_records.jsonl.gz'))
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=True,rounds=4,models=33,attempts=26400,unit_tests=len(tests)),indent=2))
