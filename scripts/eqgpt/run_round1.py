"""Run only the approved first-round main experiment, with safe stage resume."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from math_tokenizer.eqgpt.author import ROOT
from math_tokenizer.eqgpt.frozen_math import load_frozen,sha
from math_tokenizer.eqgpt.round1_runtime import runtime_sha256,verify_completed
CASES=['KdV_equation','Burgers_equation','Wave_equation','Chaffee_Infante_equation','KG_equation','Allen_Cahn','Convection_diffusion_equation','PDE_divide','Eq_6_2_12','PDE_compound','Possion_x_y']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--tokenizer-dir',type=Path,default=ROOT/'experiments/eqgpt_round1/frozen_math')
p.add_argument('--seeds',nargs='+',type=int,default=[525,526,527]);p.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
p.add_argument('--plan-only',action='store_true');a=p.parse_args()
if len(a.seeds)!=len(set(a.seeds)) or len(a.cases)!=len(set(a.cases)):raise ValueError('Repeated seeds/cases are not independent runs')
_,frozen=load_frozen(a.tokenizer_dir)
plan=[dict(equation=c,seed=s,observations=30000 if c=='Possion_x_y' else 10000,noise=0,epochs=100,lr=1e-4) for c in a.cases for s in a.seeds]
if a.plan_only:print(json.dumps(dict(round=1,jobs=len(plan),models=len(plan)*4,plan=plan),indent=2));raise SystemExit(0)
env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
subprocess.run([sys.executable,str(ROOT/'scripts/eqgpt/preflight.py'),'--require-cuda'],env=env,check=True)
a.output.mkdir(parents=True,exist_ok=True)
header=dict(round=1,frozen_tokenizers_sha256=sha(a.tokenizer_dir/'FROZEN.json'),runtime_sha256=runtime_sha256(),jobs=plan)
if (a.output/'plan.json').exists():
 if json.loads((a.output/'plan.json').read_text())!=header:raise ValueError('Existing plan differs; do not mix experiments')
else:(a.output/'plan.json').write_text(json.dumps(header,indent=2)+'\n')
for job in plan:
 out=a.output/job['equation']/f"seed{job['seed']}"
 if (out/'manifest.json').exists():
  m=verify_completed(out,header['runtime_sha256'],header['frozen_tokenizers_sha256'])
  assert (m['equation'],m['seed'],m['shared']['observations'])==(job['equation'],job['seed'],job['observations'])
  print('Completed, retained',out,flush=True);continue
 cmd=[sys.executable,str(ROOT/'scripts/eqgpt/run_latex.py'),'--device','cuda','--equation',job['equation'],
      '--seed',str(job['seed']),'--observations',str(job['observations']),'--tokenizer-dir',str(a.tokenizer_dir.resolve()),'--output',str(out)]
 log=out.parent/(out.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 print('Run/resume',job,flush=True)
 started=time.perf_counter();ok=False
 try:
  with log.open('a') as stream:subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  ok=True
 finally:
  path=a.output/'scheduler_walltimes.json';cost=json.loads(path.read_text()) if path.exists() else []
  cost.append(dict(**job,seconds=time.perf_counter()-started,completed_invocation=ok))
  path.write_text(json.dumps(cost,indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'scripts/eqgpt/collect_latex.py'),str(a.output)],env=env,check=True)
