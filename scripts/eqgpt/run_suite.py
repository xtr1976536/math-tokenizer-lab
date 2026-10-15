"""Run each paired three-arm experiment sequentially, preserving completed runs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--preset',choices=['clean','noise'],default='clean');p.add_argument('--seeds',type=int,nargs='+',default=[525,526,527]);p.add_argument('--output',type=Path,default=ROOT/'experiments/eqgpt_cloud');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
canonical=['KdV_equation','Burgers_equation','Wave_equation','Chaffee_Infante_equation','KG_equation','Allen_Cahn','Convection_diffusion_equation','PDE_divide','Eq_6_2_12','PDE_compound']
settings=[(c,10000,0) for c in canonical]+[('Possion_x_y',30000,0)] if a.preset=='clean' else [('KdV_equation',10000,n) for n in [0,10,20,30,40,50]]
plan=[dict(case=c,observations=n,noise=noise,seed=seed) for c,n,noise in settings for seed in a.seeds]
if a.plan_only:print(json.dumps(plan,indent=2));raise SystemExit(0)
env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
subprocess.run([sys.executable,str(ROOT/'scripts/eqgpt/preflight.py'),'--require-cuda'],env=env,check=True)
a.output.mkdir(parents=True,exist_ok=True)
(a.output/('plan_'+a.preset+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
for item in plan:
 out=a.output/item['case']/f"n{item['observations']}_noise{item['noise']}_seed{item['seed']}"
 if (out/'manifest.json').exists():
  m=json.loads((out/'manifest.json').read_text())
  assert not m['smoke'] and m['seed']==item['seed'] and m['shared']['noise_percent']==item['noise'] and m['shared']['observations']==item['observations']
  print('Completed, retained',out,flush=True);continue
 if out.exists():raise FileExistsError('Partial run retained. Choose a new output root: '+str(out))
 log=out.parent/(out.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(ROOT/'scripts/eqgpt/run_experiment.py'),'--device','cuda','--equation',item['case'],'--observations',str(item['observations']),'--noise',str(item['noise']),'--seed',str(item['seed']),'--output',str(out)]
 print('Running',item,flush=True)
 with log.open('w') as stream:subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/eqgpt/collect_results.py'),str(a.output)],env=env,check=True)
