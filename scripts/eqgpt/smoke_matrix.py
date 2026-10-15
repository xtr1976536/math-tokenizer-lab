"""Run the same reduced author pipeline on every canonical released case."""
import argparse,json,subprocess,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CASES=['KdV_equation','Burgers_equation','Wave_equation','Chaffee_Infante_equation',
       'KG_equation','Allen_Cahn','Convection_diffusion_equation','PDE_divide',
       'Eq_6_2_12','PDE_compound','Possion_x_y']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/eqgpt_reproduction/matrix');p.add_argument('--cases',nargs='*',default=CASES);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);records=[]
for case in a.cases:
 folder=a.output/case;log=a.output/f'{case}.log';start=time.perf_counter()
 with log.open('w') as out:
  result=subprocess.run([sys.executable,str(ROOT/'scripts/eqgpt/run_experiment.py'),'--smoke','--equation',case,'--output',str(folder)],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),stdout=out,stderr=subprocess.STDOUT)
 record=dict(case=case,exit_code=result.returncode,seconds=time.perf_counter()-start,manifest=str(folder/'manifest.json'))
 records.append(record);print(record,flush=True)
 # Checkpoint reload is already verified in the child; retain lightweight
 # audit evidence rather than many gigabytes of redundant smoke weights.
 if result.returncode==0:
  for f in folder.rglob('*'):
   if f.suffix in ('.pt','.pkl'):f.unlink()
 (a.output/'matrix.json').write_text(json.dumps(records,indent=2)+'\n')
if any(x['exit_code'] for x in records):raise SystemExit(1)
