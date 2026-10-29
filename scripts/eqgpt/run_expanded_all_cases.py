"""Run shared expanded pretraining then eight cache-only case searches.

Status and log files make the long process inspectable. A failure stops the
sequence; no smaller budget or fallback dataset is substituted automatically.
"""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from math_tokenizer.eqgpt.expanded_preflight import verified_preflight,verified_settings


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');ap.add_argument('--output',type=Path,default=ROOT/'experiments/expanded_downstream_80_v1');args=ap.parse_args()
 corpus=ROOT/'data/formulas/expanded_downstream_80_v1'
 manifest,_=verified_preflight(corpus);cfg=verified_settings(corpus,manifest)
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
 status=dict(pid=os.getpid(),device=args.device,source_records=manifest['source_records'],samples=manifest['augmented_samples'],settings=cfg,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_stages=[])
 def save():
  tmp=out/'status.json.tmp';tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');tmp.replace(out/'status.json')
 stages=[('pretrain',None)]+[('search',case) for case in SHARED8_CASES]
 for phase,case in stages:
  stage=phase if case is None else phase+'_'+case;path=logs/(stage+'.log')
  status.update(status='RUNNING',stage=stage,log=str(path));save()
  cmd=[sys.executable,str(ROOT/'scripts/eqgpt/expanded_fast_reward_case.py'),'--phase',phase,'--device',args.device,'--output',str(out)]
  if case:cmd+=['--case',case]
  with path.open('a',buffering=1) as f:
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   status['child_pid']=child.pid;save();returncode=child.wait()
  if returncode:
   status.update(status='FAILED',returncode=returncode,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save();raise SystemExit(returncode)
  status['completed_stages'].append(stage);save()
 status.update(status='COMPLETE',stage=None,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
 print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
