"""Extract the actual ZIP, verify hashes, and smoke-test its isolated source."""
import argparse,hashlib,json,os,subprocess,sys,tempfile,zipfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,default=ROOT/'deliverables/eqgpt_tokenizer_cloud_20261016.zip');a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='eqgpt_bundle_') as tmp:
 with zipfile.ZipFile(a.zip) as z:z.extractall(tmp)
 root=Path(tmp)/'eqgpt_tokenizer_cloud_20261016'
 manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text())
 for rel,entry in manifest.items():
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==entry['sha256']
 env=dict(os.environ,PYTHONPATH=str(root/'src'),TOKENIZERS_PARALLELISM='false')
 commands=[['scripts/eqgpt/preflight.py'],['-m','pytest','tests/test_eqgpt.py','-q'],
           ['scripts/eqgpt/run_experiment.py','--smoke','--output',str(root/'bundle_smoke')]]
 checks=[]
 for i,args in enumerate(commands):
  start=time.perf_counter();result=subprocess.run([sys.executable,*args],cwd=root,env=env,capture_output=True,text=True)
  (ROOT/f'experiments/eqgpt_reproduction/bundle_check_{i+1}.log').write_text(result.stdout+result.stderr)
  checks.append(dict(command=args[:3],exit_code=result.returncode,seconds=time.perf_counter()-start))
  if result.returncode:raise RuntimeError('Bundle validation failed: '+str(args))
 original=json.loads((ROOT/'experiments/eqgpt_reproduction/validated_repeat/manifest.json').read_text())
 copied=json.loads((root/'bundle_smoke/manifest.json').read_text())
 assert original==copied
 result=dict(zip_sha256=hashlib.sha256(a.zip.read_bytes()).hexdigest(),file_hashes_verified=len(manifest),checks=checks,isolated_smoke_matches=True)
 (ROOT/'deliverables/eqgpt_bundle_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
