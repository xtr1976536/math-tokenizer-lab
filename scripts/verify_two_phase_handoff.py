"""Verify both ZIPs and execute offline smoke steps from their extracted code."""
import hashlib,json,os,subprocess,sys,tarfile,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()

def main():
    manifest=ROOT/'deliverables/two_phase_manifest.json';items=json.loads(manifest.read_text());reports=[]
    for item in items:
        archive=Path(item['zip']);phase=item['phase'];prefix=archive.stem+'/'
        extraction=ROOT/'tmp/phase_zip_extract'/phase;extraction.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            spec=json.loads(z.read(prefix+'EXPERIMENT.json'))
            jobs=json.loads(z.read(prefix+'jobs.json'))
            assert len(jobs)==(8 if phase=='phase1' else 17)
            assert not any('/train-gpt2-native' in j['id'] for j in jobs)
            with z.open(prefix+'dataset/project_bundle.tar.gz') as f:
                with tarfile.open(fileobj=f,mode='r|gz') as tar:
                    names=[];files=[]
                    for member in tar:
                        names.append(member.name);target=extraction/member.name
                        if member.isdir():target.mkdir(parents=True,exist_ok=True)
                        elif member.isfile():
                            files.append(member.name)
                            target.parent.mkdir(parents=True,exist_ok=True)
                            with target.open('wb') as out:
                                source=tar.extractfile(member)
                                for c in iter(lambda:source.read(4*1024*1024),b''):out.write(c)
            assert not any('formula_only' in n or 'kaggle.json' in n for n in names)
            if phase=='phase1':assert not any('/models/baselines/' in n for n in files)
        project=extraction/'math_tokenizer_project'
        registry=json.loads((project/'models/registry.json').read_text())
        assert set(registry['models'])==set(spec['trained_methods'])
        env=dict(os.environ,PYTHONPATH=str(project/'src'),TIKTOKEN_CACHE_DIR=str(project/'tiktoken_cache'),TASK_PHASE=phase,TASK_PROJECT=str(project),TASK_OUTPUT=str(extraction/'smoke'))
        script='''import os,json,hashlib
from pathlib import Path
from math_tokenizer.downstream_gpt2.data import generate
from math_tokenizer.downstream_gpt2.orchestrate import assemble
from math_tokenizer.downstream_gpt2.audit import dataset_audit
from math_tokenizer.downstream_gpt2.phase_runtime import audit_adapters
from math_tokenizer.downstream_gpt2.phases import PHASES,freeze_coverage,PhaseAdapter,load_checkpoint,parameter_digest
from math_tokenizer.downstream_gpt2.train import train_one,generate_one
phase=os.environ['TASK_PHASE'];project=Path(os.environ['TASK_PROJECT']);out=Path(os.environ['TASK_OUTPUT'])
source=project/'tmp/SymbolicMathematics';gpt2=project/'tmp/gpt2-607a30d';raw=out/'raw';data=out/'final'
asset=json.loads((project/'asset_manifest.json').read_text())
assert hashlib.sha256((gpt2/'model.safetensors').read_bytes()).hexdigest()==asset['gpt2_weights_sha256']
generate(source,raw,.001);assemble(raw,raw,raw,data,json.loads((raw/'manifest.json').read_text())['quotas'],source)
audit=dataset_audit(data,source);adapters=audit_adapters(phase,data,gpt2);coverage=freeze_coverage(data)
formal_hashes=dict(coverage['sha256']);coverage['kept']['train']=coverage['kept']['train'][:8]
coverage['sha256']['train']=hashlib.sha256('\\n'.join(coverage['kept']['train']).encode()).hexdigest()
(data/'coverage_512.json').write_text(json.dumps(coverage))
path=train_one('ast_boundary',11,data,gpt2,out/'runs',epochs=1,microbatch=2,effective_batch=4,eval_limit=5,
               device_name='mps',mode=PHASES[phase]['mode'],prompt_style='text-v2')
model=load_checkpoint('ast_boundary',gpt2,path/'best',out/'init',PHASES[phase]['mode'])
assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
config=json.loads((path/'config.json').read_text())
if phase=='phase2':assert config['frozen_before_sha256']==config['frozen_after_sha256']
report={'phase':phase,'status':'passed','coverage_sha256':formal_hashes,'oracle':audit['oracle_valid_counts'],
        'available_tokenizers':list(adapters),'mode':config['mode'],'trainable_parameters':config['trainable_parameters']}
if phase=='phase1':
 native=load_checkpoint('gpt2_native',gpt2,None,out/'native_init','full').to('mps').eval()
 original=parameter_digest(native);adapter=PhaseAdapter('gpt2_native',gpt2)
 prompt,_=adapter.row('integral','x','x');generate_one(native,adapter,prompt,[],'mps',max_new_tokens=2)
 assert original==parameter_digest(native) and native.config.vocab_size==50257
 report['native_optimizer_steps']=0
(out/'report.json').write_text(json.dumps(report,indent=2))
'''
        subprocess.run([str(ROOT/'.venv/bin/python'),'-c',script],cwd=ROOT,env=env,check=True)
        report=json.loads((extraction/'smoke/report.json').read_text());reports.append(report)
        deliverable=ROOT/'deliverables'/archive.stem/'audits/offline_package_review.json'
        deliverable.write_text(json.dumps(report,indent=2)+'\n')
        with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_STORED) as z:z.write(deliverable,arcname=prefix+'audits/offline_package_review.json')
        item['bytes']=archive.stat().st_size;item['sha256']=sha(archive);item['offline_package_smoke']='passed'
    assert reports[0]['coverage_sha256']==reports[1]['coverage_sha256']
    manifest.write_text(json.dumps(items,indent=2)+'\n')
    (ROOT/'deliverables/two_phase_offline_review.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(items,indent=2))

if __name__=='__main__':main()
