"""Copy complete local fast reward experiments into an auditable release tree."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from math_tokenizer.corpus import ROOT

RUNS={
    'Wave_equation':'eqgpt_fast_reward_20261022',
    'Convection_diffusion_equation':'eqgpt_fast_reward_Convection_diffusion_equation_20261022',
    'KG_equation':'eqgpt_fast_reward_KG_equation_20261022',
    'Burgers_equation':'eqgpt_fast_reward_burgers_20261022',
    'Eq_6_2_12':'eqgpt_fast_reward_Eq_6_2_12_20261022',
    'PDE_divide':'eqgpt_fast_reward_PDE_divide_20261022',
    'KdV_equation':'eqgpt_fast_reward_KdV_equation_20261022',
    'Chaffee_Infante_equation':'eqgpt_fast_reward_chaffee_20261022',
}
CASE_LABELS={
    'Wave_equation':'Wave (simple)',
    'Convection_diffusion_equation':'Convection-diffusion (simple)',
    'KG_equation':'Klein-Gordon (medium)',
    'Burgers_equation':'Burgers (medium)',
    'Eq_6_2_12':'Eq. 6.2.12 (medium)',
    'PDE_divide':'PDE divide (medium)',
    'KdV_equation':'KdV (complex)',
    'Chaffee_Infante_equation':'Chafee-Infante (complex)',
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def sync_tree(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for item in source.rglob('*'):
        relative=item.relative_to(source);destination=target/relative
        if item.is_dir():destination.mkdir(parents=True,exist_ok=True)
        elif (not destination.exists() or item.stat().st_size!=destination.stat().st_size
              or item.stat().st_mtime_ns!=destination.stat().st_mtime_ns):
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item,destination)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True)
    dest=p.parse_args().destination.resolve()
    cases=dest/'cases';logs=dest/'logs';design=dest/'design'
    cases.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True);design.mkdir(exist_ok=True)
    copied=[]
    for case,dirname in RUNS.items():
        source=ROOT/'experiments'/dirname
        complete=source/'complete.json'
        assert complete.exists(),f'Incomplete case: {case}'
        record=json.loads(complete.read_text());assert record['case']==case and len(record['results'])==7
        target=cases/case
        sync_tree(source,target)
        log=ROOT/'experiments'/f'{dirname}.log'
        assert log.exists(),f'Missing console log: {case}'
        shutil.copy2(log,logs/f'{case}.log')
        copied.append(case)
    summary=ROOT/'reports/eqgpt_fast_reward_20261022'
    command=['.venv/bin/python','scripts/eqgpt/summarize_fast_reward.py']
    subprocess.run(command,cwd=ROOT,check=True)
    design.mkdir(parents=True,exist_ok=True)
    for file in summary.iterdir():
        if file.is_file():shutil.copy2(file,design/file.name)
    script_root=dest.parent.parent
    source_files=[
        'scripts/eqgpt/fast_reward_case.py','scripts/eqgpt/summarize_fast_reward.py',
        'scripts/eqgpt/package_fast_reward_release.py',
        'src/math_tokenizer/eqgpt/eqgpt_fast_config.py',
        'src/math_tokenizer/eqgpt/eqgpt_fast_tokenizers.py',
        'src/math_tokenizer/eqgpt/author.py','src/math_tokenizer/eqgpt/data.py',
        'src/math_tokenizer/eqgpt/models.py','src/math_tokenizer/eqgpt/round1_runtime.py',
    ]
    for relative in source_files:
        origin=ROOT/relative;target=script_root/relative
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target)
    requirements=ROOT/'requirements-eqgpt.lock.txt'
    if requirements.exists():shutil.copy2(requirements,design/'requirements-eqgpt.lock.txt')
    import torch
    environment=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        torch=torch.__version__,mps_available=torch.backends.mps.is_available(),
        packages={name:importlib.metadata.version(name) for name in
                  ('numpy','pandas','scipy','tokenizers','sympy','tqdm')},
        source_repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        private_repository='https://github.com/xtr1976536/math-tokenizer-lab',
        run_device='Apple MPS')
    (design/'runtime_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    frozen={'status':'complete','cases':copied,
        'protocol_sha256':sha(design/'summary.json'),
        'report_sha256':sha(design/'report.md'),
        'runtime_files':source_files,
        'runtime_sha256':{relative:sha(script_root/relative) for relative in source_files},
        'lfs_patterns':['*.pt'],
        'visibility':'repository must remain private'}
    (design/'release_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
    inventory=[]
    for file in sorted(dest.rglob('*')):
        if file.is_file() and file.name!='SHA256SUMS.json':
            inventory.append(dict(path=str(file.relative_to(dest)),bytes=file.stat().st_size,sha256=sha(file)))
    (dest/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(copied),files=len(inventory),destination=str(dest)),indent=2))


if __name__=='__main__':main()
