"""Build two independent, private, manually submitted Kaggle experiment ZIPs."""
from __future__ import annotations
import gzip,hashlib,json,shutil,subprocess,tarfile,zipfile
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.downstream_gpt2.kaggle_package import bundle,_script
from math_tokenizer.downstream_gpt2.phases import PHASES,SEEDS,LORA,PROTOCOL
from math_tokenizer.downstream_gpt2.adapters import kernel_slug

OUT=ROOT/'deliverables'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def trim_models(project,phase):
    p=project/'models/registry.json';registry=json.loads(p.read_text());methods=PHASES[phase]['trained_methods']
    entries={m:[e for e in registry['models'][m] if e['scenario']=='pretrained' or
                (e['scenario']=='broad' and e['capacity']==2048)] for m in methods}
    keep={'models/registry.json'}
    for m,values in entries.items():
        if len(values)!=1:raise RuntimeError((m,values))
        for e in values:
            if e.get('path'):keep.add(e['path'])
        if m=='meta-llama/Meta-Llama-3-8B':
            keep.add('models/baselines/llama3_provenance.json')
            keep.update(str(f.relative_to(project)) for f in (project/'models/baselines/Meta-Llama-3-8B').rglob('*') if f.is_file())
    for f in (project/'models').rglob('*'):
        if f.is_file() and str(f.relative_to(project)) not in keep:f.unlink()
    for folder in sorted((project/'models').rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):folder.rmdir()
    registry['models']=entries;registry['baselines']=list(methods[2:]);registry['release']=phase+'-downstream-v2'
    registry['capacities']=[2048];registry['scenarios']=['broad']
    p.write_text(json.dumps(registry,indent=2)+'\n')
    return keep

def make_job(root,phase,job,method=None):
    prefix='math-gpt2-'+phase
    suffix=job+('-'+kernel_slug(method) if method else '')
    slug=prefix+'-'+suffix
    folder=root/'kernels'/job/(kernel_slug(method) if method else '')
    folder.mkdir(parents=True,exist_ok=True)
    script=_script(job,method).replace('from math_tokenizer.downstream_gpt2.kaggle_runtime import main',
                                      'from math_tokenizer.downstream_gpt2.phase_runtime import main')
    script=script[:script.rfind('main(')]+f'main({phase!r},{job!r},{method!r})\n'
    if phase=='phase2':script=script.replace('"openpyxl==3.1.5"','"openpyxl==3.1.5","peft==0.21.2"')
    sources=[]
    if job in ('train','evaluate'):sources.append('xiaotongrenkk/'+prefix+'-data')
    if job=='evaluate' and method!='gpt2_native':sources.append('xiaotongrenkk/'+prefix+'-train-'+kernel_slug(method))
    if job=='aggregate':sources=['xiaotongrenkk/'+prefix+'-evaluate-'+kernel_slug(m) for m in PHASES[phase]['methods']]
    metadata={'id':'xiaotongrenkk/'+slug,'title':slug.replace('-',' '),'code_file':'run.py','language':'python',
              'kernel_type':'script','is_private':'true','enable_gpu':'true' if job in ('smoke','train','evaluate') else 'false',
              'enable_internet':'true','machine_shape':'','dataset_sources':['xiaotongrenkk/'+prefix+'-assets'],
              'kernel_sources':sources,'competition_sources':[],'model_sources':[]}
    (folder/'run.py').write_text(script);(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

def build(phase):
    name='math_gpt2_phase1_full_vs_zero_shot' if phase=='phase1' else 'math_gpt2_phase2_lora_seven'
    root=OUT/name
    if root.exists():shutil.rmtree(root)
    root.mkdir(parents=True)
    staging=ROOT/'tmp/two_phase_packages'/phase/'math_tokenizer_project'
    if staging.exists():shutil.rmtree(staging)
    shutil.copytree(ROOT/'tmp/kaggle_downstream_gpt2/project/math_tokenizer_project',staging)
    keep=trim_models(staging,phase)
    spec={**PHASES[phase],'phase':phase,'seeds':SEEDS,'protocol':PROTOCOL,'lora':LORA if phase=='phase2' else None}
    (staging/'phase_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    code_paths=[staging/'src/math_tokenizer/downstream_gpt2'/n for n in ('phases.py','train.py','phase_results.py','data.py','orchestrate.py','verify.py')]
    spec['shared_protocol_source_sha256']={p.name:sha(p) for p in code_paths}
    dataset=root/'dataset';dataset.mkdir()
    with tarfile.open(dataset/'project_bundle.tar.gz','w:gz',compresslevel=1) as tar:
        tar.add(staging,arcname='math_tokenizer_project')
    (dataset/'dataset-metadata.json').write_text(json.dumps({'id':'xiaotongrenkk/math-gpt2-'+phase+'-assets',
        'title':'Math GPT2 '+phase+' private assets','licenses':[{'name':'other'}],
        'description':'Private academic assets: pinned GPT-2, CC BY-NC 4.0 SymbolicMathematics generator, project tokenizer checkpoints; corpus SHA-256 hashes only.'},indent=2)+'\n')
    jobs=[make_job(root,phase,'smoke'),make_job(root,phase,'data')]
    jobs += [make_job(root,phase,'train',m) for m in PHASES[phase]['trained_methods']]
    jobs += [make_job(root,phase,'evaluate',m) for m in PHASES[phase]['methods']]
    jobs += [make_job(root,phase,'aggregate')]
    audit=root/'audits';audit.mkdir()
    shutil.copy2(ROOT/'tmp/two_phase_review/review.json',audit/'four_round_review.json')
    if phase=='phase1':
        timing=ROOT/'tmp/phase1_timing/report.json'
        if timing.exists():shutil.copy2(timing,audit/'phase1_timing.json')
    pilot=root/'pilot_data';shutil.copytree(ROOT/'tmp/two_phase_review/data',pilot)
    native_note=('Original GPT-2 is evaluated ONCE with its unchanged 50,257-ID vocabulary and unchanged pretrained weights. '
                 'There is no native GPT-2 training job. It is an untrained reference, so improvement over this row includes task training, '
                 'and is not evidence of tokenizer-only gain.' if phase=='phase1' else
                 'Only the seven formal tokenizer methods are evaluated. Official PEFT LoRA rank 8/alpha 16/dropout .05 adapts GPT-2 attention and MLP projections. '
                 'New tied vocabulary embeddings are fully trainable; every other pretrained parameter is frozen and checked by SHA-256.')
    readme=f'''# {name}

Self-contained manual Kaggle package, phase `{phase}`. Local-only smoke checks are in `audits/`.
Methods: {', '.join(PHASES[phase]['methods'])}.
Trained methods: {', '.join(PHASES[phase]['trained_methods'])}; mode `{PHASES[phase]['mode']}`;
**2 epochs**, effective batch 32, one training seed: 11. Full training starts from the same pinned PRETRAINED GPT-2, not random initialization.

{native_note}

Both packages use identical raw mathematical tasks, data-generator seeds, template-disjoint splits,
readable text prompts, raw-byte eligibility, generation caps, first-line answer extraction, symbolic verifier,
greedy/beam-10 evaluation and OOD tests. No new learned task marker is sent to the zero-shot reference.
The full task dataset is NOT pre-generated locally. The pinned source generator and GPT-2 weights are bundled;
the CPU `kernels/data` job generates and freezes the full dataset without fetching the paper's multi-GB archives.
`pilot_data/` contains only the small audited smoke examples (8 eligible training examples); never use it for reported results.
All data-generation and model source licenses are retained in the resource bundle. Internet is needed for pinned Python packages only.

## Manual submission order

1. Upload `dataset/` as a PRIVATE Kaggle dataset with the supplied metadata, or use:
   `kaggle datasets create -p dataset -t` (use `datasets version` if the dataset already exists).
2. Submit `kaggle kernels push -p kernels/smoke`. Check logs and successful completion.
3. Submit `kaggle kernels push -p kernels/data`; wait for the CPU job and audits to complete.
4. Submit each directory under `kernels/train/`. Complete all trained methods with seed 11 before test jobs.
5. Submit every directory under `kernels/evaluate/` (including the zero-shot native row in phase1), then `kernels/aggregate`.
6. Download `final_results.json` and `metrics.csv` from the aggregate job. Check input/suite hashes before interpretation.

The supplied account is `xiaotongrenkk`. Every kernel is private. No upload/run is triggered by extracting this ZIP.
Kaggle can automatically unpack archives and use nested `/kaggle/input/datasets/...` paths; bootstrap code discovers
the asset manifest and checks the GPT-2 weight hash. Training checkpoints and data are mounted from prior kernel outputs.
The task ODE generator is the explicitly documented linear-in-constant subfamily of Lample–Charton, not the original paper distribution.

Stage1 is a preliminary capability experiment; stage2 provides an equal adaptation protocol among the seven systems.
Large pretrained tokenizer vocabularies have different embedding parameter counts and softmax costs: they are deployment references,
not equal-total-parameter controls. LoRA reduces backbone optimizer updates but does not remove vocabulary projection or evaluation costs.
With one training seed per method, the results cannot estimate variation across training seeds; paired intervals describe question-group uncertainty only.
The local 2026-10-14 budget test is recorded in `audits/phase1_timing.json` in Phase 1. It did not establish a
4–5-hour end-to-end Kaggle runtime; this bundle has been prepared for manual submission, not automatically run.
'''
    (root/'README.md').write_text(readme)
    spec['resource_sha256']=sha(dataset/'project_bundle.tar.gz');spec['resource_bytes']=(dataset/'project_bundle.tar.gz').stat().st_size
    spec['repository_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (root/'EXPERIMENT.json').write_text(json.dumps(spec,indent=2)+'\n')
    (root/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    archive=OUT/(name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,arcname=str(Path(name)/p.relative_to(root)))
    return {'phase':phase,'zip':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'jobs':len(jobs),'model_asset_paths':sorted(keep)}

def main():
    OUT.mkdir(exist_ok=True);bundle()
    reports=[build(p) for p in PHASES]
    (OUT/'two_phase_manifest.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
