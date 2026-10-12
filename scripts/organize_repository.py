"""One-time separation of seven active systems from immutable research history."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
from math_tokenizer.corpus import ROOT

NAMES=['ast_boundary','official_atomic_bpe','hf_byte_bpe','sentencepiece_unigram',
       'Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def zipped(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as out:
            for row in rows:out.write((json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n').encode())


def main():
    if (ROOT/'models/registry.json').exists():raise RuntimeError('Repository already organized')
    history=ROOT/'history';history.mkdir(exist_ok=True)
    previous=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    entries=json.loads((ROOT/'results/v8/full/registry.json').read_text())
    selected=[e for e in entries if e['name'] in NAMES]
    assert len(selected)==19
    active=[]
    for original in selected:
        e=dict(original);name=e['name'];role='proposed' if name in NAMES[:2] else 'baseline'
        if e.get('path'):
            src=ROOT/e['path'];suffix=src.suffix
            if e['scenario']=='pretrained':dest=ROOT/'models/baselines'/name.split('/')[-1]/src.name
            else:dest=ROOT/'models'/role/name/f"{e['scenario']}-{e['capacity']}{suffix}"
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            assert sha(src)==sha(dest)==e['checkpoint_sha256'];e['path']=str(dest.relative_to(ROOT))
        e['role']=role;active.append(e)
    by_method={n:[e for e in active if e['name']==n] for n in NAMES}
    write(ROOT/'models/registry.json',{'release':'active-seven-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'proposed':NAMES[:2],'baselines':NAMES[2:],'capacities':[1024,2048],
        'scenarios':['broad','specialized'],'default':'ast_boundary','models':by_method,
        'frozen_experiment_commit':previous,'RL_in_production':False})
    llama=next(r for r in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()) if r['repo']==NAMES[5])
    write(ROOT/'models/baselines/llama3_provenance.json',llama)
    benchmark=ROOT/'data/benchmark';allrows=[];partitions={};source_manifests={}
    for release in ('v6','v7','v8'):
        folder=ROOT/f'data/releases/{release}'
        source=folder/('curated/corpus.jsonl.gz' if release=='v6' else 'external_test.jsonl.gz')
        records=list(map(json.loads,gzip.decompress(source.read_bytes()).decode().splitlines()))
        source_manifests[release]={'archive_sha256':sha(source),'records':len(records)}
        for r in records:
            r['catalog_release']=release
            r['formula_partition']='pde' if r['domain']=='pde' else 'other'
            allrows.append(r)
        if release=='v6':
            splits=json.loads((folder/'splits.json').read_text())
            partitions.update({f'v6_{k}':v for k,v in splits.items()})
        else:
            splits=json.loads((folder/'splits.json').read_text());partitions.update({f'{release}_{k}':v for k,v in splits.items()})
    assert len(allrows)==56000 and len({r['id'] for r in allrows})==56000
    zipped(benchmark/'metadata/formulas.jsonl.gz',allrows)
    write(benchmark/'splits.json',partitions)
    buckets=defaultdict(list)
    for r in allrows:
        path=(Path('formula_only')/('pde' if r['domain']=='pde' else 'other/'+r['domain'])/r['origin']/(r['complexity']['class']+'.jsonl'))
        buckets[str(path)].append(r)
    catalog=[]
    for relative,rs in sorted(buckets.items()):
        dest=benchmark/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r['latex'],ensure_ascii=False)+'\n' for r in rs))
        ids=benchmark/'ids'/relative.removeprefix('formula_only/');ids=ids.with_suffix('.ids.jsonl');ids.parent.mkdir(parents=True,exist_ok=True)
        ids.write_text(''.join(json.dumps(r['id'])+'\n' for r in rs))
        catalog.append({'path':relative,'id_path':str(ids.relative_to(benchmark)),'formulas':len(rs),'sha256':sha(dest)})
    zipped(benchmark/'stress/formulas.jsonl.gz',list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open())))
    manifest={'release':'active-benchmark-v1','formulas':56000,'domain':dict(Counter(r['domain'] for r in allrows)),
        'paper_domain':dict(Counter(r['paper_domain'] for r in allrows)),
        'origin':dict(Counter(r['origin'] for r in allrows)),'complexity':dict(Counter(r['complexity']['class'] for r in allrows)),
        'formula_partition':dict(Counter(r['formula_partition'] for r in allrows)),
        'source_releases':source_manifests,'splits':{k:len(v) for k,v in partitions.items()},'catalog':catalog,
        'metadata_sha256':sha(benchmark/'metadata/formulas.jsonl.gz'),'splits_sha256':sha(benchmark/'splits.json'),
        'content_scope':'Formula strings only; metadata and source provenance are separate. Historical human origin inferred from pre2020 versions.',
        'frozen_splits_preserved':True,'whole_catalog_must_not_be_used_for_training':True,
        'quality_process':'Automated syntax, provenance, macro fidelity, duplicate and serialization checks; no external certification workflow.'}
    write(benchmark/'manifest.json',manifest)
    # Preserve the selected exact experiment records; no reranking on a new test.
    dest=ROOT/'experiments/current';dest.mkdir(parents=True,exist_ok=True)
    metrics=json.loads((ROOT/'results/v8/full/metrics.json').read_text());keys={e['key'] for e in active}
    filtered=[m for m in metrics if m['key'] in keys];write(dest/'metrics.json',filtered);write(dest/'registry.json',active)
    def filtered_rows():
        with gzip.open(ROOT/'results/v8/full/per_formula.jsonl.gz','rt') as stream:
            for row in map(json.loads,stream):
                if row['key'] in keys:yield row
    zipped(dest/'per_formula.jsonl.gz',filtered_rows())
    for name in ('evaluation_protocol.json','first_test_score.json','fairness_audit.json','multimetric_paired_effects.json'):
        shutil.copy2(ROOT/'results/v8/full'/name,dest/('historical_'+name))
    for scenario in ('broad','specialized'):
        shutil.copy2(ROOT/f'results/v7/{scenario}_1024/fairness_protocol.json',dest/f'{scenario}_scope.json')
    # Expert-review material is removed, not presented as completed work.
    removed=[]
    for path in (ROOT/'data/releases').glob('*/review_materials'):
        removed.append(str(path.relative_to(ROOT)));shutil.rmtree(path)
    moves=[]
    for relative in ('results','docs','tests','data/releases','data/curated','data/quarantine','models/current','models/final_intrinsic'):
        src=ROOT/relative
        if src.exists():
            target=history/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(target));moves.append(relative)
    # Old scripts are historical recipes; the original commit remains runnable.
    oldscripts=history/'scripts';oldscripts.mkdir(exist_ok=True)
    for p in (ROOT/'scripts').iterdir():
        if p.is_file() and p.name!='organize_repository.py':shutil.move(str(p),str(oldscripts/p.name))
    (ROOT/'docs').mkdir(exist_ok=True);(ROOT/'tests').mkdir(exist_ok=True)
    write(history/'migration.json',{'base_commit':previous,'moved_directories':moves,'expert_review_material_removed':removed,
        'historical_artifacts_unchanged_except_removed_review_materials':True,'historical_paths_in_logs_refer_to_base_commit':True,
        'active_methods':NAMES,'current_metric_records':len(filtered),'current_formula_rows':sum(m['formulas'] for m in filtered)})
    print('Active methods',len(NAMES),'configs',len(active),'corpus',len(allrows),'records',len(filtered),flush=True)


if __name__=='__main__':main()
