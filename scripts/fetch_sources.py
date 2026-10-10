"""Fetch only pinned small source artifacts, never full manuscript repositories."""
import hashlib,json,subprocess,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw';RAW.mkdir(parents=True,exist_ok=True)

def api(path):
    return json.loads(subprocess.check_output(['gh','api',path]))

def fetch(url,path):
    if not path.exists():
        for attempt in range(4):
            try:
                r=requests.get(url,timeout=60);r.raise_for_status();path.write_bytes(r.content);break
            except requests.RequestException:
                if attempt==3:raise
                time.sleep(1+attempt)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    snapshot=ROOT/'data/source_registry.json'
    if snapshot.exists():
        registry=json.loads(snapshot.read_text())
        for source in registry:
            if source['id']=='sympy':
                url='https://codeload.github.com/sympy/sympy/tar.gz/refs/tags/'+source['revision']
            else:
                url=f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{source['path']}"
            target=ROOT/source['local'];target.parent.mkdir(parents=True,exist_ok=True)
            digest=fetch(url,target)
            if digest!=source['sha256']:raise RuntimeError('checksum mismatch: '+source['id'])
        print('Pinned source replay verified',len(registry),'artifacts');return
    registry=[]
    eqsha='dc35fc0ac860ff224293f01d2a62204042095127'
    url=f'https://raw.githubusercontent.com/woshixuhao/EqGPT/{eqsha}/code/PDE_dataset_0725.xlsx'
    registry.append(dict(id='eqgpt',repo='woshixuhao/EqGPT',revision=eqsha,path='code/PDE_dataset_0725.xlsx',local='data/raw/eqgpt.xlsx',sha256=fetch(url,RAW/'eqgpt.xlsx'),license='Apache-2.0',origin='human',evidence='Paper Methods states manual structure-string transcription by experts; provenance applies to structure, not AI-assisted LaTeX conversion.'))
    sha=json.loads((ROOT/'references/openai_tree.json').read_text())['sha']
    folders=json.loads((ROOT/'references/preprints.json').read_text())
    specifications=[('Navier-Stokes','pde'),('Vlasov','pde'),('heat','pde'),('Quasi-Riemann','number_theory'),('pure-type','logic'),('Universal-Tensor','algebra'),('Heisenberg','mathematical_physics'),('parity','combinatorics'),('Mezard','probability')]
    selected=[]
    for needle,domain in specifications:
        matches=[x['name'] for x in folders if needle.lower() in x['name'].lower()]
        for folder in matches[:2]:selected.append((folder,domain))
    for folder,domain in selected:
        directory=f'preprints/{folder}'
        listing=api(f'repos/openai/math/contents/{directory}?ref={sha}')
        build=next((x for x in listing if x['name']=='build'),None)
        if not build:continue
        tree=api(f'repos/openai/math/git/trees/{build["sha"]}?recursive=1')
        for item in tree['tree']:
            if item['path'].endswith('.tex'):
                path=f'{directory}/build/{item["path"]}'
                local='data/raw/openai/'+folder+'/'+item['path']
                dest=ROOT/local;dest.parent.mkdir(parents=True,exist_ok=True)
                url=f'https://raw.githubusercontent.com/openai/math/{sha}/{path}'
                digest=fetch(url,dest)
                registry.append(dict(id='openai:'+folder+':'+item['path'],repo='openai/math',revision=sha,path=path,local=local,sha256=digest,license='Apache-2.0',origin='possible_mixed',domain=domain,group='openai:'+folder,evidence='Official README: model-produced manuscripts, with human-edited exceptions; per-formula absence of human edits is not certified. Conservative possible_mixed label.'))
        print('fetched',folder,flush=True)
    sym=RAW/'sympy-1.10.tar.gz'
    registry.append(dict(id='sympy',repo='sympy/sympy',revision='sympy-1.10',local='data/raw/sympy-1.10.tar.gz',sha256=fetch('https://codeload.github.com/sympy/sympy/tar.gz/refs/tags/sympy-1.10',sym),license='BSD-3-Clause',origin='unknown',evidence='Historical human-maintained documentation; release age does not certify absence of AI assistance. Kept unknown pending contributor attestation.'))
    (ROOT/'data/source_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
