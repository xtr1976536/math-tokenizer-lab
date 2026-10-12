"""Replay approved, immutable sources; no fresh discovery or license inference."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from harvest_arxiv import OfficialClient,ROOT

def main():
    client=OfficialClient()
    for r in json.loads((ROOT/'data/arxiv_sources.json').read_text()):
        path=ROOT/r['local'];data=client.get(r['source_url'],path)
        if hashlib.sha256(data).hexdigest()!=r['sha256']:raise RuntimeError('source hash mismatch '+r['id'])
        print('verified',r['id']+r['version'],flush=True)
    source=json.loads((ROOT/'data/fungrim_source.json').read_text());path=ROOT/'vendor/fungrim'
    if not path.exists():
        subprocess.run(['git','clone','https://github.com/'+source['repo']+'.git',str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',source['revision']],check=True)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=source['revision']:raise RuntimeError('Fungrim revision mismatch')
    subprocess.run([__import__('sys').executable,str(ROOT/'scripts/export_fungrim.py')],check=True)
    export=ROOT/source['export_local'];assert hashlib.sha256(export.read_bytes()).hexdigest()==source['export_sha256']
    print('Pinned source replay complete')
if __name__=='__main__':main()
