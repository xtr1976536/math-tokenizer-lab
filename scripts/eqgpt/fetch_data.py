"""Fetch pinned author data and optionally the two large figshare data files."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]

def fetch(url,dest,algorithm,expected):
    if dest.exists() and hashlib.new(algorithm,dest.read_bytes()).hexdigest()==expected:return
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.part')
    digest=hashlib.new(algorithm)
    with requests.get(url,stream=True,timeout=120) as r:
        r.raise_for_status()
        with tmp.open('wb') as out:
            for chunk in r.iter_content(1024*1024):out.write(chunk);digest.update(chunk)
    if digest.hexdigest()!=expected:tmp.unlink();raise RuntimeError('Hash mismatch: '+url)
    tmp.replace(dest)

def main():
    p=argparse.ArgumentParser();p.add_argument('--large',action='store_true');a=p.parse_args()
    provenance=json.loads((ROOT/'third_party/eqgpt/provenance.json').read_text())
    for entry in provenance['data_files']:
        dest=ROOT/'third_party/eqgpt'/entry['path']
        fetch('https://raw.githubusercontent.com/woshixuhao/EqGPT/'+provenance['revision']+'/'+entry['path'],dest,'sha256',entry['sha256'])
    if a.large:
        listing=json.loads((ROOT/'docs/eqgpt/figshare_files.json').read_text())
        for entry in listing['files']:
            if entry['name'] not in ('Laplacian_H.dat','Laplacian_shuttle.dat'):continue
            name=entry['name'].removesuffix('.dat');dest=ROOT/'third_party/eqgpt/data'/name/entry['name']
            fetch(entry['download_url'],dest,'md5',entry['computed_md5'])
    print('Author data verified. Large cases:',a.large)
if __name__=='__main__':main()
