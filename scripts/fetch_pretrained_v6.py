"""Download only pinned public tokenizer configurations, no model weights."""
import hashlib
import json
import requests
from math_tokenizer.corpus import ROOT
from math_tokenizer.baselines_v6 import prepare_llama3_ranks

for record in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()):
    if record['status']!='downloaded':continue
    path=ROOT/record['local']
    if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']):
        download_repo=record.get('download_repo',record['repo'])
        download_revision=record.get('download_revision',record['revision'])
        response=requests.get(f"https://huggingface.co/{download_repo}/resolve/{download_revision}/tokenizer.json",timeout=120)
        response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==record['sha256']
        if record.get('official_git_blob_sha1'):
            blob=hashlib.sha1(b'blob '+str(len(response.content)).encode()+b'\0'+response.content).hexdigest()
            assert blob==record['official_git_blob_sha1']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
        print(record['repo'],record['revision'],flush=True)
    if record.get('kind')=='llama3_official':
        for filename,source in record['official_code_files'].items():
            destination=ROOT/source['local']
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==source['sha256']:continue
            response=requests.get(f"https://raw.githubusercontent.com/{record['official_code_repo']}/{record['official_code_revision']}/{filename}",timeout=60)
            response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==source['sha256']
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(response.content)
        prepare_llama3_ranks(record)
