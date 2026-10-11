"""Download only pinned public tokenizer configurations, no model weights."""
import hashlib
import json
import requests
from math_tokenizer.corpus import ROOT

for record in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()):
    if record['status']!='downloaded':continue
    path=ROOT/record['local']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']:continue
    response=requests.get(f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/tokenizer.json",timeout=120)
    response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==record['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
    print(record['repo'],record['revision'],flush=True)
