import hashlib,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
repo='Qwen/Qwen2.5-0.5B'
registry=ROOT/'data/baseline_registry.json'
if registry.exists():
    metadata=json.loads(registry.read_text())['qwen'];sha=metadata['revision']
else:
    r=requests.get('https://huggingface.co/api/models/'+repo,timeout=60);r.raise_for_status();sha=r.json()['sha']
r=requests.get(f'https://huggingface.co/{repo}/resolve/{sha}/tokenizer.json',timeout=90);r.raise_for_status()
dest=ROOT/'data/raw/qwen-tokenizer.json';dest.parent.mkdir(parents=True,exist_ok=True)
if registry.exists():
    assert hashlib.sha256(r.content).hexdigest()==metadata['sha256']
dest.write_bytes(r.content)
(ROOT/'data/baseline_registry.json').write_text(json.dumps({'qwen':{'repo':repo,'revision':sha,'file':'tokenizer.json','sha256':hashlib.sha256(r.content).hexdigest(),'local':'data/raw/qwen-tokenizer.json','role':'fixed_pretrained_reference','implementation':'huggingface/tokenizers; unmodified tokenizer.json'}},indent=2))
print(repo,sha)
