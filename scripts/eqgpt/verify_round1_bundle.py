"""Verify archive content before executing any experiment."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'BUNDLE_MANIFEST.json'
if not p.exists():raise FileNotFoundError('Run this command inside the extracted first-round ZIP')
m=json.loads(p.read_text())
for name,record in m.items():
 path=(ROOT/name).resolve()
 if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Missing/invalid path: '+name)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed bundle file: '+name)
from math_tokenizer.eqgpt.author import verify_source
from math_tokenizer.eqgpt.frozen_math import load_frozen
verify_source();_,f=load_frozen(ROOT/'experiments/eqgpt_round1/frozen_math')
print(json.dumps(dict(bundle_files_verified=len(m),frozen_ids=f['vocab_size'],math_training_formulas=f['corpus']['accepted']),indent=2))
