"""Finalize source provenance after token-equivalent native micro-optimization."""
import json,sys
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import file_hash
out=ROOT/sys.argv[1];digest=file_hash(ROOT/'native/decoder_v4.c')
for path in (out/'models').glob('*.json'):
 d=json.loads(path.read_text());d['native_source_sha256']=digest;path.write_text(json.dumps(d,indent=2))
manifest=json.loads((out/'manifest.json').read_text());manifest['native_source_sha256']=digest;manifest['models']={p.name:file_hash(p) for p in (out/'models').glob('*.json')};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Final source hash',digest)
