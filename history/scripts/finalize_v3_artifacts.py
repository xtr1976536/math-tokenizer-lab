"""Add complete training-cost lineage and current exact native source metadata."""
import json,sys,hashlib
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import file_hash
out=ROOT/sys.argv[1];selection=json.loads((out/'selection.json').read_text());protocol=json.loads((out/'protocol.json').read_text());models=out/'models';metrics=json.loads((out/'metrics.json').read_text());digest=file_hash(ROOT/'native/decoder.c')
# Atomic models record complete vocabulary training. RL policy training originally
# measured separately; now publish both costs without changing learned fields.
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.evaluate import read_jsonl
texts=[r['latex'] for r in read_jsonl(ROOT/'data/releases/v2/curated/splits/train.jsonl')]
atomrl=int(selection['rl_byte'].split('-')[0][4:]);base_rl=AtomicASTBPE(False).train(texts,protocol['capacity'],command_budget=atomrl)
base_astcost=json.loads((models/f'ast-{selection["ast"]}.json').read_text())['train_seconds']
for path in models.glob('*.json'):
 d=json.loads(path.read_text());d['native_source_sha256']=digest
 if path.name.startswith('rl_bpe_v3-'):
  d['policy_train_seconds']=d['train_seconds'];d['vocab_train_seconds']=base_rl.train_seconds;d['total_train_seconds']=d['policy_train_seconds']+d['vocab_train_seconds']
  d['training_cost_scope']='vocab and policy measured separately; base vocabulary retraining deterministic, used only for cost accounting'
  d['vocabulary_training_log']=base_rl.training_log
 elif path.name.startswith('rl_ast_bpe_v3-'):
  d['policy_train_seconds']=d['train_seconds'];d['vocab_train_seconds']=base_astcost;d['total_train_seconds']=d['policy_train_seconds']+base_astcost
 path.write_text(json.dumps(d,indent=2))
for m in metrics:
 if m['method'] in ('rl_bpe_v3','rl_ast_bpe_v3'):
  d=json.loads((models/f'{m["method"]}-{m["seed"]}.json').read_text());m['policy_train_seconds']=d['policy_train_seconds'];m['vocab_train_seconds']=d['vocab_train_seconds'];m['total_train_seconds']=d['total_train_seconds']
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
manifest=json.loads((out/'manifest.json').read_text());manifest['models']={p.name:file_hash(p) for p in models.glob('*.json')};manifest['native_source_sha256']=digest;manifest['metrics_sha256']=file_hash(out/'metrics.json');(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Artifact source and training accounting finalized')
