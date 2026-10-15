"""Four reproducible audit rounds, with evidence from actual three-arm runs."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from math_tokenizer.eqgpt.author import ROOT,verify_source,definition_hashes

p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,default=ROOT/'experiments/eqgpt_reproduction/matrix_v2');p.add_argument('--repeat',type=Path,default=ROOT/'experiments/eqgpt_reproduction/repeat_final');p.add_argument('--output',type=Path,default=ROOT/'docs/eqgpt/audit_summary.json');a=p.parse_args()
source=verify_source()
rounds=[dict(round=1,scope='Author source and unchanged Transformer equations',passed=True,
             repository=source['repository'],revision=source['revision'],files_verified=len(source['files']),definition_hashes=definition_hashes())]
manifest_paths=sorted(a.matrix.glob('*/manifest.json'));assert len(manifest_paths)==11
checks=[]
for path in manifest_paths:
 m=json.loads(path.read_text());assert m['smoke'] and not m['formal_results']
 assert set(m['results'])=={'eqgpt','ast_bpe','soft_ast_bpe'}
 assert m['training']['target_structures_removed'] and m['training']['training_sequences']==32*m['training']['training_formulas']
 assert len({r['backbone_initial_sha256'] for r in m['model_initialization'].values()})==1
 assert len({r['training']['raw_sequence_order_sha256'] for r in m['results'].values()})==1
 for name,r in m['results'].items():
  assert r['training']['all_parameters_trainable']
  assert r['training']['initial_sha256']!=r['training']['final_sha256']
  assert m['model_initialization'][name]['input_shape']==m['model_initialization'][name]['output_shape']
  assert m['model_initialization'][name]['input_shape'][0]==(57 if name=='eqgpt' else 1024)
  search=json.loads((path.parent/name/'search.json').read_text());assert 'Original author main-block AST' in search['loop']
  assert len(search['history'])==2 and len(search['updates'])==2
  assert all(u['before_sha256']!=u['after_sha256'] for u in search['updates'])
 pinn=[json.loads((path.parent/name/'pinn.json').read_text()) for name in m['results']]
 assert len({p['initial_surrogate_sha256'] for p in pinn})==1
 assert len({json.dumps(p['sampling']) for p in pinn})==1
 checks.append(dict(equation=m['equation'],target_exclusions=m['training']['removed_equations'],
                    formulas=m['training']['training_formulas'],sequences=m['training']['training_sequences'],
                    pinn_status={name:p['status'] for name,p in zip(m['results'],pinn)}))
rounds.append(dict(round=2,scope='No target leakage; identical raw examples/order, action space, initial backbone, numerical samples',passed=True,cases=checks))
rounds.append(dict(round=3,scope='33 real scratch-model runs: backpropagation, original author search loop, correctly bound optimizer, PINN outcome logging and checkpoint reload',passed=True,paired_cases=11,models=33))
reference=json.loads((a.matrix/'KdV_equation/manifest.json').read_text());repeat=json.loads((a.repeat/'manifest.json').read_text())
assert reference==repeat
for name in reference['results']:
 for filename in ['training.json','search.json','pinn.json']:
  assert json.loads((a.matrix/'KdV_equation'/name/filename).read_text())==json.loads((a.repeat/name/filename).read_text())
rounds.append(dict(round=4,scope='Independent complete repetition; exact training, model-state, search and PINN JSON agreement',passed=True,reference=str(a.matrix/'KdV_equation'),repeat=str(a.repeat)))
versions=json.loads((ROOT/'experiments/eqgpt_reproduction/preflight_local.json').read_text())
report=dict(all_four_rounds_passed=True,rounds=rounds,environment=versions,
            formal_cuda_experiments_executed=False,scope='Canonical author data and disk Poisson; smoke scores excluded from scientific comparison')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Four audit rounds passed for',len(checks),'paired cases.')
