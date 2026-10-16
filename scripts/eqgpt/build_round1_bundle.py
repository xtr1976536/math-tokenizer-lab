"""Package only the first-round entry points, dependency closure and frozen data."""
from pathlib import Path
import ast,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
from math_tokenizer.eqgpt.round1_runtime import runtime_sha256
audit=json.loads((ROOT/'docs/eqgpt/round1_evidence/audit.json').read_text())
assert audit['all_four_rounds_passed'] and audit['runtime_sha256']==runtime_sha256()
PREFIX='eqgpt_round1_autodl_20261016';OUT=ROOT/'deliverables'/f'{PREFIX}.zip'
scripts=['run_round1.py','run_latex.py','collect_latex.py','preflight.py','install_round1.sh','verify_round1_bundle.py','calibrate_round1.py','profile_round1.py','build_round1_bundle.py','audit_round1.py','audit_equivalents.py']
files=[ROOT/'scripts/eqgpt'/s for s in scripts]
# Walk relative/absolute Python imports, including imports inside functions.
queue=[ROOT/'src/math_tokenizer/eqgpt'/n for n in ['latex_experiment.py','round1_runtime.py','frozen_math.py']]
seen=set()
while queue:
 path=queue.pop()
 if path in seen:continue
 seen.add(path);package=list(path.relative_to(ROOT/'src').with_suffix('').parts[:-1])
 for node in ast.walk(ast.parse(path.read_text())):
  modules=[]
  if isinstance(node,ast.ImportFrom):
   if node.level:modules=[package[:len(package)-node.level+1]+((node.module or '').split('.') if node.module else [])]
   elif node.module and node.module.startswith('math_tokenizer'):modules=[node.module.split('.')]
  elif isinstance(node,ast.Import):modules=[a.name.split('.') for a in node.names if a.name.startswith('math_tokenizer')]
  for mod in modules:
   target=(ROOT/'src').joinpath(*mod).with_suffix('.py')
   if not target.exists():target=(ROOT/'src').joinpath(*mod)/'__init__.py'
   if target.exists():queue.append(target)
files.extend(seen);files.extend([ROOT/'src/math_tokenizer/__init__.py',ROOT/'src/math_tokenizer/eqgpt/__init__.py'])
files.extend((ROOT/'native').glob('*.c'))
provenance=json.loads((ROOT/'third_party/eqgpt/provenance.json').read_text())
files.extend(ROOT/'third_party/eqgpt'/s for s in provenance['files'])
files.extend(ROOT/'third_party/eqgpt'/r['path'] for r in provenance['data_files'])
files.append(ROOT/'third_party/eqgpt/provenance.json')
files.extend(p for p in (ROOT/'experiments/eqgpt_round1/frozen_math').rglob('*') if p.is_file())
files.extend(p for p in (ROOT/'docs/eqgpt/round1_evidence').rglob('*') if p.is_file())
files.extend(ROOT/p for p in ['requirements-eqgpt.lock.txt','LICENSE','pyproject.toml','tests/test_eqgpt.py','tests/test_eqgpt_latex.py','tests/test_eqgpt_round1.py',
    'docs/eqgpt/round1_autodl.md','docs/eqgpt/round1_changes.json','docs/eqgpt/round1_preflight_local.json','docs/eqgpt/round1_plan.json','docs/eqgpt/round1_workload.json','docs/eqgpt/literature/training_fairness_20261016.md'])
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))}
manifest['README.md']=dict(bytes=len((ROOT/'docs/eqgpt/round1_autodl.md').read_bytes()),sha256=hashlib.sha256((ROOT/'docs/eqgpt/round1_autodl.md').read_bytes()).hexdigest())
OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for rel in manifest:
  z.write(ROOT/(rel if rel!='README.md' else 'docs/eqgpt/round1_autodl.md'),PREFIX+'/'+rel)
 z.writestr(PREFIX+'/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
summary=dict(path=str(OUT),bytes=OUT.stat().st_size,sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),files=len(manifest),
 first_round_only=True,frozen_math_formulas=32270,frozen_vocab_ids=1024,models=132,smoke_models=44,second_round_run=False)
(ROOT/'deliverables/eqgpt_round1_bundle_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
