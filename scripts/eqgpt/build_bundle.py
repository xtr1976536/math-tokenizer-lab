"""Create a portable, credential-free author-code experiment ZIP."""
from pathlib import Path
import ast,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'deliverables/eqgpt_tokenizer_cloud_20261016.zip'
PREFIX='eqgpt_tokenizer_cloud_20261016'
files=[]
# Include only the existing tokenizer implementation's dependency closure.
queue=list((ROOT/'src/math_tokenizer/eqgpt').glob('*.py'));seen=set()
while queue:
 path=queue.pop()
 if path in seen:continue
 seen.add(path);package=list(path.relative_to(ROOT/'src').with_suffix('').parts[:-1])
 for node in ast.walk(ast.parse(path.read_text())):
  modules=[]
  if isinstance(node,ast.ImportFrom):
   if node.level:modules=[package[:len(package)-node.level+1]+((node.module or '').split('.') if node.module else [])]
   elif node.module and node.module.startswith('math_tokenizer'):modules=[node.module.split('.')]
  elif isinstance(node,ast.Import):modules=[item.name.split('.') for item in node.names if item.name.startswith('math_tokenizer')]
  for mod in modules:
   target=(ROOT/'src').joinpath(*mod).with_suffix('.py')
   if not target.exists():target=(ROOT/'src').joinpath(*mod)/'__init__.py'
   if target.exists():queue.append(target)
files.extend(seen);files.append(ROOT/'src/math_tokenizer/__init__.py')
for directory in ['native','scripts/eqgpt','third_party/eqgpt','docs/eqgpt']:
 for path in (ROOT/directory).rglob('*'):
  if not path.is_file() or any(part in ('__pycache__','.DS_Store','downstream_gpt2') for part in path.parts):continue
  if path.suffix in ('.pyc','.dat','.part'):continue
  files.append(path)
files += [ROOT/x for x in ['pyproject.toml','requirements-eqgpt.lock.txt','tests/test_eqgpt.py','LICENSE']]
# Include compact audit evidence; no checkpoints, private data or prior project
# work is bundled. The pinned public data in third_party is sufficient for clean.
for directory in ['experiments/eqgpt_reproduction/validated_smoke','experiments/eqgpt_reproduction/validated_repeat']:
 for path in (ROOT/directory).rglob('*.json'):
  if path.name!='train_gre.json':files.append(path)
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))}
OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for rel in manifest:z.write(ROOT/rel,PREFIX+'/'+rel)
 z.writestr(PREFIX+'/README.md',(ROOT/'docs/eqgpt/README.md').read_text())
 z.writestr(PREFIX+'/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
summary=dict(path=str(OUT),bytes=OUT.stat().st_size,sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),files=len(manifest),
             public_author_data_included=True,large_figshare_files_included=False,credentials_included=False)
(ROOT/'deliverables/eqgpt_tokenizer_cloud_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
