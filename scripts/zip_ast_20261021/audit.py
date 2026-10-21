"""Three separately recorded audits; deliberately independent count aggregation."""
import argparse,bisect,gzip,hashlib,importlib.util,json,math,random,re,sys,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('study',HERE/'run.py');study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
OUT=study.OUT

def audit_definitions():
 p=json.loads((OUT/'protocol.json').read_text());prov=json.loads((OUT/'zip_provenance.json').read_text())
 if study.ZIP.exists():assert study.sha(study.ZIP.read_bytes())==prov['zip_sha256']
 for name,digest in prov['files'].items():assert study.sha((OUT/'vendor'/name).read_bytes())==digest
 assert not list((OUT/'vendor').rglob('*.json')),'No pretrained vocabularies allowed inside isolated vendor'
 snapshot=(OUT/'protocol_runner_snapshot.py.txt').read_text()
 assert study.h(snapshot)==p['scripts_sha256']
 import ast
 def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name!='extract'}
 assert functions(snapshot)==functions((HERE/'run.py').read_text()),'Metric, splitting, training and evaluation code changed after freezing'
 baseline=json.loads((OUT/'baseline_provenance.json').read_text())
 for name,digest in baseline['adapter_and_registry_source_sha256'].items():assert study.sha((study.ROOT/name).read_bytes())==digest
 for entry in baseline['baselines'].values():
  if entry.get('path'):assert study.sha((study.ROOT/entry['path']).read_bytes())==entry['checkpoint_sha256']
 llama=baseline['llama3_official_provenance']
 for entry in llama['official_code_files'].values():assert study.sha((study.ROOT/entry['local']).read_bytes())==entry['sha256']
 assert study.sha((study.ROOT/llama['official_ranks_path']).read_bytes())==llama['official_ranks_sha256']
 from math_tokenizer.active_benchmark import read_catalog
 rows=read_catalog();ps=study.protocols(rows)
 for x,y in zip(ps,p['splits']):
  assert x['name']==y['name']
  for split in ('train','dev','test'):assert [r['id'] for r in x[split]]==y[split]
  study.split_audit(x)
 # Naive interval definition and production sweep must agree on 5000 random cases.
 rng=random.Random(1729)
 for _ in range(5000):
  n=rng.randrange(2,80);bounds=[0]+sorted(rng.sample(range(1,n),rng.randrange(n)))+[n]
  spans=[tuple(sorted(rng.sample(range(n+1),2))) for _ in range(rng.randrange(30))]
  naive=sum(any(a<s<b<e or s<a<e<b for s,e in spans) for a,b in zip(bounds,bounds[1:]))
  assert study.cross_fast(bounds,spans)==naive
 class Pieces:
  def __init__(self,parts):self.parts=[s.encode() for s in parts]
  def encode(self,t):return list(range(len(self.parts)))
  def decode(self,ids):return b''.join(self.parts[i] for i in ids).decode()
 t=r'\frac{12.50}{x}';row=dict(id='hand',latex=t,split_group='hand',domain='hand',complexity={'class':'simple'},reference={'spans':[[0,5],[6,11],[13,14]]})
 unit=Pieces(list(t));whole=Pieces([t])
 a=study.summarize([study.per_formula(unit,row,lambda m,i:m.parts[i])]);b=study.summarize([study.per_formula(whole,row,lambda m,i:m.parts[i])]);byte_result=a;whole_result=b
 assert a==dict(fragmentation_pct=100,cmd_pct=0,cross_preservation_pct=100,num_pct=0)
 assert b==dict(fragmentation_pct=0,cmd_pct=100,cross_preservation_pct=100,num_pct=100)
 # This is a constructive counterexample to a guaranteed inverse correlation.
 assert all(0<=a<b<=len(r['latex'].encode()) for r in rows for a,b in r['reference']['spans'])
 cmd_count=sum(bool(re.fullmatch(r'\\[A-Za-z]+',t)) for r in rows for _,_,t in study.lex(r['latex']))
 number_lengths=Counter(len(t) for r in rows for _,_,t in study.lex(r['latex']) if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',t))
 pde=[r for r in rows if r['formula_partition']=='pde'];pn=Counter(len(t) for r in pde for _,_,t in study.lex(r['latex']) if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',t))
 overlap={}
 for family in ('pde_id','benchmark_id','benchmark_simple'):
  selected=[r for r in p['splits'] if r['name'].startswith(family)]
  for i,a in enumerate(selected):
   for b in selected[i+1:]:
    x=set(a['test']);y=set(b['test']);overlap[a['name']+'__'+b['name']]=len(x&y)/len(x|y)
 return dict(status='PASS',checks=['ZIP hash and byte-identical implementation snapshot','frozen runner snapshot matches initial hash; all computational functions unchanged','baseline adapters and registry hashes unchanged','no ZIP learned vocabulary extracted','all twelve source/canonical/template separated splits','5000 independent interval-crossing tests','hand-calculated byte and whole-formula metric endpoints'],catalog_formulas=len(rows),pde_formulas=len(pde),command_occurrences=cmd_count,number_length_counts=dict(sorted(number_lengths.items())),pde_number_length_counts=dict(sorted(pn.items())),repeated_split_test_jaccard=overlap,counterexample=dict(byte_tokenizer=byte_result,whole_formula_tokenizer=whole_result),limitation='Cross preservation alone cannot measure structural quality: byte tokenization obtains100 even though command and numeric integrity can be0.')

def audit_results(complete=True):
 protocol=json.loads((OUT/'protocol.json').read_text());byid={r['id']:r for r in __import__('math_tokenizer.active_benchmark',fromlist=['read_catalog']).read_catalog()}
 checked=0;formula_evaluations=0;signatures={};nulls=[]
 for p in protocol['splits']:
  folder=OUT/p['name'];expected=2*len(study.SIZES)+3;paths=sorted(folder.glob('*.json'))
  expected_train_strings_sha=study.h(json.dumps([byid[id]['latex'] for id in p['train']],ensure_ascii=False))
  if complete:assert len(paths)==expected,(p['name'],len(paths),expected)
  for path in paths:
   s=json.loads(path.read_text())
   with gzip.open(path.with_suffix('.jsonl.gz'),'rt') as f:rr=[json.loads(line) for line in f]
   assert [r['id'] for r in rr]==p['test']
   assert all(r['bytes']==len(byid[r['id']]['latex'].encode()) and 1<=r['tokens']<=r['bytes'] for r in rr)
   assert all(0<=r['crossing_tokens']<=r['tokens'] and 0<=r['commands_intact']<=r['commands'] and 0<=r['numbers_intact']<=r['numbers'] for r in rr)
   def ratio(num,den):
    d=math.fsum(r[den] for r in rr);return 100*math.fsum(r[num] for r in rr)/d if d else None
   independent=dict(fragmentation_pct=100*math.fsum(r['tokens']-1 for r in rr)/math.fsum(r['bytes']-1 for r in rr),cmd_pct=ratio('commands_intact','commands'),cross_preservation_pct=100-ratio('crossing_tokens','tokens'),num_pct=ratio('numbers_intact','numbers'))
   for k,v in independent.items():
    assert (v is None and s[k] is None) or (v is not None and abs(v-s[k])<1e-10),(path,k)
    if v is None:nulls.append([str(path),k])
   signature=(s['test_ids_sha256'],s['formulas'],s['denominators']['bytes'],s['denominators']['commands'],s['denominators']['numbers'],s['denominators']['possible_cuts'])
   if p['name'] in signatures:assert signatures[p['name']]==signature
   signatures[p['name']]=signature
   if s['requested_total_ids']:
    cp=folder/'models'/f"{s['method']}_{s['requested_total_ids']}.json";pr=json.loads(cp.with_suffix('.provenance.json').read_text())
    assert study.sha(cp.read_bytes())==pr['checkpoint_sha256']
    assert pr['train_ids_sha256']==study.h('\n'.join(p['train']))
    assert pr['train_strings_sha256']==expected_train_strings_sha
    d=json.loads(cp.read_text());assert len(d['vocab_hex'])+3==s['actual_total_ids']<=s['requested_total_ids']
   checked+=1;formula_evaluations+=len(rr)
 return dict(status='PASS',result_files_checked=checked,expected_result_files=12*19,formula_evaluations_checked=formula_evaluations,checks=['Independent count aggregation of all four metrics','Exact same ordered evaluation IDs for every model and budget','Same raw bytes, commands, numbers and possible-cut denominators','Valid counts and no zero-length payload or more-than-byte tokenization','Checkpoint hashes and actual ID counts','Training-ID hashes match frozen protocol'],null_metrics=nulls,per_protocol_common_signatures=signatures)

def load_vendor():
 for module in list(sys.modules):
  if module=='math_tokenizer' or module.startswith('math_tokenizer.'):del sys.modules[module]
 sys.path.insert(0,str(OUT/'vendor/src'))
 from math_tokenizer.hard_ast_2026 import HardASTBPE
 from math_tokenizer.ast_boundary_v8 import BoundaryAST,brace_boundaries
 from math_tokenizer.frontend_v6 import tree_ranges
 from math_tokenizer.atomic_bpe_v7 import AtomicBPE
 from math_tokenizer.rl_route_v6 import derive_atoms
 import ast
 source=OUT/'vendor/src/math_tokenizer/eqgpt/latex_tokenizers.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_one')
 ns=dict(HardASTBPE=HardASTBPE,BoundaryAST=BoundaryAST,AtomicBPE=AtomicBPE,derive_atoms=derive_atoms)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 return HardASTBPE,BoundaryAST,brace_boundaries,tree_ranges,ns['train_one']

def audit_backend(complete=True):
 from math_tokenizer.active_benchmark import read_catalog
 byid={r['id']:r for r in read_catalog()};protocol=json.loads((OUT/'protocol.json').read_text());Hard,Soft,brace,tree,train=load_vendor()
 hard_checked=soft_checked=0;retrained=[];rng=random.Random(43)
 for p in protocol['splits']:
  for size in study.SIZES:
   for name,cls in [('ast_bpe',Hard),('soft_ast_bpe',Soft)]:
    path=OUT/p['name']/'models'/f'{name}_{size}.json'
    if not path.exists():
     assert not complete,path;continue
    model=cls.load(path)
    samples=rng.sample(p['test'],min(8,len(p['test'])))
    for id in samples:
     text=byid[id]['latex'];ids=model.encode(text);assert model.decode(ids)==text
     if name=='ast_bpe':assert ids==model.encode_reference(text);hard_checked+=1
    if name=='soft_ast_bpe':
     # Independent exhaustive candidate-span dynamic program, without native DP.
     for text in [r'\frac{12}{x}',r'\partial_t u+2.50u=0','x+y+z']:
      raw=text.encode();spans=tree(text).tolist();endpoints=brace(text);cost=[0]+[float('inf')]*len(raw)
      for b in range(1,len(raw)+1):
       for piece in model.vocab:
        a=b-len(piece)
        if a<0 or raw[a:b]!=piece:continue
        crossing=any(a<s<b<e or s<a<e<b for s,e in spans)
        hidden=sum(a<k<b for k in endpoints)
        cost[b]=min(cost[b],cost[a]+1+model.tree_penalty*crossing+model.boundary_penalty*hidden)
      assert abs(cost[-1]-model.objective(text))<1e-7,(p['name'],size,text,cost[-1],model.objective(text))
      soft_checked+=1
  # Refit two endpoint capacities on one small protocol; no inherited dictionaries.
  if p['name']=='pde_id_17':
   for size in [320,1024]:
    for name,cls in [('ast_bpe',Hard),('soft_ast_bpe',Soft)]:
     fresh=train(name,[byid[id]['latex'] for id in p['train']],size);stored=cls.load(OUT/p['name']/'models'/f'{name}_{size}.json')
     assert fresh.vocab==stored.vocab
     assert all(fresh.encode(byid[id]['latex'])==stored.encode(byid[id]['latex']) for id in p['test'])
     retrained.append(dict(method=name,size=size,identical_vocabulary=True,identical_all_test_encodings=True))
 if complete:assert hard_checked==768 and soft_checked==288
 return dict(status='PASS',hard_native_python_reference_cases=hard_checked,soft_native_independent_dynamic_program_cases=soft_checked,fresh_retraining=retrained,checks=['Native hard AST ranked merges equal independent Python reference','Soft AST native objective equals independent byte-span DP optimum','Fresh retraining reproduces vocabulary and held-out encodings at320 and1024'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('round',choices=['1','2','3']);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
 result=audit_definitions() if a.round=='1' else audit_backend(not a.partial) if a.round=='2' else audit_results(not a.partial)
 result['scope']='partial' if a.partial else 'complete'
 study.write(OUT/f'audit_round_{a.round}{"_partial" if a.partial else ""}.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
