"""Concrete token-level explanations for the four metrics (not additional metrics)."""
import bisect,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('study',HERE/'run.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
from math_tokenizer.final_models import load_final
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.active_benchmark import read_catalog
rows=read_catalog();byid={r['id']:r for r in rows};baselines={n:load_final(n) for n in s.BASELINES}
for module in list(sys.modules):
 if module=='math_tokenizer' or module.startswith('math_tokenizer.'):del sys.modules[module]
sys.path.insert(0,str(s.OUT/'vendor/src'))
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
models={**baselines,'ast_bpe':HardASTBPE.load(s.OUT/'benchmark_id_17/models/ast_bpe_1024.json'),'soft_ast_bpe':BoundaryAST.load(s.OUT/'benchmark_id_17/models/soft_ast_bpe_1024.json')}
examples=[]
texts=[r'\partial_t u=\frac{12.50}{x}',r'\alpha+\beta=123456.789',r'\frac{x+y}{z}']
protocol=json.loads((s.OUT/'protocol.json').read_text());p=next(p for p in protocol['splits'] if p['name']=='pde_id_43')
# Real test cases where number integrity differs (if present).
for id in p['test']:
 r=byid[id]
 if any(len(t)>1 for a,b,t in s.lex(r['latex']) if __import__('re').fullmatch(r'[0-9]+(?:\.[0-9]+)?',t)):
  texts.append(r['latex'])
  if len(texts)>=6:break
for text in texts:
 record=dict(text=text,models={})
 for name,model in models.items():
  ids=model.encode(text);parts=[token_bytes(model,int(i)) for i in ids];bounds=[0]
  for part in parts:bounds.append(bounds[-1]+len(part))
  lexemes=[]
  for a,b,t in s.lex(text):
   if __import__('re').fullmatch(r'\\[A-Za-z]+|[0-9]+(?:\.[0-9]+)?',t):lexemes.append(dict(text=t,byte_span=[a,b],intact=bisect.bisect_right(bounds[1:-1],a)==bisect.bisect_left(bounds[1:-1],b)))
  record['models'][name]=dict(tokens=[dict(id=int(i),bytes_hex=part.hex(),display=part.decode(errors='backslashreplace')) for i,part in zip(ids,parts)],lexemes=lexemes)
 examples.append(record)
s.write(s.OUT/'metric_examples.json',examples);print('Wrote',len(examples),'examples')
