"""Frozen stress suites derived mechanically from heldout formulas.

Not separate naturally authored corpus, not semantic correctness supervision.
Transformations and lineage recorded; stress samples never enter RL reward.
"""
import json,hashlib,re
from collections import defaultdict
from .corpus import ROOT
from .evaluate import read_jsonl
from .structure_v2 import complexity

def generate():
 folder=ROOT/'data/releases/v5/curated/splits';fresh=read_jsonl(folder/'fresh_paper_test.jsonl')
 chosen=sorted(fresh,key=lambda r:hashlib.sha256(('v5-stress:'+r['id']).encode()).digest())[:256];rows=[]
 def add(parent,text,kind):
  if text==parent['latex']:return
  id=hashlib.sha256((parent['id']+kind+text).encode()).hexdigest()[:24]
  rows.append(dict(parent,id=id,latex=text,parent_id=parent['id'],stress=kind,origin='ai_generated',origin_evidence='Deterministic authored stress transformation of heldout source formula, not independently sampled natural paper data',benchmark_split='synthetic_stress_test',
    transformation='heldout-formula '+kind+'; source syntax subject to original limits, no proof of mathematical truth',split_group=parent['split_group'],complexity=complexity(text)))
 for r in chosen:
  text=r['latex'];add(r,re.sub(r'\s+','',text),'whitespace_removed')
  # Renaming only free-looking individual symbols outside command names; this
  # is a representation shift, not a theorem/semantic-preservation guarantee.
  parts=re.findall(r'\\[A-Za-z]+|\\[^A-Za-z]|[A-Za-z]|[^A-Za-z\\]+',text);mapping=str.maketrans('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ','nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM')
  rewritten=[]
  for part in parts:
   if re.fullmatch('[A-Za-z]',part):rewritten.append(part.translate(mapping))
   else:rewritten.append(part)
  add(r,''.join(rewritten),'symbol_renaming')
  add(r,'('+text+')+'+'('+text+')','long_composition_2')
 # Independent extremes, always explicitly synthetic and outside reward data.
 base=chosen[0] if chosen else None
 if base:
  for n in [32,128,512,2048]:add(base,'+'.join(r'\frac{x_{'+str(k)+'}^{2}}{1+x_{'+str(k)+'}^{2}}' for k in range(n)),f'long_formula_terms_{n}')
  for n in [16,64,256]:add(base,r'\frac{'*n+'x'+'}{1}'*n,f'deep_nesting_{n}')
  for text,kind in [(r'\newrareoperator{𝒜}_{999999999999}=𝔽+∂α','unknown_macro_unicode'),('αβγ𝒜𝔽𝕏🙂+1=2','rare_unicode'),('9'*4096+'+'+'1'*4096,'huge_numbers'),(r'\frac{x}{','malformed_tex')]:add(base,text,kind)
 path=folder/'synthetic_stress_test.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
 return {'count':len(rows),'parents':len(chosen),'test_only':True,'natural_formulae':False,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
if __name__=='__main__':print(generate())
