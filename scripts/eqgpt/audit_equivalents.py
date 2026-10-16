"""Independent symbolic support audit for known target identities, not generation."""
import argparse,json,tempfile
from pathlib import Path
from math_tokenizer.eqgpt.author import ROOT
from math_tokenizer.eqgpt.data import case_environment,augment_training,canonical
from math_tokenizer.eqgpt.latex_experiment import MAIN_CASES
import sympy as s,pandas as pd
from math_tokenizer.eqgpt.author import CODE,DICTIONARY
from math_tokenizer.eqgpt.latex_codec import GREParser
from math_tokenizer.eqgpt.evaluation import TRUTH
v={c:s.Symbol(c) for c in 'xyzt'};u=s.Function('u')(*[v[c] for c in 'xyzt'])
def symbolic(n):
 k=n[0]
 if k=='v':return u if n[1]=='u' else v[n[1]]
 if k=='n':return s.Integer(n[1])
 if k=='add':return sum(symbolic(x) for x in n[1:])
 if k=='mul':return s.Mul(*(symbolic(x) for x in n[1:]))
 if k=='div':return symbolic(n[1])/symbolic(n[2])
 if k=='pow':return symbolic(n[1])**n[2]
 if k=='neg':return -symbolic(n[1])
 if k=='fn':return {'sin':s.sin,'sinh':s.sinh,'exp':s.exp,'sqrt':s.sqrt}[n[1]](symbolic(n[2]))
 if k=='d':return s.diff(symbolic(n[1]),*[v[c] for c in n[2]])
 if k=='lap':
  expr=symbolic(n[1])
  for _ in range(n[2]):expr=s.diff(expr,v['x'],2)+s.diff(expr,v['y'],2)
  return expr
 raise Exception(n)
def support(raw):
 expr=s.expand(symbolic(GREParser(raw.replace(',','')).parse()))
 return frozenset(str(t.as_coeff_Mul()[1]) for t in s.Add.make_args(expr))

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'docs/eqgpt/round1_evidence/target_exclusions.json');a=p.parse_args()
x=pd.read_excel(CODE/'PDE_dataset_0725.xlsx')
rows=[(str(r['Equation name']),r['Terms'],support(r['Terms'])) for _,r in x.iterrows()]
records=[]
with tempfile.TemporaryDirectory() as temp:
 for case in MAIN_CASES:
  target=support(',+,'.join(t for t,c in TRUTH[case]));matches=[(n,t) for n,t,k in rows if k==target]
  env=case_environment(case,32,0,'cpu',525)
  seq,info=augment_training(env['Delete_equation_name'],Path(temp)/case,525,[['ut','+','(uux)x']] if case=='PDE_compound' else [])
  allowed={canonical([DICTIONARY['id2word'][i] for i in s]) for s in seq}
  assert all(canonical(raw.split(',')) not in allowed for n,raw in matches),(case,matches)
  records.append(dict(case=case,all_symbolic_support_matches=matches,removed_equations=info['removed_equations'],training_formulas=info['training_formulas']))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,raw_formulas=221,target_cases=11,
 policy='Independent SymPy expansion of formal GRE meanings; compare coefficient-free monomial support, including product differentiation and named 2-D Laplacian. Used only to audit target exclusions, never to repair generated expressions or alter author numerical functions.',cases=records),indent=2)+'\n')
print('All 11 targets: every identified symbolic-support match removed.')
