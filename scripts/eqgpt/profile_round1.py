"""Count full raw-data training workloads; never run formal training locally."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from math_tokenizer.eqgpt.author import ROOT,verify_source
from math_tokenizer.eqgpt.data import augment_training,case_environment
from math_tokenizer.eqgpt.frozen_math import load_frozen
from math_tokenizer.eqgpt.latex_codec import structure_to_latex,latex_to_structure,LatexParser,key
from math_tokenizer.eqgpt.latex_experiment import MAIN_CASES
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'docs/eqgpt/round1_workload.json');a=p.parse_args()
adapters,frozen=load_frozen(ROOT/'experiments/eqgpt_round1/frozen_math');adapters['eqgpt_reference']=Adapter('eqgpt')
rows=[]
with tempfile.TemporaryDirectory() as folder:
 for case in MAIN_CASES:
  env=case_environment(case,30000 if case=='Possion_x_y' else 10000,0,'cpu',525)
  seq,info=augment_training(env['Delete_equation_name'],Path(folder)/case,525,[['ut','+','(uux)x']] if case=='PDE_compound' else [])
  texts=[structure_to_latex(s) for s in seq]
  for t in set(texts):assert key(LatexParser(t).parse())==key(LatexParser(structure_to_latex(latex_to_structure(t))).parse())
  result=dict(case=case,source=info,observation_shape=list(env['un'].shape),methods={})
  for name,adapter in adapters.items():
   lengths=np.array([len(adapter.sequence(s)) for s in (seq if name=='eqgpt_reference' else texts)])
   padded=sum(len(lengths[i:i+128])*(int(lengths[i:i+128].max())-1) for i in range(0,len(lengths),128))
   assert lengths.max()<=512
   result['methods'][name]=dict(ids=adapter.vocab_size,samples=len(lengths),steps_100_epochs=100*((len(lengths)+127)//128),
       token_max=int(lengths.max()),token_mean=float(lengths.mean()),target_tokens_100_epochs=int((lengths-1).sum()*100),
       padded_positions_100_epochs=padded*100)
  rows.append(result);print(case,info['training_sequences'],{k:v['token_max'] for k,v in result['methods'].items()},flush=True)
report=dict(first_round_only=True,length_profile_seed=525,padded_lengths_may_vary_by_seed=True,cases=11,seeds=3,paired_runs=33,model_runs=132,source_revision=verify_source()['revision'],
 frozen_training_formulas=frozen['corpus']['accepted'],surrogate_updates=33*50000,
 optimizer_updates_pretrain=sum(m['steps_100_epochs'] for r in rows for m in r['methods'].values())*3,
 generation_attempts=132*400*5,case_workloads=rows)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print('Workload report:',a.output)
