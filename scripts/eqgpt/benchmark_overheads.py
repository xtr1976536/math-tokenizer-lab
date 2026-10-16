"""CPU wrapper microbenchmarks and five-epoch fitting check; not formal results."""
import argparse,copy,gc,hashlib,json,tempfile,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from math_tokenizer.eqgpt.author import ROOT,seed_all,tensor_hash,training_namespace
from math_tokenizer.eqgpt.data import augment_training
from math_tokenizer.eqgpt.cached_batches import cache_batches
from math_tokenizer.eqgpt.frozen_math import load_frozen
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.round1_config import LATEX_METHODS,METHODS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'docs/eqgpt/round1_evidence/performance.json');a=p.parse_args()
torch.set_num_threads(2);warnings.filterwarnings('ignore',category=FutureWarning)
with tempfile.TemporaryDirectory() as folder:seq,source=augment_training('KdV equation',folder,525)
adapters,_=load_frozen(ROOT/'experiments/eqgpt_round1/frozen_math');adapters={n:adapters[n] for n in LATEX_METHODS};adapters['eqgpt_reference']=Adapter('eqgpt')
models,meta=build_models(adapters,512,525);env=training_namespace('cpu');reports={}
for name,model in models.items():
 encoded=[adapters[name].sequence(s if name=='eqgpt_reference' else structure_to_latex(s)) for s in seq]
 def loader(rows):
  ds=env['MyDataSet'](rows);return DataLoader(ds,batch_size=128,collate_fn=ds.padding_batch,shuffle=False)
 original=loader(encoded);start=time.perf_counter();cached=cache_batches(original,'cpu');build=time.perf_counter()-start
 assert all(torch.equal(x,u) and torch.equal(y,v) for (x,y),(u,v) in zip(original,cached))
 input_times={}
 for label,l in [('original',original),('cached',cached)]:
  trials=[]
  for _ in range(3):
   start=time.perf_counter()
   for epoch in range(10):
    for x,y in l:pass
   trials.append(time.perf_counter()-start)
  input_times[label]=float(np.median(trials))
 initial=copy.deepcopy(model.state_dict());trajectories=[];fit_times={'original':[],'cached':[]}
 # ABBA ordering reduces the effect of cold kernels and process drift.
 for label in ('original','cached','cached','original'):
  model.load_state_dict(initial);seed_all(525)
  l=loader(encoded[:128]);l=cache_batches(l,'cpu') if label=='cached' else l
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);criterion=nn.CrossEntropyLoss(ignore_index=0)
  start=time.perf_counter();losses=[]
  for epoch in range(5):losses.append(env['train_step'](model,l,optimizer,criterion,1))
  fit_times[label].append(time.perf_counter()-start)
  trajectories.append(dict(losses=losses,final_sha256=tensor_hash(model.state_dict()),rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
  del optimizer;gc.collect()
 assert all(r==trajectories[0] for r in trajectories)
 assert trajectories[0]['losses'][-1]<trajectories[0]['losses'][0]
 reports[name]=dict(input_batch_build_seconds=build,input_iteration_seconds_10_epochs=input_times,
  input_iteration_speedup=input_times['original']/input_times['cached'],training_seconds_5_epochs={k:float(np.median(v)) for k,v in fit_times.items()},
  training_speedup=float(np.median(fit_times['original'])/np.median(fit_times['cached'])),
  exact_losses_parameters_rng=True,pilot_epochs=5,pilot_samples=128,pilot_optimizer_steps=5,trajectory=trajectories[0])
 del initial;gc.collect();print(name,reports[name],flush=True)
class Analytic(torch.nn.Module):
 def forward(self,xt):return torch.exp(-2*xt[:,1:2])*torch.sin(xt[:,0:1])
grid=torch.stack([torch.linspace(.2,1,10000),torch.linspace(.1,.8,10000)],1).requires_grad_()
reward_times={};records=[]
for cached in (False,True):
 e=RewardEvaluator(Analytic(),grid,'KdV_equation',['x','t'],'cpu',.2);sample=dict(stopped=True,text='u_t+u_{xx}=0')
 e.evaluate(sample);start=time.perf_counter();outputs=[]
 for i in range(500):
  if not cached:e.reward_cache.clear()
  outputs.append(e.evaluate(sample))
 reward_times['cached' if cached else 'original']=time.perf_counter()-start;records.append(outputs)
assert records[0]==records[1]
report=dict(formal_results=False,device='CPU',torch_threads=2,methods=list(METHODS),models=reports,
 reward_repeat_500_seconds=reward_times,reward_repeat_speedup=reward_times['original']/reward_times['cached'],
 note='Input iteration isolates batching overhead. Neural fitting uses exactly 128 common formulas, five optimizer steps, ABBA timing, same original six-layer class and original train_step. Reward timing isolates 500 repeats of one supported candidate on 10000 points. These ratios are not an end-to-end GPU speed claim.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
