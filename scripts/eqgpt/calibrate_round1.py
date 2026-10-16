"""GPU timing only: preserve first-round science; write no experiment results."""
import argparse,gc,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from math_tokenizer.eqgpt.author import ROOT,seed_all,training_namespace
from math_tokenizer.eqgpt.data import augment_training,case_environment,build_surrogate
from math_tokenizer.eqgpt.frozen_math import load_frozen
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.round1_config import MAIN_CASES,LATEX_METHODS,SEEDS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('gpu_calibration.json'));p.add_argument('--steps',type=int,default=8);p.add_argument('--hourly-price',type=float,default=None);a=p.parse_args()
if not torch.cuda.is_available():raise RuntimeError('Run GPU calibration on AutoDL')
torch.set_num_threads(8);seed_all(525)
adapters,_=load_frozen(ROOT/'experiments/eqgpt_round1/frozen_math');adapters={n:adapters[n] for n in LATEX_METHODS};adapters['eqgpt_reference']=Adapter('eqgpt')
with tempfile.TemporaryDirectory() as temp:
 seq,_=augment_training('KdV equation',temp,525)
texts=[structure_to_latex(s) for s in seq];models,metadata=build_models(adapters,512,525,'cuda');env=training_namespace('cuda');measurements={}
for name,m in models.items():
 torch.cuda.reset_peak_memory_stats();strings=seq if name=='eqgpt_reference' else texts
 encoded=[adapters[name].sequence(s) for s in strings];ds=env['MyDataSet'](encoded)
 batches=list(DataLoader(ds,batch_size=128,collate_fn=ds.padding_batch,shuffle=False))
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);criterion=nn.CrossEntropyLoss(ignore_index=0);durations=[]
 # Sample throughout the original order; warm-up is not timed.
 for i,index in enumerate(np.linspace(0,len(batches)-1,a.steps+2,dtype=int)):
  x,y=(v.cuda() for v in batches[index]);torch.cuda.synchronize();started=time.perf_counter()
  opt.zero_grad();out,_=m(x);loss=criterion(out,y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  torch.cuda.synchronize()
  if i>=2:durations.append(time.perf_counter()-started)
 infer={};m.eval()
 for length in ([8,16,32,48] if name=='eqgpt_reference' else [16,32,64,128,255]):
  count=1 if name=='eqgpt_reference' else 16
  x=torch.full((count,length),6 if name=='eqgpt_reference' else 3,dtype=torch.long,device='cuda')
  with torch.no_grad():
   m(x);torch.cuda.synchronize();started=time.perf_counter()
   for _ in range(5):m(x)
   torch.cuda.synchronize()
  infer[str(length)]=(time.perf_counter()-started)/5
 measurements[name]=dict(train_step_seconds=float(np.median(durations)),train_samples=len(encoded),steps_per_epoch=len(batches),
  forward_seconds_by_prefix=infer,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
  projected_pretrain_hours_per_method=len(MAIN_CASES)*len(SEEDS)*100*len(batches)*float(np.median(durations))/3600)
 del opt,out,loss,batches,encoded;gc.collect();torch.cuda.empty_cache()
# Exact surrogate NN and original MSE training objective on the registered batch.
e=case_environment('KdV_equation',10000,0,'cuda',525);net=build_surrogate(e,'cuda',525)
noise=e['un'].copy();values=e['random_data'](e['total'],e['choose'],e['choose_validate'],e['x'],e['t'],noise,e['x_num'],e['t_num'])
h,_,x,_=values;h=h.cuda();x=x.cuda();opt=torch.optim.Adam(net.parameters(),lr=.001);durations=[]
for i in range(202):
 torch.cuda.synchronize();started=time.perf_counter();opt.zero_grad();loss=(net(x)-h).square().mean();loss.backward();opt.step();torch.cuda.synchronize()
 if i>=2:durations.append(time.perf_counter()-started)
training_hours=sum(r['projected_pretrain_hours_per_method'] for r in measurements.values())
surrogate_hours=float(np.median(durations))*(len(MAIN_CASES)*len(SEEDS)*50000)/3600
# Discovery includes sequential SAT, unknown generated lengths and candidate-specific derivatives/PINN.
# Report measured components plus an explicit planning allowance, not a fabricated total benchmark.
report=dict(gpu=torch.cuda.get_device_name(),fp32=True,tf32=False,formal_results=False,models=measurements,
 measured_surrogate_step_seconds=float(np.median(durations)),projected_pretrain_hours=training_hours,
 projected_surrogate_train_only_hours=surrogate_hours,
 estimated_total_hours_range=[(training_hours+surrogate_hours)*1.3,(training_hours+surrogate_hours)*2.2],
 estimate_note='Same data order and registered raw batches. Total includes a 30–120% planning allowance for generation, CPU least squares, validation, high derivatives, PINN and I/O; replace by actual per-case timings after the first full job.')
if a.hourly_price is not None:report['estimated_compute_yuan_range']=[h*a.hourly_price for h in report['estimated_total_hours_range']]
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
