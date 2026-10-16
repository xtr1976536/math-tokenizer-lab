"""LaTeX no-SAT experiment using the unmodified author's numerical stages."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,time,importlib.metadata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .author import ROOT,CODE,DICTIONARY,ARCHITECTURE,seed_all,tensor_hash,training_namespace,numerical_namespace,verify_source,model_namespace
from .data import augment_training,CASES
from .experiment import run_pinn,dump
from .models import build_models
from .tokenizer_adapters import Adapter
from .latex_codec import (structure_to_latex,latex_to_structure,LatexParser,LatexError,Unsupported,key)
from .frozen_math import load_frozen,sha
from .round1_runtime import shared_surrogate,train_round1,runtime_sha256
from .latex_generation import generate
from .evaluation import classify,coefficient_error


MAIN_CASES=['KdV_equation','Burgers_equation','Wave_equation','Chaffee_Infante_equation','KG_equation',
    'Allen_Cahn','Convection_diffusion_equation','PDE_divide','Eq_6_2_12','PDE_compound','Possion_x_y']

def sync(device):
    if device=='cuda':torch.cuda.synchronize()


class RewardEvaluator:
    def __init__(self,net,database,equation,variables,device,epi):
        self.env=numerical_namespace(device,equation);self.net=net;self.database=database
        self.variables=variables;self.epi=epi;self.cache={};self.mask=self.env['get_mask_invalid'](variables)
    def evaluate(self,attempt):
        r=copy.deepcopy(attempt)
        if not r['stopped']:r.update(status='incomplete',reward=None);return r
        if r.get('text') is None:r.update(status='decode_error',reward=None);return r
        try:gre=r['gre'] if 'gre' in r else latex_to_structure(r['text'])
        except (LatexError,ValueError) as exc:
            r.update(status='unsupported_expression' if isinstance(exc,Unsupported) else 'syntax_error',error=str(exc),reward=None);return r
        # Check physical domain only AFTER generation. Never change token probs.
        if any(float(self.mask[i])==0 for i in gre[1:-1] if i>=6):
            r.update(status='variable_or_numeric_unavailable',gre=gre,reward=None);return r
        try:
            value,self.cache,self.mask=self.env['calculate_reward'](gre[1:].copy(),self.net,self.database,self.cache,self.mask,self.variables,self.epi)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite reward')
        except (np.linalg.LinAlgError,FloatingPointError,ValueError,TypeError,IndexError,ZeroDivisionError,RuntimeError) as exc:
            r.update(status='numerical_failure',gre=gre,error=type(exc).__name__+': '+str(exc),reward=None);return r
        r.update(status='evaluated',gre=gre,reward=float(value));return r


def select_top(evaluated,env):
    if not evaluated:return []
    rewards=torch.tensor([r['reward'] if r['status']=='evaluated' else -float('inf') for r in evaluated],dtype=torch.float32)
    indices,_=env['find_min_no_repeat'](rewards,samples=len(rewards))
    selected=[(i,copy.deepcopy(evaluated[i])) for i in indices if torch.isfinite(rewards[i])]
    for i,r in selected:r['reward']=float(rewards[i])
    return [r for _,r in selected]


def merge_top(previous,current):
    # Original reward ordering and reward deduplication, with the approved
    # missing-candidate rule: retain up to ten actually available outputs.
    if not previous:return copy.deepcopy(current)
    result=copy.deepcopy(previous)
    for r in current:
        arr=np.asarray([x['reward'] for x in result]);positions=np.where(arr-r['reward']<0)[0]
        if np.isin(r['reward'],arr):continue
        if len(positions):result.insert(int(positions[0]),copy.deepcopy(r))
        elif len(result)<10:result.append(copy.deepcopy(r))
        if len(result)>10:result.pop(-1)
    return result


def search_latex(model,adapter,net,database,equation,variables,folder,device,seed,
                 attempts=400,cycles=5,finetune_epochs=5,max_tokens=256,batch_size=16,epi=.2,reference=False,gre_limit=50):
    sync(device);started=time.perf_counter()
    evaluator=RewardEvaluator(net,database,equation,variables,device,epi)
    env=training_namespace(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    criterion=nn.CrossEntropyLoss(ignore_index=0);best=[];history=[];trace=[]
    for cycle in range(cycles):
        if reference:
            seed_all(seed+cycle);samples=[];evaluated=[];model.eval()
            for i in range(attempts):
                sentence=[5]
                while len(sentence)<gre_limit-1:
                    with torch.no_grad():value,_=model.step(sentence,evaluator.mask)
                    sentence.append(int(value))
                    if value==1:break
                # Match the author's reward-side normalization of truncated GRE
                # sequences; direct LaTeX truncation is explicitly not repaired.
                cleaned=evaluator.env['delete_duplicate'](sentence[1:].copy())
                samples.append(dict(attempt=i,ids=sentence,stopped=True,stop_reason='eos' if sentence[-1]==1 else 'author_length_limit',
                    gre=[5]+cleaned,text=structure_to_latex([5]+cleaned)))
                # As in the original loop, reward may update the SAT mask
                # before the next candidate is sampled.
                evaluated.append(evaluator.evaluate(samples[-1]))
        else:
            samples=generate(model,adapter,attempts,max_tokens,2048,batch_size,seed+cycle,1.)
            evaluated=[evaluator.evaluate(r) for r in samples]
        for r in evaluated:r['cycle']=cycle+1
        trace.extend(evaluated);best=merge_top(best,select_top(evaluated,evaluator.env))
        before=tensor_hash(model.state_dict());losses=[]
        if best:
            # Fine-tune on exactly the generated source text; no canonical repair
            # or replacement by a known PDE from the source collection.
            ds=env['MyDataSet']([adapter.sequence(r['gre'] if reference else r['text']) for r in best])
            loader=DataLoader(ds,batch_size=len(best),collate_fn=ds.padding_batch,shuffle=False)
            for _ in range(finetune_epochs):losses.append(env['train_step'](model,loader,optimizer,criterion,1,print_every=None))
        after=tensor_hash(model.state_dict())
        if best and before==after:raise RuntimeError('Continuation model did not update')
        record=dict(cycle=cycle+1,attempts=len(samples),statuses=dict(Counter(r['status'] for r in evaluated)),
            evaluated=sum(r['status']=='evaluated' for r in evaluated),rewards=[r['reward'] for r in best],
            gre=[r['gre'] for r in best],equations=[r['text'] for r in best],before_sha256=before,after_sha256=after,
            continuation_examples=len(best),continuation_epochs=finetune_epochs if best else 0,continuation_losses=losses)
        history.append(record);print('sample',adapter.name,cycle+1,record['statuses'],flush=True)
    report=dict(sat=reference,probability='Author SAT, GRE 80/20 sampling' if reference else 'Full vocabulary softmax, temperature=1; no semantic or special-ID masks',
        attempts_per_cycle=attempts,max_new_tokens=max_tokens,max_raw_bytes=2048,batch_size=batch_size,
        cycles=cycles,continuation_epochs=finetune_epochs,lr=1e-5,epi=epi,
        insufficient_candidates='No refill; train on available ranked outputs; if none, skip update and continue next fixed 400 attempts',
        history=history,attempts=trace,elapsed_seconds=time.perf_counter()-started)
    dump(folder/'search.json',report);torch.save(model.state_dict(),folder/'searched.pt');return history


def result_summary(history,equation,pinn):
    last=history[-1];gre=last['gre'];result=dict(top1='no_evaluable_candidate',top10_correct=False,
        best_reward=last['rewards'][0] if gre else None,pinn_status=pinn['status'],coefficient_relative_error=None)
    if gre:
        result['top1']=classify(gre[0],equation)['status'];result['top10_correct']=any(classify(s,equation)['status']=='correct' for s in gre)
        matches=[r for r in pinn.get('coefficients',[]) if r.get('gre')==list(map(int,gre[0][1:]))]
        if matches and pinn['status']=='completed':result['coefficient_relative_error']=coefficient_error(gre[0],matches[-1]['values'],equation)
    return result


def run(folder,equation='KdV_equation',mode='fixed',capacity=1024,device='cpu',seed=525,
        observations=10000,noise=0,smoke=False,attempts=400,smoke_tokens=16,reference=True,frozen_assets=None):
    folder=Path(folder).resolve()
    if (folder/'manifest.json').exists():raise FileExistsError('Completed run retained')
    if not smoke:
        if device!='cuda' or attempts!=400:raise ValueError('Formal run requires CUDA and exactly 400 attempts per cycle')
        if not reference or noise!=0 or observations!=(30000 if equation=='Possion_x_y' else 10000):
            raise ValueError('First-round main protocol requires all four arms, no noise, and the registered observation count')
        if equation not in MAIN_CASES:raise ValueError('Case outside the first-round main experiment')
    if mode!='fixed' or capacity!=1024:raise ValueError('Only fixed 1024-ID first-round main experiment is enabled')
    if frozen_assets is None:raise ValueError('Independent pretrained frozen tokenizers are required; no handbook vocabulary fitting')
    adapters,frozen=load_frozen(frozen_assets)
    fingerprint=dict(equation=equation,seed=seed,observations=observations,noise=noise,smoke=smoke,
        attempts=attempts,smoke_tokens=smoke_tokens,reference=reference,frozen_sha256=sha(Path(frozen_assets)/'FROZEN.json'),runtime_sha256=runtime_sha256())
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RUN.json').exists():
        if json.loads((folder/'RUN.json').read_text())!=fingerprint:raise ValueError('Partial-run configuration or source changed')
    else:dump(folder/'RUN.json',fingerprint)
    verify_source();torch.set_num_threads(2 if smoke else 8)
    env,net,database,state,shared=shared_surrogate(folder/'shared',equation,observations,noise,seed,device,smoke)
    env['_shared_folder']=folder/'shared'
    extra=[['ut','+','(uux)x']] if equation=='PDE_compound' else []
    sequences,source=augment_training(env['Delete_equation_name'],folder/'shared',seed,extra)
    texts=[structure_to_latex(s) for s in sequences]
    # Each original GRE structure has exactly the same mathematical content.
    for s,t in zip(sequences,texts):
        converted=structure_to_latex(latex_to_structure(t));assert key(LatexParser(t).parse())==key(LatexParser(converted).parse())
    vocab_report=json.loads((Path(frozen_assets)/'tokenizers/manifest.json').read_text())['methods']
    used=texts[:4] if smoke else texts
    max_training=max(len(a.sequence(t)) for a in adapters.values() for t in texts)
    generation_tokens=smoke_tokens if smoke else 256
    positions=512
    if max_training>positions:raise ValueError('A common context increase requires a new reviewed configuration; no truncation')
    if reference:adapters['eqgpt_reference']=Adapter('eqgpt')
    models,initial=build_models(adapters,positions,seed,device)
    variables=['x','t'] if equation in env['canonical_PDEs'] else ['x','y'];epi=.2 if len(variables)==2 and 't' in variables else 1.
    results={};timings={}
    for name,bridge in models.items():
        out=folder/name;out.mkdir(exist_ok=True);adapter=adapters[name]
        sync(device);started=time.perf_counter()
        encoded=[adapter.sequence(s) for s in (sequences[:4] if smoke else sequences)] if name=='eqgpt_reference' else [adapter.sequence(t) for t in used]
        training=train_round1(bridge,encoded,out,device,seed,smoke)
        sync(device);after_train=time.perf_counter();net.load_state_dict(state);net.eval();database.grad=None
        if (out/'search.json').exists() and (out/'searched.pt').exists():
            history=json.loads((out/'search.json').read_text())['history']
            saved=torch.load(out/'searched.pt',map_location=device,weights_only=True)
            if tensor_hash(saved)!=history[-1]['after_sha256']:raise ValueError('Search checkpoint mismatch')
            bridge.load_state_dict(saved)
        else:
            history=search_latex(bridge if name=='eqgpt_reference' else bridge.core,adapter,net,database,equation,variables,out,device,seed,
                     attempts=attempts,cycles=2 if smoke else 5,finetune_epochs=1 if smoke else 5,
                     max_tokens=generation_tokens,batch_size=16,epi=epi,reference=name=='eqgpt_reference',gre_limit=10 if smoke else 50)
        sync(device);after_search=time.perf_counter()
        if not (out/'pinn.json').exists():
            if history[-1]['gre']:run_pinn(env.copy(),net,state,history,out,device,smoke,seed)
            else:dump(out/'pinn.json',dict(status='no_evaluable_candidate',coefficients=[],initial_surrogate_sha256=tensor_hash(state)))
        sync(device);after_pinn=time.perf_counter()
        pinn=json.loads((out/'pinn.json').read_text())
        results[name]=dict(training=training,**result_summary(history,equation,pinn))
        saved=torch.load(out/'searched.pt',map_location=device,weights_only=True)
        assert tensor_hash(saved)==tensor_hash(bridge.state_dict())
        reloaded=model_namespace(adapter.vocab_size,positions,device)['GPT']().to(device)
        reloaded.load_state_dict(saved);reloaded.eval();bridge.core.eval()
        sample=adapter.sequence(sequences[0] if name=='eqgpt_reference' else texts[0])
        sample=torch.tensor([sample],dtype=torch.long,device=device)
        with torch.no_grad():
            original,_=bridge.core(sample);replayed,_=reloaded(sample)
        if not torch.equal(original,replayed):raise RuntimeError('Checkpoint inference differs after reload')
        results[name]['checkpoint_reload_exact']=True
        del reloaded,saved,original,replayed
        timings[name]=dict(train_seconds=sum(h['seconds'] for h in training['history']),
            search_seconds=json.loads((out/'search.json').read_text())['elapsed_seconds'],pinn_seconds=pinn.get('elapsed_seconds',0),
            invocation_seconds=after_pinn-started)
    manifest=dict(version='eqgpt-round1-main-v1',equation=equation,seed=seed,smoke=smoke,formal_results=not smoke,
        mode=mode,requested_capacity=capacity if mode=='fixed' else None,shared=shared,training_source=source,
        raw_latex_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
        model_initialization=initial,frozen_tokenizers_sha256=fingerprint['frozen_sha256'],tokenizer_corpus=frozen['corpus'],tokenizers=vocab_report,positions=positions,architecture=ARCHITECTURE,
        runtime_sha256=fingerprint['runtime_sha256'],versions={n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','scipy','tokenizers')},
        protocol=dict(attempts_per_cycle=attempts,cycles=2 if smoke else 5,sat_for_latex=False,
           training_epochs=1 if smoke else 100,generation_max_tokens=generation_tokens,generation_max_bytes=2048,
           temperature=1.,uniform_scientific_exploration=False,whole_formula_tokenization=True,
           reference='Author GRE/SAT system comparison' if reference else None),results=results)
    dump(folder/'manifest.json',manifest);dump(folder/'timing.json',timings);return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--equation',choices=CASES,default='KdV_equation')
    p.add_argument('--mode',choices=['fixed'],default='fixed');p.add_argument('--capacity',type=int,default=1024)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--seed',type=int,default=525)
    p.add_argument('--observations',type=int,default=10000);p.add_argument('--noise',type=float,default=0)
    p.add_argument('--smoke',action='store_true');p.add_argument('--attempts',type=int,default=400)
    p.add_argument('--smoke-tokens',type=int,default=16);p.add_argument('--no-reference',action='store_true')
    p.add_argument('--tokenizer-dir',type=Path,default=ROOT/'experiments/eqgpt_round1/frozen_math')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=run(a.output,a.equation,a.mode,a.capacity,a.device,a.seed,a.observations,a.noise,a.smoke,a.attempts,a.smoke_tokens,not a.no_reference,a.tokenizer_dir)
    print(json.dumps(dict(results=m['results'],mode=m['mode'],positions=m['positions']),indent=2))
if __name__=='__main__':main()
