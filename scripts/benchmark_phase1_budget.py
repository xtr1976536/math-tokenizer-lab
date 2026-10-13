"""Local 2-epoch timing smoke and transparent Phase 1 cost extrapolation."""
from __future__ import annotations
import json,math,time
from pathlib import Path
import torch
from math_tokenizer.corpus import ROOT
from math_tokenizer.downstream_gpt2.orchestrate import assemble
from math_tokenizer.downstream_gpt2.audit import dataset_audit
from math_tokenizer.downstream_gpt2.phases import PhaseAdapter,SEEDS,freeze_coverage,load_checkpoint
from math_tokenizer.downstream_gpt2.ood import generate as generate_ood
from math_tokenizer.downstream_gpt2.train import train_one,read_split,examples_for,evaluate
from math_tokenizer.downstream_gpt2.verify import Verifier
from math_tokenizer.downstream_gpt2.adapters import method_slug

OUT=ROOT/'tmp/phase1_timing'
METHODS=('ast_boundary','official_atomic_bpe')

def save(report):
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')

def project(report):
    formal_train=48000
    steps_per_method=math.ceil(formal_train/32)*2*len(SEEDS)
    total_steps=steps_per_method*len(METHODS)
    projected={}
    for adaptation in ('full','lora'):
        seconds=0
        for method in METHODS:
            t=report['trainings'][adaptation+'|'+method]
            seconds+=(t['per_epoch_train_seconds'][-1]/t['updates_per_epoch'])*steps_per_method
        projected[adaptation]={'local_mps_training_seconds':seconds,'training_hours':seconds/3600,
                               'seconds_per_update_for_five_hours_training_only':18000/total_steps}
    report['projection']={'training_seeds':list(SEEDS),'formal_optimizer_updates_total':total_steps,
        'training_only_mps':projected,
        'serial_cpu_data_hours_at_100x_sample':(report['generation']['seconds']+report['ood_generation_seconds']+report['data_audit_seconds'])*100/3600,
        'planned_validation_questions_before_length_filter':3600*2*len(METHODS)*len(SEEDS),
        'planned_test_greedy_questions_before_length_filter':(12000+7000)*(len(METHODS)*len(SEEDS)+1),
        'planned_beam10_questions_before_length_filter':1000*(len(METHODS)*len(SEEDS)+1),
        'caution':'Kaggle CPU/GPU speed and queue are unmeasured; do not claim a 5-hour guarantee from MPS timing.'}
    return report

def main():
    raw=OUT/'raw_verified';data=OUT/'data_verified';gpt2=ROOT/'tmp/gpt2-607a30d';source=ROOT/'tmp/SymbolicMathematics'
    manifest=json.loads((raw/'manifest.json').read_text())
    clock=time.perf_counter();assembled=assemble(raw,raw,raw,data,manifest['quotas'],source)
    audit=dataset_audit(data,source);coverage=freeze_coverage(data)
    data_audit_seconds=time.perf_counter()-clock
    if any(audit['oracle_unverified'].values()):raise RuntimeError('oracle prefilter failed')
    report={'status':'running','generation':json.loads((raw/'generation_timing.json').read_text()),
            'data_audit_seconds':data_audit_seconds,'raw_quotas':manifest['quotas'],
            'eligible_counts':{k:len(v) for k,v in coverage['kept'].items()},
            'quota_deviation':assembled['quota_deviation'],'trainings':{},'inference':{}}
    save(report)
    clock=time.perf_counter()
    for mode in ('length','composition'):generate_ood(mode,source,data,OUT/'ood_verified',.01)
    report['ood_generation_seconds']=time.perf_counter()-clock;save(report)
    for adaptation in ('full','lora'):
        for method in METHODS:
            start=time.perf_counter()
            path=train_one(method,11,data,gpt2,OUT/adaptation/'runs',epochs=2,microbatch=2,
                           effective_batch=32,eval_limit=5,device_name='mps',mode=adaptation,prompt_style='text-v2')
            history=json.loads((path/'history.json').read_text());config=json.loads((path/'config.json').read_text())
            report['trainings'][adaptation+'|'+method]={'wall_seconds':time.perf_counter()-start,
                'per_epoch_train_seconds':[r['train_seconds'] for r in history],
                'per_epoch_validation_seconds':[r['validation_seconds'] for r in history],
                'updates_per_epoch':math.ceil(config['train_examples']/32),
                'eligible_train_examples':config['train_examples'],
                'trainable_parameters':config['trainable_parameters'],
                'frozen_unchanged':config.get('frozen_before_sha256')==config.get('frozen_after_sha256') if adaptation=='lora' else None}
            save(report)
    verifier=Verifier(source);test=read_split(data,'test');allowed=set(coverage['kept']['test'])
    sample=[];seen={}
    for row in test:
        if row['id'] not in allowed:continue
        key=row['method'] if row['task']=='integral' else row['task']
        if seen.get(key,0)<2:sample.append(row);seen[key]=seen.get(key,0)+1
    for name,mode in [('full|ast_boundary','full'),('lora|ast_boundary','lora'),('gpt2_native','full')]:
        method='gpt2_native' if name=='gpt2_native' else 'ast_boundary'
        folder=None if method=='gpt2_native' else OUT/mode/'runs'/method_slug(method)/'11'/'best'
        model=load_checkpoint(method,gpt2,folder,OUT/'eval_init',mode).to('mps').eval()
        adapter=PhaseAdapter(method,gpt2);examples=examples_for(adapter,sample,{r['id'] for r in sample})
        start=time.perf_counter();scores,pred=evaluate(model,adapter,examples,verifier,adapter.masked_ids(),'mps',protocol='text-v2')
        report['inference'][name]={'greedy_seconds':time.perf_counter()-start,'greedy_questions':len(examples),
                                   'scores_smoke_only':scores}
        start=time.perf_counter()
        _score,_pred=evaluate(model,adapter,examples[:2],verifier,adapter.masked_ids(),'mps',beams=10,protocol='text-v2')
        report['inference'][name].update(beam10_seconds=time.perf_counter()-start,beam10_questions=min(2,len(examples)))
        del model;torch.mps.empty_cache();save(report)
    project(report)
    report['status']='complete';save(report)
    print(json.dumps(report['projection'],indent=2),flush=True)

if __name__=='__main__':main()
