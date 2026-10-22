"""One-case, reward-preserving EqGPT tokenizer comparison without final PINN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import verify_source
from math_tokenizer.eqgpt.data import augment_training
from math_tokenizer.eqgpt.evaluation import classify
from math_tokenizer.eqgpt.experiment import dump
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.latex_experiment import search_latex
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.round1_runtime import shared_surrogate, train_round1
from math_tokenizer.eqgpt.shared8_config import SHARED8_TARGETS
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.eqgpt_fast_config import ALPHABET, CASES, SIZES
from math_tokenizer.eqgpt.eqgpt_fast_config import EPOCHS, LAYERS, ATTEMPTS, CYCLES, CONTINUATION
from math_tokenizer.eqgpt.eqgpt_fast_config import OBSERVATIONS, SEED
from math_tokenizer.eqgpt.eqgpt_fast_config import load_or_train_tokenizers


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--case',choices=CASES,default='Wave_equation')
    cli.add_argument('--device',choices=('cpu','mps','cuda'),default='mps')
    cli.add_argument('--output',type=Path,default=ROOT/'experiments/eqgpt_fast_reward_20261022')
    cli.add_argument('--shared-pretrain',type=Path)
    args=cli.parse_args()
    run_started=time.perf_counter()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    verify_source()
    torch.set_num_threads(8)
    source_dir=out/'source';source_dir.mkdir(exist_ok=True)
    sequences,source=augment_training(SHARED8_TARGETS,source_dir,SEED)
    texts=[structure_to_latex(seq) for seq in sequences]
    assert source['training_sequences']==6784
    adapters={'eqgpt_reference':Adapter('eqgpt')}
    adapters.update(load_or_train_tokenizers(texts,out/'tokenizers'))
    assert set(adapters)=={'eqgpt_reference',*(f'{m}_{s}' for m in ('ast_bpe','soft_ast_bpe') for s in SIZES)}
    encoded={name:[adapter.sequence(seq if name=='eqgpt_reference' else text)
                    for seq,text in zip(sequences,texts)] for name,adapter in adapters.items()}
    max_train=max(map(len,(seq for rows in encoded.values() for seq in rows)))
    positions=max(50,max_train)
    assert positions<=100,positions
    protocol=dict(case=args.case,seed=SEED,device=args.device,layers=LAYERS,epochs=EPOCHS,
        attempts_per_cycle=ATTEMPTS,cycles=CYCLES,continuation_epochs=CONTINUATION,
        observations=OBSERVATIONS,noise=0,sizes=SIZES,alphabet=ALPHABET,
        positions=positions,max_training_tokens=max_train,target_exclusion=SHARED8_TARGETS,
        reward='unchanged author calculate_reward using shared fitted surrogate',
        pinn='omitted; no coefficient accuracy measured',
        corpus_sha256=source['training_sha256'])
    p=out/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:dump(p,protocol)
    started=time.perf_counter()
    shared_dir=out/'surrogate'/args.case
    env,net,database,state,shared=shared_surrogate(shared_dir,args.case,OBSERVATIONS,0,SEED,args.device,False)
    surrogate_seconds=time.perf_counter()-started
    loss_path=next((shared_dir/'model_save'/args.case).glob('*/loss.txt'))
    loss_lines=loss_path.read_text().splitlines()
    assert len(loss_lines)>=100
    dump(out/'surrogate_quality.json',dict(case=args.case,shared=shared,
        loss_first=loss_lines[0],loss_last=loss_lines[-1],
        loss_file_sha256=digest(loss_path),fit_or_cache_load_seconds=surrogate_seconds,
        cache_sha256=digest(shared_dir/'surrogate_cache.pt')))
    print('SURROGATE_READY',args.case,'seconds',round(surrogate_seconds,2),flush=True)
    models,initial=build_models(adapters,positions,SEED,args.device,n_layers=LAYERS)
    dump(out/'initial_models.json',initial)
    results={}
    for name,bridge in models.items():
        folder=out/args.case/name;folder.mkdir(parents=True,exist_ok=True)
        adapter=adapters[name]
        if args.shared_pretrain:
            origin=args.shared_pretrain.resolve()/'Wave_equation'/name
            for filename in ('pretrained.pt','training.json'):
                target=folder/filename
                if not target.exists():os.link(origin/filename,target)
        started=time.perf_counter()
        training=train_round1(bridge,encoded[name],folder,args.device,SEED,epochs_override=EPOCHS)
        train_seconds=time.perf_counter()-started
        print('TRAIN_COMPLETE',name,'seconds',round(train_seconds,2),flush=True)
        net.load_state_dict(state);net.eval();database.grad=None
        if (folder/'search.json').exists():
            history=json.loads((folder/'search.json').read_text())['history']
        else:
            history=search_latex(bridge if name=='eqgpt_reference' else bridge.core,
                adapter,net,database,args.case,['x','t'],folder,args.device,SEED,
                attempts=ATTEMPTS,cycles=CYCLES,finetune_epochs=CONTINUATION,
                max_tokens=positions,batch_size=16,epi=.2,
                reference=name=='eqgpt_reference',gre_limit=50,
                normalize_ast=name!='eqgpt_reference')
        search=json.loads((folder/'search.json').read_text())
        last=history[-1]
        classes=[classify(seq,args.case)['status'] for seq in last['gre']]
        attempts=[]
        for row in search['attempts']:
            if row.get('status')=='evaluated' and row.get('gre'):
                attempts.append(dict(cycle=row['cycle'],attempt=row['attempt'],
                    status=classify(row['gre'],args.case)['status']))
        results[name]=dict(training_seconds=train_seconds,pretraining_reused=bool(args.shared_pretrain),
            original_pretraining_seconds=sum(r['seconds'] for r in training['history']),
            search_seconds=search['elapsed_seconds'],
            top1=classes[0] if classes else 'no_evaluable_candidate',
            top10_correct='correct' in classes,top10_redundant='redundant' in classes,
            ranked_classes=classes,attempted_statuses={s:sum(r['status']==s for r in attempts)
                for s in ('correct','redundant','incorrect')},
            first_correct=next(((r['cycle']-1)*ATTEMPTS+r['attempt']+1 for r in attempts if r['status']=='correct'),None),
            reward_calls=search['performance']['author_reward_calls'],
            cache_hits=search['performance']['exact_reward_cache_hits'])
        dump(folder/'fast_summary.json',results[name])
        dump(out/'results.json',results)
        print('CASE_ARM_COMPLETE',name,results[name],flush=True)
    dump(out/'complete.json',dict(case=args.case,results=results,
        elapsed_seconds=time.perf_counter()-run_started,surrogate_sha256=shared['surrogate_sha256']))
    print('CASE_COMPLETE',args.case,flush=True)


if __name__=='__main__':main()
