"""Dev-only route optimization. No test split is loaded by this script."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import RELEASE, load_split
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.rl_route_v6 import AtomicTrajectoryV5, AtomicVocab
from math_tokenizer.rl_vocab_v5 import RLVocab, VocabRLConfig, internal_split, MergeSession
from math_tokenizer.trajectory_v5 import TrajectoryConfig, TrajectoryVocab
from math_tokenizer.measure_v6 import measure, selection_utility

OUT=ROOT/'results/v6'
AST_OPTIONS=[{'atom_budget':256,'subtree_budget':0},{'atom_budget':384,'subtree_budget':0},{'atom_budget':256,'subtree_budget':128}]
RL_OPTIONS={'atomic_terminal':{'atom_budget':384,'temperature':1.,'tree_weight':.25},
            'atomic_explore':{'atom_budget':384,'temperature':3.,'tree_weight':.25}}


def write(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def fixed_dev(rows,limit=1200):
    groups={}
    for r in rows:groups.setdefault(r['split_group'],[]).append(r)
    rs=[r for g,records in sorted(groups.items()) for r in sorted(records,key=lambda r:hashlib.sha256(('v6-dev:'+r['id']).encode()).digest())[:24]]
    return sorted(rs,key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:limit]


def score(model,dev,name):
    m,_=measure(model,dev,name,repeats=1)
    return {k:m[k] for k in ('tokens_per_formula','document_macro_tokens_per_byte','command_integrity','number_integrity','independent_tree_crossing','encode_mib_s')}|{'utility':selection_utility(m)}


def atomic_frequency(config,atoms,train,capacity):
    started=time.perf_counter();model=AtomicVocab(config,atoms);session=MergeSession(model,train)
    while len(model.vocab)<capacity:
        pairs,x,rewards,parts=session.candidates(frequency_only=True)
        if not pairs:break
        session.commit(pairs[0],parts[0],np.zeros(22),float(rewards[0]),'atomic_frequency_control')
    session.finish(capacity,0,'atomic_frequency_control',time.perf_counter()-started);return model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capacity',type=int,default=1024);parser.add_argument('--scenario',choices=['broad','specialized'],default='broad');parser.add_argument('--phase',choices=['ast','rl','all'],default='all');parser.add_argument('--rebuild-native',action='store_true');args=parser.parse_args()
    capacity=args.capacity;scenario=args.scenario;out=OUT/f'{scenario}_{capacity}';out.mkdir(parents=True,exist_ok=True);models=out/'models';models.mkdir(exist_ok=True)
    manifest=json.loads((RELEASE/'manifest.json').read_text());quality=json.loads((RELEASE/'quality_audit.json').read_text());seal=json.loads((RELEASE/'seal.json').read_text())
    assert quality['all_passed'] and manifest['corpus_sha256']==quality['corpus_sha256']==seal['corpus_sha256']
    conf=VocabRLConfig(fit_limit=1000000,reward_limit=512,candidates=12,initial_frequency_prior=0.)
    train=load_split('train' if scenario=='broad' else 'transfer_train');fit,reward=internal_split(train,conf)
    dev=fixed_dev(load_split('dev' if scenario=='broad' else 'transfer_dev'))
    write(out/'scope.json',{'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],'dev_ids':[r['id'] for r in dev],
        'fit_groups':sorted({r['split_group'] for r in fit}),'reward_groups':sorted({r['split_group'] for r in reward}),
        'all_scenario_training_ids':[r['id'] for r in train],'config':asdict(conf),'corpus_sha256':manifest['corpus_sha256']})
    write(out/'search_protocol.json',{'capacity':capacity,'scenario':scenario,'ast_options':AST_OPTIONS,'rl_options':RL_OPTIONS,
        'rl_selection_seeds':[17,29],'formal_seeds':[17,29,43],'rl_epochs':2,'rollouts_per_epoch':3,
        'selection':'docmacro tokens/byte+.15 command error+.03 number error+.10 independent KaTeX crossing',
        'test_loads':0,'previous_theta_used':False,'creation_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    phases=['ast','rl'] if args.phase=='all' else [args.phase]
    for phase in phases:
        if (out/(phase+'_selection.json')).exists() and not (phase=='ast' and args.rebuild_native):continue
        results=[];pool={}
        if phase=='ast':
            if (models/'ast_v4_previous.json').exists():parent=ConstrainedTokenizer.load(models/'ast_v4_previous.json')
            else:
                trainer=GrammarBPE(True).train([r['latex'] for r in fit],capacity,atom_budget=256)
                parent=ConstrainedTokenizer(trainer.vocab,True,'commands','shortest');parent.train_seconds=trainer.train_seconds;parent.capacity=capacity
                parent.save(models/'ast_v4_previous.json')
            results.append({'name':'ast_v4_previous','score':score(parent,dev,'ast_v4_previous'),'train_seconds':parent.train_seconds});print(scenario,capacity,results[-1],flush=True)
            # Current active dictionary route is the real parent control.
            dictionary=augment_dictionary(parent,fit,reward,capacity,extra_subtrees=512)
            dictionary.save(models/'ast_v4_previous_dictionary.json')
            results.append({'name':'ast_v4_previous_dictionary','score':score(dictionary,dev,'ast_v4_previous_dictionary'),'train_seconds':dictionary.train_seconds});print(scenario,capacity,results[-1],flush=True)
            for i,option in enumerate(AST_OPTIONS):
                name='native_'+str(i);model=NativeASTV4.train([r['latex'] for r in fit],capacity,**option);model.save(models/(name+'.json'));pool[name]=model
                results.append({'name':name,'option':option,'score':score(model,dev,name),'train_seconds':model.train_seconds});write(out/'ast_dev.json',results);print(scenario,capacity,results[-1],flush=True)
            selected=min([r for r in results if r['name'] in pool],key=lambda r:r['score']['utility'])
            pool[selected['name']].save(models/'ast_v4_selected.json')
            write(out/'ast_selection.json',{'selected':selected,'test_used':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        else:
            # Baseline from scratch on v6; no previously exposed OOD weights.
            parent=RLVocab(conf).train(train,capacity,17,'rl');parent.save(models/'v5_previous_local.json')
            results.append({'name':'v5_previous_local','seed':17,'score':score(parent,dev,'v5_previous_local'),'train_seconds':parent.train_seconds});print(scenario,capacity,results[-1],flush=True)
            for name,option in RL_OPTIONS.items():
                for seed in (17,29):
                    trajectory=TrajectoryConfig(epochs=2,samples=3,temperature=option['temperature'],tree_weight=option['tree_weight'])
                    model=AtomicTrajectoryV5(conf,trajectory,option['atom_budget']).train(train,capacity,seed,progress=lambda e,s,v:print('rollout',scenario,capacity,name,seed,e,s,round(v['utility'],6),flush=True))
                    model.save(models/(name+'_'+str(seed)+'.json'));pool[(name,seed)]=model
                    results.append({'name':name,'seed':seed,'option':option,'score':score(model,dev,name),'train_seconds':model.train_seconds,'initial_terminal':model.initial_stats,'final_terminal':model.final_stats,'policy_change_norm':float(np.linalg.norm(model.theta-model.initial_theta))})
                    write(out/'rl_dev.json',results);print(scenario,capacity,results[-1],flush=True)
            means={name:float(np.mean([r['score']['utility'] for r in results if r['name']==name])) for name in RL_OPTIONS};chosen=min(means,key=means.get)
            write(out/'rl_selection.json',{'selected':chosen,'option':RL_OPTIONS[chosen],'mean_dev_utility':means,'test_used':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            # Same search budget / atomization controls are compulsory.
            option=RL_OPTIONS[chosen];trajectory=TrajectoryConfig(epochs=2,samples=3,temperature=option['temperature'],tree_weight=option['tree_weight'])
            for seed in (17,29):pool[(chosen,seed)].save(models/('v5_selected_'+str(seed)+'.json'))
            model=AtomicTrajectoryV5(conf,trajectory,option['atom_budget']).train(train,capacity,43,progress=lambda e,s,v:print('formal rollout',scenario,capacity,43,e,s,round(v['utility'],6),flush=True));model.save(models/'v5_selected_43.json')
            fixed=AtomicTrajectoryV5(conf,trajectory,option['atom_budget']).train(train,capacity,17,update=False);fixed.save(models/'v5_no_update.json');fixed.best_sample.save(models/'v5_search_control.json')
            frequency=atomic_frequency(conf,fixed.atoms,train,capacity);frequency.save(models/'v5_atomic_frequency.json')
            write(out/'controls.json',{'same_atoms':True,'same_no_update_rollout_budget':6,'fixed_terminal':fixed.final_stats,'search_terminal':fixed.best_sample_stats,'frequency_train_seconds':frequency.train_seconds})
    write(out/'ready.json',{'scenario':scenario,'capacity':capacity,'test_used_for_selection':False,'all_test_experiments_pending':True})


if __name__=='__main__':main()
