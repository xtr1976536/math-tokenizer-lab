"""Capacity generalization: inherit 1024 dev choices, never tune on test."""
import argparse,json,time
from dataclasses import asdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_vocab_v5 import VocabRLConfig,internal_split,RLVocab
from math_tokenizer.rl_route_v6 import AtomicTrajectoryV5,derive_atoms
from math_tokenizer.trajectory_v5 import TrajectoryConfig
from math_tokenizer.rl_structural_v6 import train_structural
from math_tokenizer.rl_mask_v6 import MaskConfig,BarrierPolicyV5
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.dictionary_v6 import augment_native
from train_routes_v6 import write,atomic_frequency

p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');a=p.parse_args();scenario=a.scenario;capacity=2048
parent=ROOT/f'results/v6/{scenario}_1024';out=ROOT/f'results/v6/{scenario}_2048';models=out/'models';models.mkdir(parents=True,exist_ok=True)
final=json.loads((parent/'final_selection.json').read_text());ast=json.loads((parent/'ast_selection.json').read_text());rl=json.loads((parent/'rl_selection.json').read_text());barrier=json.loads((parent/'barrier_selection.json').read_text());structural=json.loads((parent/'structural_selection.json').read_text())
conf=VocabRLConfig(fit_limit=1000000,reward_limit=512);train=load_split('train' if scenario=='broad' else 'transfer_train');fit,reward=internal_split(train,conf)
scope=json.loads((parent/'scope.json').read_text());write(out/'scope.json',scope)
write(out/'fixed_protocol.json',{'inherited_from':str(parent.relative_to(ROOT)),'capacity':2048,'selection_seeds':[],'formal_seeds':[17,29,43],'test_used':False,'configuration_transfer':True})
if not (out/'ast_selection.json').exists():
    trainer=GrammarBPE(True).train([r['latex'] for r in fit],capacity,atom_budget=256);previous=ConstrainedTokenizer(trainer.vocab,True,'commands','shortest');previous.capacity=capacity;previous.train_seconds=trainer.train_seconds;previous.save(models/'ast_v4_previous.json')
    augment_dictionary(previous,fit,reward,capacity,extra_subtrees=512).save(models/'ast_v4_previous_dictionary.json')
    model=NativeASTV4.train([r['latex'] for r in fit],capacity,**ast['selected']['option']);model.save(models/'ast_v4_native_before_dictionary.json')
    extra=ast['dictionary_refinement']['selected_extra']
    if extra:model=augment_native(model,fit,reward,capacity,extra)
    model.save(models/'ast_v4_selected.json');write(out/'ast_selection.json',{'inherited_1024':ast,'test_used':False})
if not (models/'v5_previous_local.json').exists():RLVocab(conf).train(train,capacity,17).save(models/'v5_previous_local.json')
weight=structural['selected']['tree_weight']
if not (out/'structural_selection.json').exists():
    model=train_structural(train,capacity,conf,17,weight)
    for seed in (17,29,43):model.seed=seed;model.save(models/f'v5_structural_selected_{seed}.json')
    for method in ('frequency','reward_greedy'):train_structural(train,capacity,conf,17,weight,method).save(models/f'v5_structural_{method}.json')
    write(out/'structural_selection.json',{'inherited_1024':structural,'deterministic_expected_gradient':True,'test_used':False})
option=rl['option'];tc=TrajectoryConfig(epochs=2,samples=3,temperature=option['temperature'],tree_weight=option['tree_weight'])
for seed in ((17,29,43) if final['v5_branch']=='trajectory' else ()):
    path=models/f'v5_selected_{seed}.json'
    if not path.exists():
        model=AtomicTrajectoryV5(conf,tc,option['atom_budget']).train(train,capacity,seed,progress=lambda e,s,m:print('2048 trajectory',scenario,seed,e,s,round(m['utility'],6),flush=True));model.save(path)
if final['v5_branch']=='trajectory' and not (models/'v5_no_update.json').exists():
    fixed=AtomicTrajectoryV5(conf,tc,option['atom_budget']).train(train,capacity,17,update=False);fixed.save(models/'v5_no_update.json');fixed.best_sample.save(models/'v5_search_control.json');atomic_frequency(conf,fixed.atoms,train,capacity).save(models/'v5_atomic_frequency.json')
if not (models/'v5_atomic_frequency.json').exists():atomic_frequency(conf,derive_atoms(fit,capacity,384),train,capacity).save(models/'v5_atomic_frequency.json')
write(out/'rl_selection.json',{'inherited_1024':rl,'test_used':False});write(out/'barrier_selection.json',{'inherited_1024':barrier,'test_used':False})
final=dict(final,inherited_from_1024=True,test_used=False,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
final['selected_v5_paths']=[str(models.joinpath('v5_structural_selected_17.json').relative_to(ROOT))] if final['v5_branch']=='structural' else [str(models.joinpath(f'v5_selected_{s}.json').relative_to(ROOT)) for s in (17,29,43)]
write(out/'final_selection.json',final);print('fixed 2048 ready',scenario,flush=True)
