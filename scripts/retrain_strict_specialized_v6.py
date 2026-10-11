"""Source-only OOD repair; all hyperparameters inherited from pre-test dev."""
import json,time,subprocess
from dataclasses import asdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split,RELEASE
from math_tokenizer.rl_vocab_v5 import VocabRLConfig,internal_split,RLVocab
from math_tokenizer.rl_route_v6 import AtomicTrajectoryV5
from math_tokenizer.trajectory_v5 import TrajectoryConfig,TrajectoryVocab
from math_tokenizer.rl_structural_v6 import train_structural
from math_tokenizer.rl_mask_v6 import BarrierPolicyV5,MaskConfig
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.dictionary_v6 import augment_native
from train_routes_v6 import write,atomic_frequency,fixed_dev


def main():
    saved=json.loads((ROOT/'results/v6/invalidated_specialized_scope_1024.json').read_text())['stats']
    ast=json.loads(saved['ast_selection.json']);rl=json.loads(saved['rl_selection.json']);barrier=json.loads(saved['barrier_selection.json']);structural=json.loads(saved['structural_selection.json']);final=json.loads(saved['final_selection.json'])
    config=VocabRLConfig(fit_limit=1000000,reward_limit=512);train=load_split('transfer_train');dev=fixed_dev(load_split('transfer_dev'));fit,reward=internal_split(train,config)
    manifest=json.loads((RELEASE/'manifest.json').read_text())
    for capacity in (1024,2048):
        out=ROOT/f'results/v6/specialized_{capacity}';models=out/'models';models.mkdir(parents=True,exist_ok=True)
        scope={'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],'dev_ids':[r['id'] for r in dev],
            'fit_groups':sorted({r['split_group'] for r in fit}),'reward_groups':sorted({r['split_group'] for r in reward}),
            'all_scenario_training_ids':[r['id'] for r in train],'config':asdict(config),'corpus_sha256':manifest['corpus_sha256'],
            'verified_arxiv_only':True,'hyperparameters_fixed_before_partial_test':True}
        write(out/'scope.json',scope)
        old=GrammarBPE(True).train([r['latex'] for r in fit],capacity,atom_budget=256);previous=ConstrainedTokenizer(old.vocab,True,'commands','shortest');previous.capacity=capacity;previous.train_seconds=old.train_seconds;previous.save(models/'ast_v4_previous.json')
        augment_dictionary(previous,fit,reward,capacity,extra_subtrees=512).save(models/'ast_v4_previous_dictionary.json')
        model=NativeASTV4.train([r['latex'] for r in fit],capacity,**ast['selected']['option']);model.save(models/'ast_v4_native_before_dictionary.json')
        extra=ast['dictionary_refinement']['selected_extra']
        if extra:model=augment_native(model,fit,reward,capacity,extra)
        model.save(models/'ast_v4_selected.json');write(out/'ast_selection.json',{'inherited_pretest':ast,'test_used':False,'fixed_hyperparameters_after_source_repair':True})
        parent=RLVocab(config).train(train,capacity,17);parent.save(models/'v5_previous_local.json')
        weight=structural['selected']['tree_weight'];model=train_structural(train,capacity,config,17,weight)
        for seed in (17,29,43):model.seed=seed;model.save(models/f'v5_structural_selected_{seed}.json')
        for method in ('frequency','reward_greedy'):train_structural(train,capacity,config,17,weight,method).save(models/f'v5_structural_{method}.json')
        write(out/'structural_selection.json',{'inherited_pretest':structural,'test_used':False,'deterministic_expected_gradient':True})
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train_baselines_v6.py'),'--scenario','specialized','--capacity',str(capacity)],check=True)
        if capacity==1024:
            option=rl['option'];tc=TrajectoryConfig(epochs=2,samples=3,temperature=option['temperature'],tree_weight=option['tree_weight'])
            for seed in (17,29,43):
                model=AtomicTrajectoryV5(config,tc,option['atom_budget']).train(train,capacity,seed,progress=lambda e,s,m:print('strict source trajectory',seed,e,s,round(m['utility'],6),flush=True));model.save(models/f'v5_selected_{seed}.json')
            fixed=AtomicTrajectoryV5(config,tc,option['atom_budget']).train(train,capacity,17,update=False);fixed.save(models/'v5_no_update.json');fixed.best_sample.save(models/'v5_search_control.json');atomic_frequency(config,fixed.atoms,train,capacity).save(models/'v5_atomic_frequency.json')
            mc=MaskConfig(**barrier['config'])
            for seed in (17,29,43):
                trainer=BarrierPolicyV5(config,mc);model=trainer.train(train,capacity,seed);model.save(models/f'v5_barrier_selected_{seed}.json');trainer.frequency.save(models/'v5_barrier_frequency.json')
            trainer=BarrierPolicyV5(config,mc);trainer.train(train,capacity,17,update=False).save(models/'v5_barrier_no_update.json');trainer.best_sample.save(models/'v5_barrier_search_control.json')
            old=TrajectoryVocab(config,TrajectoryConfig(epochs=3,samples=3,temperature=3.)).train(train,capacity,17,parent.theta,progress=lambda e,s,m:print('strict source previous trajectory',e,s,round(m['utility'],6),flush=True));old.train_seconds+=parent.train_seconds;old.save(models/'v5_previous_trajectory.json')
        else:
            from math_tokenizer.rl_route_v6 import derive_atoms
            atomic_frequency(config,derive_atoms(fit,capacity,384),train,capacity).save(models/'v5_atomic_frequency.json')
        write(out/'rl_selection.json',{'inherited_pretest':rl,'test_used':False});write(out/'barrier_selection.json',{'inherited_pretest':barrier,'test_used':False})
        result=dict(final,test_used=False,fixed_hyperparameters_after_source_repair=True,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        result['selected_v5_paths']=[str((models/'v5_structural_selected_17.json').relative_to(ROOT))];write(out/'final_selection.json',result)
        print('strict source model scope finished',capacity,flush=True)


if __name__=='__main__':main()
