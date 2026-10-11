"""Prepare both deterministic current routes at 2048, dev choices from1024."""
import argparse,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_vocab_v5 import VocabRLConfig,internal_split,RLVocab
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.rl_structural_v6 import train_structural
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.dictionary_v6 import augment_native
from train_routes_v6 import write,atomic_frequency

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');a=p.parse_args();scenario=a.scenario;capacity=2048
    parent=ROOT/f'results/v6/{scenario}_1024';out=ROOT/f'results/v6/{scenario}_2048';models=out/'models';models.mkdir(parents=True,exist_ok=True)
    ast=json.loads((parent/'ast_selection.json').read_text());structural=json.loads((parent/'structural_selection.json').read_text())
    conf=VocabRLConfig(fit_limit=1000000,reward_limit=512);train=load_split('train' if scenario=='broad' else 'transfer_train');fit,reward=internal_split(train,conf)
    write(out/'scope.json',json.loads((parent/'scope.json').read_text()))
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
    if not (models/'v5_atomic_frequency.json').exists():atomic_frequency(conf,derive_atoms(fit,capacity,384),train,capacity).save(models/'v5_atomic_frequency.json')
    write(out/'core_ready.json',{'capacity':capacity,'scenario':scenario,'final_v5_selection_pending':True});print('core2048ready',scenario,flush=True)

if __name__=='__main__':main()
