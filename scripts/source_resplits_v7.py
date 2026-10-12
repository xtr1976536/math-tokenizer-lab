"""Three source-disjoint resplits, fresh vocabularies/policies per split.

This is replication on already available development sources, not a new sealed
test. All methods share atoms, fit/reward scopes and actual raw held-out rows.
"""
from collections import defaultdict
import hashlib,json,time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.rl_transfer_v7 import TransferConfig,train_policy,rollout,task_rows
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.measure_v6 import measure,fit_usage


def main():
    folder=ROOT/'results/v7/source_resplits';folder.mkdir(parents=True,exist_ok=True)
    pool=[r for r in load_split('train')+load_split('dev') if r['source_repo']=='arxiv.org'];groups=defaultdict(list)
    for r in pool:groups[r['split_group']].append(r)
    results=[]
    for seed in (71,109,173):
        dest=folder/str(seed);dest.mkdir(exist_ok=True)
        ordered=sorted(groups,key=lambda g:hashlib.sha256(f'v7-split:{seed}:{g}'.encode()).digest())
        hold=set(ordered[:max(4,len(ordered)//5)]);reward_groups=set(ordered[len(hold):len(hold)+max(4,len(ordered)//6)])
        fit=task_rows([r for r in pool if r['split_group'] not in hold|reward_groups],4000,seed)
        reward=task_rows([r for r in pool if r['split_group'] in reward_groups],64,seed)
        test=task_rows([r for r in pool if r['split_group'] in hold],256,seed)
        assert not ({r['split_group'] for r in fit}&{r['split_group'] for r in reward})
        assert not ({r['split_group'] for r in fit+reward}&{r['split_group'] for r in test})
        atoms=derive_atoms(fit,1024,256);config=TransferConfig(epochs=2,samples=3,fit_task_limit=640,reward_task_limit=64,training_capacity=768)
        scope={'seed':seed,'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],
            'heldout_ids':[r['id'] for r in test],'heldout_groups':sorted(hold),'primitive_hex':[p.hex() for p in atoms],
            'scope':'Source-disjoint resampling from previously available v6 train/dev; not unseen historical sources'}
        (dest/'scope.json').write_text(json.dumps(scope,indent=2)+'\n')
        base=AtomicBPE.train(fit+reward,atoms,1024)
        policy,_=train_policy(fit,reward,atoms,config,seed)
        (dest/'RL_training.json').write_text(json.dumps(policy,indent=2)+'\n')
        learned,_,_=rollout(fit,atoms,1024,np.array(policy['policy']),config)
        learned.train_seconds+=policy['training_seconds']
        learned.reward_ids=[r['id'] for r in reward]
        unchanged,_,_=rollout(fit,atoms,1024,np.array(policy['initial_policy']),config)
        balanced=BalancedAST(list(dict.fromkeys(base.vocab)),.75)
        models={'official_atomic_union':base,'AST_balanced_same_vocab':balanced,'RL_terminal_transfer':learned,'RL_no_update':unchanged}
        for name,model in models.items():
            model.save(dest/f'{name}.json')
            m,_=measure(model,test,name,repeats=1,training_usage=fit_usage(model,fit));m.update(seed=seed,
                paper_resplit=True,phase='development_replication',fit_reward_information_available=True)
            results.append(m);(folder/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
            print('source split',seed,name,round(m['document_macro_tokens_per_byte'],5),round(m['independent_tree_crossing'],4),flush=True)
    (folder/'completion.json').write_text(json.dumps({'folds':3,'methods_per_fold':4,'same_heldout_inputs':True,
        'policies_and_vocabularies_trained_from_scratch_each_fold':True,'sealed_external_test_loaded':False},indent=2)+'\n')


if __name__=='__main__':main()
