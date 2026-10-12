"""Train future-credit policy on subsets, transfer once to full fit vocabulary."""
import argparse,json,time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_transfer_v7 import TransferConfig,train_policy,rollout
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--tree-weight',type=float,default=.5);p.add_argument('--epochs',type=int,default=4);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];reward=[byid[id] for id in scope['reward_ids']];dev=[byid[id] for id in scope['dev_ids']]
    atoms=[bytes.fromhex(p) for p in scope['shared_atom_hex']];config=TransferConfig(epochs=a.epochs,tree_weight=a.tree_weight,training_capacity=min(1024,a.capacity))
    name=f'v5_transfer_tree{a.tree_weight}_e{a.epochs}_s{a.seed}'
    policy,best=train_policy(fit,reward,atoms,config,a.seed,progress=lambda e,s,m:print('RL rollout',e,s,round(m['utility'],6),flush=True))
    (folder/f'{name}_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    stats=[]
    for variant,theta,method in [('learned',np.array(policy['policy']),'policy'),
                                 ('no_update',np.array(policy['initial_policy']),'policy'),
                                 ('frequency',np.zeros(24),'frequency')]:
        shared=folder/'models'/f'v5_transfer_tree0.5_e4_s17_{variant}.json'
        if variant!='learned' and shared.exists() and name!='v5_transfer_tree0.5_e4_s17':
            from math_tokenizer.rl_route_v6 import AtomicVocab
            model=AtomicVocab.load(shared)
            assert model.atoms==atoms and model.fit_ids==[r['id'] for r in fit]
            export_seconds=model.train_seconds
            model.logs[0]['shared_deterministic_control']=str(shared.relative_to(ROOT))
        else:
            started=time.perf_counter();model,_,_=rollout(fit,atoms,a.capacity,theta,config,method=method)
            export_seconds=time.perf_counter()-started;model.train_seconds=policy['training_seconds']+export_seconds if variant=='learned' else export_seconds
        model.logs[0].update(policy_learning_seconds=policy['training_seconds'] if variant=='learned' else 0,
                            export_seconds=export_seconds,training_policy_path=f'{name}_policy.json',variant=variant)
        model.reward_ids=scope['reward_ids'] if variant=='learned' else []
        model.save(folder/'models'/f'{name}_{variant}.json')
        m,_=measure(model,dev,f'{name}_{variant}',repeats=1,training_usage=fit_usage(model,fit));m['dev_selection_utility']=selection_utility(m)
        m.update(phase='development',offline_policy_only=True,export_seconds=export_seconds,policy_seconds=policy['training_seconds'] if variant=='learned' else 0)
        stats.append(m);print(m['method'],round(m['document_macro_tokens_per_byte'],5),round(m['independent_tree_crossing'],4),round(m['dev_selection_utility'],5),flush=True)
        (folder/f'{name}_dev.json').write_text(json.dumps(stats,indent=2)+'\n')
    (folder/f'{name}_dev.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__=='__main__':main()
