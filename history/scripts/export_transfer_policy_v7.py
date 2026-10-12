"""Freeze trained policy and transfer across capacity/domain without updates."""
import argparse,json,time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_transfer_v7 import TransferConfig,rollout
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='specialized');p.add_argument('--capacity',type=int,default=1024)
    p.add_argument('--policy-scenario',choices=['broad','specialized'],default='broad');a=p.parse_args()
    source=ROOT/f'results/v7/{a.policy_scenario}_1024/v5_transfer_tree0.5_e4_s17_policy.json';policy=json.loads(source.read_text())
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];dev=[byid[id] for id in scope['dev_ids']]
    atoms=[bytes.fromhex(p) for p in scope['shared_atom_hex']];config=TransferConfig(**policy['config']);stats=[]
    for variant,theta,method in [('learned',np.array(policy['policy']),'policy'),('no_update',np.array(policy['initial_policy']),'policy')]:
        name='v5_cross_task_'+variant+('_strict' if a.policy_scenario=='specialized' else '');path=folder/'models'/f'{name}.json'
        if path.exists():
            from math_tokenizer.rl_route_v6 import AtomicVocab
            model=AtomicVocab.load(path)
        else:
            start=time.perf_counter();model,_,_=rollout(fit,atoms,a.capacity,theta,config,method=method);export=time.perf_counter()-start
            model.train_seconds=export+(policy['training_seconds'] if variant=='learned' else 0)
            model.logs[0].update(policy_source=str(source.relative_to(ROOT)),policy_unchanged=True,
                reward_used_at_export=False,policy_training_seconds=policy['training_seconds'] if variant=='learned' else 0,
                export_seconds=export,policy_cost_amortization_not_assumed=True)
            model.logs[0]['strict_ood_eligible']=a.scenario!='specialized' or a.policy_scenario=='specialized'
            model.save(path)
        m,_=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit));m.update(dev_selection_utility=selection_utility(m),phase='development',policy_frozen_transfer=True,capacity=a.capacity)
        m['strict_ood_eligible']=a.scenario!='specialized' or a.policy_scenario=='specialized'
        stats.append(m);(folder/('cross_task_transfer'+('_strict' if a.policy_scenario=='specialized' else '')+'_dev.json')).write_text(json.dumps(stats,indent=2)+'\n')
        print(a.scenario,a.capacity,name,round(m['document_macro_tokens_per_byte'],5),round(m['independent_tree_crossing'],4),flush=True)


if __name__=='__main__':main()
