"""Same terminal objective and number of task evaluations, no policy gradient.

Eight fixed candidate policies each evaluated on both tasks =16 exported
encoder terminal scores, matching four epochs x four RL sampled trajectories.
Candidate selection uses only training-internal reward, never dev/test.
"""
import argparse,json,time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_transfer_v7 import TransferConfig,task_rows,rollout
from math_tokenizer.rl_route_v6 import ReferenceReward
from math_tokenizer.trajectory_v5 import TrajectoryConfig
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];reward=[byid[id] for id in scope['reward_ids']];dev=[byid[id] for id in scope['dev_ids']]
    atoms=[bytes.fromhex(p) for p in scope['shared_atom_hex']];config=TransferConfig();rng=np.random.default_rng(891)
    initial=np.zeros(24);initial[0]=2.;pool=[initial]+[initial+rng.normal(0,.5,24) for _ in range(7)]
    started=time.perf_counter();tasks=[]
    for task in (0,1):
        f=task_rows(fit,config.fit_task_limit,task);r=task_rows(reward,config.reward_task_limit,task)
        tasks.append((f,ReferenceReward(f,r,TrajectoryConfig(tree_weight=config.tree_weight))))
    results=[]
    for i,theta in enumerate(pool):
        values=[]
        for f,objective in tasks:
            model,_,_=rollout(f,atoms,1024,theta,config);values.append(objective.score(model))
        value=float(np.mean([v['utility'] for v in values]));results.append({'index':i,'policy':theta.tolist(),'tasks':values,'mean_training_reward_utility':value})
        print('search control',a.scenario,i,round(value,6),flush=True)
    best=min(results,key=lambda r:r['mean_training_reward_utility']);search_seconds=time.perf_counter()-started
    model,_,_=rollout(fit,atoms,a.capacity,np.array(best['policy']),config);model.train_seconds=time.perf_counter()-started;model.reward_ids=scope['reward_ids']
    model.logs[0].update(method='random_policy_search_no_RL_updates',terminal_evaluations=16,search_seconds=search_seconds)
    path=folder/'models/v5_policy_search_control.json';model.save(path)
    m,_=measure(model,dev,'v5_policy_search_control',repeats=1,training_usage=fit_usage(model,fit));m.update(dev_selection_utility=selection_utility(m),phase='development',terminal_evaluations=16)
    (folder/'policy_search_dev.json').write_text(json.dumps(m,indent=2)+'\n')
    (folder/'policy_search_training.json').write_text(json.dumps({'candidates':results,'selected':best['index'],'selection_test_or_dev_used':False,
        'terminal_evaluations':16,'matches_RL_task_evaluation_count':True,'same_terminal_objective':True,'search_seconds':search_seconds},indent=2)+'\n')
    print(a.scenario,m['document_macro_tokens_per_byte'],m['independent_tree_crossing'],flush=True)


if __name__=='__main__':main()
