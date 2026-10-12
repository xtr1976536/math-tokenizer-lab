"""Full scenario pools; actual target-capacity search; three frozen seeds."""
import json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.terminal_search_v8 import SearchConfig,optimize,export,reward_config,rollout
from math_tokenizer.rl_transfer_v7 import TransferConfig
from math_tokenizer.rl_route_v6 import ReferenceReward
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.independent_v7 import additional,source_scopes
import numpy as np


def main():
    byid={r['id']:r for r in load_split('train')+load_split('dev')};metrics=[]
    for scenario in ('broad','specialized'):
        for capacity in (1024,2048):
            folder=ROOT/f'results/v8/{scenario}_{capacity}';models=folder/'models';models.mkdir(parents=True,exist_ok=True)
            scope=json.loads((ROOT/f'results/v7/{scenario}_{capacity}/fairness_protocol.json').read_text())
            fit=[byid[i] for i in scope['fit_ids']];reward=[byid[i] for i in scope['reward_ids']];dev=[byid[i] for i in scope['dev_ids']]
            atoms=[bytes.fromhex(p) for p in scope['shared_atom_hex']]
            config=SearchConfig(fit_limit=1600,reward_limit=128,training_capacity=capacity)
            objective=ReferenceReward(fit,dev,reward_config());cache={r['id']:source_scopes(r['latex']) for r in dev}
            initial=np.zeros(24);initial[0]=2.
            unchanged,_,_=rollout(fit,atoms,capacity,initial,TransferConfig());unchanged.method='v8_no_update';unchanged.save(models/'no_update.json')
            m,_=measure(unchanged,dev,'no_update_v8',repeats=1,training_usage=fit_usage(unchanged,fit));extra,_=additional(unchanged,dev,scope_cache=cache,probe_limit=0)
            m.update(extra);m.update(capacity=capacity,scenario=scenario,seed=0,terminal_validation_utility=objective.score(unchanged)['utility']);metrics.append(m)
            for seed in (17,29,43):
                for algorithm in ('parameter_rl','random_search','cem_search'):
                    path=folder/f'{algorithm}_{seed}_policy.json'
                    if path.exists():policy=json.loads(path.read_text())
                    else:
                        policy=optimize(fit,reward,atoms,config,seed,algorithm);path.write_text(json.dumps(policy,indent=2)+'\n')
                    model=export(fit,reward,atoms,capacity,policy);model.save(models/f'{algorithm}_{seed}.json')
                    m,_=measure(model,dev,algorithm,repeats=1,training_usage=fit_usage(model,fit));extra,_=additional(model,dev,scope_cache=cache,probe_limit=0)
                    m.update(extra);m.update(capacity=capacity,scenario=scenario,seed=seed,terminal_validation_utility=objective.score(model)['utility'],terminal_evaluations=policy['terminal_evaluations'])
                    metrics.append(m);(ROOT/'results/v8/full_terminal_development.json').write_text(json.dumps(metrics,indent=2)+'\n')
                    print('full terminal',capacity,scenario,seed,algorithm,round(m['terminal_validation_utility'],6),flush=True)
    (ROOT/'results/v8/full_training_completion.json').write_text(json.dumps({'models':len(metrics),'all_three_seeds_reported':True,'same_terminal_evaluations':34,'new_external_scores_used':False},indent=2)+'\n')


if __name__=='__main__':main()
