import numpy as np
from math_tokenizer.terminal_search_v8 import gaussian_score, optimize, SearchConfig


def test_gaussian_policy_score_matches_numerical_log_density_derivative():
    theta=np.array([.4,-.2,.7]); mean=np.array([.1,.3,-.1]); sigma=.35
    def logdensity(center): return -np.sum((theta-center)**2)/(2*sigma**2)
    eps=1e-6
    numeric=[]
    for i in range(3):
        offset=np.zeros(3);offset[i]=eps
        numeric.append((logdensity(mean+offset)-logdensity(mean-offset))/(2*eps))
    assert np.allclose(gaussian_score(theta,mean,sigma),numeric,rtol=1e-6)


def test_terminal_algorithms_share_first_batch_reward_and_exact_budget():
    atoms=[bytes([b]) for b in range(256)]
    fit=[{'id':f'f{i}','latex':'x+x=y' if i%2 else 'y*x+x=1','split_group':f'f:{i}'} for i in range(6)]
    reward=[{'id':f'r{i}','latex':'x+y=1','split_group':f'r:{i}','reference':{'spans':[[0,3]]}} for i in range(4)]
    config=SearchConfig(rounds=2,samples=2,fit_limit=6,reward_limit=4,training_capacity=264,candidates=4)
    records=[optimize(fit,reward,atoms,config,17,a) for a in ('parameter_rl','random_search','cem_search')]
    assert all(r['terminal_evaluations']==10 and not r['validation_or_test_used'] for r in records)
    assert all(r['reward_config']==records[0]['reward_config'] for r in records)
    assert all(r['evaluations'][:3]==records[0]['evaluations'][:3] for r in records)
    for r in records:
        assert r['incumbent_utility']==min(e['utility'] for e in r['evaluations'])
        assert all(set(ids)<={v['id'] for v in fit} for ids in r['task_fit_ids'])
        assert all(set(ids)<={v['id'] for v in reward} for ids in r['task_reward_ids'])
