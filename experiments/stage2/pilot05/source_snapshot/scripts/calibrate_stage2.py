"""Additional raw-router comparison with fit/reward-only matched expected rate.

Post-hoc fairness sensitivity, not a new confirmatory experiment. No prediction
loss is consulted when calibrating. All methods get the same 24-step calibration.
"""
import json
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.signal_stage2 import contexts
from math_tokenizer.autonomous_stage2 import PolicyConfig,PatchPredictor,probabilities,evaluate_policy
from run_stage2 import balanced,paired_interval,sha


def expected_rate(rows,logits,max_patch):
    values=[]
    for row in rows:
        raw=row['latex'].encode();p=probabilities(logits[contexts(raw)]);ages=np.zeros(max_patch+1)
        ages[1]=1.;total=0.
        for i in range(1,len(raw)):
            cut=float(ages[:-1].sum()*p[i]+ages[-1]);total+=cut
            new=np.zeros(max_patch+1);new[1]=cut
            new[2:]=(1-p[i])*ages[1:-1];ages=new
        values.append(total/max(1,len(raw)-1))
    return float(np.mean(values))


def main():
    p=ROOT/'experiments/stage2/pilot04';manifest=json.loads((p/'run_manifest.json').read_text());byid={r['id']:r for r in read_catalog()}
    reward=balanced([byid[i] for i in manifest['scopes']['reward']],48)
    dev=balanced([byid[i] for i in manifest['scopes']['dev']],128)
    with np.load(p/'models/byte_signals.npz',allow_pickle=False) as d:base=d['base'].copy()
    results=[]
    for seed in (17,29,43):
        for algorithm in ('boundary_rl','random_search','cem_search','no_update'):
            name=f'{algorithm}_{seed}';meta=json.loads((p/f'{name}_training.json').read_text());config=PolicyConfig(**meta['config'])
            with np.load(p/'models'/f'{name}_router.npz',allow_pickle=False) as d:logits=d['logits'].copy();initial=d['initial'].copy()
            target=meta['target_boundary_rate'];lo=-8.;hi=8.
            for _ in range(24):
                mid=(lo+hi)/2;rate=expected_rate(reward,logits+mid,config.max_patch)
                if rate>target:hi=mid
                else:lo=mid
            offset=(lo+hi)/2;calibrated=logits+offset
            predictor=PatchPredictor(base,config).fit([byid[i] for i in meta['fit_ids']],initial,seed+17011)
            score=evaluate_policy(dev,calibrated,predictor,99173+seed,draws=8)
            results.append({'seed':seed,'algorithm':algorithm,'offset':offset,'target':target,
                            'calibrated_expected_reward_rate':expected_rate(reward,calibrated,config.max_patch),
                            'development':score})
    comparisons=[]
    for seed in (17,29,43):
        a=next(r['development'] for r in results if r['seed']==seed and r['algorithm']=='boundary_rl')
        for control in ('random_search','cem_search','no_update'):
            b=next(r['development'] for r in results if r['seed']==seed and r['algorithm']==control)
            comparisons.append({'seed':seed,'control':control,'relative_bpb_improvement':1-a['paper_macro_causal_byte_prediction_bpb']/b['paper_macro_causal_byte_prediction_bpb'],
                                'rl_dev_rate':a['boundary_rate'],'control_dev_rate':b['boundary_rate'],
                                **paired_interval(a['rows'],b['rows'])})
    output={'status':'post-hoc sensitivity; calibration does not inspect dev or loss',
            'calibration':'24 bisection iterations per policy; analytical expected patch boundaries on same 48 reward formulas',
            'source_sha256':sha(Path(__file__)),'results':results,'comparisons':comparisons,
            'oracle_training_budget_unchanged':True,'evaluation_draws':8,
            'limitations':'reward rate exactly matched; held-development rate may differ; compute still not neural FLOPs'}
    (ROOT/'experiments/stage2/matched_rate_router.json').write_text(json.dumps(output,indent=2)+'\n')
    for r in comparisons:print(r['seed'],r['control'],round(100*r['relative_bpb_improvement'],3),'%',r['paper_cluster_ci95'])


if __name__=='__main__':main()
