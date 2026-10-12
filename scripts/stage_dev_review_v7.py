"""Uniform development scoring and a frozen gate before external evaluation."""
from collections import defaultdict
import hashlib,json
from datetime import datetime,timezone
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.experiment_v6 import load_entry
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility
from math_tokenizer.independent_v7 import additional,source_scopes


def main():
    byid={r['id']:r for r in load_split('train')+load_split('dev')};output=[];selection={}
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            folder=ROOT/f'results/v7/{scenario}_{capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
            fit=[byid[id] for id in scope['fit_ids']];dev=[byid[id] for id in scope['dev_ids']]
            cache={r['id']:source_scopes(r['latex']) for r in dev};available={}
            for path in folder.glob('*_dev.json'):
                data=json.loads(path.read_text());records=data if isinstance(data,list) else [data]
                for r in records:
                    if not isinstance(r,dict) or 'method' not in r:continue
                    file=folder/'models'/f'{r["method"]}.json'
                    if file.exists() and (r['method'].startswith(('ast_','v5_transfer','v5_cross_task','v5_policy_search','atom_budget'))):
                        if scenario=='specialized' and r['method'].startswith('v5_cross_task') and not r.get('strict_ood_eligible',False):continue
                        available[r['method']]={'name':r['method'],'kind':'checkpoint','path':str(file.relative_to(ROOT)),'metric':r}
            baseline=json.loads((folder/'closest_baseline_dev.json').read_text())
            for r in baseline: r['dev_selection_utility']=selection_utility(r)
            # Only final experiment candidates and declared controls need full
            # independent probes, not every discarded development parameter.
            candidates=[e for e in available.values() if e['name'].startswith(('ast_balanced_official_atomic_union','ast_lexicographic','ast_risk','ast_whole','atom_budget','v5_transfer','v5_cross_task','v5_policy_search'))]
            candidates=[e for e in candidates if not e['name'].startswith('ast_whole') or e['name'].endswith('_fixedmask')]
            for e in candidates:e['metric']['dev_selection_utility']=selection_utility(e['metric'])
            ast=[e for e in candidates if e['name'].startswith('ast_balanced_official_atomic_union')]
            chosen_ast=next(e for e in candidates if e['name']=='ast_lexicographic')
            # RL seed selection is avoided: pick a route, retain all independent
            # seeds. Only broad1024 offers 3 seeds; strict routes reported once.
            rl=[e for e in candidates if e['name'].startswith('v5_transfer_tree0.5_e4') and e['name'].endswith('_learned')]
            if not rl:rl=[e for e in candidates if e['name']=='v5_cross_task_learned' or e['name']=='v5_cross_task_learned_strict']
            mean_rl=sum(e['metric']['dev_selection_utility'] for e in rl)/len(rl) if rl else None
            parent=next(m for m in baseline if m['method']=='v5_v6_parent')
            parent_ast=next(m for m in baseline if m['method']=='ast_v4_v6_parent')
            probe=[chosen_ast]+rl+[e for e in candidates if e['name']=='v5_policy_search_control']
            for e in probe:
                model=load_checkpoint(ROOT/e['path']);usage=fit_usage(model,fit)
                extra,_=additional(model,dev,usage,cache)
                output.append({'capacity':capacity,'scenario':scenario,'method':e['name'],'metric':e['metric'],**extra})
            selection[f'{capacity}/{scenario}']={
                'ast_compression_candidate':{'name':chosen_ast['name'],'path':chosen_ast['path'],
                    'dev_utility':chosen_ast['metric']['dev_selection_utility'],
                    'parent_utility':parent_ast['dev_selection_utility'],
                    'purpose':'Explicit compression/structure tradeoff branch; does not supersede hard-structure AST parent.'},
                'rl_future_credit_candidates':[{'name':e['name'],'path':e['path'],'dev_utility':e['metric']['dev_selection_utility']} for e in rl],
                'rl_mean_dev_utility':mean_rl,'rl_parent_dev_utility':parent['dev_selection_utility'],
                'rl_promoted_by_dev':mean_rl is not None and mean_rl<parent['dev_selection_utility'],
                'seed_selection':False,'all_future_credit_seeds_reported':True,
                'explicitly_discarded_cross_domain_policy_for_strict_ood':scenario=='specialized',
                'all_candidates_only_fit_reward_dev':True,'external_test_score_access':False,
                'development_model_count':len(candidates)}
            (ROOT/'results/v7/development_probe_review.json').write_text(json.dumps(output,indent=2)+'\n')
            print(capacity,scenario,'AST',chosen_ast['name'],'RLmean',mean_rl,'parent',parent['dev_selection_utility'],flush=True)
    (ROOT/'results/v7/development_selection.json').write_text(json.dumps({'selection':selection,
        'decision_utc':datetime.now(timezone.utc).isoformat(),
        'active_registry_changed':False,'scope':'Freeze experimental candidates, keep parent models; no claim of achieved large gain.'},indent=2)+'\n')


if __name__=='__main__':main()
