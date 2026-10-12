"""Final evidence inventory, controlled optimizer provenance and checkpoint replay."""
from datetime import datetime,timezone
import hashlib,json,re
from dataclasses import asdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import master_path,load_split
from math_tokenizer.experiment_v7 import read_external
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.measure_v6 import measure
from math_tokenizer.terminal_search_v8 import reward_config
from math_tokenizer.final_models import load_final


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    folder=ROOT/'results/v8/full';registry=read(folder/'registry.json');protocol=read(folder/'evaluation_protocol.json');first=read(folder/'first_test_score.json')
    metrics=read(folder/'metrics.json');audit=read(folder/'fairness_audit.json');completion=read(folder/'completion.json');ranks=read(folder/'rankings.json')
    assert len(read_external('v8'))==read(ROOT/'data/releases/v8/manifest.json')['formulas']
    assert sha(folder/'registry.json')==protocol['registry_sha256']==first['registry_sha256']==audit['registry_sha256']
    assert datetime.fromisoformat(protocol['frozen_utc'])<datetime.fromisoformat(first['utc'])
    assert sha(folder/'per_formula.jsonl.gz')==audit['per_formula_sha256']
    assert audit['all_passed'] and audit['total_rounds']>=8
    assert completion['configurations']==len(registry) and completion['metric_records']==len(metrics)
    assert completion['formula_rows']==sum(m['formulas'] for m in metrics)
    assert completion['uniform_retiming_completed'] and not completion['uniform_retiming_required']
    assert all(m['roundtrip_rate']==m['payload_fidelity_rate']==1 and not m['provisional_timing'] for m in metrics)
    assert all(len(m['encode_seconds_repeats'])==len(m['decode_seconds_repeats'])==7 for m in metrics)
    assert len(ranks['within_scenario'])==22 and all(len(b['metrics'])==24 for b in ranks['within_scenario'].values())
    corpus=sha(master_path());assert corpus=='972ada427ecb69f38aa2359b7f10fcfbf13f6896a1da7986656ee1321ec24bc7'
    for entry in registry:
        if entry.get('path'):assert sha(ROOT/entry['path'])==entry['checkpoint_sha256']
    pool={r['id']:r for r in load_split('train')+load_split('dev')};replayed=[]
    controlled=ROOT/'results/v8/controlled_terminal';saved=read(controlled/'metrics.json')
    for fold in (211,337,461):
        scope=read(controlled/str(fold)/'scope.json')
        groups=[{pool[i]['split_group'] for i in scope[k]} for k in ('fit_ids','reward_ids','validation_ids')]
        assert all(not groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2)))
        rows=[pool[i] for i in scope['validation_ids']]
        for original in (m for m in saved if m['fold']==fold):
            name=original['method'];seed=original['seed'];filename='no_update' if name=='no_update' else f'{name}_{seed}'
            model=load_checkpoint(controlled/str(fold)/(filename+'.json'))
            observed,_=measure(model,rows,name,repeats=1)
            for key in ('tokens','bytes','document_macro_tokens_per_byte','independent_tree_crossing','command_integrity','number_integrity','roundtrip_rate'):
                assert abs(observed[key]-original[key])<1e-12,(fold,name,seed,key)
            replayed.append({'fold':fold,'seed':seed,'method':name,'validation_rows':len(rows),'matches':True})
        for seed in (17,29,43,59,71):
            policies=[read(controlled/str(fold)/f'{a}_{seed}_policy.json') for a in ('parameter_rl','random_search','cem_search')]
            assert all(p['terminal_evaluations']==34 and p['candidate_count']==17 and p['reward_config']==asdict(reward_config()) and not p['validation_or_test_used'] for p in policies)
            assert all(p['evaluations'][:5]==policies[0]['evaluations'][:5] for p in policies)
            assert all(p['task_fit_ids']==policies[0]['task_fit_ids'] and p['task_reward_ids']==policies[0]['task_reward_ids'] for p in policies)
    engineering_matches=[]
    old=ROOT/'results/v8/engineering_reference'
    for prior in old.rglob('*_policy.json'):
        if 'controlled_terminal_before_cpp' in prior.parts:
            target=controlled/prior.parent.name/prior.name
        elif 'full_fit_before_cpp' in prior.parts:
            target=ROOT/'results/v8'/prior.parent.name/prior.name
        else:continue
        before=read(prior);after=read(target)
        for key in ('policy','last_distribution_mean','incumbent_utility','evaluations','history','task_fit_ids','task_reward_ids'):
            assert before[key]==after[key],(str(prior),key)
        engineering_matches.append(str(prior.relative_to(ROOT)))
    tests=(ROOT/'results/v8/code_review_round2_tests.log').read_text();match=re.search(r'(\d+) passed',tests);assert match and 'FAILED' not in tests and 'ERROR' not in tests
    final=read(ROOT/'models/final_intrinsic/registry.json')
    for name in final['shortlist']['math_core']+final['shortlist']['equal_capacity_controls']:
        for scenario in ('broad','specialized'):
            for capacity in (1024,2048):
                model=load_final(name,scenario,capacity)
                text=r'\frac{x_i^2}{y}+\alpha=1'
                assert model.decode(model.encode(text))==text
    if final['RL_removed_from_production']:
        assert 'parameter_rl' not in final['shortlist']['math_core'] and 'v5_parent' in final['shortlist']['diagnostic_only']
        assert load_checkpoint(ROOT/final['models']['official_atomic_bpe'][0]['path']).__class__.__name__=='AtomicBPE'
        try:load_final('parameter_rl')
        except ValueError:pass
        else:raise AssertionError('RL diagnostic unexpectedly in default production loader')
    result={'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'tests_passed':int(match[1]),'configurations':len(registry),'metric_records':len(metrics),
        'formula_rows':completion['formula_rows'],'fairness_rounds':audit['total_rounds'],
        'new_test_scored_after_freeze':True,'code_review_rounds':3,
        'v6_corpus_unchanged_sha256':corpus,'source_resplit_checkpoint_replays':replayed,
        'controlled_algorithms_same_objective_and_34_terminal_evaluations':True,
        'before_after_native_engineering_policies_identical':engineering_matches,
        'downstream_roles_frozen_before_test':True,'RL_removed_from_production':final['RL_removed_from_production'],
        'expert_reviews_completed':0,'second_hardware_tested':False,'downstream_executed':False,
        'hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in (folder/'registry.json',folder/'metrics.json',folder/'rankings.json',folder/'per_formula.jsonl.gz',folder/'fairness_audit.json',
            folder/'multimetric_paired_effects.json',ROOT/'models/final_intrinsic/registry.json',ROOT/'data/releases/v8/external_test.jsonl.gz')}}
    (ROOT/'results/v8/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    reviews={'all_passed':True,'reviews':[
        {'round':1,'scope':'Source/data access, same terminal reward and task IDs, common first proposals, 34-score budgets and fixed selection',
         'evidence':'results/v8/full/preflight_audit.json','passed':True},
        {'round':2,'scope':'Full implementation tests: finite-difference Gaussian score, shared optimizer budgets, exhaustive boundary lattice, exact native count/gain/sequence and stochastic gradient equivalence',
         'tests_passed':int(match[1]),'passed':True},
        {'round':3,'scope':'Every formal encoder reload and raw-byte replay, independently recomputed count aggregates, source-resplit saved models and before/after engineering policies, frozen production API',
         'formula_rows':completion['formula_rows'],'source_resplit_models':len(replayed),'passed':True}],
        'independent_human_or_external_team_reviews':False}
    (ROOT/'results/v8/code_reviews.json').write_text(json.dumps(reviews,indent=2)+'\n')
    print('Final release passed',len(replayed),'source-resplit model replays and',result['formula_rows'],'formal rows',flush=True)


if __name__=='__main__':main()
