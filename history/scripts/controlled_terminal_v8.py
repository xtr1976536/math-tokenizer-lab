"""Three paper splits x five seeds; equal 34-score terminal optimizer controls."""
from collections import defaultdict
import hashlib, json
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_route_v6 import derive_atoms, ReferenceReward
from math_tokenizer.rl_transfer_v7 import task_rows, TransferConfig
from math_tokenizer.terminal_search_v8 import optimize, export, SearchConfig, reward_config,rollout
from math_tokenizer.measure_v6 import measure, fit_usage
from math_tokenizer.independent_v7 import additional, source_scopes


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')


def main():
    folder = ROOT/'results/v8/controlled_terminal'
    pool = [r for r in load_split('train')+load_split('dev') if r['source_repo']=='arxiv.org']
    groups = defaultdict(list)
    for r in pool: groups[r['split_group']].append(r)
    metrics = []
    for fold in (211, 337, 461):
        ordered = sorted(groups, key=lambda g: hashlib.sha256(f'v8-fold:{fold}:{g}'.encode()).digest())
        n = max(4, len(ordered)//5); held = set(ordered[:n]); reward_groups = set(ordered[n:2*n])
        fit = task_rows([r for r in pool if r['split_group'] not in held|reward_groups], 4000, fold)
        reward = task_rows([r for r in pool if r['split_group'] in reward_groups], 128, fold)
        dev = task_rows([r for r in pool if r['split_group'] in held], 256, fold)
        atoms = derive_atoms(fit, 1024, 256)
        config = SearchConfig()
        dest = folder/str(fold)
        write(dest/'scope.json', {'fit_ids': [r['id'] for r in fit], 'reward_ids': [r['id'] for r in reward],
              'validation_ids': [r['id'] for r in dev], 'atom_hex': [p.hex() for p in atoms],
              'source_disjoint': True, 'data_phase': 'previously exposed source resplits, not fresh final test'})
        objective = ReferenceReward(fit, dev, reward_config())
        scope_cache = {r['id']: source_scopes(r['latex']) for r in dev}
        initial = np.zeros(24); initial[0] = 2.
        unchanged, _, _ = rollout(fit, atoms, 1024, initial, TransferConfig())
        unchanged.save(dest/'no_update.json')
        m, _ = measure(unchanged, dev, 'no_update', repeats=1, training_usage=fit_usage(unchanged, fit))
        extra, _ = additional(unchanged, dev, scope_cache=scope_cache, probe_limit=0)
        m.update(extra); m.update(fold=fold, seed=0, terminal_validation_utility=objective.score(unchanged)['utility'])
        metrics.append(m)
        for seed in (17, 29, 43, 59, 71):
            for algorithm in ('parameter_rl', 'random_search', 'cem_search'):
                checkpoint = dest/f'{algorithm}_{seed}.json'
                policy_path = dest/f'{algorithm}_{seed}_policy.json'
                if policy_path.exists(): policy = json.loads(policy_path.read_text())
                else:
                    policy = optimize(fit, reward, atoms, config, seed, algorithm)
                    write(policy_path, policy)
                model = export(fit, reward, atoms, 1024, policy); model.save(checkpoint)
                m, _ = measure(model, dev, algorithm, repeats=1, training_usage=fit_usage(model, fit))
                extra, _ = additional(model, dev, scope_cache=scope_cache, probe_limit=0)
                m.update(extra); m.update(fold=fold, seed=seed, terminal_evaluations=policy['terminal_evaluations'],
                    terminal_validation_utility=objective.score(model)['utility'], checkpoint=str(checkpoint.relative_to(ROOT)))
                metrics.append(m); write(folder/'metrics.json', metrics)
                print('controlled', fold, seed, algorithm, round(m['terminal_validation_utility'], 6), flush=True)
    pairs = {}
    for control in ('random_search', 'cem_search'):
        differences = []; relative = []; regressions = []
        for fold in (211, 337, 461):
            for seed in (17, 29, 43, 59, 71):
                a = next(m for m in metrics if m['fold']==fold and m['seed']==seed and m['method']=='parameter_rl')
                b = next(m for m in metrics if m['fold']==fold and m['seed']==seed and m['method']==control)
                diff = b['terminal_validation_utility']-a['terminal_validation_utility']
                differences.append(diff); relative.append(diff/b['terminal_validation_utility'])
                regressions.append({'fold': fold, 'seed': seed,
                    'compression_relative_regression': a['document_macro_tokens_per_byte']/b['document_macro_tokens_per_byte']-1,
                    'command_recall_loss': b['command_integrity']-a['command_integrity'],
                    'number_recall_loss': b['number_integrity']-a['number_integrity'],
                    'scope_recall_loss': b['independent_scope_boundary_recall']-a['independent_scope_boundary_recall']})
        # Training seeds resampled jointly over folds; overlap between folds
        # explicitly prevents treating their fifteen entries as independent.
        delta = np.asarray(differences).reshape(3, 5)
        rng = np.random.default_rng(8128)
        boot = delta[:, rng.integers(0, 5, (20000, 5))].mean(axis=(0, 2))
        ci = np.quantile(boot, [.025, .975]).tolist()
        guards = all(r['compression_relative_regression'] <= .01 and all(r[k] <= .01 for k in
            ('command_recall_loss', 'number_recall_loss', 'scope_recall_loss')) for r in regressions)
        passed = float(np.mean(relative)) >= .005 and sum(d>0 for d in differences) >= 12 and ci[0] > 0 and guards
        pairs[control] = {'utility_improvement_mean': float(np.mean(differences)),
            'relative_improvement_mean': float(np.mean(relative)), 'winning_pairs': sum(d>0 for d in differences),
            'pairs': 15, 'seed_cluster_ci95': ci, 'all_noninferiority_guards_passed': guards,
            'regressions': regressions, 'promotion_passed': passed,
            'inference_scope': 'five training seeds conditional on three possibly overlapping paper resplits'}
    write(folder/'decision.json', {'comparisons': pairs, 'rl_promoted': all(p['promotion_passed'] for p in pairs.values()),
        'total_models': len(metrics), 'same_reward_config': True, 'same_terminal_evaluation_budget': 34,
        'new_test_used_for_selection': False, 'interpretation': 'Practical promotion evidence, not proof all RL is ineffective'})
    print('RL promotion', all(p['promotion_passed'] for p in pairs.values()), flush=True)


if __name__ == '__main__': main()
