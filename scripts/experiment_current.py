"""Current-only AST-v4/v5 experiments; dev selection precedes benchmark replay.

The v5 suites are previously exposed benchmarks, not new pristine holdouts.
No source corpus, original result, or retired checkpoint is rewritten.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import bootstrap_difference, file_hash, read_jsonl
from math_tokenizer.measure_v5 import measure
from math_tokenizer.rl_vocab_v5 import RLVocab, internal_split
from math_tokenizer.trajectory_v5 import TrajectoryConfig, TrajectoryVocab

AST_OPTIONS = [
    dict(atom_budget=256, subtree_budget=0),
    dict(atom_budget=256, subtree_budget=64),
    dict(atom_budget=256, subtree_budget=128),
    dict(atom_budget=256, subtree_budget=192),
    dict(atom_budget=192, subtree_budget=128),
    dict(atom_budget=128, subtree_budget=128),
]
RL_OPTIONS = {f'terminal_temp{int(t)}': TrajectoryConfig(epochs=3, samples=3, temperature=t) for t in (3., 8.)}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dev_sample(records, limit=1600):
    groups = defaultdict(list)
    for row in records:
        groups[row['split_group']].append(row)
    selected = [r for group, rows in sorted(groups.items()) for r in sorted(rows, key=lambda r: r['id'])[:32]]
    return sorted(selected, key=lambda r: r['id'])[:limit]


def score_dev(model, records):
    metric, rows = measure(model, records, 'dev', repeats=1)
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        groups[row['group']][0] += row['tokens']
        groups[row['group']][1] += row['bytes']
    cost = float(np.mean([t / b for t, b in groups.values()]))
    command = metric['command_integrity_rate'] if metric['command_integrity_rate'] is not None else 1.
    number = metric['number_integrity_rate'] if metric['number_integrity_rate'] is not None else 1.
    utility = cost + .15 * (1 - command) + .03 * (1 - number) + .10 * metric['hybrid_tree_crossing_rate']
    return {'utility': utility, 'document_macro_tokens_per_byte': cost,
            'tokens_per_formula': metric['tokens_per_formula'], 'command_integrity': command,
            'number_integrity': number, 'tree_crossing': metric['hybrid_tree_crossing_rate']}


def build_ast(fit, option, capacity):
    trainer = GrammarBPE(True).train([r['latex'] for r in fit], capacity, **option)
    model = ConstrainedTokenizer(trainer.vocab, True, 'commands', 'shortest')
    model.capacity = capacity
    model.train_seconds = trainer.train_seconds
    model.training_log = trainer.training_log
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/current_1024')
    parser.add_argument('--capacity', type=int, default=1024)
    parser.add_argument('--phase', choices=['ast', 'rl', 'all'], default='all')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    data = ROOT / 'data/releases/v5/curated'
    suites = ['fresh_paper_test', 'extreme_domain_test', 'synthetic_stress_test']
    # Protocol stores hashes, never test text, before any new training/selection.
    protocol = {'families': ['ast_v4', 'v5'], 'capacity': args.capacity,
                'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                'suite_hashes': {s: file_hash(data / f'splits/{s}.jsonl') for s in suites},
                'test_exposure': 'Previously exposed v5 benchmark replay; no new-unseen claim',
                'ast_options': AST_OPTIONS, 'rl_options': {k: asdict(c) for k, c in RL_OPTIONS.items()},
                'dev_seeds': [17, 29], 'formal_seeds': [17, 29, 43],
                'selection_rule': 'common docmacro token/byte + .15 command error + .03 number error + .10 tree crossing; AST command drop <=.002 vs same-fit no-subtree control',
                'reward_scope': 'train-internal paper-disjoint fit/reward, exact v5 IDs; no dev/test reward',
                'training_exposure': 'both trained families and greedy use identical scenario fit; RL/greedy additionally use internal reward; frozen parents separately labeled',
                'online': 'AST-v4 constrained C shortest; v5 unchanged official Rust BPE, no online policy/parser',
                'selection_before_test_load': True,
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    path = out / 'protocol.json'
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ('families', 'capacity', 'corpus_sha256', 'suite_hashes', 'ast_options', 'rl_options'):
            value = existing[key]
            if key == 'rl_options':value = {k: asdict(TrajectoryConfig(**v)) for k, v in value.items()}
            assert value == protocol[key], key
    else:
        write(path, protocol)
    phases = ['ast', 'rl'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        destination = out / f'{phase}_selection.json'
        if destination.exists():
            print('skip immutable finished phase', phase, flush=True)
            continue
        ablations, selections, pool, control_pool = [], {}, {}, {}
        for scenario in ('broad', 'specialized'):
            train_split = 'train' if scenario == 'broad' else 'transfer_train'
            dev_split = 'dev' if scenario == 'broad' else 'transfer_dev'
            train = read_jsonl(data / f'splits/{train_split}.jsonl')
            dev = dev_sample(read_jsonl(data / f'splits/{dev_split}.jsonl'))
            parent = RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json')
            fit, reward = internal_split(train, parent.config)
            assert [r['id'] for r in fit] == parent.fit_ids
            assert [r['id'] for r in reward] == parent.reward_ids
            write(out / f'{scenario}_scope.json', {'fit_ids': parent.fit_ids, 'reward_ids': parent.reward_ids,
                                                  'dev_selection_ids': [r['id'] for r in dev],
                                                  'fit_groups': sorted({r['split_group'] for r in fit}),
                                                  'reward_groups': sorted({r['split_group'] for r in reward})})
            if phase == 'ast':
                for index, option in enumerate(AST_OPTIONS):
                    model = build_ast(fit, option, args.capacity)
                    name = f'atoms{option["atom_budget"]}-subtrees{option["subtree_budget"]}'
                    stats = score_dev(model, dev)
                    pool[(scenario, name, 0)] = model
                    ablations.append({'scenario': scenario, 'name': name, 'option': option, 'stats': stats,
                                      'train_seconds': model.train_seconds})
                    write(out / 'ast_dev.json', ablations)
                    print('AST dev', scenario, name, stats, flush=True)
                base = ablations[-len(AST_OPTIONS)]
                eligible = [r for r in ablations if r['scenario'] == scenario and r['stats']['command_integrity'] >= base['stats']['command_integrity'] - .002]
                chosen = min(eligible, key=lambda r: (r['stats']['utility'], r['name']))
                selections[scenario] = {'name': chosen['name'], 'option': chosen['option'], 'stats': chosen['stats']}
                control_pool[scenario] = pool[(scenario, base['name'], 0)]
            else:
                # Frozen v5 parent and same-family greedy are controls, not old
                # superseded architectures. Parent scores cannot train policy.
                parent_stats = score_dev(parent, dev)
                for name, configuration in RL_OPTIONS.items():
                    for seed in (17, 29):
                        callback = lambda epoch, sample, stats, sc=scenario, n=name, se=seed: print('RL rollout', sc, n, se, epoch, sample, round(stats['utility'], 6), flush=True)
                        model = TrajectoryVocab(parent.config, configuration).train(train, args.capacity, seed, parent.theta, progress=callback)
                        stats = score_dev(model, dev)
                        pool[(scenario, name, seed)] = model
                        ablations.append({'scenario': scenario, 'name': name, 'seed': seed,
                                          'stats': stats, 'train_seconds': model.train_seconds,
                                          'initial_terminal': model.initial_stats, 'final_terminal': model.final_stats,
                                          'history': model.history})
                        write(out / 'rl_dev.json', ablations)
                        print('RL dev', scenario, name, seed, stats, flush=True)
                scores = {name: float(np.mean([r['stats']['utility'] for r in ablations if r['scenario'] == scenario and r['name'] == name])) for name in RL_OPTIONS}
                chosen = min(scores, key=lambda name: (scores[name], name))
                selections[scenario] = {'name': chosen, 'config_mean_scores': scores,
                                        'parent_stats': parent_stats,
                                        'promote': scores[chosen] < parent_stats['utility']}
        write(destination, {'phase': phase, 'selections': selections, 'test_used_for_selection': False,
                            'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print('selection frozen', phase, selections, flush=True)
        # All scenario selections for this phase precede reading any test text.
        test_records = {s: read_jsonl(data / f'splits/{s}.jsonl') for s in suites}
        assert all(file_hash(data / f'splits/{s}.jsonl') == protocol['suite_hashes'][s] for s in suites)
        metrics, rows, by, pairwise = [], [], {}, {}
        def evaluate(model, name, seed, scenario, suite):
            result, rr = measure(model, test_records[suite], name, seed, repeats=5)
            result.update(scenario=scenario, suite=suite, phase=phase)
            for record in rr:
                record.update(scenario=scenario, suite=suite, phase=phase)
            metrics.append(result); rows.extend(rr); by[(scenario, suite, name, seed)] = rr
            write(out / f'{phase}_metrics.json', metrics)
            (out / f'{phase}_per_formula.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            print('benchmark', phase, scenario, suite, name, seed, round(result['tokens_per_formula'], 4), round(result['encode_mib_s'], 3), flush=True)
        for scenario in ('broad', 'specialized'):
            active = ['fresh_paper_test', 'synthetic_stress_test'] if scenario == 'broad' else ['extreme_domain_test', 'synthetic_stress_test']
            chosen = selections[scenario]['name']
            if phase == 'ast':
                selected = pool[(scenario, chosen, 0)]
                selected.save(checkpoints / f'{scenario}-ast-v4-selected.json')
                controls = [('ast_v4_same_fit_base', control_pool[scenario]),
                            ('ast_v4_frozen_prior_train', ConstrainedTokenizer.load(ROOT / 'models/current/ast_v4/base-1024.json'))]
                for suite in active:
                    evaluate(selected, 'ast_v4_subtree', 0, scenario, suite)
                    for name, model in controls:
                        evaluate(model, name, 0, scenario, suite)
                    pairwise[f'{scenario}/{suite}/subtree-vs-same-fit'] = bootstrap_difference(by[(scenario, suite, 'ast_v4_subtree', 0)], by[(scenario, suite, 'ast_v4_same_fit_base', 0)])
            else:
                train = read_jsonl(data / f'splits/{"train" if scenario == "broad" else "transfer_train"}.jsonl')
                parent = RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json')
                greedy = RLVocab.load(ROOT / f'models/current/v5/controls/{scenario}-greedy.json')
                for suite in active:
                    evaluate(parent, 'v5_local_parent', 17, scenario, suite)
                    evaluate(greedy, 'v5_local_greedy', 17, scenario, suite)
                for seed in (17, 29, 43):
                    model = pool.get((scenario, chosen, seed))
                    if model is None:
                        model = TrajectoryVocab(parent.config, RL_OPTIONS[chosen]).train(train, args.capacity, seed, parent.theta,
                                progress=lambda e, s, v: print('formal rollout', scenario, seed, e, s, round(v['utility'], 6), flush=True))
                    model.save(checkpoints / f'{scenario}-v5-trajectory-{seed}.json')
                    for suite in active:
                        evaluate(model, 'v5_trajectory_rl', seed, scenario, suite)
                        for name in ('v5_local_parent', 'v5_local_greedy'):
                            pairwise[f'{scenario}/{suite}/trajectory-vs-{name}/{seed}'] = bootstrap_difference(by[(scenario, suite, 'v5_trajectory_rl', seed)], by[(scenario, suite, name, 17)])
                # Same rollout budget with updates disabled: measures policy
                # learning separately from simply searching sampled tables.
                no_update = TrajectoryVocab(parent.config, RL_OPTIONS[chosen]).train(train, args.capacity, 17, parent.theta, update=False,
                            progress=lambda e, s, v: print('no-update rollout', scenario, e, s, round(v['utility'], 6), flush=True))
                no_update.save(checkpoints / f'{scenario}-v5-no-update.json')
                search = no_update.best_sample
                search.method = 'terminal_search_no_policy_update'
                search.save(checkpoints / f'{scenario}-v5-search-control.json')
                for suite in active:
                    evaluate(no_update, 'v5_fixed_initial_policy', 17, scenario, suite)
                    evaluate(search, 'v5_equal_budget_search', 17, scenario, suite)
                    pairwise[f'{scenario}/{suite}/trajectory-vs-no-update/17'] = bootstrap_difference(by[(scenario, suite, 'v5_trajectory_rl', 17)], by[(scenario, suite, 'v5_fixed_initial_policy', 17)])
                    pairwise[f'{scenario}/{suite}/trajectory-vs-search/17'] = bootstrap_difference(by[(scenario, suite, 'v5_trajectory_rl', 17)], by[(scenario, suite, 'v5_equal_budget_search', 17)])
        write(out / f'{phase}_pairwise.json', pairwise)
        write(out / f'{phase}_manifest.json', {'protocol_sha256': file_hash(path), 'selection_sha256': file_hash(destination),
                                             'records': len(rows), 'method_seed_suite_runs': len(metrics),
                                             'checkpoints': {p.name: file_hash(p) for p in checkpoints.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
