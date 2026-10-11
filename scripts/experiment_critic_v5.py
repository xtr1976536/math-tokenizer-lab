"""Paired v5 whole-trajectory RL estimator change, same reward and rollout budget."""
from dataclasses import asdict
import argparse
import json
import time

import numpy as np

from experiment_current import dev_sample, score_dev, write
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import bootstrap_difference, file_hash, read_jsonl
from math_tokenizer.measure_v5 import measure
from math_tokenizer.rl_vocab_v5 import RLVocab
from math_tokenizer.trajectory_v5 import TrajectoryConfig, TrajectoryVocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/current_v5_critic1024')
    args = p.parse_args()
    out = ROOT / args.output
    if (out / 'protocol.json').exists():raise RuntimeError('immutable run')
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data/releases/v5/curated'
    old = ROOT / 'results/current_rl_1024'
    config = TrajectoryConfig(epochs=3, samples=3, temperature=3., baseline='crossfit_critic', critic_ridge=.1)
    suites = ['fresh_paper_test', 'extreme_domain_test', 'synthetic_stress_test']
    protocol = {'family': 'v5', 'capacity': 1024, 'configuration': asdict(config),
                'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                'suite_hashes': {s: file_hash(data / f'splits/{s}.jsonl') for s in suites},
                'paired_rloo_source_manifest_sha256': file_hash(old / 'rl_manifest.json'),
                'estimator_difference_only': 'cross-fitted state-value baseline vs scalar RLOO; same initial policy, rewards, seeds, 9 sampled trajectories and two deterministic tables',
                'critic_targets': 'other independent rollout terminal returns; never own return or gold segmentation',
                'selection_rule': 'common dev utility averaged seeds17/29; compare critic/rloo/parent before loading test',
                'test_exposure': 'v5 and preceding estimator benchmarks exposed; repeated benchmark, not new pristine holdout',
                'selection_before_test_load': True}
    write(out / 'protocol.json', protocol)
    pool, selections, dev_rows = {}, {}, []
    previous = json.loads((old / 'rl_selection.json').read_text())['selections']
    for scenario in ('broad', 'specialized'):
        train = read_jsonl(data / f'splits/{"train" if scenario == "broad" else "transfer_train"}.jsonl')
        dev = dev_sample(read_jsonl(data / f'splits/{"dev" if scenario == "broad" else "transfer_dev"}.jsonl'))
        parent = RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json')
        scope = json.loads((old / f'{scenario}_scope.json').read_text())
        assert scope['fit_ids'] == parent.fit_ids and scope['reward_ids'] == parent.reward_ids
        assert scope['dev_selection_ids'] == [r['id'] for r in dev]
        write(out / f'{scenario}_scope.json', scope)
        for seed in (17, 29):
            model = TrajectoryVocab(parent.config, config).train(train, 1024, seed, parent.theta,
                    progress=lambda e, s, v: print('critic rollout', scenario, seed, e, s, round(v['utility'], 6), flush=True))
            pool[(scenario, seed)] = model
            stats = score_dev(model, dev)
            dev_rows.append({'scenario': scenario, 'seed': seed, 'stats': stats, 'history': model.history,
                             'train_seconds': model.train_seconds})
            write(out / 'dev_ablations.json', dev_rows)
            print('critic dev', scenario, seed, stats, flush=True)
        critic_score = float(np.mean([r['stats']['utility'] for r in dev_rows if r['scenario'] == scenario]))
        rloo_name = previous[scenario]['name']
        rloo_score = previous[scenario]['config_mean_scores'][rloo_name]
        parent_score = previous[scenario]['parent_stats']['utility']
        scores = {'crossfit_critic': critic_score, 'rloo': rloo_score, 'local_parent': parent_score}
        selections[scenario] = {'method': min(scores, key=lambda name: (scores[name], name)), 'scores': scores,
                                'deployment_seed': 17}
    write(out / 'selection.json', {'selections': selections, 'test_used_for_selection': False,
                                   'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('critic selection frozen', selections, flush=True)
    tests = {s: read_jsonl(data / f'splits/{s}.jsonl') for s in suites}
    checkpoints = out / 'checkpoints'; checkpoints.mkdir()
    metrics, rows, buckets, pairs = [], [], {}, {}
    for scenario in ('broad', 'specialized'):
        active = ['fresh_paper_test', 'synthetic_stress_test'] if scenario == 'broad' else ['extreme_domain_test', 'synthetic_stress_test']
        controls = {
            'v5_local_parent': RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json'),
            'v5_local_greedy': RLVocab.load(ROOT / f'models/current/v5/controls/{scenario}-greedy.json'),
            'v5_fixed_initial_policy': TrajectoryVocab.load(old / f'checkpoints/{scenario}-v5-no-update.json'),
            'v5_equal_budget_search': RLVocab.load(old / f'checkpoints/{scenario}-v5-search-control.json'),
        }
        methods = [(name, 17, model) for name, model in controls.items()]
        for seed in (17, 29, 43):
            model = pool.get((scenario, seed))
            if model is None:
                train = read_jsonl(data / f'splits/{"train" if scenario == "broad" else "transfer_train"}.jsonl')
                model = TrajectoryVocab(controls['v5_local_parent'].config, config).train(train, 1024, seed, controls['v5_local_parent'].theta,
                    progress=lambda e, s, v: print('critic formal', scenario, seed, e, s, round(v['utility'], 6), flush=True))
            model.save(checkpoints / f'{scenario}-v5-critic-{seed}.json')
            methods.append(('v5_crossfit_critic', seed, model))
            # RLOO model trained with identical seed/budget is an estimator peer.
            methods.append(('v5_trajectory_rloo', seed, TrajectoryVocab.load(old / f'checkpoints/{scenario}-v5-trajectory-{seed}.json')))
        for name, seed, model in methods:
            for suite in active:
                summary, rr = measure(model, tests[suite], name, seed, repeats=5)
                summary.update(scenario=scenario, suite=suite)
                for r in rr:r.update(scenario=scenario, suite=suite)
                metrics.append(summary); rows.extend(rr); buckets[(scenario, suite, name, seed)] = rr
                write(out / 'metrics.json', metrics)
                (out / 'per_formula.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
                print('critic benchmark', scenario, suite, name, seed, round(summary['tokens_per_formula'], 4), flush=True)
        for suite in active:
            for seed in (17, 29, 43):
                a = buckets[(scenario, suite, 'v5_crossfit_critic', seed)]
                for name, s in [('v5_trajectory_rloo', seed)] + [(n, 17) for n in controls]:
                    pairs[f'{scenario}/{suite}/critic-vs-{name}/{seed}'] = bootstrap_difference(a, buckets[(scenario, suite, name, s)])
    write(out / 'pairwise.json', pairs)
    write(out / 'manifest.json', {'protocol_sha256': file_hash(out / 'protocol.json'),
                                'selection_sha256': file_hash(out / 'selection.json'),
                                'checkpoints': {p.name: file_hash(p) for p in checkpoints.iterdir()},
                                'runs': len(metrics), 'records': len(rows),
                                'reused_controls': 'identical frozen policy and stochastic rollout budget; no extra critic update affects their generated tables'})


if __name__ == '__main__':main()
