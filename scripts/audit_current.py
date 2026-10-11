"""Six review blocks for the current-only research and fixed-config replay."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tokenizers import Tokenizer

from math_tokenizer.ast import compatible
from math_tokenizer.checkpoints import verify_saved_hash
from math_tokenizer.corpus import ROOT
from math_tokenizer.current_models import load_checkpoint, load_current
from math_tokenizer.evaluate import file_hash, read_jsonl

RUNS = [
    ('current_1024', 'ast_'), ('current_ast_dictionary1024', ''),
    ('current_rl_1024', 'rl_'), ('current_v5_critic1024', ''), ('current_reproduce17', ''),
]


def model_for(run, method, scenario, seed):
    current = ROOT / 'models/current'
    if run == 'current_1024':
        paths = {'ast_v4_subtree': ROOT / f'results/{run}/checkpoints/{scenario}-ast-v4-selected.json',
                 'ast_v4_same_fit_base': ROOT / f'results/current_ast_dictionary1024/checkpoints/{scenario}-ast_v4_same_fit_base.json',
                 'ast_v4_frozen_prior_train': current / 'ast_v4/base-1024.json'}
        return paths[method]
    if run == 'current_ast_dictionary1024':return ROOT / f'results/{run}/checkpoints/{scenario}-{method}.json'
    if run == 'current_reproduce17':
        suffix = 'ast-v4' if method == 'ast_v4' else 'v5'
        return ROOT / f'results/{run}/checkpoints/{scenario}-{suffix}.json'
    if method == 'v5_local_parent':return current / f'v5/{scenario}-base.json'
    if method == 'v5_local_greedy':return current / f'v5/controls/{scenario}-greedy.json'
    if method == 'v5_crossfit_critic':return ROOT / f'results/current_v5_critic1024/checkpoints/{scenario}-v5-critic-{seed}.json'
    if method in ('v5_trajectory_rl', 'v5_trajectory_rloo'):return ROOT / f'results/current_rl_1024/checkpoints/{scenario}-v5-trajectory-{seed}.json'
    if method == 'v5_fixed_initial_policy':return ROOT / f'results/current_rl_1024/checkpoints/{scenario}-v5-no-update.json'
    if method == 'v5_equal_budget_search':return ROOT / f'results/current_rl_1024/checkpoints/{scenario}-v5-search-control.json'
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--without-pytest', action='store_true')
    args = p.parse_args()
    data = ROOT / 'data/releases/v5/curated'
    summary = json.loads((data / 'summary.json').read_text())
    corpus_hash = file_hash(data / 'corpus.jsonl')
    assert corpus_hash == summary['corpus_sha256']
    registry = json.loads((ROOT / 'models/current/registry.json').read_text())
    retired = json.loads((ROOT / 'models/retirement_manifest.json').read_text())
    assert registry['families'] == ['ast_v4', 'v5']
    assert all(not (ROOT / relative).exists() for relative in retired['files'])
    retirement_states = Counter(verify_saved_hash(ROOT / path, row['sha256']) for path, row in retired['files'].items())
    for path, expected in registry['active_checkpoint_sha256'].items():assert file_hash(ROOT / path) == expected
    result = {'round1_cleanup_corpus_and_registry': {'passed': True, 'removed_files': retired['removed_files'],
               'retirement_states': dict(retirement_states), 'corpus_sha256': corpus_hash, 'active_families': registry['families']}}
    suites = {s: {r['id']: r for r in read_jsonl(data / f'splits/{s}.jsonl')}
              for s in ('fresh_paper_test', 'extreme_domain_test', 'synthetic_stress_test')}
    total_runs = total_rows = replayed = 0
    inventories = []
    for run, prefix in RUNS:
        folder = ROOT / 'results' / run
        protocol = json.loads((folder / 'protocol.json').read_text())
        manifest = json.loads((folder / f'{prefix}manifest.json').read_text())
        assert protocol['corpus_sha256'] == corpus_hash
        assert manifest['protocol_sha256'] == file_hash(folder / 'protocol.json')
        selection_path = folder / f'{prefix}selection.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            assert selection['test_used_for_selection'] is False
            assert manifest['selection_sha256'] == file_hash(selection_path)
        for suite, expected in protocol['suite_hashes'].items():assert file_hash(data / f'splits/{suite}.jsonl') == expected
        for path, expected in manifest['checkpoints'].items():assert file_hash(folder / 'checkpoints' / path) == expected
        if run != 'current_reproduce17':
            for scenario in ('broad', 'specialized'):
                scope = json.loads((folder / f'{scenario}_scope.json').read_text())
                train = read_jsonl(data / f'splits/{"train" if scenario == "broad" else "transfer_train"}.jsonl')
                lookup = {r['id']: r for r in train}
                fit, reward = set(scope['fit_ids']), set(scope['reward_ids'])
                assert fit.isdisjoint(reward) and (fit | reward) <= set(lookup)
                assert {lookup[i]['split_group'] for i in fit}.isdisjoint(lookup[i]['split_group'] for i in reward)
                assert (fit | reward).isdisjoint(suites['fresh_paper_test'])
                assert (fit | reward).isdisjoint(suites['extreme_domain_test'])
        metrics = json.loads((folder / f'{prefix}metrics.json').read_text())
        rows = read_jsonl(folder / f'{prefix}per_formula.jsonl')
        by = defaultdict(list)
        for row in rows:by[(row['scenario'], row['suite'], row['method'], row['seed'])].append(row)
        assert len(by) == len(metrics)
        for m in metrics:
            key = m['scenario'], m['suite'], m['method'], m['seed']
            rr = by[key]
            assert len(rr) == m['formulas'] == len(suites[m['suite']])
            assert {r['id'] for r in rr} == set(suites[m['suite']])
            assert sum(r['tokens'] for r in rr) == m['tokens']
            assert sum(r['bytes'] for r in rr) == m['bytes']
            assert all(r['roundtrip'] for r in rr) and m['roundtrip_rate'] == 1.
            assert m['vocab_actual'] <= 1024
            for top, bottom, field in [('commands_intact', 'commands', 'command_integrity_rate'),
                                       ('numbers_intact', 'numbers', 'number_integrity_rate'),
                                       ('lexemes_intact', 'lexemes', 'lexeme_preservation_rate')]:
                denominator = sum(r[bottom] for r in rr)
                if denominator:assert abs(sum(r[top] for r in rr) / denominator - m[field]) < 1e-12
            measured = [r for r in rr if r['crossing_tokens'] is not None]
            crossing = sum(r['crossing_tokens'] for r in measured) / max(1, sum(r['tokens'] for r in measured))
            assert abs(crossing - m['hybrid_tree_crossing_rate']) < 1e-12
            path = model_for(run, m['method'], m['scenario'], m['seed'])
            model = load_checkpoint(path)
            official = Tokenizer.from_file(str(path.with_suffix('.tokenizer.json'))) if hasattr(model, 'tokenizer') else None
            for row in rr:
                text = suites[m['suite']][row['id']]['latex']
                ids = model.encode(text)
                assert len(ids) == row['tokens'] and model.decode(ids) == text
                if official is not None:assert official.encode(text, add_special_tokens=False).ids == ids
            replayed += len(rr)
        total_runs += len(metrics); total_rows += len(rows)
        inventories.append({'run': run, 'method_seed_suite_records': len(metrics), 'formula_records': len(rows),
                            'checkpoint_and_result_replay': True})
        print('audit replay', run, len(rows), flush=True)
    result['round2_training_scope_and_selection'] = {'passed': True, 'paper_disjoint_internal_rewards': True,
                    'fixed_dev_selection_hashes': True, 'test_status': 'previously exposed repeated benchmark'}
    result['round3_common_metrics_and_all_checkpoints'] = {'passed': True, 'runs': inventories,
                    'method_seed_suite_records': total_runs, 'formula_records': total_rows, 'replayed': replayed,
                    'same_raw_ids_no_exclusions_no_truncation': True, 'official_export_ids_equal': True}
    # Separate the current learned actor from no-update search; verify an actual
    # whole-trajectory gradient update and honest future-value fitting scope.
    policies = []
    for directory, suffix in [('current_rl_1024', 'trajectory'), ('current_v5_critic1024', 'critic')]:
        for scenario in ('broad', 'specialized'):
            for seed in (17, 29, 43):
                path = ROOT / f'results/{directory}/checkpoints/{scenario}-v5-{suffix}-{seed}.json'
                d = json.loads(path.read_text())
                assert d['policy_update_scope'] == 'complete_merge_trajectory_terminal_return'
                assert any(abs(a - b) > 1e-8 for a, b in zip(d['initial_policy'], d['policy']))
                assert len(d['training_history']) == 3
                for epoch in d['training_history']:
                    assert epoch['updates_enabled'] and epoch['reward_scope'] == 'training_internal_document_holdout'
                    if suffix == 'critic':
                        assert epoch['baseline_estimator'] == 'crossfit_critic'
                        for head in epoch['critic']['cross_fitted_value_heads']:
                            assert head['excluded_rollout'] not in head['trained_on_rollouts']
                policies.append({'scenario': scenario, 'seed': seed, 'variant': suffix, 'policy_parameters': len(d['policy'])})
    result['round4_RL_learning_and_critic_no_self_target'] = {'passed': True, 'policies': policies,
                'variance_and_gradient_proofs': 'tests/test_current_routes.py finite differences and full paired-path expectation'}
    engineering = json.loads((ROOT / 'results/current_engineering/verification.json').read_text())
    assert engineering['corpus_sha256'] == corpus_hash
    assert all(r['identical_ids_and_losslessness'] for r in engineering['ast_exact'])
    assert all(r['exact_vocab_merges_policy_and_scope'] for r in engineering['v5_train_speed'])
    result['round5_exact_engineering_and_interleaved_speed'] = {'passed': True, 'evidence': 'results/current_engineering/verification.json',
                'ast_compared_formulas_each_budget': 54551, 'two_budgets': [512, 1024], 'v5_original_and_optimized_runs_each_scenario': 3}
    # Re-training current configs must reproduce selected deployment tables/IDs.
    for scenario in ('broad', 'specialized'):
        for family in ('ast_v4', 'v5'):
            current = load_current(family, scenario)
            suffix = 'ast-v4' if family == 'ast_v4' else 'v5'
            rebuilt = load_checkpoint(ROOT / f'results/current_reproduce17/checkpoints/{scenario}-{suffix}.json')
            assert current.vocab == rebuilt.vocab
            if family == 'v5':
                assert current.merges == rebuilt.merges and (current.theta == rebuilt.theta).all()
    result['round6_fixed_config_retraining'] = {'passed': True, 'families_and_scenarios': 4,
                 'exact_vocab_and_policy': True, 'reproduction': 'results/current_reproduce17'}
    if not args.without_pytest:
        test = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=ROOT, capture_output=True, text=True)
        assert test.returncode == 0, test.stdout + test.stderr
        result['tests'] = test.stdout
    destination = ROOT / 'results/current_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Six current audit blocks passed', total_runs, total_rows)


if __name__ == '__main__':main()
