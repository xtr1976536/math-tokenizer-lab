"""Publish the already dev-selected current models, never choose by test rank."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.corpus import ROOT
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.trajectory_v5 import TrajectoryVocab


def main():
    destination = ROOT / 'models/current/registry.json'
    old = json.loads(destination.read_text())
    ast = json.loads((ROOT / 'results/current_ast_dictionary1024/selection.json').read_text())
    rl = json.loads((ROOT / 'results/current_v5_critic1024/selection.json').read_text())
    assert ast['test_used_for_selection'] is False and rl['test_used_for_selection'] is False
    for scenario, row in rl['selections'].items():
        assert row['method'] == min(row['scores'], key=lambda name: (row['scores'][name], name))
        assert row['deployment_seed'] == 17
    latest_ast, latest_v5, training = {}, {}, {'ast_v4': {}, 'v5': {}}
    for scenario in ('broad', 'specialized'):
        ast_source = ROOT / f'results/current_ast_dictionary1024/checkpoints/{scenario}-ast_v4_dictionary.json'
        ast_model = load_checkpoint(ast_source)
        relative = f'models/current/ast_v4/{scenario}-dictionary-1024.json'
        ast_model.save(ROOT / relative); latest_ast[scenario] = relative
        training['ast_v4'][scenario] = {'atom_budget': 256, 'extra_subtrees': ast['selections'][scenario]['extras']}
        selected = rl['selections'][scenario]['method']
        if selected == 'crossfit_critic':
            source = ROOT / f'results/current_v5_critic1024/checkpoints/{scenario}-v5-critic-17.json'
        elif selected == 'rloo':
            source = ROOT / f'results/current_rl_1024/checkpoints/{scenario}-v5-trajectory-17.json'
        else:
            source = ROOT / f'models/current/v5/{scenario}-base.json'
        model = load_checkpoint(source)
        relative = f'models/current/v5/{scenario}-selected.json'
        model.save(ROOT / relative); latest_v5[scenario] = relative
        training['v5'][scenario] = {'method': selected}
        if isinstance(model, TrajectoryVocab):
            training['v5'][scenario]['trajectory_config'] = asdict(model.trajectory)
    fast512 = ConstrainedTokenizer.load(ROOT / 'models/current/ast_v4/base-512.json')
    fast512.save(ROOT / 'models/current/ast_v4/fast-512.json')
    registry = {'families': ['ast_v4', 'v5'], 'base_commit': old['base_commit'],
                'ast_v4': {'512': 'models/current/ast_v4/fast-512.json', '1024': latest_ast['broad']},
                'ast_v4_specialized': latest_ast['specialized'], 'v5': latest_v5,
                'base_snapshots': {'ast_v4': {str(c): f'models/current/ast_v4/base-{c}.json' for c in (512, 1024)},
                                   'v5': {s: f'models/current/v5/{s}-base.json' for s in ('broad', 'specialized')}},
                'controls': old['controls'], 'training': training,
                'selection_provenance': {'ast_v4': 'results/current_ast_dictionary1024/selection.json',
                                          'v5': 'results/current_v5_critic1024/selection.json'},
                'deployment_seed': 17, 'test_exposure': 'previously exposed repeated benchmark',
                'interpretation': 'dev-selected model; no claim of winning every metric or search control'}
    paths = list(registry['ast_v4'].values()) + [registry['ast_v4_specialized']] + list(registry['v5'].values())
    registry['active_checkpoint_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    destination.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'active_ast_v4': latest_ast, 'active_v5': latest_v5, 'v5_choices': training['v5']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':main()
