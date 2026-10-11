"""Public entry point: only the two user-selected active model families."""
import json
from pathlib import Path

from .corpus import ROOT


def load_checkpoint(path):
    path = Path(path)
    saved = json.loads(path.read_text())
    if saved.get('format') in ('ast-v4-constrained', 'masked-rl-math-v4'):
        from .constrained_v4 import ConstrainedTokenizer
        model = ConstrainedTokenizer.load(path)
        if model.inference != 'shortest':
            raise ValueError('retired RL-v4 model is not an active AST-v4 checkpoint')
        return model
    if saved.get('format') == 'v5-offline-trajectory-rl':
        from .trajectory_v5 import TrajectoryVocab
        return TrajectoryVocab.load(path)
    if saved.get('format') == 'rl-vocab-v5':
        from .rl_vocab_v5 import RLVocab
        return RLVocab.load(path)
    raise ValueError('unsupported current model family')


def load_current(family, scenario='broad', capacity=1024):
    family = family.replace('-', '_')
    if scenario not in ('broad', 'specialized'):
        raise ValueError('scenario must be broad or specialized')
    registry = json.loads((ROOT / 'models/current/registry.json').read_text())
    if family == 'ast_v4':
        if scenario == 'specialized':
            if capacity != 1024 or 'ast_v4_specialized' not in registry:
                raise ValueError('specialized AST-v4 is only available at capacity 1024')
            relative = registry['ast_v4_specialized']
        else:
            relative = registry['ast_v4'].get(str(capacity))
    elif family == 'v5':
        if capacity != 1024:
            raise ValueError('current v5 is only available at capacity 1024')
        relative = registry['v5'][scenario]
    else:
        raise ValueError('active families are ast_v4 and v5')
    if not relative:
        raise ValueError('requested current model capacity not available')
    model = load_checkpoint(ROOT / relative)
    assert model.capacity == capacity
    return model
