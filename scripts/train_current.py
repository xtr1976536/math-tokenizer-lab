"""Train just the selected AST-v4/v5 configurations, with no old-model imports.

The fixed registry configurations are reused; this is not a new test-based
selection sweep. New outputs are immutable. Corpus inputs remain unchanged.
"""
import argparse
from dataclasses import asdict
import json
import time

from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.corpus import ROOT
from math_tokenizer.corpus_archive_v5 import ensure_master
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.evaluate import file_hash, read_jsonl
from math_tokenizer.measure_v5 import measure
from math_tokenizer.rl_vocab_v5 import RLVocab, internal_split
from math_tokenizer.trajectory_v5 import TrajectoryVocab, TrajectoryConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    ensure_master()
    out = ROOT / args.output
    if out.exists():raise RuntimeError('choose a new output directory; completed experiments are immutable')
    out.mkdir(parents=True)
    registry = json.loads((ROOT / 'models/current/registry.json').read_text())
    if 'training' not in registry:raise RuntimeError('current registry has not been finalized')
    data = ROOT / 'data/releases/v5/curated'
    suites = ['fresh_paper_test', 'extreme_domain_test', 'synthetic_stress_test']
    protocol = {'families': ['ast_v4', 'v5'], 'capacity': 1024, 'seed': args.seed,
                'registry_sha256': file_hash(ROOT / 'models/current/registry.json'),
                'training': registry['training'], 'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                'suite_hashes': {s: file_hash(data / f'splits/{s}.jsonl') for s in suites},
                'test_exposure': 'previously exposed benchmark; fixed-config reproduction, not new selection',
                'normalization': 'same raw UTF8, no truncation or special tokens',
                'reward_scope': 'paper-disjoint split inside scenario train; no dev/test reward'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    checkpoints = out / 'checkpoints'; checkpoints.mkdir()
    trained = {}
    for scenario in ('broad', 'specialized'):
        records = read_jsonl(data / f'splits/{"train" if scenario == "broad" else "transfer_train"}.jsonl')
        initial = RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json')
        fit, reward = internal_split(records, initial.config)
        ast_config = registry['training']['ast_v4'][scenario]
        trainer = GrammarBPE(True).train([r['latex'] for r in fit], 1024, atom_budget=ast_config['atom_budget'])
        base = ConstrainedTokenizer(trainer.vocab, True, 'commands', 'shortest')
        base.capacity = 1024; base.train_seconds = trainer.train_seconds; base.training_log = trainer.training_log
        ast = augment_dictionary(base, fit, reward, 1024, extra_subtrees=ast_config['extra_subtrees'])
        ast.save(checkpoints / f'{scenario}-ast-v4.json'); trained[(scenario, 'ast_v4')] = ast
        choice = registry['training']['v5'][scenario]
        if choice['method'] == 'local_parent':
            v5 = RLVocab(initial.config).train(records, 1024, args.seed)
        else:
            config = TrajectoryConfig(**choice['trajectory_config'])
            v5 = TrajectoryVocab(initial.config, config).train(records, 1024, args.seed, initial.theta,
                    progress=lambda e, s, v: print('train v5', scenario, e, s, round(v['utility'], 6), flush=True))
        v5.save(checkpoints / f'{scenario}-v5.json'); trained[(scenario, 'v5')] = v5
        print('trained current families', scenario, flush=True)
    # All training complete before the repeated benchmark text is loaded.
    tests = {s: read_jsonl(data / f'splits/{s}.jsonl') for s in suites}
    metrics, rows = [], []
    for (scenario, family), model in trained.items():
        active = ['fresh_paper_test', 'synthetic_stress_test'] if scenario == 'broad' else ['extreme_domain_test', 'synthetic_stress_test']
        for suite in active:
            summary, rr = measure(model, tests[suite], family, args.seed, repeats=5)
            summary.update(scenario=scenario, suite=suite)
            for r in rr:r.update(scenario=scenario, suite=suite)
            metrics.append(summary); rows.extend(rr)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'per_formula.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (out / 'manifest.json').write_text(json.dumps({'protocol_sha256': file_hash(out / 'protocol.json'),
        'checkpoints': {p.name: file_hash(p) for p in checkpoints.iterdir()}, 'runs': len(metrics), 'records': len(rows)}, indent=2))
    print('current-only fixed-configuration reproduction complete', len(metrics), len(rows))


if __name__ == '__main__':main()
