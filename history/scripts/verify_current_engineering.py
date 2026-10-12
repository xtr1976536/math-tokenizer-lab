"""Isolated equivalence/timing checks after training jobs have finished.

Original AST-v4/v5 code is read from the pre-cleanup Git revision as a reference;
no retired checkpoint is restored. Only current-family snapshots are loaded.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl, file_hash
from math_tokenizer.rl_vocab_v5 import RLVocab

COMMIT = 'c75de3c'


def reference_module(name, relative):
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:{relative}'], cwd=ROOT, text=True)
    module = types.ModuleType(f'math_tokenizer._reference_{name}')
    module.__package__ = 'math_tokenizer'
    sys.modules[module.__name__] = module
    exec(compile(source, f'{COMMIT}:{relative}', 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def main():
    out = ROOT / 'results/current_engineering'
    out.mkdir(exist_ok=True)
    original_ast, ast_sha = reference_module('ast', 'src/math_tokenizer/constrained_v4.py')
    original_v5, v5_sha = reference_module('v5', 'src/math_tokenizer/rl_vocab_v5.py')
    corpus = read_jsonl(ROOT / 'data/releases/v5/curated/corpus.jsonl')
    stress = read_jsonl(ROOT / 'data/releases/v5/curated/splits/synthetic_stress_test.jsonl')
    result = {'reference_git_commit': COMMIT, 'reference_source_sha256': {'ast': ast_sha, 'v5': v5_sha},
              'corpus_sha256': file_hash(ROOT / 'data/releases/v5/curated/corpus.jsonl'),
              'timing': 'one process, isolated from training workers; interleaved cold public per-formula encoding; load/compile excluded',
              'ast_exact': [], 'ast_speed': [], 'v5_train_speed': []}
    for cap in (512, 1024):
        path = ROOT / f'models/current/ast_v4/base-{cap}.json'
        old, new = original_ast.ConstrainedTokenizer.load(path), ConstrainedTokenizer.load(path)
        for row in corpus + stress:
            text = row['latex']
            old_ids, new_ids = old.encode(text), new.encode(text)
            assert old_ids == new_ids, (cap, row['id'])
            assert new.decode(new_ids) == text
        result['ast_exact'].append({'capacity': cap, 'natural_formulas': len(corpus), 'stress_formulas': len(stress),
                                    'identical_ids_and_losslessness': True})
        print('AST exact all corpus/stress', cap, len(corpus) + len(stress), flush=True)
        samples = read_jsonl(ROOT / 'data/releases/v5/curated/splits/fresh_paper_test.jsonl')
        times = {'original': [], 'optimized': []}
        total_bytes = sum(len(r['latex'].encode()) for r in samples)
        for repetition in range(5):
            order = [('original', old), ('optimized', new)] if repetition % 2 == 0 else [('optimized', new), ('original', old)]
            for name, model in order:
                start = time.perf_counter()
                encodings = [model.encode(r['latex']) for r in samples]
                times[name].append(time.perf_counter() - start)
        medians = {k: statistics.median(v) for k, v in times.items()}
        result['ast_speed'].append({'capacity': cap, 'formulas': len(samples), 'seconds': times,
                                    'mib_s': {k: total_bytes / (1024 ** 2 * value) for k, value in medians.items()},
                                    'speedup': medians['original'] / medians['optimized']})
    for scenario, split in [('broad', 'train'), ('specialized', 'transfer_train')]:
        parent = RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json')
        records = read_jsonl(ROOT / f'data/releases/v5/curated/splits/{split}.jsonl')
        timings = {'original': [], 'optimized': []}
        for repetition in range(3):
            order = ['original', 'optimized'] if repetition % 2 == 0 else ['optimized', 'original']
            for name in order:
                cls = original_v5.RLVocab if name == 'original' else RLVocab
                config_cls = original_v5.VocabRLConfig if name == 'original' else type(parent.config)
                config = config_cls(**asdict(parent.config))
                started = time.perf_counter()
                model = cls(config).train(records, 1024, 17, 'rl')
                timings[name].append(time.perf_counter() - started)
                assert model.vocab == parent.vocab and model.merges == parent.merges
                assert (model.theta == parent.theta).all()
                assert model.fit_ids == parent.fit_ids and model.reward_ids == parent.reward_ids
                print('v5 equivalent training', scenario, name, repetition, round(timings[name][-1], 3), flush=True)
        median = {k: statistics.median(v) for k, v in timings.items()}
        result['v5_train_speed'].append({'scenario': scenario, 'capacity': 1024, 'wall_seconds': timings,
                                         'median_wall_seconds': median, 'speedup': median['original'] / median['optimized'],
                                         'exact_vocab_merges_policy_and_scope': True})
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
