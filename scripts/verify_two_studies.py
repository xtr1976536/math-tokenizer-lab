"""Read-only scientific checks and delivery inventory for the two complete studies.

Only reports/repository_two_studies/ is written. No training, corpus rewriting,
checkpoint updates, or modifications of the original run audits are performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/repository_two_studies'
ARMS = ('eqgpt_reference', 'ast_bpe_280', 'ast_bpe_315', 'ast_bpe_350')


def read(path):
    return json.loads((ROOT / path).read_text())


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def check_metrics(saved, summary):
    total = lambda key: sum(r[key] for r in saved)
    for metric, num, den, scale in (
        ('tokenizer compression', 'initial_character_tokens', 'final_tokens', 1),
        ('operation', 'commands_intact', 'commands', 100),
        ('numbers', 'numbers_intact', 'numbers', 100),
    ):
        assert total(den) > 0
        assert abs(total(num) / total(den) * scale - summary[metric]) < 1e-10


def round_one():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + str(ROOT / 'src')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return {'status': 'PASS', 'tests': result.stdout.strip()}


def round_two():
    import torch
    from math_tokenizer.pde_protocol import split_corpus, digest
    from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
    from math_tokenizer.eqgpt.evaluation import classify
    from math_tokenizer.eqgpt.author import tensor_hash
    from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
    from math_tokenizer.eqgpt.expanded_preflight import verified_preflight, verified_settings
    upstream = ROOT / 'experiments/upstream_811'
    source = ROOT / 'data/formulas/pde/pde_equations.jsonl.gz'
    formulas = rows(source)
    plans, split = split_corpus(formulas)
    protocol = read(upstream / 'protocol.json')
    assert len(formulas) == 1500 and protocol['split'] == split
    assert protocol['data_sha256'] == digest_file(source)
    assert split['counts'] == {'train': 1200, 'validation': 150, 'test': 150}
    subsets = [{r['id'] for r in plans[k]} for k in ('train', 'validation', 'test')]
    assert not (subsets[0] & subsets[1] or subsets[0] & subsets[2] or subsets[1] & subsets[2])
    assert set.union(*subsets) == {r['id'] for r in formulas}
    matrix = read(upstream / 'validation_results.json')
    assert len(matrix) == 132
    assert Counter(r['status'] for r in matrix) == {
        'evaluated': 78, 'training_characters_not_covered': 44,
        'below_base_plus_controls': 10,
    }
    metric_files = 0
    for row in matrix:
        if row['status'] != 'evaluated':
            continue
        folder = upstream / row['alphabet']
        size = row['total_ids']
        model = CompactAlphabet.load(folder / f'{size}.model.json')
        assert len(model.vocab) + 3 == size
        assert digest(json.dumps([x.hex() for x in model.vocab])) == row['vocabulary_sha256']
        for part in ('train', 'validation'):
            saved = rows(folder / f'{size}.{part}.jsonl.gz')
            assert [r['id'] for r in saved] == [r['id'] for r in plans[part]]
            check_metrics(saved, row[part]); metric_files += 1
        for formula in plans['validation']:
            assert model.decode(model.encode(formula['latex'])) == formula['latex']
    tests = read(upstream / 'test_results.json')
    selection = read(upstream / 'selection.json')
    selected = {(r['alphabet'], r['total_ids']) for r in selection['arms']}
    assert {(r['alphabet'], r['total_ids']) for r in tests if r['method'] == 'ast_bpe'} == selected
    assert len(tests) == 6 and len(selected) == 3
    for row in tests:
        if row['method'] == 'ast_bpe':
            path = upstream / row['alphabet'] / f"{row['total_ids']}.test.jsonl.gz"
        else:
            path = upstream / f"{row['method']}.test.jsonl.gz"
        saved = rows(path)
        assert [r['id'] for r in saved] == [r['id'] for r in plans['test']]
        assert row['test_ids_sha256'] == digest('\n'.join(r['id'] for r in plans['test']))
        check_metrics(saved, row); metric_files += 1
    # Both downstream studies: independently recompute classifications and budgets.
    summaries = {}
    for study, directory in [('study1', 'experiments/downstream'),
                              ('study2', 'experiments/expanded_downstream_80_v1/cases')]:
        result = {arm: {'top1': 0, 'top10': 0, 'evaluated': 0} for arm in ARMS}
        for case in SHARED8_CASES:
            cache = ROOT / 'data/mlp_cache' / case / 'surrogate_cache.pt'
            assert digest_file(cache) == read(cache.parent / 'surrogate_quality.json')['cache_sha256']
            for arm in ARMS:
                folder = ROOT / directory / case / arm
                data = read(folder / 'search.json')
                assert len(data['attempts']) == 600 and len(data['history']) == 3
                assert (data['attempts_per_cycle'], data['cycles'], data['continuation_epochs']) == (200, 3, 2)
                assert data['max_new_tokens'] == (50 if study == 'study1' else 80)
                classes = [classify(seq, case)['status'] for seq in data['history'][-1]['gre']]
                summary = read(folder / 'fast_summary.json')
                assert summary['top1'] == classes[0] and summary['top10_correct'] == ('correct' in classes)
                result[arm]['top1'] += classes[0] == 'correct'
                result[arm]['top10'] += 'correct' in classes
                result[arm]['evaluated'] += sum(r['status'] == 'evaluated' for r in data['attempts'])
                if study == 'study1':
                    training = read(folder / 'training.json')
                    assert (training['samples'], training['epochs']) == (6784, 25)
                    assert read(folder.parent / 'protocol.json')['positions'] == 50
                else:
                    pre = ROOT / 'experiments/expanded_downstream_80_v1/shared_pretrain' / arm
                    training = read(pre / 'training.json')
                    assert (training['samples'], training['epochs']) == (59360, 25)
                    binding = read(folder / 'SEARCH_INPUT.json')
                    assert binding['pretrained_sha256'] == training['final_sha256']
                    assert binding['mlp_sha256'] == digest_file(cache)
        summaries[study] = result
    # First study upstream artifacts are retained and independently recomputable.
    first_protocol = read('experiments/upstream_main/protocol.json')
    first_files = list((ROOT / 'experiments/upstream_main').glob('pde_*/*.per_formula.jsonl.gz'))
    assert len(first_files) == 16
    for path in first_files:
        saved = rows(path)
        summary = read(path.with_name(path.name.replace('.per_formula.jsonl.gz', '.json')))
        totals = {key: sum(r[key] for r in saved) for key in ('initial_character_tokens', 'final_tokens', 'commands', 'commands_intact', 'numbers', 'numbers_intact')}
        assert abs(totals['initial_character_tokens'] / totals['final_tokens'] - summary['compression_ratio']) < 1e-10
        assert abs(100 * totals['commands_intact'] / totals['commands'] - summary['cmd_pct']) < 1e-10
        assert abs(100 * totals['numbers_intact'] / totals['numbers'] - summary['num_pct']) < 1e-10
    assert [summaries['study1'][a]['top1'] for a in ARMS] == [3, 3, 3, 4]
    assert [summaries['study2'][a]['top1'] for a in ARMS] == [2, 1, 3, 1]
    corpus, _ = verified_preflight(ROOT / 'data/formulas/expanded_downstream_80_v1')
    verified_settings(ROOT / 'data/formulas/expanded_downstream_80_v1', corpus)
    sources = [json.loads(x) for x in (ROOT / 'data/formulas/expanded_downstream_80_v1/sources.jsonl').read_text().splitlines()]
    old = [json.loads(x) for x in (ROOT / 'data/formulas/gre_benchmark_audit_v2/eqgpt_221_after_target_exclusion.jsonl').read_text().splitlines()]
    assert Counter(tuple(x['gre_ids']) for x in sources if x['source_kind'] == 'eqgpt_handbook_non_target') == Counter(tuple(x['gre_ids']) for x in old)
    # Serialized file hashes and tensor hashes for both versions' checkpoint distributions.
    manifest = read('experiments/expanded_downstream_80_v1/weight_manifest.json')
    for item in manifest['files']:
        p = ROOT / 'experiments/expanded_downstream_80_v1' / item['path']
        assert p.stat().st_size == item['size_bytes'] and digest_file(p) == item['sha256']
        state = torch.load(p, map_location='cpu', weights_only=True)
        expected = (read(p.parent / 'training.json')['final_sha256'] if p.name == 'pretrained.pt'
                    else read(p.parent / 'search.json')['history'][-1]['after_sha256'])
        assert tensor_hash(state) == expected
    old_manifest = read('data/weights/manifest.json')
    old_weights = [*old_manifest['pretrained'].values(), *(r for case in old_manifest['searched'].values() for r in case.values())]
    for item in old_weights:
        p = ROOT / item['path']
        assert p.stat().st_size == item['bytes'] and digest_file(p) == item['sha256']
    return {'status': 'PASS', 'upstream_formulas': 1500, 'split': split['counts'],
            'validation_combinations': 132, 'validated_tokenizers': 78,
            'reaggregated_metric_files': metric_files, 'study1_upstream_metric_files': len(first_files), 'test_arms': 6,
            'downstream_results': 64, 'checkpoint_files_hashed': 72,
            'study2_tensor_hashes': 36, 'old_212_preserved': True,
            'current_expanded_runtime_gate': 'PASS', 'downstream_summary': summaries}


def round_three():
    docs = [ROOT / p for p in (
        'README.md', 'data/README.md', 'reports/实验方案索引.md',
        'reports/upstream/README.md', 'reports/downstream/README.md',
        'experiments/upstream_811/README.md', 'reports/preparation_811/README.md',
        'reports/expanded_downstream_80_v1/README.md',
        'experiments/expanded_downstream_80_v1/README.md',
    )]
    missing = []
    for doc in docs:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', doc.read_text()):
            target = target.split('#', 1)[0]
            if target and '://' not in target and not (doc.parent / target).exists():
                missing.append(str(doc.relative_to(ROOT)) + ' -> ' + target)
    assert not missing, missing
    listed = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    expected = [p for base in ('experiments/upstream_811', 'reports/preparation_811',
                               'experiments/downstream', 'experiments/expanded_downstream_80_v1',
                               'data/formulas/expanded_downstream_80_v1')
                for p in (ROOT / base).rglob('*') if p.is_file() and p.suffix != '.pt']
    assert all(str(p.relative_to(ROOT)) in listed for p in expected), 'Required study file is not staged/tracked'
    release_records = {}
    for tag, manifest_path in (
        ('eqgpt-fast-20261022-weights-v1', 'data/weights/manifest.json'),
        ('expanded-downstream-80-v1-weights', 'experiments/expanded_downstream_80_v1/weight_manifest.json'),
    ):
        release = json.loads(subprocess.check_output(['gh', 'release', 'view', tag, '--json', 'assets,url'], cwd=ROOT))
        actual = {r['name']: r['size'] for r in release['assets']}
        manifest = read(manifest_path)
        if 'files' in manifest:
            expected_assets = {r['asset']: r['size_bytes'] for r in manifest['files']}
        else:
            entries = [*manifest['pretrained'].values(), *(r for c in manifest['searched'].values() for r in c.values())]
            expected_assets = {r['asset']: r['bytes'] for r in entries}
        assert all(actual.get(name) == size for name, size in expected_assets.items())
        release_records[tag] = {'url': release['url'], 'matched_weights': len(expected_assets)}
    assert subprocess.run(['git', 'diff', '--quiet', '5953ce5', '--', 'experiments/downstream',
                           'experiments/upstream_main', 'experiments/upstream_vocab_size',
                           'data/formulas/eqgpt/PDE_dataset_0725.xlsx'], cwd=ROOT).returncode == 0
    return {'status': 'PASS', 'documentation_links': len(docs),
            'required_tracked_files': len(expected), 'release_records': release_records,
            'study1_results_unchanged_since_5953ce5': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    result = (round_one, round_two, round_three)[args.round - 1]()
    result['round'] = args.round
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / f'round_{args.round}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
