"""Verify frozen artifacts and reload all twelve source-resplit checkpoints.

This complements the complete public-encoder replay in audit_results_v7.py;
it never trains, selects, refreezes, or labels automatic reviews as expert gold.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import re

from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split, master_path
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.experiment_v7 import read_external
from math_tokenizer.measure_v6 import measure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    folder = ROOT/'results/v7/full'
    registry = read(folder/'registry.json')
    protocol = read(folder/'evaluation_protocol.json')
    first = read(folder/'first_test_score.json')
    audit = read(folder/'fairness_audit.json')
    metrics = read(folder/'metrics.json')
    ranks = read(folder/'rankings.json')
    completion = read(folder/'completion.json')
    external = read_external()
    assert len(external) == 3933
    assert len(registry) == completion['configurations'] == 79
    assert len(metrics) == completion['metric_records'] == 204
    assert completion['formula_rows'] == 376996
    assert sha(folder/'registry.json') == protocol['registry_sha256'] == first['registry_sha256'] == audit['registry_sha256']
    assert sha(folder/'per_formula.jsonl.gz') == audit['per_formula_sha256']
    assert audit['all_passed'] and audit['total_rounds'] == 8
    assert datetime.fromisoformat(protocol['frozen_utc']) < datetime.fromisoformat(first['utc'])
    for entry in registry:
        if 'checkpoint_sha256' in entry:
            assert sha(ROOT/entry['path']) == entry['checkpoint_sha256'], entry['key']
    assert all(not m['provisional_timing'] and len(m['encode_seconds_repeats']) == len(m['decode_seconds_repeats']) == 7 for m in metrics)
    assert len(ranks['within_scenario']) == 10
    assert all(len(b['metrics']) == 20 for b in ranks['within_scenario'].values())
    with (folder/'all_main_metrics.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 200
    active_sha = '585feae1398244fc4f99350e74af3d76bb2608cb8b122e68ef083c79228a5294'
    assert sha(ROOT/'models/current/registry.json') == active_sha
    old_corpus_sha = sha(master_path())
    assert old_corpus_sha == '972ada427ecb69f38aa2359b7f10fcfbf13f6896a1da7986656ee1321ec24bc7'
    pool = {r['id']: r for r in load_split('train') + load_split('dev')}
    replication = []
    saved_metrics = read(ROOT/'results/v7/source_resplits/metrics.json')
    assert len(saved_metrics) == 12
    for seed in (71, 109, 173):
        dest = ROOT/f'results/v7/source_resplits/{seed}'
        scope = read(dest/'scope.json')
        sets = [set(scope[k]) for k in ('fit_ids', 'reward_ids', 'heldout_ids')]
        groups = [{pool[i]['split_group'] for i in ids} for ids in sets]
        assert all(not groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        records = [pool[i] for i in scope['heldout_ids']]
        assert len(records) == 256
        for original in (m for m in saved_metrics if m['seed'] == seed):
            model = load_checkpoint(dest/f"{original['method']}.json")
            observed, _ = measure(model, records, original['method'], repeats=1)
            for key in ('tokens', 'bytes', 'test_id_sha256', 'document_macro_tokens_per_byte', 'independent_tree_crossing', 'command_integrity', 'number_integrity', 'roundtrip_rate', 'payload_fidelity_rate'):
                if isinstance(original[key], float):
                    assert abs(observed[key] - original[key]) < 1e-12, (seed, original['method'], key)
                else:
                    assert observed[key] == original[key], (seed, original['method'], key)
            assert Counter(str(i) for r in records for i in model.encode(r['latex'])) == Counter(original['usage'])
            replication.append({'seed': seed, 'method': original['method'], 'heldout_rows': len(records), 'reloaded_metrics_match': True})
    test_log = (ROOT/'results/v7/final_tests.log').read_text()
    match = re.search(r'(\d+) passed', test_log)
    assert match and 'FAILED' not in test_log and 'ERROR' not in test_log
    completion['uniform_retiming_required'] = False
    completion['uniform_retiming_completed'] = True
    (folder/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    verified = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
        'tests_passed': int(match[1]), 'formal_configurations': len(registry), 'formal_metric_records': len(metrics),
        'formal_formula_rows': completion['formula_rows'], 'fairness_review_rounds': 8,
        'frozen_before_first_external_score': True, 'full_public_encoder_replay_passed': True,
        'active_registry_unchanged_sha256': active_sha, 'v6_master_unchanged_sha256': old_corpus_sha,
        'source_resplit_checkpoint_replays': replication, 'expert_reviews_completed': 0,
        'second_hardware_tested': False, 'downstream_tested': False,
        'hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in (
            folder/'registry.json', folder/'metrics.json', folder/'per_formula.jsonl.gz',
            folder/'rankings.json', folder/'all_main_metrics.csv', folder/'fairness_audit.json',
            ROOT/'data/releases/v7/external_test.jsonl.gz')}}
    (ROOT/'results/v7/verification.json').write_text(json.dumps(verified, indent=2)+'\n')
    print('Release verified:', verified['tests_passed'], 'tests,', len(replication), 'reloaded source-split models')


if __name__ == '__main__':
    main()
