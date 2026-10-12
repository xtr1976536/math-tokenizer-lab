"""Harmonize auxiliary readouts, then directly compare all frozen v6 methods.

No model selection, corpus changes, retraining or timing changes. This reporting
amendment is made after the original experiments, as requested by the user.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json

import numpy as np

from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.comparison_v6 import (
    CALIBRATION_METRICS, calibrate, comparison_boards, id_hash,
)
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v6 import load_entry
from math_tokenizer.measure_v6 import fit_usage
from summarize_full_v6 import aggregate_rows, paired_bootstrap

OUT = ROOT / 'results/v6/full'
CALIBRATION_FIELDS = set(CALIBRATION_METRICS) | {'fit_calibrations', 'fit_calibration_scenario'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def immutable_metric_digest(metrics):
    unchanged = [{k: v for k, v in m.items() if k not in CALIBRATION_FIELDS} for m in metrics]
    return hashlib.sha256(json.dumps(unchanged, sort_keys=True).encode()).hexdigest()


def immutable_row_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        original = {k: v for k, v in row.items()
                    if k not in {'predictive_bits', 'fit_predictive_bits_by_scenario'}}
        digest.update(json.dumps(original, sort_keys=True).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def main():
    metrics = json.loads((OUT / 'metrics.json').read_text())
    registry = json.loads((OUT / 'registry.json').read_text())
    original_protocol = json.loads((ROOT / 'results/v6/evaluation_protocol.json').read_text())
    weights = original_protocol['comprehensive_weights']
    scopes = {}
    for scenario in ('broad', 'specialized'):
        a = json.loads((ROOT / f'results/v6/{scenario}_1024/scope.json').read_text())['fit_ids']
        b = json.loads((ROOT / f'results/v6/{scenario}_2048/scope.json').read_text())['fit_ids']
        assert set(a) == set(b), 'Cannot reuse fixed-tokenizer calibration across different fit corpora'
        scopes[scenario] = sorted(a)
    protocol_path = OUT / 'comparison_protocol.json'
    if not protocol_path.exists():
        write_json(protocol_path, {
            'amendment_utc': datetime.now(timezone.utc).isoformat(),
            'user_request': 'Direct identical-experiment comparison of all eight mature tokenizers with the proposed mathematical tokenizers.',
            'post_test_reporting_amendment': True,
            'model_selection_or_training_changed': False,
            'corpus_or_timing_changed': False,
            'weights_changed': False, 'weights': weights,
            'original_protocol_sha256': hashlib.sha256((ROOT / 'results/v6/evaluation_protocol.json').read_bytes()).hexdigest(),
            'comparison_scope': 'Deployed systems on common complete raw inputs, official pretrained vocabularies retained; not a matched-training causal algorithm comparison.',
            'scenarios_equal_weight': ['broad/fresh_test', 'specialized/extreme_domain_test'],
            'additional_full_suites': ['broad/extreme_domain_test', 'broad/stress_test', 'specialized/stress_test'],
            'unigram_calibration': 'Every comparator uses identical scenario fit IDs, add-half smoothing and its actual vocabulary; broad-model OOD retains broad fit calibration.',
            'fit_ids': {s: {'count': len(ids), 'sha256': id_hash(ids)} for s, ids in scopes.items()},
            'limitations': ['External pretraining and historical paper exposure are unknown.',
                            'Intrinsic unigram BPB is not language-model perplexity or a downstream performance measure.',
                            'Composite rankings depend on weights and the comparator set.'],
        })
    all_rows = list(map(json.loads, gzip.open(OUT / 'per_formula.jsonl.gz', 'rt')))
    metric_before = immutable_metric_digest(metrics)
    row_before = immutable_row_digest(all_rows)
    rows = defaultdict(list)
    for r in all_rows:
        rows[(r['key'], r['suite'])].append(r)
    samples = {s: load_split(s) for s in ('fresh_test', 'extreme_domain_test')}
    samples['stress_test'] = list(map(json.loads, (ROOT / 'results/v6/stress_test.jsonl').open()))
    expected = {s: {r['id']: r for r in rs} for s, rs in samples.items()}
    # Review 1: common complete raw inputs, exact IDs, byte totals and source groups.
    for m in metrics:
        rr = rows[(m['key'], m['suite'])]
        original = expected[m['suite']]
        assert len(rr) == len(original) == len({r['id'] for r in rr})
        assert {r['id'] for r in rr} == set(original)
        assert all(r['bytes'] == len(original[r['id']]['latex'].encode()) and
                   r['group'] == original[r['id']]['split_group'] for r in rr)
        assert m['test_id_sha256'] == id_hash(original)
    logs = [{'round': 1, 'review': 'All configurations have identical full suite IDs, raw bytes and paper groups; no filtering or truncation.', 'passed': True, 'rows': len(all_rows)}]
    fit_byid = {r['id']: r for r in load_split('train')}
    test_ids = set().union(*(set(s) for s in expected.values()))
    for ids in scopes.values():
        assert set(ids) <= set(fit_byid) and not set(ids) & test_ids
    amended_path = OUT / 'calibration_amendment.json'
    previous = json.loads(amended_path.read_text()) if amended_path.exists() else {
        'old_readouts': [{k: m[k] for k in ('key', 'suite', *CALIBRATION_METRICS)}
                         for m in metrics if m['group'] == 'fixed_pretrained_reference'],
        'old_per_formula_gz_sha256': hashlib.sha256((OUT / 'per_formula.jsonl.gz').read_bytes()).hexdigest(),
        'reason': 'Fixed tokenizers previously used broad fit counts for extreme-domain readouts. Both fit scopes are now recorded and the primary extreme readout uses specialized fit counts.',
    }
    # Review 2: reload original official tokenizers; independent count-based
    # recomputation validates both auxiliary readouts and every per-formula bit sum.
    calibrated = 0
    for entry in [e for e in registry if e['group'] == 'fixed_pretrained_reference']:
        model = load_entry(entry)
        if entry.get('checkpoint_sha256'):
            assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['checkpoint_sha256']
        usages = {s: fit_usage(model, [fit_byid[id] for id in ids]) for s, ids in scopes.items()}
        for m in [m for m in metrics if m['key'] == entry['key']]:
            rr = rows[(m['key'], m['suite'])]
            encodings = [model.encode(expected[m['suite']][r['id']]['latex']) for r in rr]
            assert all(len(ids) == r['tokens'] and model.decode(ids) == expected[m['suite']][r['id']]['latex']
                       for r, ids in zip(rr, encodings))
            m['fit_calibrations'] = {}
            for scenario, usage in usages.items():
                calibration, bits = calibrate(encodings, usage, m['vocab_actual'], m['bytes'])
                test_usage = Counter(token for ids in encodings for token in ids)
                denominator = sum(usage.values()) + .5 * m['vocab_actual']
                independent_bits = sum(count * -np.log2((usage[token] + .5) / denominator)
                                       for token, count in test_usage.items())
                assert abs(calibration['fit_predictive_unigram_bits_per_byte'] - independent_bits / m['bytes']) < 1e-10
                assert abs(calibration['fit_rare_token_rate'] - sum(count for token, count in test_usage.items() if usage[token] <= 5) / m['tokens']) < 1e-12
                calibration['fit_id_sha256'] = id_hash(scopes[scenario])
                m['fit_calibrations'][scenario] = calibration
                for r, value in zip(rr, bits):
                    r.setdefault('fit_predictive_bits_by_scenario', {})[scenario] = value
                calibrated += 1
            primary = 'specialized' if m['suite'] == 'extreme_domain_test' else 'broad'
            m['fit_calibration_scenario'] = primary
            for k in CALIBRATION_METRICS:
                m[k] = m['fit_calibrations'][primary][k]
            for r in rr:
                r['predictive_bits'] = r['fit_predictive_bits_by_scenario'][primary]
        print('official replay and equal-fit calibration:', entry['name'], flush=True)
    logs.append({'round': 2, 'review': 'All eight official tokenizers replayed; same disjoint fit IDs at both budgets; independent frequency-count BPB/rare-rate validation for both fit scopes.', 'passed': True, 'calibration_records': calibrated})
    # Review 3: all preexisting encoding, quality, timing and model records stay
    # semantically identical; 7 timing repetitions are present in every record.
    assert metric_before == immutable_metric_digest(metrics)
    assert row_before == immutable_row_digest(all_rows)
    timing = json.loads((OUT / 'timing_protocol.json').read_text())
    assert timing['repeats'] == 7 and timing['training_not_running']
    assert all(len(m['encode_seconds_repeats']) == len(m['decode_seconds_repeats']) == 7 for m in metrics)
    for m in metrics:
        rr = rows[(m['key'], m['suite'])]
        a = aggregate_rows(rr)
        for k in a:
            if k in m and a[k] is not None:
                assert abs(a[k] - m[k]) < 1e-10, (m['key'], k)
        assert abs(sum(r['predictive_bits'] for r in rr) / m['bytes'] - m['fit_predictive_unigram_bits_per_byte']) < 1e-10
    logs.append({'round': 3, 'review': 'All old noncalibration metrics and rows unchanged; independent aggregate agreement; same original 7 public-interface timing repetitions including frontends.', 'passed': True,
                 'immutable_metrics_sha256': metric_before, 'immutable_rows_sha256': row_before})
    write_json(OUT / 'metrics.json', metrics)
    temporary = OUT / 'per_formula.harmonized.jsonl.gz'
    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
        for r in all_rows:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    temporary.replace(OUT / 'per_formula.jsonl.gz')
    previous.update({'noncalibration_metric_digest': metric_before, 'noncalibration_row_digest': row_before,
                     'new_per_formula_gz_sha256': hashlib.sha256((OUT / 'per_formula.jsonl.gz').read_bytes()).hexdigest(),
                     'model_checkpoints_changed': False, 'timings_changed': False})
    write_json(amended_path, previous)
    boards, combined = comparison_boards(metrics, weights)
    pairs = {}
    for capacity in (1024, 2048):
        for label in ('broad', 'specialized', 'broad_ood'):
            board = boards[f'{capacity}/{label}']
            mature = [n for n, m in board['metrics'].items() if m['model_group'] == 'fixed_pretrained_reference']
            assert len(mature) == 8 and len(board['comprehensive_ranking']) == 15
            for left in ('ast_v4', 'v5'):
                for right in mature:
                    lkeys, rkeys = board['metrics'][left]['keys'], board['metrics'][right]['keys']
                    assert len(lkeys) == len(rkeys) == 1, 'Do not treat seed rows as independent formulas'
                    pairs[f'{capacity}/{label}/{left}-vs-{right}'] = paired_bootstrap(
                        rows[(lkeys[0], board['suite'])], rows[(rkeys[0], board['suite'])])
    sensitivity = []
    for speed in (.10, .20, .30, .40):
        alternate = {k: v * (1 - speed) / (1 - weights['encode_mib_s'])
                     for k, v in weights.items() if k != 'encode_mib_s'}
        alternate['encode_mib_s'] = speed
        _, scores = comparison_boards(metrics, alternate)
        sensitivity.append({'speed_weight': speed, 'combined': scores})
    logs.append({'round': 4, 'review': 'Every main/OOD/stress comparison board includes all eight mature tokenizers and seven matched methods, same calibration scope and lossless eligibility; paired paper CIs and fixed-weight sensitivity.', 'passed': True, 'boards': len(boards), 'paired_comparisons': len(pairs)})
    write_json(OUT / 'general_vs_math_rankings.json', {
        'weights': weights, 'scenario_weight': {'broad': .5, 'specialized': .5},
        'within_scenario': boards, 'combined': combined, 'weight_sensitivity': sensitivity,
        'paired_paper_bootstraps': pairs,
        'scope': 'Practical deployed-tokenizer comparison with identical evaluation protocol; mature models retain actual external vocabulary. Matched algorithm rankings remain in rankings.json.',
        'uncertainty': 'Paper-cluster bootstrap conditional on frozen vocabulary; pairwise 95% CIs are descriptive and unadjusted for multiple comparisons.',
    })
    write_json(OUT / 'comparison_fairness_audit.json', {
        'rounds': logs, 'all_passed': True, 'total_comparison_fairness_rounds': 4,
        'corpus_sha256': original_protocol['corpus_sha256'],
        'per_formula_gz_sha256': previous['new_per_formula_gz_sha256'],
    })
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == '__main__':
    main()
