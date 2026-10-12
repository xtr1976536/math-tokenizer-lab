"""Common calibration and practical rankings for frozen deployed tokenizers.

This comparison holds evaluation inputs fixed, not external training history.
The original matched-training algorithm comparison remains separately available.
"""
from collections import defaultdict
import hashlib
import math

import numpy as np
from scipy.stats import rankdata

LOWER = {'document_macro_tokens_per_byte', 'independent_tree_crossing',
         'fit_predictive_unigram_bits_per_byte'}
CALIBRATION_METRICS = ('fit_predictive_unigram_bits_per_byte', 'fit_rare_token_rate')


def id_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def calibrate(encodings, training_usage, vocab_size, raw_bytes):
    """Same add-half fit-only unigram readout used by measure_v6.measure."""
    denominator = sum(training_usage.values()) + .5 * vocab_size
    bits = [-sum(math.log2((training_usage[token] + .5) / denominator)
                 for token in ids) for ids in encodings]
    rare = sum(training_usage[token] <= 5 for ids in encodings for token in ids)
    total = sum(map(len, encodings))
    return {
        'fit_predictive_unigram_bits_per_byte': sum(bits) / max(1, raw_bytes),
        'fit_rare_token_rate': rare / max(1, total),
    }, bits


def comparison_boards(metrics, weights):
    """15 methods per board: seven matched methods plus eight official models.

    A pretrained tokenizer retains its actual vocabulary. Its auxiliary unigram
    readout uses exactly the fit IDs of the matched methods in each scenario.
    Stress and broad-model OOD are reported without adding to combined scores.
    """
    boards = {}
    for capacity in (1024, 2048):
        for scenario, suite, label in (
            ('broad', 'fresh_test', 'broad'),
            ('specialized', 'extreme_domain_test', 'specialized'),
            ('broad', 'extreme_domain_test', 'broad_ood'),
            ('broad', 'stress_test', 'broad_stress'),
            ('specialized', 'stress_test', 'specialized_stress'),
        ):
            selected = [m for m in metrics if m['suite'] == suite and (
                m['group'] == 'fixed_pretrained_reference' or
                (m['group'] == 'matched_primary' and m['capacity'] == capacity
                 and m['scenario'] == scenario))]
            assert len({m['test_id_sha256'] for m in selected}) == 1
            assert len({(m['formulas'], m['bytes']) for m in selected}) == 1
            grouped = defaultdict(list)
            for original in selected:
                m = dict(original)
                if m['group'] == 'fixed_pretrained_reference':
                    m.update(m['fit_calibrations'][scenario])
                name = 'v5' if m['method'] in {'v5_structural', 'v5_trajectory'} else m['method']
                grouped[name].append(m)
            means = {}
            for name, records in grouped.items():
                # Report all scalar intrinsic metrics, not only scored metrics.
                scalar_keys = [k for k, v in records[0].items()
                               if isinstance(v, (float, int)) and not isinstance(v, bool)]
                means[name] = {k: float(np.mean([r[k] for r in records])) for k in scalar_keys}
                for alpha in records[0].get('renyi_efficiency', {}):
                    means[name]['renyi_efficiency_' + alpha] = float(np.mean([
                        r['renyi_efficiency'][alpha] for r in records]))
                for index, quantile in enumerate(('p50', 'p95', 'p99')):
                    if 'length_p50_p95_p99' in records[0]:
                        means[name]['length_' + quantile] = float(np.mean([
                            r['length_p50_p95_p99'][index] for r in records]))
                means[name]['keys'] = [r['key'] for r in records]
                means[name]['model_group'] = records[0]['group']
                means[name]['scored_seed_count'] = len(records)
            eligible = sorted(name for name, records in grouped.items()
                              if all(r['rank_eligible_lossless'] for r in records))
            scores = {name: 0. for name in eligible}
            metric_rankings = {}
            metric_ranks = {}
            for metric, weight in weights.items():
                values = [means[name][metric] * (1 if metric in LOWER else -1) for name in eligible]
                percentile = (rankdata(values, method='average') - 1) / max(1, len(eligible) - 1)
                for name, value in zip(eligible, percentile):
                    scores[name] += weight * value
                metric_rankings[metric] = sorted(eligible, key=lambda n: (
                    means[n][metric] * (1 if metric in LOWER else -1), n))
                metric_ranks[metric] = dict(zip(eligible, map(int, rankdata(values, method='min'))))
            pareto = []
            points = {n: np.array([means[n][k] * (1 if k in LOWER else -1) for k in weights])
                      for n in eligible}
            for name in eligible:
                if not any(np.all(points[other] <= points[name]) and np.any(points[other] < points[name])
                           for other in eligible if other != name):
                    pareto.append(name)
            boards[f'{capacity}/{label}'] = {
                'suite': suite, 'fit_calibration_scenario': scenario,
                'test_id_sha256': selected[0]['test_id_sha256'],
                'primary_ranking': metric_rankings['document_macro_tokens_per_byte'],
                'comprehensive_ranking': sorted(eligible, key=lambda n: (scores[n], n)),
                'metric_rankings': metric_rankings, 'scores': scores, 'pareto': pareto,
                'metric_ranks': metric_ranks,
                'comprehensive_ranks': dict(zip(eligible, map(int, rankdata(
                    [scores[n] for n in eligible], method='min')))),
                'metrics': means, 'excluded_nonlossless': sorted(set(grouped) - set(eligible)),
            }
    combined = {}
    for capacity in (1024, 2048):
        a, b = boards[f'{capacity}/broad'], boards[f'{capacity}/specialized']
        assert set(a['scores']) == set(b['scores'])
        scores = {n: .5 * (a['scores'][n] + b['scores'][n]) for n in a['scores']}
        combined[str(capacity)] = {'ranking': sorted(scores, key=lambda n: (scores[n], n)), 'scores': scores,
                                   'ranks': dict(zip(scores, map(int, rankdata(list(scores.values()), method='min'))))}
    return boards, combined
