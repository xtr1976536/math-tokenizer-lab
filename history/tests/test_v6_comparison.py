"""Regression checks for common-input and common-fit practical comparisons."""
from collections import Counter
import copy
import math

import pytest

from math_tokenizer.baselines import ByteTokenizer
from math_tokenizer.comparison_v6 import calibrate, comparison_boards
from math_tokenizer.measure_v6 import measure


WEIGHTS = {'document_macro_tokens_per_byte': .3, 'encode_mib_s': .25,
           'independent_tree_crossing': .2, 'command_integrity': .1,
           'number_integrity': .05, 'fit_predictive_unigram_bits_per_byte': .1}


def test_fit_readout_matches_full_measure_without_test_fitting():
    text = r'\alpha+x=7'
    record = {'id': 'heldout', 'latex': text, 'split_group': 'paper:heldout',
              'domain': 'analysis', 'paper_domain': 'analysis', 'origin': 'human',
              'complexity': {'class': 'simple'}, 'reference': {'spans': [[0, 6]]}}
    model = ByteTokenizer()
    usage = Counter(b'xxx=1')  # Held-out command and digit must remain unseen.
    metric, rows = measure(model, [record], 'bytes', repeats=1, training_usage=usage)
    readout, bits = calibrate([model.encode(text)], usage, model.vocab_size, len(text.encode()))
    for key, value in readout.items():
        assert value == pytest.approx(metric[key], abs=1e-12)
    assert bits[0] == rows[0]['predictive_bits']
    assert sum(usage.values()) == 5
    expected = -sum(math.log2((usage[token] + .5) / (5 + 128)) for token in text.encode())
    assert bits[0] == pytest.approx(expected)


def synthetic_metrics():
    records = []
    for capacity in (1024, 2048):
        for scenario in ('broad', 'specialized'):
            suites = ['fresh_test', 'extreme_domain_test', 'stress_test'] if scenario == 'broad' else ['extreme_domain_test', 'stress_test']
            for method in ('ast_v4', 'v5_structural'):
                for suite in suites:
                    records.append({'method': method, 'key': f'{capacity}/{scenario}/{method}',
                                    'scenario': scenario, 'capacity': capacity,
                                    'suite': suite, 'group': 'matched_primary', 'formulas': 2,
                                    'bytes': 9, 'test_id_sha256': suite, 'seed': 0,
                                    'rank_eligible_lossless': True, 'document_macro_tokens_per_byte': .5,
                                    'encode_mib_s': 10., 'independent_tree_crossing': .1,
                                    'command_integrity': .9, 'number_integrity': .9,
                                    'fit_predictive_unigram_bits_per_byte': 3.})
    for suite in ('fresh_test', 'extreme_domain_test', 'stress_test'):
        records.append({'method': 'official', 'key': 'official', 'scenario': 'pretrained',
                        'capacity': None, 'suite': suite, 'group': 'fixed_pretrained_reference',
                        'formulas': 2, 'bytes': 9, 'test_id_sha256': suite, 'seed': 0,
                        'rank_eligible_lossless': True, 'document_macro_tokens_per_byte': .6,
                        'encode_mib_s': 11., 'independent_tree_crossing': .2,
                        'command_integrity': .3, 'number_integrity': .9,
                        'fit_predictive_unigram_bits_per_byte': 999.,
                        'fit_calibrations': {'broad': {'fit_predictive_unigram_bits_per_byte': 4.},
                                             'specialized': {'fit_predictive_unigram_bits_per_byte': 7.}}})
    return records


def test_official_comparator_uses_scenario_fit_even_for_same_ood_strings():
    records = synthetic_metrics()
    before = copy.deepcopy(records)
    boards, combined = comparison_boards(records, WEIGHTS)
    assert boards['1024/specialized']['metrics']['official']['fit_predictive_unigram_bits_per_byte'] == 7.
    assert boards['1024/broad_ood']['metrics']['official']['fit_predictive_unigram_bits_per_byte'] == 4.
    assert boards['2048/specialized_stress']['metrics']['official']['fit_predictive_unigram_bits_per_byte'] == 7.
    assert boards['2048/broad_stress']['metrics']['official']['fit_predictive_unigram_bits_per_byte'] == 4.
    assert set(combined['1024']['ranking']) == {'ast_v4', 'v5', 'official'}
    assert records == before


@pytest.mark.parametrize('field,value', [('test_id_sha256', 'other'), ('formulas', 1), ('bytes', 8)])
def test_input_mismatch_cannot_silently_enter_comparison(field, value):
    records = synthetic_metrics()
    records[-1][field] = value
    with pytest.raises(AssertionError):
        comparison_boards(records, WEIGHTS)


def test_nonlossless_official_is_excluded_explicitly():
    records = synthetic_metrics()
    for record in records:
        if record['method'] == 'official':
            record['rank_eligible_lossless'] = False
    boards, combined = comparison_boards(records, WEIGHTS)
    assert 'official' not in combined['2048']['ranking']
    assert boards['1024/broad']['excluded_nonlossless'] == ['official']


def test_identical_values_get_tied_ranks_regardless_of_method_name():
    boards, combined = comparison_boards(synthetic_metrics(), WEIGHTS)
    board = boards['1024/broad']
    assert board['metric_ranks']['document_macro_tokens_per_byte']['ast_v4'] == 1
    assert board['metric_ranks']['document_macro_tokens_per_byte']['v5'] == 1
    assert board['metric_ranks']['document_macro_tokens_per_byte']['official'] == 3
    assert combined['1024']['ranks']['ast_v4'] == combined['1024']['ranks']['v5']
