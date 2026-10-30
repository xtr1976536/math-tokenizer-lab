from collections import Counter
from itertools import permutations
import math

import pytest

from math_tokenizer.eqgpt.study3_corpus import (
    PermutationSpace, paired_rows, sample_ranks, multiset_count, rank_of, unrank)
from math_tokenizer.eqgpt.expanded_corpus import gre_signature


@pytest.mark.parametrize('items', [[], [1], [1, 1, 2], [1, 2, 3, 4], [(1,), (1,), (2,)]])
def test_multiset_ranking_bijection_against_exhaustive_oracle(items):
    expected = sorted(set(permutations(items)))
    assert multiset_count(items) == len(expected)
    assert [tuple(unrank(items, r)) for r in range(len(expected))] == expected
    assert [rank_of(items, v) for v in expected] == list(range(len(expected)))


@pytest.mark.parametrize('ids,count', [
    ([5, 6, 2, 7, 3, 8, 2, 9, 1], 12),
    ([5, 6, 2, 7, 3, 7, 1], 2),
    ([5, 7, 3, 8, 2, 8, 3, 7, 1], 4),
    ([5, 6, 3, 7, 4, 8, 4, 9, 1], 4),
])
def test_exact_space_counts_unique_outputs_and_preserves_division(ids, count):
    space = PermutationSpace(ids)
    outputs = [tuple(space.at(r)) for r in range(space.count)]
    assert space.count == count == len(set(outputs))
    assert all(gre_signature(list(v)) == gre_signature(ids) for v in outputs)
    assert all(v[0] == 5 and v[-1] == 1 and v.count(1) == v.count(5) == 1 for v in outputs)


def test_more_than32_is_unique_seed_stable_and_source_order_independent():
    a = dict(source_id='a', gre_ids=[5, 6, 2, 7, 3, 8, 3, 9, 2, 10, 2, 11, 1])
    b = dict(a, source_id='b')
    rows = list(paired_rows(a))
    assert len(rows) == len({tuple(v['gre_ids']) for v in rows}) == 32
    assert rows == list(paired_rows(a))
    assert rows != list(paired_rows(b))
    assert rows[0]['gre_ids'] == a['gre_ids']


def test_small_space_repetition_is_balanced():
    source = dict(source_id='a', gre_ids=[5, 6, 2, 7, 3, 8, 2, 9, 1])
    counts = Counter(tuple(r['gre_ids']) for r in paired_rows(source))
    assert len(counts) == 12 and sum(counts.values()) == 32
    assert max(counts.values()) - min(counts.values()) <= 1


def test_factorial_space_beyond_machine_integer_is_not_enumerated():
    ids = [5, 6]
    for v in range(7, 35):
        ids.extend([2, v])
    ids.append(1)
    space = PermutationSpace(ids)
    assert space.count == math.factorial(29) > 2**63
    ranks = sample_ranks(space, 'huge')
    assert len(set(ranks)) == 32
    assert len({tuple(space.at(r)) for r in ranks}) == 32


def test_internal_eos_rejected_not_repaired():
    with pytest.raises(ValueError, match='boundaries'):
        PermutationSpace([5, 6, 2, 7, 1, 3, 8])
