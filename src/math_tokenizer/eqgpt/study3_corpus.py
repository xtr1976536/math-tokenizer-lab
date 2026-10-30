"""Study 3: bounded, unique multiset permutations and paired GRE/LaTeX.

Only scalar products and sums commute. Division sides never interchange;
named GRE atoms stay atomic. No coefficients, derivatives or terms are added.
"""
from collections import Counter
import hashlib
import json
import math
import random

from .expanded_corpus import gre_terms, gre_signature
from .latex_codec import structure_to_latex

VERSION = 'study3-corpus-effect-v1'
TIMES = 32
LIMIT = 60
SEED = 525
SIZES = (280, 315, 350)
ARMS = ('eqgpt_reference', *(f'ast_bpe_{s}' for s in SIZES))
SETTINGS = dict(seed=SEED, layers=3, epochs=25, batch_size=128, learning_rate=1e-4,
                positions=LIMIT, search_limit=LIMIT, attempts=200, cycles=3,
                continuation_epochs=2, continuation_lr=1e-5, epi=0.2,
                samples_per_source=TIMES, vocabulary_sizes=list(SIZES),
                text='coefficient-free GRE rendered with structure_to_latex',
                augmentation='unique multiset rank sampling; original normalized variant +31; balanced repeat below32',
                budget='equal epochs, not equal updates/tokens/FLOPs',
                tokenizer='each cohort learns its own Hard vocabulary from its unaugmented GRE-rendered LaTeX',
                generation='original EqGPT SAT and Hard unconstrained sampling; legacy search behavior retained')


def multiset_count(items):
    counts = Counter(items)
    return math.factorial(len(items)) // math.prod(math.factorial(c) for c in counts.values())


def unrank(items, rank):
    """Distinct lexicographic multiset permutation, without enumeration."""
    counts = Counter(items)
    ways = multiset_count(items)
    if not 0 <= rank < ways:
        raise ValueError('Permutation rank out of range')
    remaining = len(items)
    result = []
    while remaining:
        for value in sorted(counts):
            if not counts[value]:
                continue
            block = ways * counts[value] // remaining
            if rank < block:
                result.append(value)
                counts[value] -= 1
                ways = block
                remaining -= 1
                break
            rank -= block
    return result


def rank_of(items, ordered):
    if Counter(items) != Counter(ordered):
        raise ValueError('Not a permutation')
    counts = Counter(items)
    ways = multiset_count(items)
    remaining = len(items)
    rank = 0
    for value in ordered:
        rank += sum(ways * counts[v] // remaining for v in counts if v < value)
        ways = ways * counts[value] // remaining
        counts[value] -= 1
        remaining -= 1
    return rank


def split_factors(term):
    numerator, denominator = [], []
    destination = numerator
    for token in term:
        if token == 3:
            destination = numerator
        elif token == 4:
            destination = denominator
        elif token >= 6:
            destination.append(token)
            destination = numerator
        else:
            raise ValueError('Unexpected factor/control token')
    if not numerator:
        raise ValueError('Leading reciprocal not supported')
    return numerator, denominator


class PermutationSpace:
    def __init__(self, ids):
        self.parts = [split_factors(t) for t in gre_terms(ids)]
        self.groups = [(tuple(sorted(n)), tuple(sorted(d))) for n, d in self.parts]
        self.factor_ways = {g: multiset_count(g[0]) * multiset_count(g[1]) for g in self.groups}
        self.inner = math.prod(self.factor_ways[g] for g in self.groups)
        self.count = multiset_count(self.groups) * self.inner
        inner_rank = 0
        for group, (n, d) in zip(self.groups, self.parts):
            local = rank_of(group[0], n) * multiset_count(group[1]) + rank_of(group[1], d)
            inner_rank = inner_rank * self.factor_ways[group] + local
        self.original_rank = rank_of(self.groups, self.groups) * self.inner + inner_rank
        self.signature = gre_signature(ids)

    def at(self, rank):
        outer, inner = divmod(rank, self.inner)
        order = unrank(self.groups, outer)
        local_ranks = []
        for group in reversed(order):
            inner, local = divmod(inner, self.factor_ways[group])
            local_ranks.append(local)
        result = [5]
        for index, (group, local) in enumerate(zip(order, reversed(local_ranks))):
            if index:
                result.append(2)
            n_rank, d_rank = divmod(local, multiset_count(group[1]))
            numerator, denominator = unrank(group[0], n_rank), unrank(group[1], d_rank)
            for j, factor in enumerate(numerator):
                if j:
                    result.append(3)
                result.append(factor)
            for factor in denominator:
                result.extend([4, factor])
        result.append(1)
        if gre_signature(result) != self.signature:
            raise RuntimeError('Permutation changed structure')
        return result


def sample_ranks(space, source_id, seed=SEED, times=TIMES):
    local = hashlib.sha256(f'{seed}:{source_id}'.encode()).digest()
    rng = random.Random(int.from_bytes(local, 'big'))
    if space.count >= times:
        # Floyd sampling works even when factorial counts exceed sys.maxsize.
        chosen = set()
        for j in range(space.count - 1 - (times - 1), space.count - 1):
            candidate = rng.randrange(j + 1)
            chosen.add(j if candidate in chosen else candidate)
        others = [r + (r >= space.original_rank) for r in sorted(chosen)]
        rng.shuffle(others)
        return [space.original_rank, *others]
    ranks = [space.original_rank, *[r for r in range(space.count) if r != space.original_rank]]
    full, extra = divmod(times, space.count)
    result = ranks * full
    remainder = ranks.copy()
    rng.shuffle(remainder)
    return result + remainder[:extra]


def paired_rows(source, seed=SEED):
    space = PermutationSpace(source['gre_ids'])
    for index, rank in enumerate(sample_ranks(space, source['source_id'], seed)):
        ids = space.at(rank)
        yield dict(source_id=source['source_id'], augmentation_index=index,
                   permutation_rank=str(rank), distinct_permutation_space=str(space.count),
                   gre_ids=ids, latex=structure_to_latex(ids))


def rows_digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
    return h.hexdigest()
