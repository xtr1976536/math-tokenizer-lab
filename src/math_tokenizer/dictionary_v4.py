"""AST-v4 dictionary augmentation with held-paper replacement-benefit pruning.

The initial BPE vocabulary is unchanged. Extra payloads come from repeated
fit-only source subtrees. Pruning uses an independent train-internal reward
paper set, preserves byte/covered grammar atoms, and keeps the same hard decoder.
Replacement costs are local isolated-payload proxies, not exact global gains.
"""
from collections import Counter, defaultdict
import time

from .ast import compatible
from .constrained_v4 import ConstrainedTokenizer
from .structure_v3 import structure_fast


def augment_dictionary(base, fit, reward, capacity=1024, extra_subtrees=256, max_bytes=48):
    started = time.perf_counter()
    counts = Counter()
    grammar_units = set()
    for row in fit:
        text = row['latex']
        raw = text.encode()
        structure = structure_fast(text)
        grammar_units.update(t.encode() for a, b, t in structure.lexemes if t.startswith('\\'))
        grammar_units.update(c.encode() for c in text if not c.isascii())
        if structure.kind != 'byte_fallback':
            for a, b in structure.spans:
                if 3 <= b - a <= max_bytes:
                    counts[raw[a:b]] += 1
    present = set(base.vocab)
    added = [t for t, count in sorted(counts.items(), key=lambda x: (-x[1] * (len(x[0]) - 1), x[0]))
             if count >= 2 and t not in present][:extra_subtrees]
    vocabulary = list(base.vocab) + added
    model = ConstrainedTokenizer(vocabulary, True, 'commands', 'shortest')
    # Preserve the original explicit grammar coverage, including complete
    # command/codepoint tokens learned by BPE after the original atom seeding.
    protected = {i for i, token in enumerate(vocabulary) if i < 256 or token in grammar_units}
    if len(protected) > capacity:
        raise ValueError('protected atoms exceed declared vocabulary budget')
    group_bytes = Counter()
    for row in reward:
        group_bytes[row['split_group']] += len(row['latex'].encode())
    usage = defaultdict(float)
    for row in reward:
        weight = 1 / max(1, group_bytes[row['split_group']])
        for id in model.encode(row['latex']):
            usage[id] += weight

    def replacement_count(id):
        token = vocabulary[id]
        text = token.decode('utf-8', errors='strict') if _utf8(token) else None
        # Incomplete UTF8 payloads remain byte-composable. The unconstrained
        # local cost is only a pruning proxy, never a change to actual masks.
        if text is None:
            return len(token)
        raw, spans, safe, interior, allowed, exceptions, structure = model.prepare(text)
        starts, ends, ids = model.matches(text)
        options = defaultdict(list)
        for a, b, other in zip(starts, ends, ids):
            a, b, other = int(a), int(b), int(other)
            if other == id or not allowed[a] or not allowed[b]:
                continue
            if structure.kind != 'byte_fallback' and not compatible(a, b, structure.spans):
                continue
            options[a].append(b)
        best = [len(raw) + 1] * (len(raw) + 1)
        best[-1] = 0
        for a in range(len(raw) - 1, -1, -1):
            if options[a]:
                best[a] = min(1 + best[b] for b in options[a])
        return min(best[0], len(raw))

    scores = {}
    for id in range(256, len(vocabulary)):
        if id not in protected:
            scores[id] = usage[id] * max(0, replacement_count(id) - 1) if usage[id] else 0.
    order = sorted(scores, key=lambda id: (-scores[id], id >= len(base.vocab), id))
    selected = protected | set(order[:capacity - len(protected)])
    kept = [token for id, token in enumerate(vocabulary) if id in selected]
    final = ConstrainedTokenizer(kept, True, 'commands', 'shortest')
    final.capacity = capacity
    final.train_seconds = base.train_seconds + time.perf_counter() - started
    final.training_log = list(base.training_log) + [{
        'stage': 'fit_subtree_augmentation_reward_paper_pruning',
        'scope': 'train_only_paper_disjoint_fit_and_reward',
        'extra_subtrees_requested': extra_subtrees, 'extra_subtrees_proposed': len(added),
        'extra_subtrees_retained': sum(id >= len(base.vocab) for id in selected),
        'protected_grammar_types': len(protected) - 256,
        'proxy': 'reward_document_normalized_usage_times_local_alternative_token_cost_minus_one',
        'fit_ids': [r['id'] for r in fit], 'reward_ids': [r['id'] for r in reward],
    }]
    assert final.vocab[:256] == [bytes([b]) for b in range(256)]
    assert len(final.vocab) <= capacity
    assert {vocabulary[id] for id in protected} <= set(final.vocab)
    return final


def _utf8(payload):
    try:
        payload.decode()
        return True
    except UnicodeDecodeError:
        return False
