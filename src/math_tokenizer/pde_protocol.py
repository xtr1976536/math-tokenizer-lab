"""Reproducible full-corpus source/template connected 80/10/10 split."""
from collections import defaultdict
import hashlib
import json


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def split_corpus(rows, seed=20261028):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen = {}
    fields = ('source_group', 'template_sha256', 'canonical_sha256')
    for i, row in enumerate(rows):
        for field in fields:
            if row.get(field):
                key = (field, row[field])
                if key in seen:
                    union(i, seen[key])
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    names = ('train', 'validation', 'test')
    targets = [len(rows)*p for p in (.8, .1, .1)]
    output = {name: [] for name in names}
    counts = [0, 0, 0]
    ordered = sorted(groups.values(), key=lambda rr: (-len(rr), digest(str(seed)+':'+min(r['id'] for r in rr))))
    for group in ordered:
        # Largest absolute deficit keeps large connected groups in training.
        choice = max(range(3), key=lambda i: targets[i]-counts[i])
        output[names[choice]].extend(group)
        counts[choice] += len(group)
    for name in names:
        output[name].sort(key=lambda r: r['id'])
    assert sum(map(len, output.values())) == len(rows)
    assert len({r['id'] for rr in output.values() for r in rr}) == len(rows)
    for field in ('id', *fields):
        sets = [{r[field] for r in output[n] if r.get(field)} for n in names]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    manifest = dict(seed=seed, requested_ratios=[.8, .1, .1], counts=dict(zip(names, counts)),
                    connected_groups=len(groups), largest_connected_group=max(map(len, groups.values())),
                    grouping_fields=list(fields), assignment='descending connected component size; largest absolute deficit; SHA256 tie-break',
                    splits={n: [r['id'] for r in output[n]] for n in names},
                    ordered_corpus_sha256=digest(json.dumps([(r['id'], r['latex']) for r in rows], ensure_ascii=False)))
    return output, manifest
