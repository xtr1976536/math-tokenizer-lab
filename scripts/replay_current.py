"""Replay both active families against saved raw rows and serialized exports."""
import argparse
import json
from collections import defaultdict

from tokenizers import Tokenizer

from math_tokenizer.corpus import ROOT
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.evaluate import file_hash, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    out = ROOT / args.output
    manifest = json.loads((out / 'manifest.json').read_text())
    assert file_hash(out / 'protocol.json') == manifest['protocol_sha256']
    for name, expected in manifest['checkpoints'].items():
        assert file_hash(out / 'checkpoints' / name) == expected
    by = defaultdict(list)
    for row in read_jsonl(out / 'per_formula.jsonl'):
        by[(row['scenario'], row['method'], row['suite'])].append(row)
    results = []
    for (scenario, family, suite), rows in by.items():
        filename = f'{scenario}-ast-v4.json' if family == 'ast_v4' else f'{scenario}-v5.json'
        path = out / 'checkpoints' / filename
        model = load_checkpoint(path)
        records = {r['id']: r for r in read_jsonl(ROOT / f'data/releases/v5/curated/splits/{suite}.jsonl')}
        official = Tokenizer.from_file(str(path.with_suffix('.tokenizer.json'))) if family == 'v5' else None
        for row in rows:
            text = records[row['id']]['latex']
            ids = model.encode(text)
            assert len(ids) == row['tokens']
            assert model.decode(ids) == text
            if official is not None:assert ids == official.encode(text, add_special_tokens=False).ids
        results.append({'scenario': scenario, 'family': family, 'suite': suite, 'formulas': len(rows), 'same_counts_losslessness_and_official_export': True})
    result = {'replayed': results, 'all_equal': True}
    (out / 'checkpoint_replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print('current-only replay verified', len(results), sum(r['formulas'] for r in results))


if __name__ == '__main__':main()
