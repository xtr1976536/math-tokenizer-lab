"""Prepare only: tokenizers and paired data, NEVER Transformer training/search."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import source_records, sha
from math_tokenizer.eqgpt.eqgpt_fast_config import ALPHABET
from math_tokenizer.eqgpt.eqgpt_fast_tokenizers import train_or_load
from math_tokenizer.eqgpt.evaluation import classify
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from math_tokenizer.eqgpt.study3_corpus import VERSION, SETTINGS, SIZES, LIMIT, paired_rows, rows_digest

DEFAULT = ROOT / 'data/formulas/study3_corpus_effect_v1'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare_cohort(sources, folder, mandatory):
    folder.mkdir()
    keep = list(sources)
    excluded = []
    for iteration in range(len(sources) + 1):
        rows = [v for source in keep for v in paired_rows(source)]
        tok_dir = folder / f'tokenizers_round_{iteration}'
        texts = [structure_to_latex(source['gre_ids']) for source in keep]
        models = train_or_load(texts, tok_dir, SIZES, ALPHABET)
        lengths = {name: [] for name in ('eqgpt_reference', *(f'ast_bpe_{s}' for s in SIZES))}
        bad = {}
        for row in rows:
            ll = dict(eqgpt_reference=len(row['gre_ids']))
            for (_, size), model in models.items():
                ids = model.encode(row['latex'])
                if model.decode(ids) != row['latex']:
                    raise ValueError('Lossy encoding')
                ll[f'ast_bpe_{size}'] = len(ids) + 2
            for name, length in ll.items():
                lengths[name].append(length)
                if length > LIMIT:
                    bad.setdefault(row['source_id'], {})[name] = max(length, bad.get(row['source_id'], {}).get(name, 0))
        print('CLOSURE', folder.name, iteration, len(keep), 'excluded', len(bad), flush=True)
        if mandatory & bad.keys():
            raise ValueError('Would remove an original small-cohort source: ' + repr(mandatory & bad.keys()))
        if not bad:
            break
        excluded.extend(dict(source_id=s, reason='Hard common60 closure', lengths=v, round=iteration) for s, v in bad.items())
        keep = [s for s in keep if s['source_id'] not in bad]
        if not keep:
            raise ValueError('Empty common corpus')
    else:
        raise RuntimeError('Closure did not stabilize')
    with (folder / 'sources.jsonl').open('w') as f:
        for source in keep:
            f.write(json.dumps(source, ensure_ascii=False) + '\n')
    with gzip.open(folder / 'paired.jsonl.gz', 'wt') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    dump(folder / 'excluded.json', excluded)
    manifest = dict(sources=len(keep), samples=len(rows), source_kinds=dict(Counter(s['source_kind'] for s in keep)),
                    paired_rows_sha256=rows_digest(rows), tokenizer_folder=tok_dir.name,
                    lengths={k: dict(min=min(v), max=max(v), mean=sum(v)/len(v)) for k, v in lengths.items()},
                    unique_structures_note='Source records, not a claim of independent structures',
                    exclusions=len(excluded))
    dump(folder / 'manifest.json', manifest)
    return keep, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    # New directory only; historical studies are never overwritten.
    if args.output.resolve() != DEFAULT.resolve():
        raise ValueError('Use the dedicated study3 corpus path')
    args.output.mkdir(exist_ok=False)
    all_sources = source_records()
    small = [s for s in all_sources if s['source_kind'] == 'eqgpt_handbook_non_target']
    if len(small) != 212:
        raise ValueError('Small source pool must contain all 212 historical rows')
    small_ids = {s['source_id'] for s in small}
    eligible, excluded = [], []
    for source in all_sources:
        if any(classify(source['gre_ids'], case)['status'] == 'correct' for case in SHARED8_CASES):
            raise ValueError('Target leakage: ' + source['source_id'])
        if len(source['gre_ids']) > LIMIT:
            excluded.append(dict(source_id=source['source_id'], reason='GRE exceeds60', length=len(source['gre_ids'])))
        else:
            eligible.append(source)
    if small_ids - {s['source_id'] for s in eligible}:
        raise ValueError('Original212 exceed common GRE capacity')
    _, small_rows = prepare_cohort(small, args.output / 'small', small_ids)
    large_sources = small + [s for s in eligible if s['source_id'] not in small_ids]
    _, large_rows = prepare_cohort(large_sources, args.output / 'large', small_ids)
    shared = [r for r in large_rows if r['source_id'] in small_ids]
    if shared != small_rows:
        raise ValueError('Small augmentations not exactly nested in large cohort')
    dump(args.output / 'excluded_gre60.json', excluded)
    input_names = ['data/formulas/gre_benchmark_filtered/benchmark.jsonl',
                   'data/formulas/gre_benchmark_audit_v2/eqgpt_221_after_target_exclusion.jsonl']
    files = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    dump(args.output / 'manifest.json', dict(version=VERSION, settings=SETTINGS, files=files,
         inputs={s: sha(ROOT / s) for s in input_names}, small_rows_exact_subset=True,
         status='PREPARED_NOT_AUTHORIZED_TO_RUN', formal_training_started=False))
    print('PREPARED ONLY. No Transformer training or PDE search started.', flush=True)


if __name__ == '__main__':
    main()
