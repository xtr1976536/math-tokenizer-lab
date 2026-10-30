"""Independent, read-only verification. Does not train/search any model."""
import argparse
from collections import Counter
import json
import hashlib

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import sha, gre_signature
from math_tokenizer.eqgpt.evaluation import classify
from math_tokenizer.eqgpt.latex_codec import structure_to_latex, latex_to_structure, LatexParser, key
from math_tokenizer.eqgpt.study3_corpus import SETTINGS, LIMIT, SIZES, paired_rows, rows_digest
from math_tokenizer.eqgpt.study3_protocol import DATA, RUN, REPORT, adapters, read_rows, verify_prepared, code_digest
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES


def data_audit():
    verify_prepared()
    by_cohort = {}
    all_rows = {}
    tokenizer_hashes = {}
    for cohort in ('small', 'large'):
        folder = DATA/cohort
        meta = json.loads((folder/'manifest.json').read_text())
        sources = [json.loads(s) for s in (folder/'sources.jsonl').read_text().splitlines()]
        lookup = {s['source_id']: s for s in sources}
        assert len(lookup) == len(sources) == meta['sources']
        rows = read_rows(cohort)
        assert len(rows) == len(sources)*32 == meta['samples']
        assert rows_digest(rows) == meta['paired_rows_sha256']
        assert Counter(r['source_id'] for r in rows) == Counter({s:32 for s in lookup})
        grouped = {}
        for row in rows:
            grouped.setdefault(row['source_id'], []).append(row)
        for variants in grouped.values():
            counts = Counter(tuple(v['gre_ids']) for v in variants)
            assert len(counts) == min(32, int(variants[0]['distinct_permutation_space']))
            assert max(counts.values()) - min(counts.values()) <= 1
        regenerated = [r for source in sources for r in paired_rows(source)]
        assert rows == regenerated
        aa = adapters(cohort)
        lengths = {arm: [] for arm in aa}
        for source in sources:
            assert not any(classify(source['gre_ids'], case)['status']=='correct' for case in SHARED8_CASES)
        # Every row is examined. No random sampling of encoding/round-trip checks.
        for row in rows:
            gre = row['gre_ids']
            assert gre[0]==5 and gre[-1]==1 and gre.count(5)==gre.count(1)==1
            assert gre_signature(gre) == gre_signature(lookup[row['source_id']]['gre_ids'])
            assert row['latex'] == structure_to_latex(gre)
            for name, adapter in aa.items():
                seq = adapter.sequence(gre if name=='eqgpt_reference' else row['latex'])
                assert len(seq)<=LIMIT
                assert seq[0]==adapter.bos and seq[-1]==adapter.eos
                assert seq.count(adapter.bos)==seq.count(adapter.eos)==1
                if name != 'eqgpt_reference':
                    assert adapter.decode(seq[1:-1]) == row['latex']
                lengths[name].append(len(seq))
        for text in {r['latex'] for r in rows}:
            assert key(LatexParser(text).parse()) == key(LatexParser(structure_to_latex(latex_to_structure(text))).parse())
        for name, values in lengths.items():
            assert max(values)==meta['lengths'][name]['max']
        all_rows[cohort] = rows
        tok = folder/meta['tokenizer_folder']
        training_texts = [structure_to_latex(source['gre_ids']) for source in sources]
        training_digest = hashlib.sha256(json.dumps(training_texts, ensure_ascii=False).encode()).hexdigest()
        assert json.loads((tok/'manifest.json').read_text())['corpus_sha256'] == training_digest
        tokenizer_hashes[cohort] = {p.name:sha(p) for p in tok.glob('*.json')}
        by_cohort[cohort] = dict(sources=len(sources), samples=len(rows),
                               maxima={k:max(v) for k,v in lengths.items()},
                               source_kinds=dict(Counter(s['source_kind'] for s in sources)))
    small_ids = {r['source_id'] for r in all_rows['small']}
    assert len(small_ids)==212
    assert [r for r in all_rows['large'] if r['source_id'] in small_ids] == all_rows['small']
    for size in SIZES:
        # Check the learned payload, not only corpus fingerprints in manifest.
        assert tokenizer_hashes['small'][f'ast_bpe_{size}.model.backend.json'] != tokenizer_hashes['large'][f'ast_bpe_{size}.model.backend.json']
    return dict(status='PASS', cohorts=by_cohort, full_row_checks=True,
                exact_small_augmentation_subset=True, separate_unaugmented_tokenizer_corpora_verified=True,
                target_exclusion_scope='upstream audited external exact/high-similarity exclusions + current exact structure recheck')


def safety_audit():
    before = json.loads((REPORT/'historical_files_before.json').read_text())
    changed = [s for s,h in before.items() if not (ROOT/s).is_file() or sha(ROOT/s)!=h]
    assert not changed, changed
    # No actual experiment output should exist before user review.
    assert not RUN.exists(), 'Formal run directory exists before review'
    caches = {}
    for case in SHARED8_CASES:
        folder = ROOT/'data/mlp_cache'/case
        quality = json.loads((folder/'surrogate_quality.json').read_text())
        assert sha(folder/'surrogate_cache.pt')==quality['cache_sha256']
        caches[case]=quality['cache_sha256']
    return dict(status='PASS', historical_tracked_files_unchanged=len(before),
                mlp_caches=caches, formal_run_directory_absent=True,
                code_sha256=code_digest(), corpus_manifest_sha256=sha(DATA/'manifest.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('data','safety'), required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    result=data_audit() if args.mode=='data' else safety_audit()
    from pathlib import Path
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
