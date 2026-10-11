"""Independent audit passes: corpus, evaluator protocol, result recomputation."""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
from .corpus import ROOT,canonical,validate
from .evaluate import read_jsonl,file_hash
from .checkpoints import verify_saved_hash

def audit_corpus():
    rows=read_jsonl(ROOT/'data/curated/corpus.jsonl');summary=json.loads((ROOT/'data/curated/summary.json').read_text())
    assert file_hash(ROOT/'data/curated/corpus.jsonl')==summary['corpus_sha256']
    assert len(rows)==summary['count']
    groups=defaultdict(set);canonical_groups=defaultdict(set)
    for r in rows:
        assert validate(r['latex']) is None
        groups[r['split_group']].add(r['split']);canonical_groups[canonical(r['latex'])].add(r['split'])
    assert all(len(x)==1 for x in groups.values())
    assert all(len(x)==1 for x in canonical_groups.values())
    assert len(canonical_groups)==len(rows)
    assert all(r['origin_evidence'] for r in rows)
    return {'passed':True,'count':len(rows),'groups':len(groups),'checks':['formula_filter_revalidation','canonical_deduplication','document_family_disjointness','origin_evidence_nonempty','corpus_hash']}

def audit_protocol(out):
    manifest=json.loads((out/'manifest.json').read_text());rows=read_jsonl(out/'per_formula.jsonl')
    assert manifest['special_tokens'] is False and manifest['truncation'] is False
    assert manifest['train_sha256']==file_hash(ROOT/'data/curated/splits/train.jsonl')
    assert manifest['test_sha256']==file_hash(ROOT/'data/curated/splits/test.jsonl')
    train=set(manifest['train_ids']);test=set(manifest['test_ids']);assert train.isdisjoint(test)
    by=defaultdict(list)
    for row in rows:by[(row['method'],row['seed'])].append(row)
    assert len(by)>=14
    for key,items in by.items():
        assert len(items)==len(test)
        assert {r['id'] for r in items}==test
        assert all(r['roundtrip'] for r in items),f'roundtrip failure {key}'
    # All sampled formula byte lengths are method-independent.
    lengths=defaultdict(set)
    for row in rows:lengths[row['id']].add(row['bytes'])
    assert all(len(v)==1 for v in lengths.values())
    return {'passed':True,'method_seed_runs':len(by),'test_formula_count':len(test),'checks':['identical_test_ids','identical_raw_byte_lengths','no_truncation_or_specials','train_test_disjoint','all_lossless']}

def audit_results(out):
    manifest=json.loads((out/'manifest.json').read_text());metrics=json.loads((out/'metrics.json').read_text());rows=read_jsonl(out/'per_formula.jsonl')
    for m in metrics:
        rr=[r for r in rows if r['method']==m['method'] and r['seed']==m['seed']]
        assert len(rr)==m['n_formulas']
        assert sum(r['tokens'] for r in rr)==m['tokens']
        assert sum(r['bytes'] for r in rr)==m['bytes']
        assert abs(m['tokens_per_formula']-m['tokens']/len(rr))<1e-12
        assert len(m['encode_seconds_repeats'])==5
        if m['comparison_group']=='trained':
            assert m['capacity_budget']==manifest['capacity']
            assert m['vocab_actual']<=manifest['capacity']
        elif m['comparison_group']=='pretrained_reference':
            assert m['capacity_budget'] is None
    modeldir=out/'models'
    states=Counter(verify_saved_hash(modeldir/filename,digest) for filename,digest in manifest['model_hashes'].items())
    return {'passed':True,'checkpoint_evidence_states':dict(states),'checks':['raw_result_reaggregation','same_trained_capacity_budget','fixed_tokenizers_separate_reference_group','timing_five_repeats','actual_or_relocated_hashes_verified_retired_hashes_provenance_only']}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('result',default='results/pilot01',nargs='?');a=p.parse_args()
    out=ROOT/a.result
    reports={
        'fairness_round_1_corpus':audit_corpus(),
        'fairness_round_2_protocol':audit_protocol(out),
        'fairness_round_3_reaggregation':audit_results(out)}
    import subprocess,sys
    check=subprocess.run([sys.executable,'-m','pytest','tests/test_audit_sensitivity.py','tests/test_evaluate.py','-q'],cwd=ROOT,capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    reports['fairness_round_4_adversarial_protocol']={'passed':True,'checks':['audit_rejects_missing_formula','audit_rejects_changed_raw_bytes','audit_rejects_test_leakage','audit_rejects_mismatched_vocab_or_totals','official_adapters_unicode_and_whitespace'],'test_output':check.stdout}
    (out/'fairness_audit.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
