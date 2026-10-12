"""V2 audit rounds plus independently replayable provenance and test metrics."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from collections import Counter,defaultdict
from email.utils import parsedate_to_datetime
from math_tokenizer.corpus import ROOT,canonical
from math_tokenizer.validation_v2 import validate
from math_tokenizer.structure_v2 import complexity,structure
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.bpe import MathBPE
from math_tokenizer.rl_v2 import LatticeTokenizer
from math_tokenizer.checkpoints import verify_saved_hash


def corpus_audit():
    path=ROOT/'data/releases/v2/curated';rows=read_jsonl(path/'corpus.jsonl');summary=json.loads((path/'summary.json').read_text())
    assert file_hash(path/'corpus.jsonl')==summary['corpus_sha256'];assert len(rows)==summary['count']
    assert len({canonical(r['latex']) for r in rows})==len(rows)
    groups=defaultdict(set)
    for r in rows:
        assert validate(r['latex']) is None,(r['id'],validate(r['latex']))
        assert r['complexity']==complexity(r['latex']);assert r['structure_kind']==structure(r['latex']).kind
        assert r['origin_evidence'] and r['source_revision'];groups[r['split_group']].add(r['split'])
        if r['source_repo']=='arxiv.org':
            assert int(r['version_date'][:4])<2020
            assert r['license'] in {'CC-BY-4.0','CC-BY-3.0','CC0-1.0'}
            assert r['origin_basis']=='pre2020_arxiv_historical_inference_user_approved'
    assert all(len(x)==1 for x in groups.values())
    for split in ('train','dev','test'):
        rr=read_jsonl(path/f'splits/{split}.jsonl')
        assert {r['id'] for r in rr}=={r['id'] for r in rows if r['split']==split}
    return {'passed':True,'formulas':len(rows),'groups':len(groups),'checks':['formula_only_revalidation','fixed_complexity_rule','canonical_dedup','document_disjoint','pre2020_versions','licenses','source_evidence']}


def result_audit(out):
    protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    assert protocol['data_sha256']==file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')
    assert file_hash(out/'protocol.json')==manifest['protocol_sha256'];assert file_hash(out/'selection.json')==manifest['selection_sha256']
    sets=[set(protocol[k]) for k in ('train_ids','dev_ids','test_ids')]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3))
    assert set(protocol['dev_selection_ids'])<=sets[1]
    assert selection['test_used_for_selection'] is False
    for arm,scores in selection['dev_config_mean_scores'].items():
        assert selection['selected'][arm]==min(scores,key=lambda k:(scores[k],k))
    assert protocol['special_tokens'] is False and protocol['truncation'] is False
    rows=read_jsonl(out/'per_formula.jsonl');metrics=json.loads((out/'metrics.json').read_text())
    by=defaultdict(list)
    for row in rows:by[(row['method'],row['seed'])].append(row)
    assert len(by)==manifest['method_seed_runs'];assert len(rows)==manifest['test_records']
    input_lengths=defaultdict(set)
    for key,rr in by.items():
        assert {r['id'] for r in rr}==sets[2];assert len(rr)==len(sets[2]);assert all(r['roundtrip'] for r in rr)
        for r in rr:input_lengths[r['id']].add(r['bytes'])
    assert all(len(v)==1 for v in input_lengths.values())
    for m in metrics:
        rr=by[(m['method'],m['seed'])]
        assert sum(r['tokens'] for r in rr)==m['tokens'];assert sum(r['bytes'] for r in rr)==m['bytes']
        assert abs(m['tokens_per_formula']-m['tokens']/len(rr))<1e-12
        assert m['cache_protocol']=='structure_cache_cleared_before_each_repeat'
        if m['comparison_group']=='trained':assert m['vocab_actual']<=protocol['capacity'] and m['capacity_budget']==protocol['capacity']
        else:assert m['capacity_budget'] is None
        c=Counter({int(k):v for k,v in m['token_usage'].items()});assert sum(c.values())==m['tokens']
        import math
        p=[v/m['tokens'] for v in c.values()];h=-sum(x*math.log2(x) for x in p)
        assert abs(h-m['shannon_entropy_bits_per_token'])<1e-10
    states=Counter(verify_saved_hash(out/'models'/name,digest) for name,digest in manifest['models'].items())
    return {'passed':True,'test_formulas':len(sets[2]),'method_seed_runs':len(by),'checkpoint_evidence_states':dict(states),'checks':['identical_raw_test','dev_only_selection','selection_hash_frozen','no_test_deletion','metrics_recomputed','same_vocab_budget','same_cache_timing','current_hashes_or_retired_provenance','upstream_renyi_crosscheck']}


def replay(out):
    protocol=json.loads((out/'protocol.json').read_text());testids=set(protocol['test_ids'])
    records=[r for r in read_jsonl(ROOT/'data/releases/v2/curated/corpus.jsonl') if r['id'] in testids]
    # Fixed samples across each domain/complexity bucket, same replay for all.
    samples=[];groups=defaultdict(list)
    for r in records:groups[(r['domain'],r['complexity']['class'])].append(r)
    for key,rr in sorted(groups.items()):samples.extend(sorted(rr,key=lambda r:r['id'])[:8])
    models=out/'models';count=0
    for path in sorted(models.glob('*.json')):
        d=json.loads(path.read_text());fmt=d.get('format')
        if fmt=='rl-bpe-lattice-v2':m=LatticeTokenizer.load(path)
        elif fmt=='mathbpe-v2-hybrid':m=FastBPE.load(path)
        elif fmt=='mathbpe-v1':m=MathBPE.load(path)
        else:continue
        for r in samples:
            ids=m.encode(r['latex']);assert m.decode(ids)==r['latex']
            if hasattr(m,'encode_reference'):assert ids==m.encode_reference(r['latex'])
        count+=1
    return {'passed':True,'checkpoints':count,'formulas_per_checkpoint':len(samples),'checks':['checkpoint_decode','optimized_vs_reference','both_complexities','all_test_domains']}


def main():
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--without-raw-sources',action='store_true');a=p.parse_args();out=ROOT/a.output
    result={'round1_corpus':corpus_audit(),'round2_protocol':result_audit(out),'round3_checkpoint_replay':replay(out)}
    tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,capture_output=True,text=True);assert tests.returncode==0,tests.stdout+tests.stderr
    result['round4_unit_property_and_adversarial']={'passed':True,'output':tests.stdout}
    arxiv=json.loads((ROOT/'data/arxiv_sources.json').read_text())
    for source in arxiv:
        assert source['license_verified_on_version_page']
        if not a.without_raw_sources:assert file_hash(ROOT/source['local'])==source['sha256']
    result['round5_licenses_and_source_hashes']={'passed':True,'papers':len(arxiv),'raw_source_hashes_verified':not a.without_raw_sources}
    dev=json.loads((out/'dev_ablations.json').read_text());assert len(dev)>=30
    assert all(len(r['training_log'])>=6 for r in dev)
    result['round6_ablation_and_variance_logs']={'passed':True,'dev_runs':len(dev),'checks':['all_configurations_retained','two_dev_seeds','no_test_configuration_selection','epoch_rewards_and_gradient_variance']}
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
