"""Four pre-test fairness reviews over every frozen model, not just proposals."""
from collections import Counter
import hashlib,json,subprocess,math
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import RELEASE,master_path,load_split
from math_tokenizer.experiment_v6 import entries,pretrained_entries,load_entry
from math_tokenizer.measure_v6 import measure
from math_tokenizer.metrics_v2 import token_bytes,upstream_information
from math_tokenizer.frontend_v6 import reference_lex


def main():
    out=ROOT/'results/v6';protocol=json.loads((out/'evaluation_protocol.json').read_text());seal=json.loads((RELEASE/'seal.json').read_text());manifest=json.loads((RELEASE/'manifest.json').read_text())
    assert protocol['corpus_sha256']==seal['corpus_sha256']==manifest['corpus_sha256']
    splits=json.loads((RELEASE/'splits.json').read_text());test=set(splits['fresh_test']);allentries=[];logs=[]
    from math_tokenizer.quality_v6 import TARGET_CATEGORIES
    for name in ('transfer_train','transfer_dev'):
        assert all(r['source_repo']=='arxiv.org' and not set(r['subject_categories'])&TARGET_CATEGORIES for r in load_split(name))
    # Round 1: identical fit IDs, true source-paper/category exclusion and pre-test freeze.
    for capacity in protocol['capacities']:
        for scenario in protocol['scenarios']:
            folder=out/f'{scenario}_{capacity}';scope=json.loads((folder/'scope.json').read_text());final=json.loads((folder/'final_selection.json').read_text())
            assert not final['test_used'];assert not (set(scope['fit_ids'])|set(scope['reward_ids'])|set(scope['dev_ids']))&test
            assert not set(scope['fit_groups'])&set(scope['reward_groups'])
            assert scope['corpus_sha256']==seal['corpus_sha256']
            for row in json.loads((folder/'baselines.json').read_text()):assert row['fit_id_count']==len(scope['fit_ids']) and not row['upstream_modified']
            es=entries(capacity,scenario)
            for e in es:
                model=load_entry(e)
                if e['kind']=='proposed' and hasattr(model,'fit_ids'):assert set(model.fit_ids)==set(scope['fit_ids'])
                if e['kind']=='proposed' and hasattr(model,'reward_ids'):assert set(model.reward_ids)==set(scope['reward_ids'])
            allentries+=es
    allentries+=pretrained_entries();logs.append({'round':1,'review':'all_frozen_models_fit_reward_dev_and_test_scope','passed':True,'configurations':len(allentries)})
    # Round 2: all reserved slots and actual public encode/decode, no special wrappers.
    probes=['',r'\frac{x_{i}^2}{y}+\alpha=12345',r'\begin{aligned}x&=1\\y&=2\end{aligned}',
            'α+β=λ',r'\unknownCorpusMacro{a+b}=c','x+y=z\n',r'x_{ij}^{n+1}+y=z']
    probe_failures=[]
    for e in allentries:
        model=load_entry(e);actual=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size
        if e['capacity'] is not None:assert actual<=e['capacity']
        for text in probes:
            ids=model.encode(text);payload=b''.join(token_bytes(model,id) for id in ids)
            if payload!=text.encode() or model.decode(ids)!=text:probe_failures.append({'method':e['name'],'input':text,'lossless':False})
        assert all(0<=id<actual for text in probes for id in model.encode(text))
    assert not probe_failures,probe_failures
    logs.append({'round':2,'review':'capacity_all_slots_raw_interface_unicode_unknown_and_multiline','passed':True,'probes_per_model':len(probes)})
    # Round 3: common metric direction/denominator, known failures and official Rényi.
    synthetic={'id':'audit','latex':r'\alpha+x=12345','split_group':'audit:paper','domain':'analysis','paper_domain':'analysis','origin':'human','complexity':{'class':'simple'},'reference':{'spans':[[0,6],[7,8],[9,14]]}}
    from math_tokenizer.baselines import ByteTokenizer
    m,rows=measure(ByteTokenizer(),[synthetic],'byte_audit',repeats=1)
    assert m['command_integrity']==0 and m['number_integrity']==0 and m['roundtrip_rate']==1
    upstream=upstream_information('byte_audit',[ByteTokenizer().encode(synthetic['latex'])],[synthetic['latex']],256)
    for alpha in ('1.0','2.0','2.5','3.0'):
        value=upstream['renyi_efficiency']['per_tokenizer']['byte_audit']['renyi_'+alpha]['overall'];assert abs(value-m['renyi_efficiency'][alpha])<1e-12
    assert protocol['comprehensive_weights'] and abs(sum(protocol['comprehensive_weights'].values())-1)<1e-12
    logs.append({'round':3,'review':'known_bad_segmentations_common_byte_denominators_official_information_metrics','passed':True})
    # Round 4: source/version integrity and independently reloaded model equality.
    for e in allentries:
        one=load_entry(e);two=load_entry(e)
        for text in probes:assert one.encode(text)==two.encode(text)
    for repo,revision in [('vendor/pathpiece','1764d9fc91ade17f65edbfe7a54ad4a2c8117e43'),('vendor/tokeval','7d03a61c2e2b4b557b796519141ccb52d144e10f')]:
        assert subprocess.check_output(['git','-C',str(ROOT/repo),'rev-parse','HEAD'],text=True).strip()==revision
        assert not subprocess.check_output(['git','-C',str(ROOT/repo),'diff','--name-only'],text=True).strip()
    for r in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()):
        if r['status']=='downloaded':assert hashlib.sha256((ROOT/r['local']).read_bytes()).hexdigest()==r['sha256']
    logs.append({'round':4,'review':'all_checkpoint_reload_and_unmodified_pinned_official_source','passed':True})
    report={'rounds':logs,'all_passed':True,'test_scores_read':False,'corpus_sha256':seal['corpus_sha256'],'note':'Post-test coverage, independent full replay, metric reaggregation and isolated timing receive additional audits.'}
    (out/'fairness_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
