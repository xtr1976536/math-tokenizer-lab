"""Post-score coverage, independent full encoder replay and original metrics."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import bisect,gzip,hashlib,json,math
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v7 import load_model,read_external
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.independent_v7 import source_scopes
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.frontend_v6 import reference_lex,tree_ranges
from math_tokenizer.measure_v5 import cross_count
from summarize_full_v6 import aggregate_rows


def main():
    folder=ROOT/'results/v7/full';metrics=json.loads((folder/'metrics.json').read_text());entries=json.loads((folder/'registry.json').read_text())
    protocol=json.loads((folder/'evaluation_protocol.json').read_text());external=read_external()
    extreme=set(json.loads((ROOT/'data/releases/v7/splits.json').read_text())['extreme_domain_test'])
    suites={'external_test':external,'extreme_domain_test':[r for r in external if r['id'] in extreme],
            'stress_test':list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))}
    expected={s:{r['id']:r for r in rs} for s,rs in suites.items()};scope_cache={s:{r['id']:source_scopes(r['latex']) for r in rs} for s,rs in suites.items()}
    rows=defaultdict(list)
    for r in map(json.loads,gzip.open(folder/'per_formula.jsonl.gz','rt')):rows[(r['key'],r['suite'])].append(r)
    logs=[];count=0
    for m in metrics:
        rr=rows[(m['key'],m['suite'])];want=expected[m['suite']]
        assert len(rr)==len(want)==len({r['id'] for r in rr}) and {r['id'] for r in rr}==set(want)
        assert all(r['group']==want[r['id']]['split_group'] and r['bytes']==len(want[r['id']]['latex'].encode()) for r in rr)
        assert m['test_id_sha256']==hashlib.sha256('\n'.join(sorted(want)).encode()).hexdigest()
        aggregate=aggregate_rows(rr)
        for key,value in aggregate.items():
            if key in m and value is not None:assert abs(value-m[key])<1e-10,(m['key'],key)
        assert abs(sum(r['predictive_bits'] for r in rr)/m['bytes']-m['fit_predictive_unigram_bits_per_byte'])<1e-10
        count+=len(rr)
    logs.append({'round':5,'review':'Whole result IDs/raw bytes/source groups and independent paper/micro aggregates, no exclusions or duplicates','passed':True,'rows':count})
    replay=0;reference_agreement=0
    for entry in entries:
        if entry.get('checkpoint_sha256'):assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['checkpoint_sha256']
        model=load_model(entry)
        for m in [v for v in metrics if v['key']==entry['key']]:
            tp=total=parsed=0
            for r in rows[(m['key'],m['suite'])]:
                record=expected[m['suite']][r['id']];text=record['latex'];raw=text.encode();ids=model.encode(text)
                assert len(ids)==r['tokens'] and model.decode(ids)==text
                payload=[token_bytes(model,id) for id in ids];assert b''.join(payload)==raw
                bounds=[0]
                for piece in payload:bounds.append(bounds[-1]+len(piece))
                assert cross_count(bounds,record['reference']['spans'])==r['crossing_tokens']
                commands=good=0
                for a,b in reference_lex(text):
                    if raw[a:b].startswith(b'\\'):
                        commands+=1;good+=not any(a<p<b for p in bounds)
                assert (commands,good)==(r['commands'],r['commands_intact'])
                spans=scope_cache[m['suite']][r['id']]
                if spans is not None:
                    parsed+=1;gold={p for a,b in spans for p in (a,b)}-{0,len(raw)}
                    assert len(gold)==r['scope_boundaries'] and len(gold&set(bounds))==r['scope_boundaries_preserved']
                    total+=len(gold);tp+=len(gold&set(bounds));reference_agreement+=1
                replay+=1
            assert parsed/len(expected[m['suite']])==m['independent_scope_parser_coverage']
            if total:assert abs(tp/total-m['independent_scope_boundary_recall'])<1e-12
        print('full replay',entry['key'],flush=True)
    logs.append({'round':6,'review':'Every reloaded public encoder, raw payload/decode, independent lexical cuts/KaTeX crossings and pylatexenc scope checks','passed':True,'rows':replay,'independent_parser_rows':reference_agreement})
    official_checked=0
    for entry in entries:
        for m in [v for v in metrics if v['key']==entry['key'] and v['suite']!='stress_test']:
            official=json.loads((folder/'upstream_information'/entry['key'].split(':',1)[0]/(m['suite']+'.json')).read_text())
            for alpha in ('1.0','2.0','2.5','3.0'):
                value=official['renyi_efficiency']['per_tokenizer'][entry['name']]['renyi_'+alpha]['overall']
                assert abs(value-m['renyi_efficiency'][alpha])<1e-12
            official_checked+=1
    timings=json.loads((folder/'timing_protocol.json').read_text())
    assert timings['repeats']==7 and timings['training_not_running']
    assert all(len(m['encode_seconds_repeats'])==len(m['decode_seconds_repeats'])==7 and not m['provisional_timing'] for m in metrics)
    logs.append({'round':7,'review':'Original author information metrics equal independent formulas, uniform seven-round full-interface timings','passed':True,'natural_records':official_checked})
    same_dict=[]
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            chosen=[m for m in metrics if m['capacity']==capacity and m['scenario']==scenario]
            def find(name,suite):return next(m for m in chosen if m['method']==name and m['suite']==suite)
            for suite in sorted({m['suite'] for m in chosen}):
                ast=find('ast_lexicographic',suite);short=find('lexical_shortest',suite);bpe=find('official_atomic_bpe',suite)
                a={r['id']:r for r in rows[(ast['key'],suite)]};s={r['id']:r for r in rows[(short['key'],suite)]}
                # Mathematical primary objective guarantee only for same graph.
                assert all(a[id]['tokens']==s[id]['tokens'] for id in a)
                same_dict.append({'capacity':capacity,'scenario':scenario,'suite':suite,'all_formula_lengths_equal_to_same_mask_shortest':True,
                    'independent_crossing_difference':ast['independent_tree_crossing']-short['independent_tree_crossing'],
                    'compression_difference_from_official_BPE':ast['document_macro_tokens_per_byte']-bpe['document_macro_tokens_per_byte']})
    logs.append({'round':8,'review':'Every lexicographic formula equals the same-mask shortest primary length; structure external differences reported, no semantic/global-vocab optimum claim','passed':True,'same_dictionary_checks':same_dict})
    report={'rounds':logs,'preflight_rounds':4,'total_rounds':8,'all_passed':True,'corpus_archive_sha256':protocol['external_archive_sha256'],
        'registry_sha256':protocol['registry_sha256'],'per_formula_sha256':hashlib.sha256((folder/'per_formula.jsonl.gz').read_bytes()).hexdigest(),
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    (folder/'fairness_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('Eight reviews passed',count,'rows',flush=True)


if __name__=='__main__':main()
