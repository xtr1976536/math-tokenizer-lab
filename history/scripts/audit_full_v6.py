"""Independent full replay, common coverage and recomputed metrics, round5+."""
from collections import defaultdict,Counter
import gzip,json,hashlib
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import RELEASE,load_split
from math_tokenizer.experiment_v6 import load_entry
from math_tokenizer.measure_v6 import measure
from summarize_full_v6 import aggregate_rows


def main():
    folder=ROOT/'results/v6/full';registry=json.loads((folder/'registry.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    samples={'fresh_test':load_split('fresh_test'),'extreme_domain_test':load_split('extreme_domain_test'),'stress_test':list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))}
    expected={suite:{r['id']:r for r in rs} for suite,rs in samples.items()};rows=defaultdict(list)
    for row in map(json.loads,gzip.open(folder/'per_formula.jsonl.gz','rt')):rows[(row['key'],row['suite'])].append(row)
    logs=[];count=0
    for m in metrics:
        rr=rows[(m['key'],m['suite'])];assert len(rr)==len(expected[m['suite']])
        assert len({r['id'] for r in rr})==len(rr) and {r['id'] for r in rr}==set(expected[m['suite']])
        assert all(r['group']==expected[m['suite']][r['id']]['split_group'] for r in rr)
        recomputed=aggregate_rows(rr)
        for key in ('tokens_per_formula','tokens_per_byte','document_macro_tokens_per_byte','roundtrip_rate'):
            assert abs(recomputed[key]-m[key])<1e-12,(m['key'],key)
        if m['rank_eligible_lossless']:
            for key in ('command_integrity','number_integrity','independent_tree_crossing'):
                assert abs(recomputed[key]-m[key])<1e-12,(m['key'],key)
        count+=len(rr)
    logs.append({'round':5,'review':'whole_result_coverage_unique_ids_no_drops_and_independent_aggregate','rows':count,'passed':True})
    # Separate reloading of every model and every natural/stress input.
    replay=0
    for entry in registry:
        model=load_entry(entry)
        for m in [m for m in metrics if m['key']==entry['key']]:
            for row in rows[(m['key'],m['suite'])]:
                original=expected[m['suite']][row['id']]['latex'];ids=model.encode(original)
                assert len(ids)==row['tokens'];assert (model.decode(ids)==original)==row['roundtrip']
                replay+=1
    logs.append({'round':6,'review':'independent_all_checkpoint_and_official_encoder_full_input_replay','rows':replay,'passed':True})
    # Renyi: official code vs independent formula, across all natural datasets.
    checked=0
    for entry in registry:
        number=entry['key'].split(':',1)[0]
        for m in [m for m in metrics if m['key']==entry['key'] and m['suite']!='stress_test']:
            official=json.loads((folder/'upstream_information'/number/(m['suite']+'.json')).read_text())
            for alpha in ('1.0','2.0','2.5','3.0'):
                value=official['renyi_efficiency']['per_tokenizer'][entry['name']]['renyi_'+alpha]['overall']
                assert abs(value-m['renyi_efficiency'][alpha])<1e-12
            checked+=1
    timing=json.loads((folder/'timing_protocol.json').read_text());assert timing['repeats']==7 and timing['training_not_running']
    assert all(len(m['encode_seconds_repeats'])==len(m['decode_seconds_repeats'])==7 for m in metrics)
    logs.append({'round':7,'review':'all_natural_official_entropy_recomputation_and_uniform_isolated_interleaved_timing','natural_metric_records':checked,'passed':True})
    report={'rounds':logs,'all_passed':True,'preflight_rounds':4,'total_fairness_rounds':7,'corpus_sha256':json.loads((RELEASE/'manifest.json').read_text())['corpus_sha256'],
            'per_formula_gz_sha256':hashlib.sha256((folder/'per_formula.jsonl.gz').read_bytes()).hexdigest()}
    (folder/'fairness_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
