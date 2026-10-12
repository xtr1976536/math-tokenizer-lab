"""Frozen external formal evaluation: one raw-input/metric/calibration protocol."""
from datetime import datetime,timezone
import gzip,hashlib,json,time
from collections import defaultdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.experiment_v7 import load_model,read_external,scope
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.independent_v7 import additional,source_scopes
from math_tokenizer.comparison_v6 import calibrate,id_hash
from math_tokenizer.metrics_v2 import upstream_information


def main():
    folder=ROOT/'results/v7/full';protocol=json.loads((folder/'evaluation_protocol.json').read_text())
    assert hashlib.sha256((folder/'registry.json').read_bytes()).hexdigest()==protocol['registry_sha256']
    assert json.loads((folder/'preflight_audit.json').read_text())['all_passed']
    entries=json.loads((folder/'registry.json').read_text());external=read_external()
    split=json.loads((ROOT/'data/releases/v7/splits.json').read_text());extreme=set(split['extreme_domain_test'])
    suites={'external_test':external,'extreme_domain_test':[r for r in external if r['id'] in extreme],
            'stress_test':list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))}
    cache={name:{r['id']:source_scopes(r['latex']) for r in rows} for name,rows in suites.items()}
    fit_byid={r['id']:r for r in load_split('train')};fit_ids={s:scope(s)['fit_ids'] for s in ('broad','specialized')}
    manifest=json.loads((ROOT/'data/releases/v7/manifest.json').read_text());metrics=[];first=folder/'first_test_score.json'
    if not first.exists():first.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'registry_sha256':protocol['registry_sha256'],'models_and_primary_comparisons_frozen_before_external_scores':True},indent=2)+'\n')
    with gzip.open(folder/'per_formula.jsonl.gz','wt',encoding='utf-8') as stream:
        for entry in entries:
            start=time.perf_counter();model=load_model(entry);load_time=time.perf_counter()-start
            scenarios=('broad','specialized') if entry['scenario']=='pretrained' else (entry['scenario'],)
            usages={s:fit_usage(model,[fit_byid[id] for id in fit_ids[s]]) for s in scenarios}
            active=list(suites) if entry['scenario']!='specialized' else ['extreme_domain_test','stress_test']
            for suite in active:
                calibration=('specialized' if suite=='extreme_domain_test' else 'broad') if entry['scenario']=='pretrained' else entry['scenario']
                m,rows=measure(model,suites[suite],entry['name'],entry['seed'],repeats=1,training_usage=usages[calibration])
                extra,scope_rows=additional(model,suites[suite],usages[calibration],cache[suite],probe_limit=protocol['edit_probe_limit'])
                scoped={r['id']:r for r in scope_rows}
                m.update(extra);m.update({k:entry[k] for k in ('key','capacity','scenario','group')})
                m.update(suite=suite,fit_calibration_scenario=calibration,fit_id_sha256=id_hash(fit_ids[calibration]),
                    load_seconds=load_time,checkpoint_bytes=(ROOT/entry['path']).stat().st_size if entry.get('path') else None,
                    provisional_timing=True,information_scope=entry['information'])
                if entry['scenario']=='pretrained':
                    encodings=[model.encode(r['latex']) for r in suites[suite]];m['fit_calibrations']={}
                    m['additional_by_fit_scope']={}
                    for s in scenarios:
                        values,bits=calibrate(encodings,usages[s],m['vocab_actual'],m['bytes']);values['fit_id_sha256']=id_hash(fit_ids[s])
                        m['fit_calibrations'][s]=values
                        for row,bit in zip(rows,bits):row.setdefault('fit_predictive_bits_by_scenario',{})[s]=bit
                        # Only fit-dependent readouts vary; shared boundaries/edit probes unchanged.
                        if s==calibration:extra_for_s=extra
                        else:extra_for_s,_=additional(model,suites[suite],usages[s],cache[suite],probe_limit=0)
                        m['additional_by_fit_scope'][s]={k:extra_for_s[k] for k in ('unigram_bpb_smoothing_sensitivity',)}
                for row in rows:
                    row.update(key=entry['key'],capacity=entry['capacity'],scenario=entry['scenario'],model_group=entry['group'],suite=suite)
                    row.update({k:v for k,v in scoped.get(row['id'],{}).items() if k!='id'})
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                if suite!='stress_test':
                    encodings=[model.encode(r['latex']) for r in suites[suite]]
                    upstream=upstream_information(entry['name'],encodings,[r['latex'] for r in suites[suite]],m['vocab_actual'])
                    dest=folder/'upstream_information'/entry['key'].split(':',1)[0];dest.mkdir(parents=True,exist_ok=True)
                    (dest/(suite+'.json')).write_text(json.dumps(upstream,indent=2)+'\n')
                metrics.append(m);(folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
                print(entry['key'],suite,len(rows),round(m['document_macro_tokens_per_byte'],5),round(m['independent_tree_crossing'],4),flush=True)
    (folder/'completion.json').write_text(json.dumps({'configurations':len(entries),'metric_records':len(metrics),
        'formula_rows':sum(m['formulas'] for m in metrics),'all_raw_inputs':True,'per_model_exclusions':False,
        'archive_sha256':manifest['archive_sha256'],'uniform_retiming_required':True,
        'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')


if __name__=='__main__':main()
