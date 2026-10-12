"""Append newly registered official tokenizers to frozen v6 full experiments.

Preserve checkpoint selection and data. Retime all configurations together
after this script; newly measured provisional timings must not enter rankings.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import subprocess
import sys

from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.comparison_v6 import calibrate, id_hash
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v6 import load_entry, pretrained_entries
from math_tokenizer.measure_v6 import measure, fit_usage
from math_tokenizer.metrics_v2 import upstream_information


def main():
    folder=ROOT/'results/v6/full'
    registry=json.loads((folder/'registry.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text())
    existing={e['name'] for e in registry}
    new=[e for e in pretrained_entries() if e['name'] not in existing]
    if not new:
        print('No new official configurations');return
    history=folder/'pre_llama3_comparison'
    history.mkdir(exist_ok=True)
    # Compact prior experiment evidence; no repeated large per-formula archive.
    for name in ('general_vs_math_rankings.json','comparison_fairness_audit.json',
                 'comparison_verification.json','timing_protocol.json','fairness_audit.json',
                 'completion.json','comparison_protocol.json'):
        path=history/name
        if not path.exists():path.write_bytes((folder/name).read_bytes())
    timing_path=history/'timing_readouts.json'
    if not timing_path.exists():
        timing_path.write_text(json.dumps([{k:m[k] for k in ('key','suite','encode_mib_s','decode_mib_s','encode_seconds_repeats','decode_seconds_repeats')} for m in metrics],indent=2)+'\n')
    report_path=ROOT/'docs/general_vs_math_v6.md'
    if not (history/'general_vs_math_report.md').exists():
        (history/'general_vs_math_report.md').write_bytes(report_path.read_bytes())
    immutable=[{k:v for k,v in m.items() if k not in {'encode_mib_s','decode_mib_s','encode_seconds_repeats','decode_seconds_repeats'}} for m in metrics]
    (history/'non_timing_metric_digest.json').write_text(json.dumps({'sha256':hashlib.sha256(json.dumps(immutable,sort_keys=True).encode()).hexdigest(),'records':len(metrics)},indent=2)+'\n')
    suites={s:load_split(s) for s in ('fresh_test','extreme_domain_test')}
    suites['stress_test']=list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))
    scopes={s:json.loads((ROOT/f'results/v6/{s}_1024/scope.json').read_text())['fit_ids'] for s in ('broad','specialized')}
    byid={r['id']:r for r in load_split('train')}
    additions=[]
    for entry in new:
        index=len(registry)
        entry['key']=f"{index}:{entry['name']}:None:pretrained:0"
        entry['checkpoint_sha256']=hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()
        model=load_entry(entry)
        usage={s:fit_usage(model,[byid[id] for id in sorted(ids)]) for s,ids in scopes.items()}
        for suite,records in suites.items():
            primary='specialized' if suite=='extreme_domain_test' else 'broad'
            m,rows=measure(model,records,entry['name'],repeats=1,training_usage=usage[primary])
            m.update({k:entry[k] for k in ('key','capacity','scenario','group')})
            m.update(suite=suite,checkpoint_bytes=(ROOT/entry['path']).stat().st_size,fit_calibration_scenario=primary,fit_calibrations={})
            encodings=[model.encode(r['latex']) for r in records]
            for scenario in scopes:
                readout,bits=calibrate(encodings,usage[scenario],m['vocab_actual'],m['bytes'])
                readout['fit_id_sha256']=id_hash(scopes[scenario]);m['fit_calibrations'][scenario]=readout
                for row,bit in zip(rows,bits):row.setdefault('fit_predictive_bits_by_scenario',{})[scenario]=bit
            for row in rows:row.update(key=entry['key'],capacity=None,scenario='pretrained',model_group=entry['group'],suite=suite)
            additions.extend(rows);metrics.append(m)
            if suite!='stress_test':
                dest=folder/'upstream_information'/str(index);dest.mkdir(exist_ok=True)
                upstream=upstream_information(entry['name'],encodings,[r['latex'] for r in records],m['vocab_actual'])
                (dest/(suite+'.json')).write_text(json.dumps(upstream,indent=2)+'\n')
            print(entry['name'],suite,m['formulas'],m['tokens_per_formula'],m['roundtrip_rate'],flush=True)
        registry.append(entry)
    with gzip.open(folder/'per_formula.jsonl.gz','at',encoding='utf-8') as stream:
        for row in additions:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (folder/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    profiles=json.loads((folder/'deployment_profile.json').read_text())
    for index in range(len(registry)-len(new),len(registry)):
        worker=subprocess.run([sys.executable,str(ROOT/'scripts/profile_deployment_v6.py'),'--worker',str(index)],capture_output=True,text=True,check=True)
        profiles.append(json.loads(worker.stdout.splitlines()[-1]))
    (folder/'deployment_profile.json').write_text(json.dumps(profiles,indent=2)+'\n')
    completion=json.loads((folder/'completion.json').read_text())
    completion.update(configurations=len(registry),metric_records=len(metrics),extension_utc=datetime.now(timezone.utc).isoformat(),timings_require_isolated_interleaved_remeasurement=True)
    (folder/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    (folder/'llama3_extension.json').write_text(json.dumps({
        'utc':datetime.now(timezone.utc).isoformat(),'added':[e['name'] for e in new],
        'added_formula_rows':len(additions),'prior_configs':len(registry)-len(new),
        'checkpoint_selection_changed':False,'corpus_changed':False,
        'timing_action':'Rerun all configurations, same 7 interleaved repetitions before ranking.',
        'official_source_evidence':'data/pretrained_registry_v6.json',
        'previous_results_archive':'results/v6/full/pre_llama3_comparison',
    },indent=2)+'\n')


if __name__=='__main__':main()
