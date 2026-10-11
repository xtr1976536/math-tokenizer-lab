"""All frozen models, all heldout formulas, one raw-byte metric protocol."""
import gzip,hashlib,json,platform,sys,time
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import RELEASE,load_split
from math_tokenizer.experiment_v6 import entries,pretrained_entries,load_entry
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.metrics_v2 import upstream_information

OUT=ROOT/'results/v6/full'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'results/v6/evaluation_protocol.json').read_text())
    registry=[]
    for capacity in protocol['capacities']:
        for scenario in protocol['scenarios']:registry.extend(entries(capacity,scenario))
    registry+=pretrained_entries()
    for i,e in enumerate(registry):e['key']=str(i)+':'+e['name']+':'+str(e['capacity'])+':'+e['scenario']+':'+str(e['seed'])
    for e in registry:
        if e.get('path'):e['checkpoint_sha256']=hashlib.sha256((ROOT/e['path']).read_bytes()).hexdigest()
    (OUT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    suites={'fresh_test':load_split('fresh_test'),'extreme_domain_test':load_split('extreme_domain_test')}
    stress_path=ROOT/'results/v6/stress_test.jsonl';suites['stress_test']=list(map(json.loads,stress_path.open()))
    scopes={}
    for capacity in protocol['capacities']:
        for scenario in protocol['scenarios']:
            scope=json.loads((ROOT/f'results/v6/{scenario}_{capacity}/scope.json').read_text());scopes[(capacity,scenario)]=set(scope['fit_ids'])
    allfit=load_split('train');byid={r['id']:r for r in allfit}
    metrics=[];row_path=OUT/'per_formula.jsonl.gz';first=OUT/'first_test_score.json'
    if not first.exists():first.write_text(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'all_models_selected_before_test':True,'registry_sha256':hashlib.sha256((OUT/'registry.json').read_bytes()).hexdigest()},indent=2)+'\n')
    with gzip.open(row_path,'wt',encoding='utf-8') as stream:
        for e in registry:
            started=time.perf_counter();model=load_entry(e);loadseconds=time.perf_counter()-started
            fitting=[byid[id] for id in sorted(scopes[(e['capacity'],e['scenario'])])] if e['scenario']!='pretrained' else [byid[id] for id in sorted(scopes[(1024,'broad')])]
            usage=fit_usage(model,fitting)
            active=['fresh_test','stress_test'] if e['scenario']=='broad' else ['extreme_domain_test','stress_test'] if e['scenario']=='specialized' else list(suites)
            # Both breadth models are evaluated on exactly the OOD subset as
            # well, enabling matched-input specialization/generalization gaps.
            if e['scenario']=='broad':active.insert(1,'extreme_domain_test')
            for suite in active:
                m,rows=measure(model,suites[suite],e['name'],e['seed'],repeats=1,training_usage=usage)
                m.update({k:e[k] for k in ('key','capacity','scenario','group')});m['suite']=suite;m['load_seconds_observed']=loadseconds
                m['checkpoint_bytes']=(ROOT/e['path']).stat().st_size if e.get('path') else None
                for row in rows:
                    row.update(key=e['key'],capacity=e['capacity'],scenario=e['scenario'],model_group=e['group'],suite=suite)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                # Official unmodified metric code is run on every natural suite.
                if suite!='stress_test':
                    encodings=[model.encode(r['latex']) for r in suites[suite]]
                    upstream=upstream_information(e['name'],encodings,[r['latex'] for r in suites[suite]],m['vocab_actual'])
                    destination=OUT/'upstream_information'/str(e['key'].split(':',1)[0]);destination.mkdir(parents=True,exist_ok=True)
                    (destination/(suite+'.json')).write_text(json.dumps(upstream,indent=2)+'\n')
                metrics.append(m);(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
                print(e['key'],suite,m['formulas'],round(m['tokens_per_formula'],4),round(m['command_integrity'] or 0,4),round(m['independent_tree_crossing'],4),flush=True)
            del model
    summary={'configurations':len(registry),'metric_records':len(metrics),'corpus_sha256':protocol['corpus_sha256'],
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hardware':platform.platform(),'python':sys.version,
        'all_raw_inputs':True,'no_per_model_exclusion':True,'timings_require_isolated_interleaved_remeasurement':True}
    (OUT/'completion.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)


if __name__=='__main__':main()
