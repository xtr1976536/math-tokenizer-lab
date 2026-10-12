"""All frozen configurations, same complete inputs, isolated 7-round timing."""
import json,random,time,subprocess
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v7 import load_model,read_external
from math_tokenizer.measure_v6 import clear


def main():
    processes=subprocess.check_output(['ps','-Ao','command'],text=True)
    assert not any(('scripts/develop_' in line or 'scripts/source_resplits_v7.py' in line) and 'python' in line for line in processes.splitlines()),'Training running; timing prohibited'
    folder=ROOT/'results/v7/full';entries=json.loads((folder/'registry.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    models={e['key']:load_model(e) for e in entries};external=read_external();extreme=set(json.loads((ROOT/'data/releases/v7/splits.json').read_text())['extreme_domain_test'])
    suites={'external_test':external,'extreme_domain_test':[r for r in external if r['id'] in extreme],
            'stress_test':list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))}
    pairs=[(m['key'],m['suite']) for m in metrics];enc={};times={p:[] for p in pairs};dec={p:[] for p in pairs};order=[];rng=random.Random(971)
    for repeat in range(7):
        current=pairs.copy();rng.shuffle(current);order.append(current)
        for key,suite in current:
            model=models[key];records=suites[suite];clear(model);start=time.perf_counter()
            ids=[model.encode(r['latex']) for r in records];times[(key,suite)].append(time.perf_counter()-start)
            if (key,suite) in enc:assert enc[(key,suite)]==ids
            else:enc[(key,suite)]=ids
            start=time.perf_counter();decoded=[model.decode(i) for i in ids];dec[(key,suite)].append(time.perf_counter()-start)
            assert all(s==r['latex'] for s,r in zip(decoded,records))
        print('final interleaved round',repeat+1,flush=True)
    for m in metrics:
        pair=m['key'],m['suite'];m.update(encode_seconds_repeats=times[pair],decode_seconds_repeats=dec[pair],provisional_timing=False)
        m['encode_mib_s']=m['bytes']/1024**2/float(np.median(times[pair]));m['decode_mib_s']=m['bytes']/1024**2/float(np.median(dec[pair]))
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (folder/'timing_protocol.json').write_text(json.dumps({'repeats':7,'training_not_running':True,'all_inputs':True,
        'order':order,'interfaces':'Full public raw string encode/decode, cache clear; all frontends included','repetition_statistic':'median throughput'},indent=2)+'\n')
    completion=json.loads((folder/'completion.json').read_text())
    completion.update(uniform_retiming_required=False,uniform_retiming_completed=True)
    (folder/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')


if __name__=='__main__':main()
