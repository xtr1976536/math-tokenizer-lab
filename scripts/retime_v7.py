"""All frozen configurations, same complete inputs, isolated 7-round timing."""
import argparse,json,random,time,subprocess
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v7 import load_model,read_external,evaluation_suites
from math_tokenizer.measure_v6 import clear


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=['v7','v8'],default='v7');a=parser.parse_args()
    processes=subprocess.check_output(['ps','-Ao','command'],text=True)
    assert not any(any(name in line for name in ('scripts/develop_','scripts/source_resplits_v7.py','scripts/train_terminal_full_v8.py','scripts/controlled_terminal_v8.py','scripts/harvest_arxiv')) and 'python' in line for line in processes.splitlines()),'Training/harvest running; timing prohibited'
    folder=ROOT/f'results/{a.release}/full';entries=json.loads((folder/'registry.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    models={e['key']:load_model(e) for e in entries};suites=evaluation_suites(a.release)
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
