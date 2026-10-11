"""Interleaved isolated raw-string public encode and decode, all formulas."""
import json,time,random,subprocess
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.experiment_v6 import load_entry
from math_tokenizer.measure_v6 import clear

OUT=ROOT/'results/v6/full'


def main():
    process=subprocess.check_output(['ps','-Ao','command'],text=True)
    assert not any('python scripts/train_' in line for line in process.splitlines()),'training still running; timing prohibited'
    entries=json.loads((OUT/'registry.json').read_text());metrics=json.loads((OUT/'metrics.json').read_text());models={e['key']:load_entry(e) for e in entries}
    suites={name:load_split(name) for name in ('fresh_test','extreme_domain_test')};suites['stress_test']=list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))
    pairs=[(m['key'],m['suite']) for m in metrics];enc={};times={p:[] for p in pairs};decodes={p:[] for p in pairs};order=[]
    rng=random.Random(719)
    for repeat in range(7):
        current=pairs.copy();rng.shuffle(current);order.append(current)
        for key,suite in current:
            model=models[key];rows=suites[suite];clear(model)
            start=time.perf_counter();ids=[model.encode(r['latex']) for r in rows];seconds=time.perf_counter()-start
            if (key,suite) in enc:assert enc[(key,suite)]==ids
            else:enc[(key,suite)]=ids
            times[(key,suite)].append(seconds)
            start=time.perf_counter();strings=[model.decode(x) for x in ids];decodes[(key,suite)].append(time.perf_counter()-start)
            assert len(strings)==len(rows)
        print('interleaved timing repeat',repeat+1,flush=True)
    for m in metrics:
        pair=(m['key'],m['suite']);m['encode_seconds_repeats']=times[pair];m['decode_seconds_repeats']=decodes[pair]
        m['encode_mib_s']=m['bytes']/1024**2/float(np.median(times[pair]));m['decode_mib_s']=m['bytes']/1024**2/float(np.median(decodes[pair]));m['timing_protocol']='7 interleaved repetitions, all suite inputs, full frontend, cache cleared, no training'
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(OUT/'timing_protocol.json').write_text(json.dumps({'order':order,'repeats':7,'training_not_running':True,'full_suite':True},indent=2)+'\n')


if __name__=='__main__':main()
