"""Five distinct engineering checks, with machine-readable evidence."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from math_tokenizer.corpus import ROOT,build
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.bpe import MathBPE
from math_tokenizer.audit import audit_corpus,audit_protocol,audit_results

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',default='results/pilot01');a=p.parse_args();out=ROOT/a.result
    evidence={}
    # 1: Corpus snapshot/provenance, including rejected entries.
    evidence['round_1_corpus_and_provenance']=audit_corpus()
    registry=json.loads((ROOT/'data/source_registry.json').read_text())
    for entry in registry:assert file_hash(ROOT/entry['local'])==entry['sha256']
    evidence['round_1_corpus_and_provenance']['source_checksums_verified']=len(registry)
    # 2: unit/property/differential tests including unsupported and UTF8 cases.
    result=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,capture_output=True,text=True)
    evidence['round_2_algorithm_and_unicode']={'passed':result.returncode==0,'output':result.stdout+result.stderr}
    assert result.returncode==0
    # 3: All official baseline adapters + three methods at smoke scale.
    evidence['round_3_smoke_all_adapters']=audit_protocol(ROOT/'results/smoke01')
    # 4: full pilot, all seeds, independent metric reaggregation and hashes.
    evidence['round_4_full_pilot_three_seeds']={'protocol':audit_protocol(out),'results':audit_results(out)}
    # 5: saved checkpoint replay, optimized/reference encode differential,
    # corpus rebuild determinism, plus actual timings on common test corpus.
    test=read_jsonl(ROOT/'data/curated/splits/test.jsonl');texts=[r['latex'] for r in test]
    timings={};replayed=0
    for modelpath in sorted((out/'models').glob('*.json')):
        if modelpath.name=='hf-bpe.json':continue
        model=MathBPE.load(modelpath)
        fast=[model.encode(t) for t in texts];slow=[model.encode_reference(t) for t in texts]
        assert fast==slow
        assert all(model.decode(ids)==t for t,ids in zip(texts,fast))
        replayed+=1
        speed={}
        for name,fn in [('reference',model.encode_reference),('optimized',model.encode)]:
            times=[]
            for _ in range(5):
                st=time.perf_counter();[fn(t) for t in texts];times.append(time.perf_counter()-st)
            speed[name+'_seconds_median']=float(np.median(times))
        speed['speedup']=speed['reference_seconds_median']/speed['optimized_seconds_median']
        timings[modelpath.name]=speed
    before=file_hash(ROOT/'data/curated/corpus.jsonl');build();after=file_hash(ROOT/'data/curated/corpus.jsonl')
    assert before==after
    evidence['round_5_replay_optimization_and_rebuild']={'passed':True,'checkpoints_replayed':replayed,'same_formula_ids':len(texts),'rebuild_sha256':after,'speed_benchmark':timings}
    sensitivity=ROOT/'results/sensitivity512'
    evidence['round_6_capacity512_three_seeds']={'protocol':audit_protocol(sensitivity),'results':audit_results(sensitivity),'purpose':'Fixed sensitivity setting; no test-based checkpoint or capacity selection'}
    (out/'six_round_checks.json').write_text(json.dumps(evidence,indent=2))
    print('Six distinct check rounds passed; checkpoints:',replayed)
    print('Speedups:',{k:round(v['speedup'],2) for k,v in timings.items()})
if __name__=='__main__':main()
