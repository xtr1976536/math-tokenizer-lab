"""Isolated index/end-to-end rollout timing with exact vocabulary equivalence."""
from dataclasses import asdict
import json,time,statistics,subprocess
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_transfer_v7 import rollout,TransferConfig,task_rows
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.corpus_index_v8 import CorpusFitIndex


def main():
    processes=subprocess.check_output(['ps','-Ao','command'],text=True)
    assert not any('python scripts/'+name in line for line in processes.splitlines() for name in
        ('train_terminal_full_v8.py','controlled_terminal_v8.py','evaluate_v7.py','retime_v7.py','harvest_arxiv')),'Other experiment running; isolated timing required'
    fit=task_rows(load_split('train'),4000,8928);atoms=derive_atoms(fit,1024,384)
    theta=np.arange(24)/100;config=TransferConfig(candidates=16)
    times={'v7_formula_native':[],'v8_corpus_native':[]};reference=None
    for repeat in range(5):
        order=list(times) if repeat%2==0 else list(reversed(times))
        for name in order:
            started=time.perf_counter()
            kwargs={} if name=='v7_formula_native' else {'index_class':CorpusFitIndex}
            model,score,entropy=rollout(fit,atoms,1024,theta,config,**kwargs)
            times[name].append(time.perf_counter()-started)
            content=(model.vocab,model.merges,score.tolist(),entropy)
            if reference is None:reference=content
            else:assert reference==content
        print('engineering rollout timing',repeat+1,flush=True)
    result={'formulas':len(fit),'target_capacity':1024,'actual_capacity':len(model.vocab),'config':asdict(config),
        'repeats':5,'times':times,'vocabulary_merges_and_policy_scores_identical':True,
        'median_rollout_speedup':statistics.median(times['v7_formula_native'])/statistics.median(times['v8_corpus_native']),
        'scope':'Full fit-vocabulary rollout including codec, candidate features, exact merge index and official BPE construction; excludes terminal reward evaluation',
        'RL_learning_gain':False}
    (ROOT/'results/v8/training_speed.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
