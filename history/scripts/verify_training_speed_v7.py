"""Isolated equivalent fit-index timing, identical full merge actions/states."""
import hashlib,json,random,time
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_vocab_v5 import Indexed
from math_tokenizer.fit_index_v7 import NativeFitIndexed


def main():
    rows=sorted(load_split('dev'),key=lambda r:hashlib.sha256(('index-probe:'+r['id']).encode()).digest())[:256]
    texts=[r['latex'] for r in rows];rng=random.Random(371);timings={'python':[],'native':[]};reference=None;order=[]
    for repeat in range(5):
        names=['python','native'];rng.shuffle(names);order.append(names)
        for name in names:
            cls=Indexed if name=='python' else NativeFitIndexed;started=time.perf_counter();index=cls(texts);actions=[]
            for newid in range(256,384):
                pairs=index.top(12)
                if not pairs:break
                pair=pairs[0];actions.append(pair);index.merge(pair,newid)
            elapsed=time.perf_counter()-started;timings[name].append(elapsed)
            state=(actions,index.counts,index.gains,[(s.ids,s.starts,s.ends) for s in index.seqs])
            if reference is None:reference=state
            else:assert state==reference
    result={'formulas':len(rows),'merges':len(reference[0]),'repeats':5,'timings_seconds':timings,
        'speedup_median':float(np.median(timings['python'])/np.median(timings['native'])),
        'all_actions_counts_nonoverlap_gains_and_source_positions_equal':True,'order':order,
        'scope':'Only same-ID training index construction/update; not an end-to-end RL training speed claim.'}
    (ROOT/'results/v7/training_index_speed.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
