"""Same frozen AST vocab/frontend/masks/IDs, interleaved implementation probe.

No new algorithm or selection. Compare fused CPython bridge, same C frontend
through ctypes, and same C frontend plus the older generic C DP kernel.
"""
import hashlib
import json
import random
import time

import numpy as np

from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.constrained_v4 import library4
from math_tokenizer.corpus import ROOT
from math_tokenizer.current_models import load_current
from math_tokenizer.frontend_v6 import library, tree_ranges
from math_tokenizer.measure_v6 import clear


def generic_encode(model, text):
    raw=text.encode();n=len(raw)
    if not n:return []
    allowed=np.empty(n+1,dtype=np.uint8)
    library().mt_allowed(raw,n,model.children.ctypes.data,model.terminal.ctypes.data,allowed.ctypes.data)
    spans=tree_ranges(text);out=np.empty(n,dtype=np.int32)
    safe=np.zeros(n+1,dtype=np.uint8);interior=np.zeros(n+1,dtype=np.uint8)
    count=library4().mt_decode(raw,n,model.children.ctypes.data,model.terminal.ctypes.data,
        model.maxlen,spans.ctypes.data,len(spans),int(model.structural),safe.ctypes.data,
        interior.ctypes.data,allowed.ctypes.data,model.logfreq.ctypes.data,model.theta.ctypes.data,
        len(model.theta),0,0.,0.,0.,None,out.ctypes.data)
    if count<0:raise RuntimeError('generic AST constraint graph disconnected')
    return out[:count].tolist()


def main():
    rows=load_split('fresh_test');text=[r['latex'] for r in rows]
    raw_bytes=sum(len(s.encode()) for s in text);rng=random.Random(817)
    results=[]
    for capacity in (1024,2048):
        model=load_current('ast_v4',scenario='broad',capacity=capacity)
        methods={'fused_bridge':model.encode,'ctypes_same_native_frontend':model.encode_ctypes,
                 'generic_dp_same_native_frontend':lambda t:generic_encode(model,t)}
        timings={name:[] for name in methods};order=[];reference=None
        for repeat in range(7):
            names=list(methods);rng.shuffle(names);order.append(names)
            for name in names:
                clear(model);started=time.perf_counter()
                encodings=[methods[name](s) for s in text]
                timings[name].append(time.perf_counter()-started)
                if reference is None:reference=encodings
                else:assert encodings==reference,('implementation IDs differ',capacity,name)
        assert all(model.decode(ids)==s for s,ids in zip(text,reference))
        throughput={k:raw_bytes/1024**2/float(np.median(v)) for k,v in timings.items()}
        speedups={k:float(np.median(v))/float(np.median(timings['fused_bridge'])) for k,v in timings.items()}
        result={'capacity':capacity,'scenario':'broad','suite':'fresh_test','formulas':len(rows),
            'bytes':raw_bytes,'formula_id_sha256':hashlib.sha256('\n'.join(sorted(r['id'] for r in rows)).encode()).hexdigest(),
            'all_ids_identical':True,'all_roundtrip':True,'same_vocabulary_frontend_and_constraints':True,
            'seconds_repeats':timings,'encode_mib_s':throughput,'fused_bridge_speedup':speedups,'order':order,
            'repeats':7,'scope':'Dedicated same-ID implementation ablation; not a replacement for full multi-model timings.',
            'limitations':'Single host/run; does not attribute different dictionary or frontend semantics to engineering alone.'}
        results.append(result);print(json.dumps(result,indent=2),flush=True)
    destination=ROOT/'results/v6/full/ast_speed_attribution.json'
    destination.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
