from collections import Counter
import numpy as np
from math_tokenizer.corpus_index_v8 import CorpusFitIndex
from math_tokenizer.rl_vocab_v5 import Indexed
from math_tokenizer.rl_transfer_v7 import rollout,TransferConfig


def test_whole_corpus_native_index_matches_every_python_gain_and_sequence():
    texts=['aaaaaaa+bbbb+ababab',r'\alpha+\alpha','中文🙂+x+x','aaaaabaaaa']*5
    reference=Indexed(texts);native=CorpusFitIndex(texts,sequences=reference.seqs)
    for step in range(80):
        pairs=reference.top(12,1);assert native.top(12,1)==pairs
        seqs,counts,gains=native.snapshot()
        assert seqs==[s.ids for s in reference.seqs]
        assert Counter(counts)==+reference.counts and Counter(gains)==+reference.gains
        assert all(native.gains[p]==reference.gains[p] for p in pairs)
        if not pairs:break
        pair=pairs[step%len(pairs)];reference.merge(pair,256+step);native.merge(pair,256+step)


def test_native_full_policy_rollout_preserves_stochastic_and_deterministic_choices():
    fit=[{'id':str(i),'latex':r'\frac{x+x}{y}=1' if i%2 else 'aaaaaa+x+y=y','split_group':str(i)} for i in range(20)]
    atoms=[bytes([b]) for b in range(256)]+[b'\\frac'];theta=np.arange(24)/100;config=TransferConfig(candidates=8)
    for seed in (None,17,29):
        def rng():return None if seed is None else np.random.default_rng(seed)
        a,scorea,_=rollout(fit,atoms,300,theta,config,rng())
        b,scoreb,_=rollout(fit,atoms,300,theta,config,rng(),index_class=CorpusFitIndex)
        assert a.vocab==b.vocab and a.merges==b.merges and np.array_equal(scorea,scoreb)
