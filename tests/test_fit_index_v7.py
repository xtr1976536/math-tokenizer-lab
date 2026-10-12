import random
from math_tokenizer.rl_vocab_v5 import Indexed
from math_tokenizer.fit_index_v7 import NativeFitIndexed


def test_full_native_update_sequence_matches_python_pair_counts_gains_and_positions():
    rng=random.Random(191)
    texts=[''.join(rng.choice('aaabc+{}') for _ in range(90)) for _ in range(30)]+['aaaaaaaaaaaa']
    slow=Indexed(texts);fast=NativeFitIndexed(texts)
    for newid in range(256,296):
        assert slow.counts==fast.counts and slow.gains==fast.gains
        assert slow.top(12)==fast.top(12)
        pairs=slow.top(12)
        if not pairs:break
        pair=pairs[rng.randrange(len(pairs))];slow.merge(pair,newid);fast.merge(pair,newid)
        for a,b in zip(slow.seqs,fast.seqs):assert (a.ids,a.starts,a.ends)==(b.ids,b.starts,b.ends)
