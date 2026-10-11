from collections import Counter
from math_tokenizer.bpe import Sequence
from math_tokenizer.rl_structural_v6 import TreeIndexed
from math_tokenizer.measure_v5 import cross_count


def naive_bad(index):
    return sum(cross_count([0]+s.ends,[(1,4),(2,5)]) for s in index.seqs)


def test_structural_counterfactual_and_cache_clone_match_full_reference():
    texts=['aaaaab','baaaab'];seqs=[Sequence.make(t,False) for t in texts]
    index=TreeIndexed(texts,None,seqs,{t:[(1,4),(2,5)] for t in texts})
    assert index.tree_bad==naive_bad(index)
    for pair,new in [((97,97),256),((256,97),257),((257,98),258)]:
        clone=index.clone();before=naive_bad(clone);expected_delta=clone.tree_delta[pair]
        clone.merge(pair,new)
        assert naive_bad(clone)-before==expected_delta
        assert clone.tree_bad==naive_bad(clone)
        assert index.tree_bad==naive_bad(index)
        index=clone
