import numpy as np
import pytest
from math_tokenizer.merge_scope_stage2 import MergeScopeTokenizer,extract_merges
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.distilled_scope_stage2 import distill_crossing


def test_native_learned_merge_tree_matches_python_and_official_bpe(tmp_path):
    atoms=[bytes([i]) for i in range(256)]+[b'\\alpha']
    rows=[{'latex':s} for s in ['{ab}+abc','ababc','abcabc','{ab}+abc',r'\alpha{abc}']]
    official=AtomicBPE.train(rows,atoms,280);merges=extract_merges(official.tokenizer)
    model=MergeScopeTokenizer(official.vocab,atoms,[(123,125)],merges,.3,.3)
    for text in ('','ababc','{ab}+abc',r'\alpha{abc}',r'\unknown{α😊}','\x00'):
        ids,cost=model.encode_with_cost(text);ref,value=model.encode_reference(text)
        assert ids==ref and cost==pytest.approx(value) and model.decode(ids)==text
        raw,allowed,marked,spans,path=model.reference_spans(text)
        if text in ('','ababc','{ab}+abc',r'\alpha{abc}'):
            assert path==official.encode(text)
    path=tmp_path/'m.npz';model.save(path);loaded=MergeScopeTokenizer.load(path)
    assert loaded.encode('{ab}+abc')==model.encode('{ab}+abc')


def test_heap_repeated_pairs_and_nested_merge_cost():
    atoms=[bytes([i]) for i in range(256)]
    vocab=atoms+[b'aa',b'aaaa',b'aab',b'b{',b'{aa}']
    merges=[(97,97,256,0),(256,256,257,1),(256,98,258,2)]
    m=MergeScopeTokenizer(vocab,atoms,[(123,125)],merges,.6,.3)
    for text in ('aaaaaa','aaaaaab','{aaaaaa}','aaaab{aa}'):
        ids,cost=m.encode_with_cost(text);ref,value=m.encode_reference(text)
        assert ids==ref and cost==pytest.approx(value) and m.decode(ids)==text


def test_distilled_signal_is_candidate_occurrence_probability():
    atoms=[bytes([i]) for i in range(256)]
    vocab=atoms+[b'ab',b'abc',b'bc']
    m=MergeScopeTokenizer(vocab,atoms,[],[(97,98,256,0),(256,99,257,1)])
    rates,log=distill_crossing(m,[{'id':'fit','latex':'abc'}])
    assert rates[258]==1. and rates[256]==0. and rates[257]==0.
    assert log['fit_ids']==['fit'] and log['total'][258]==1 and np.all((rates>=0)&(rates<=1))
