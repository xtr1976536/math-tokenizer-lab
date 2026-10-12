"""Exact pruning and same-alphabet official baseline regression checks."""
from collections import Counter
import numpy as np
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.dictionary_v7 import prune_whole_formula,reward_weights
from math_tokenizer.ast_route_v6 import NativeASTV4


def rows(texts):
    return [{'id':str(i),'latex':t,'split_group':'paper:'+str(i%2),'paper_domain':'analysis'} for i,t in enumerate(texts)]


def brute_prune(vocab,atoms,records,capacity):
    kept=set(range(len(vocab)));protected=set(range(len(atoms)));weights=reward_weights(records)
    while len(kept)>capacity:
        current=NativeASTV4([p for id,p in enumerate(vocab) if id in kept],False)
        before=[len(current.encode(r['latex'])) for r in records];scores=[]
        for remove in sorted(kept-protected):
            alternative=NativeASTV4([p for id,p in enumerate(vocab) if id in kept and id!=remove],False)
            loss=sum(w*(len(alternative.encode(r['latex']))-n) for r,w,n in zip(records,weights,before))
            scores.append((loss,remove))
        kept.remove(min(scores)[1])
    return [p for id,p in enumerate(vocab) if id in kept]


def test_cached_full_formula_pruning_matches_exhaustive_after_unused_alternative_removal():
    atoms=[bytes([b]) for b in range(256)]
    # 'ab' may be unused in the best path, yet be required by a counterfactual
    # path when 'abc' is excluded. Removing it must invalidate that loss.
    vocab=atoms+[b'ab',b'bc',b'abc',b'abcd',b'cd',b'cda',b'da']
    for records in [rows(['abcd','abcda','abcabc']),rows(['abc','abcdabcd','bcd'])]:
        expected=brute_prune(vocab,atoms,records,259)
        actual=prune_whole_formula(vocab,atoms,records,259,hard_tree=False)
        assert actual.vocab==expected
        assert all(actual.decode(actual.encode(r['latex']))==r['latex'] for r in records)


def test_official_atomic_bpe_keeps_commands_and_unknown_bytes_and_serializes(tmp_path):
    atoms=[bytes([b]) for b in range(256)]+[b'\\alpha',b'\\frac','α'.encode()]
    model=AtomicBPE.train(rows([r'\frac{\alpha+x}{y}=1']*6),atoms,280)
    path=tmp_path/'atomic.json';model.save(path);loaded=AtomicBPE.load(path)
    for text in [r'\alpha+x=1','α+😊=1',r'\unknownABC{x}=12345','']:
        assert model.decode(model.encode(text))==text
        assert model.encode(text)==loaded.encode(text)
    assert len(model.vocab)<=280 and b'\\alpha' in model.vocab
