import itertools
import numpy as np
import pytest
from math_tokenizer.learned_scope_stage2 import LearnedScopeTokenizer,discover_pairs
from math_tokenizer.signal_stage2 import CONTEXTS


def test_pair_discovery_has_no_fixed_delimiter_identity():
    rows=[{'id':str(i),'latex':s} for i,s in enumerate(['!ab?','!!cd??','!ef?','!gh?'])]
    pairs,info=discover_pairs(rows,min_support=2,min_confidence=.9)
    assert (ord('!'),ord('?')) in pairs and info['fit_ids']==[r['id'] for r in rows]


def test_scope_native_matches_exact_legal_paths_and_generic_atoms(tmp_path):
    atoms=[bytes([b]) for b in range(256)]+[b'ab']
    vocab=atoms+[b'{ab',b'ab}',b'{ab}',b'{ab}+c',b'+c',b'a{',b'{ab}+']
    rng=np.random.default_rng(71);m=LearnedScopeTokenizer(vocab,atoms,[(123,125)],.3,rng.normal(0,.1,CONTEXTS),rng.normal(0,.05,len(vocab)))
    for text in ('','ab','{ab}+c','a{ab}','{ab','α😊',r'\Novel{ab}'):
        ids,cost=m.encode_with_cost(text);ref,value=m.encode_reference(text)
        assert ids==ref and cost==pytest.approx(value) and m.decode(ids)==text
        raw,allowed,marked=m.prepared_reference(text)
        ends=np.cumsum([len(vocab[i]) for i in ids]);assert all(allowed[e] for e in ends)
    # A partial primitive token would shorten the count but must be illegal.
    m=LearnedScopeTokenizer(atoms+[b'!a',b'b?'],atoms,[],0.)
    ids=m.encode('!ab?');assert [m.vocab[i] for i in ids]==[b'!',b'ab',b'?']
    path=tmp_path/'m.npz';m.save(path);loaded=LearnedScopeTokenizer.load(path)
    assert loaded.encode('!ab?')==ids
