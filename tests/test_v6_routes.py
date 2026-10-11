import json
import numpy as np
from math_tokenizer.frontend_v6 import lexical_ranges, reference_lex, tree_ranges
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.rl_route_v6 import AtomicTrajectoryV5, AtomicVocab, AtomCodec
from math_tokenizer.rl_vocab_v5 import VocabRLConfig
from math_tokenizer.trajectory_v5 import TrajectoryConfig
from math_tokenizer.atoms_v4 import FastSpanSequence
from math_tokenizer.ast import compatible


def test_native_lexer_utf8_escapes_and_numeric_boundaries():
    texts=[r'\frac{x^2}{y}+α=β',r'\begin{aligned}x&=1\\y&=2\end{aligned}', 'a\\α+123.45=x',r'\unknownABC{x}+\,', '', '\\']
    for text in texts:
        assert lexical_ranges(text).tolist()==[list(x) for x in reference_lex(text)]
        assert all(0<=a<b<=len(text.encode()) for a,b in tree_ranges(text))


def test_native_nested_script_tree_has_laminar_bindings():
    for text in [r'\frac{x_{i}^2}{y}_{j}',r'{x+y}_{i}^{n}',r'\sum_{i=1}^{n}x_i^2']:
        spans=tree_ranges(text).tolist()
        for a,b in spans:
            for c,d in spans:assert not (a<c<b<d or c<a<d<b)


def test_native_arithmetic_ast_preserves_precedence_and_falls_back():
    assert [2,5] in tree_ranges('x+y*z').tolist()
    assert [0,3] in tree_ranges('x*y+z').tolist()
    assert [0,5] in tree_ranges('x+y*z').tolist()
    assert [6,9] in tree_ranges(r'\frac{x+y}{z}+a').tolist()
    # A named operator outside the supported mathematical grammar retains CST.
    spans=tree_ranges(r'\operatorname{Hom}(X,Y)').tolist()
    assert spans and all(0<=a<b<=len(r'\operatorname{Hom}(X,Y)'.encode()) for a,b in spans)


def test_training_span_index_matches_independent_interval_rule():
    text=r'\frac{x_{i}^2}{y}+α=β';raw=text.encode();spans=tree_ranges(text).tolist()
    for a in range(len(raw)):
        for b in range(a+1,len(raw)+1):
            seq=FastSpanSequence(list(raw),list(range(len(raw))),list(range(1,len(raw)+1)),spans)
            seq.ids=[0,1];seq.starts=[a,a+1];seq.ends=[a+1,b]
            assert seq.legal(0)==compatible(a,b,spans)


def test_source_ast_and_codec_have_lossless_unknown_escape(tmp_path):
    texts=[r'\frac{x^2}{y}+α=β',r'x_i^2+y_i^2=z_i^2']*3
    model=NativeASTV4.train(texts,320,atom_budget=32)
    path=tmp_path/'ast.json';model.save(path);loaded=NativeASTV4.load(path)
    atoms=[bytes([i]) for i in range(256)]+[b'\\frac','α'.encode()]
    codec=AtomCodec(atoms)
    for text in texts+[r'\newunknown{😊}=1234',r'\foo{x}_{y}=z']:
        assert loaded.encode(text)==model.encode(text)
        assert model.decode(model.encode(text))==text
        assert b''.join(atoms[codec.symbol_reverse[c]] for c in codec.symbols(text))==text.encode()
        assert [codec.symbol_reverse[c] for c in codec.symbols(text)]==[ord(c)-0xE000 for c in codec.symbols_ctypes(text)]


def test_training_cache_clone_does_not_share_mutable_rollouts():
    atoms=[bytes([i]) for i in range(256)]+[b'\\frac'];codec=AtomCodec(atoms)
    first=codec.indexed(['aaaa+x=1',r'\frac{x}{y}=2']);second=codec.indexed(['aaaa+x=1',r'\frac{x}{y}=2'])
    before=second.counts.copy();first.merge((ord('a'),ord('a')),257)
    assert second.counts==before and first.counts!=before
    third=codec.indexed(['aaaa+x=1',r'\frac{x}{y}=2'])
    assert third.counts==before and third.seqs[0].ids==second.seqs[0].ids


def test_primitive_trajectory_updates_and_serializes(tmp_path):
    records=[]
    for i in range(12):
        text=r'\frac{x^2}{y}+'+str(i)+'=z'
        records.append({'id':str(i),'split_group':f'paper:{i}','latex':text,'reference':{'spans':[[6,9],[11,12]]}})
    conf=VocabRLConfig(fit_limit=100,reward_limit=100,candidates=4)
    trajectory=TrajectoryConfig(epochs=2,samples=2)
    model=AtomicTrajectoryV5(conf,trajectory,atom_budget=8).train(records,264,17)
    assert len(model.vocab)<=264 and len(model.history)==2
    assert not np.array_equal(model.theta,model.initial_theta)
    path=tmp_path/'rl.json';model.save(path);loaded=AtomicTrajectoryV5.load(path)
    for text in [r'\frac{x^2}{y}+1=z',r'\unknownFoo{α+😊}=9']:
        assert loaded.encode(text)==model.encode(text)
        assert model.decode(model.encode(text))==text
