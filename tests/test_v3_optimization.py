from pathlib import Path
import json
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import file_hash,read_jsonl
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.rl_v2 import LatticeTokenizer


def test_frozen_corpus_sha():
    p=ROOT/'data/releases/v2/curated'
    assert file_hash(p/'corpus.jsonl')==json.loads((p/'summary.json').read_text())['corpus_sha256']


def test_native_policy_equivalence_across_real_formula_shapes():
    records=read_jsonl(ROOT/'data/releases/v2/curated/splits/dev.jsonl')
    samples=records[::17]
    for cap in (512,1024):
        for arm in ('byte','ast'):
            slow=LatticeTokenizer.load(ROOT/f'results/v2_release{cap}/models/rl_{arm}_selected-17.json')
            fast=NativeTokenizer.from_v2(slow)
            for r in samples:
                assert fast.encode(r['latex'])==slow.encode(r['latex'])
                assert fast.decode(fast.encode(r['latex']))==r['latex']


def test_native_soft_structure_penalty_respects_cost_definition():
    # Two valid choices: whole tree interval is valid, cross interval is charged.
    vocab=[bytes([b]) for b in range(256)]+[b'x+(y',b'x',b'(y*z)']
    vocab=list(dict.fromkeys(vocab))
    model=NativeTokenizer(vocab,False,'policy_structure',theta=[1.,0,0,0,0,0,0],crossfee=3.)
    for text in ['x+(y*z)','α+β=γ',r'\frac{x}{y}=1']:
        assert model.decode(model.encode(text))==text


def test_native_end_to_end_unicode_controls_and_empty():
    from math_tokenizer.ast_v3 import AtomicASTBPE
    corpus=[r'\alpha+\beta=1',r'\frac{x}{y}',r'\sum_{i=0}^{n}x_i']*4
    model=AtomicASTBPE(True).train(corpus,300,command_budget=16).decoder()
    for text in ['','\x00\x01', '\n\t x+中文🙂  ',r'\unknown{\alpha}',r'\frac{']:
        assert model.decode(model.encode(text))==text


def test_atom_ids_count_inside_capacity():
    from math_tokenizer.ast_v3 import AtomicASTBPE
    data=[r'\alpha+\beta=\gamma']*4
    for capacity in (256,260,280):
        m=AtomicASTBPE(True).train(data,capacity,command_budget=192)
        assert len(m.vocab)<=capacity and m.vocab[:256]==[bytes([b]) for b in range(256)]


def test_fast_structure_exact_all_corpus_records():
    from math_tokenizer.structure_v3 import structure_fast
    from math_tokenizer.structure_v2 import structure
    records=read_jsonl(ROOT/'data/releases/v2/curated/corpus.jsonl')
    for record in records:
        assert structure_fast(record['latex'])==structure(record['latex'])


def test_native_exact_compiled_source_is_current_after_save(tmp_path):
    from math_tokenizer.native_v3 import NativeTokenizer
    model=NativeTokenizer([bytes([b]) for b in range(256)])
    path=tmp_path/'model.json';model.save(path)
    artifact=json.loads(path.read_text())
    assert artifact['native_source_sha256']==file_hash(ROOT/'native/decoder.c')


def test_v3_protocol_budget_and_corpus_locked():
    for cap in (512,1024):
        path=ROOT/f'results/v3_{cap}_release/protocol.json'
        if not path.exists():continue
        protocol=json.loads(path.read_text())
        assert protocol['corpus_sha256']==file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')
        assert protocol['capacity']==cap and protocol['new_baselines_run'] is False
        assert set(protocol['selection_dev_ids'])<=set(protocol['dev_ids'])
        assert set(protocol['train_ids']).isdisjoint(protocol['test_ids'])
        assert protocol['formal_seeds']==[17,29,43]
