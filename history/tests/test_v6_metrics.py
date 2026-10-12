from math_tokenizer.baselines import ByteTokenizer
from math_tokenizer.measure_v6 import measure,selection_utility
from math_tokenizer.rl_mask_v6 import fit_barriers,FrozenBarrierV5,BarrierPolicyV5,MaskConfig
from math_tokenizer.rl_vocab_v5 import VocabRLConfig
from math_tokenizer.rl_route_v6 import AtomCodec
from math_tokenizer.baselines_v6 import PathPieceOfficial


def record(text,id='a'):
    return {'id':id,'latex':text,'split_group':'paper:'+id,'domain':'analysis','paper_domain':'analysis','origin':'human','complexity':{'class':'simple'},'reference':{'spans':[[0,5]]}}


def test_metric_no_integrity_becomes_a_real_penalty():
    model=ByteTokenizer();m,rows=measure(model,[record(r'\alpha+x=1')],'bytes',repeats=1)
    assert m['command_integrity']==0 and m['roundtrip_rate']==1
    assert selection_utility(m)>=m['document_macro_tokens_per_byte']+.15
    assert rows[0]['commands_intact']==0 and rows[0]['payload_exact']


def test_official_pathpiece_budget_and_utf8_payload(tmp_path):
    path=tmp_path/'pathpiece.vocab';path.write_text(''.join(bytes([b]).hex()+'\n' for b in range(1,256))+'6162\n')
    model=PathPieceOfficial(path)
    assert model.vocab_size==257
    for text in ['α+ab=β','ab\ncd','\x00abc']:
        ids=model.encode(text);assert model.decode_bytes(ids)==text.encode() and model.decode(ids)==text


def test_rl_barriers_are_atomic_and_roundtrip_serialization(tmp_path):
    atoms=[bytes([b]) for b in range(256)]+[b'\\alpha'];codec=AtomCodec(atoms)
    texts=[r'\alpha+x^2=y']*3
    model=fit_barriers(atoms,[codec.symbols(s) for s in texts],[ord('=')],280,codec)
    path=tmp_path/'barrier.json';model.save(path);loaded=FrozenBarrierV5.load(path)
    for text in texts+['α+😊=1',r'\alphaBeta{x}=2']:
        assert loaded.encode(text)==model.encode(text)
        assert loaded.decode(loaded.encode(text))==text
    ids=model.encode(texts[0]);payload=[model.vocab[id] for id in ids]
    assert b'=' in payload
