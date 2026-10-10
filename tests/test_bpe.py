import random
import pytest
from math_tokenizer.bpe import MathBPE,Sequence
from math_tokenizer.ast import byte_spans,compatible

TEXTS=[r'\frac{\partial u}{\partial t}=u+u^{2}',r'x+x=y',r'(x+2)^{3}=y',r'\sin(x)+\cos(x)=1','α+β=γ','x_{i}^{2}+1',r'\unsupported{x}',r'x  + y']*4
@pytest.mark.parametrize('structural,rl',[(False,False),(True,False),(False,True),(True,True)])
def test_roundtrip_save_and_reproducibility(structural,rl,tmp_path):
    a=MathBPE(structural,rl).train(TEXTS,288,17,8)
    b=MathBPE(structural,rl).train(TEXTS,288,17,8)
    assert a.merges==b.merges
    a.save(tmp_path/'model.json');loaded=MathBPE.load(tmp_path/'model.json')
    for text in TEXTS+['','中文数学','\n x + y\n','🙂+1']:
        ids=a.encode(text)
        assert a.decode(ids)==text
        assert loaded.encode(text)==ids
        assert a.encode_reference(text)==ids
    assert len(a.vocab)<=288
    if rl:
        assert len(a.training_log)>0
        assert any(any(abs(x)>1e-9 for x in log['policy']) for log in a.training_log)

@pytest.mark.parametrize('structural',[False,True])
def test_heap_differential_fuzz(structural):
    rng=random.Random(91)
    texts=[''.join(rng.choice('abc+^{}123 α') for _ in range(rng.randrange(1,80))) for _ in range(150)]
    model=MathBPE(structural).train(TEXTS+texts,320)
    for text in texts:
        assert model.encode(text)==model.encode_reference(text)
        assert model.decode(model.encode(text))==text


def test_structure_and_fallback():
    m=MathBPE(True).train(TEXTS,320)
    for text in TEXTS[:6]:
        spans=byte_spans(text);offset=0
        for i in m.encode(text):
            end=offset+len(m.vocab[i]);assert compatible(offset,end,spans);offset=end
    assert m.encode(r'\unsupported{x}')==list(r'\unsupported{x}'.encode())

def test_nonoverlapping_reward():
    s=Sequence.make('aaaaa',False)
    assert s.gain((97,97))==2
    s.merge((97,97),256)
    assert s.ids==[256,256,97]
