import pytest
from math_tokenizer.ast import parse,byte_spans,compatible,ParseError

@pytest.mark.parametrize('text',[r'x+y=z',r'\frac{\partial u}{\partial t}=\frac{\partial^{2}u}{\partial x^{2}}',r'(x+2)^{3}',r'\sin(x)+\cos(x)=1',r'α+β=γ',r'x_{i}^{2}',r'\sqrt{x+1}',r'\mathbb{E}(X)=0'])
def test_parse(text):
    root=parse(text)
    assert text[root.start:root.end]==text
    spans=byte_spans(text,root)
    assert all(0<=a<b<=len(text.encode()) for a,b in spans)

@pytest.mark.parametrize('text',[r'\input{secret}',r'\frac{x}',r'\unknown{x}',r'x+}',r'\begin{matrix}x\end{matrix}',r'',r'(x+y'])
def test_reject(text):
    with pytest.raises(ParseError):parse(text)

def test_semantics_and_boundaries():
    n=parse('x+y*z')
    assert n.kind=='binary:+'
    assert n.children[1].kind=='binary:*'
    spans=byte_spans('x+(y*z)')
    assert compatible(0,1,spans)
    assert not compatible(0,4,spans)
    assert compatible(0,7,spans)
