import pytest
from math_tokenizer.baselines import ByteTokenizer,HFTokenizer,SentencePieceTokenizer,TikToken
from math_tokenizer.evaluate import measure,bootstrap_difference

TEXTS=[r'\frac{x}{y}+1',r'x  + y','α+β=γ',r'\sqrt{x^{2}+1}','x_{1}=2']*3
@pytest.mark.parametrize('kind',['hf','sp_bpe','sp_unigram','gpt2','cl100k_base','o200k_base','bytes'])
def test_official_adapter_losslessness(kind,tmp_path):
    if kind=='hf':m=HFTokenizer.train(TEXTS,384,tmp_path/'hf.json')
    elif kind.startswith('sp_'):m=SentencePieceTokenizer.train(TEXTS,384,kind[3:],tmp_path/kind)
    elif kind=='bytes':m=ByteTokenizer()
    else:m=TikToken(kind)
    for text in TEXTS+['x\n+y','  x+y  ','🙂+中文=2']:
        assert m.decode(m.encode(text))==text

def test_common_evaluator():
    records=[{'id':str(i),'latex':s,'split_group':f'g{i//2}','domain':'algebra','origin':'ai_generated'} for i,s in enumerate(TEXTS)]
    summary,rows=measure(ByteTokenizer(),records,'bytes',17)
    assert summary['roundtrip_rate']==1
    assert summary['bytes']==summary['tokens']
    delta=bootstrap_difference(rows,rows,n=100)
    assert delta['delta_tokens_per_formula']==0 and delta['ci95_group_bootstrap']==[0,0]
