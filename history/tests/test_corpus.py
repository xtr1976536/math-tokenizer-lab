import json
import pytest
from math_tokenizer.corpus import ROOT,validate,eqterm,canonical
from math_tokenizer.evaluate import read_jsonl

@pytest.mark.parametrize('formula',[r'x^2+y^2=z^2',r'\frac{x}{y}=2',r'\partial_t u=0'])
def test_accept(formula):assert validate(formula) is None
@pytest.mark.parametrize('formula',[r'This is a formula',r'\text{where x is real}',r'\input{evil}',r'$x+1$',r'x+{y',r'u',r'\newcommand{\x}{y}',r'x+1 % ignore'])
def test_quarantine(formula):assert validate(formula) is not None

def test_mapping_no_equation_invention():
    assert eqterm('ut')==r'\frac{\partial u}{\partial t}'
    assert eqterm('uxx')==r'\frac{\partial^{2} u}{\partial x^{2}}'
    assert eqterm('u^2')=='u^{2}'
    with pytest.raises(ValueError):eqterm('unknown')

def test_provenance_splits_and_duplicates():
    rows=read_jsonl(ROOT/'data/curated/corpus.jsonl')
    assert len(rows)>200
    assert len({r['id'] for r in rows})==len(rows)
    assert len({canonical(r['latex']) for r in rows})==len(rows)
    groups={}
    for r in rows:
        assert validate(r['latex']) is None
        assert r['origin'] in {'human','possible_mixed','ai_generated','unknown'}
        assert r['origin_evidence'] and r['location'] and r['source_revision']
        groups.setdefault(r['split_group'],set()).add(r['split'])
        if r['origin']=='human':assert r['source_id']=='eqgpt'
    assert all(len(s)==1 for s in groups.values())
    for split in ('train','dev','test'):
        ss=read_jsonl(ROOT/f'data/curated/splits/{split}.jsonl')
        assert {r['id'] for r in ss}=={r['id'] for r in rows if r['split']==split}
