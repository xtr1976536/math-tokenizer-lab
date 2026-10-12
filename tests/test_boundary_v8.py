import pytest
from math_tokenizer.ast_boundary_v8 import BoundaryAST,brace_boundaries


def test_native_boundary_path_matches_exhaustive_distortion_optimum():
    vocab=[bytes([b]) for b in range(256)]+[b'{x',b'x+y',b'y}',b'{x+y}',b'{x+y}+z',b'+z']
    text='{x+y}+z';raw=text.encode()
    def paths(a):
        if a==len(raw):return [[]]
        return [[i]+tail for i,p in enumerate(vocab) if raw[a:a+len(p)]==p for tail in paths(a+len(p))]
    for tree,boundary in ((0,0),(.25,.15),(.75,.3),(0,2)):
        model=BoundaryAST(vocab,tree,boundary);ids,cost=model.encode_with_cost(text)
        assert cost==pytest.approx(min(model.objective(text,path) for path in paths(0)))
        assert cost==pytest.approx(model.objective(text,ids))
        assert model.decode(ids)==text


def test_boundary_scanner_escaped_braces_unicode_nested_and_empty():
    assert brace_boundaries(r'\{x\}')==set()
    assert brace_boundaries('{}')==set()
    assert brace_boundaries('{α{b}}')=={1,4,5,6}
    vocab=[bytes([b]) for b in range(256)]+[b'\\alpha','α'.encode()]
    model=BoundaryAST(vocab)
    for text in ['',r'\alpha{x}',r'\unknown{α+😊}',r'{x{y}}',r'{x']:
        ids,cost=model.encode_with_cost(text)
        assert model.decode(ids)==text and cost==pytest.approx(model.objective(text,ids))


def test_boundary_saved_model_and_weight_validation(tmp_path):
    vocab=[bytes([b]) for b in range(256)]+[b'{x}',b'\\alpha']
    model=BoundaryAST(vocab,.75,.3);path=tmp_path/'model.json';model.save(path)
    loaded=BoundaryAST.load(path)
    assert loaded.encode(r'\alpha{x}')==model.encode(r'\alpha{x}')
    for bad in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):BoundaryAST(vocab,bad,.1)
