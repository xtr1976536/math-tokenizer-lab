from math_tokenizer.independent_v7 import source_scopes,additional
from math_tokenizer.baselines import ByteTokenizer
from math_tokenizer.ast_balanced_v7 import BalancedAST


def test_independent_macro_groups_have_utf8_byte_offsets():
    text=r'\frac{α+x}{y}'
    spans=source_scopes(text);raw=text.encode()
    assert spans is not None
    assert 'α+x'.encode() in [raw[a:b] for a,b in spans]
    assert b'y' in [raw[a:b] for a,b in spans]


def test_boundary_gold_probe_distinguishes_whole_expression_from_byte_degeneracy():
    text=r'\frac{x}{y}';records=[{'id':'a','latex':text}]
    byte,_=additional(ByteTokenizer(),records)
    whole=BalancedAST([bytes([b]) for b in range(256)]+[text.encode()],0.)
    full,_=additional(whole,records)
    assert byte['independent_scope_boundary_recall']==1
    assert full['independent_scope_boundary_recall']==0
    # The probe diagnoses boundaries only: byte baseline scores high and thus
    # cannot establish a quality ranking without compression and other metrics.
    assert full['independent_scope_parser_coverage']==1
