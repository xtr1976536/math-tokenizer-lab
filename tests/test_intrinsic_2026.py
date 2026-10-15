import math
from collections import Counter
import numpy as np
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.intrinsic_2026 import remove_template_overlap,summarize
from math_tokenizer.measure_v5 import cross_count
from math_tokenizer.frontend_v6 import tree_ranges
from math_tokenizer.ast import compatible

def test_hard_ranked_native_matches_reference_and_never_crosses_source_tree():
    texts=[rf'\frac{{x+{i}}}{{y^{i%5+1}}}=\sqrt{{z+1}}' for i in range(45)]+['α+β=2','x>2\\sqrt{2}','x + y','']
    m=HardASTBPE.train(texts,360)
    assert len(m.vocab)==360 and all(bytes([i]) in m.vocab for i in range(256))
    for text in texts+[r'\frac{u+5}{1+v}',r'\operatorname{new}(x)', '𝒜∂α+3.25']:
        ids=m.encode(text);assert ids==m.encode_reference(text) and m.decode(ids)==text
        cuts=[0]
        for i in ids:cuts.append(cuts[-1]+len(m.vocab[i]))
        assert all(compatible(a,b,tree_ranges(text).tolist()) for a,b in zip(cuts,cuts[1:]))

def test_crossing_extremes_and_true_partial_overlap():
    spans=[(0,5),(5,10)]
    assert cross_count(list(range(11)),spans)==0
    assert cross_count([0,10],spans)==0
    assert cross_count([0,7,10],spans)==1

def test_template_and_render_equivalence_both_block_leakage():
    r=lambda id,template,render:dict(id=id,template_sha256=template,reference={'mathml_sha256':render})
    train=[r('a','T1','R1')];dev=[r('b','T2','R1'),r('c','T1','R2'),r('d','T3','R3')]
    test=[r('e','T3','R4'),r('f','T4','R3'),r('g','T5','R5')]
    tr,dv,te,info=remove_template_overlap(train,dev,test)
    assert [r['id'] for r in dv]==['d'] and [r['id'] for r in te]==['g']
    assert info=={'dev_removed':2,'test_removed':2}

def test_declared_vocabulary_renyi_matches_published_function():
    from tokenizer_analysis.metrics.information_theoretic import InformationTheoreticMetrics
    tool=InformationTheoreticMetrics.__new__(InformationTheoreticMetrics)
    counts=Counter({0:9,1:3,2:1});h=tool.compute_renyi_entropy(counts,2.5)
    p=np.array([9,3,1])/13
    assert abs(h-np.log2(np.sum(p**2.5))/(1-2.5))<1e-12
    assert h/math.log2(1024)<h/math.log2(3)
