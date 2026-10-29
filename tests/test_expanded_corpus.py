from collections import Counter
import random
import pytest
from math_tokenizer.eqgpt.expanded_corpus import (equation_sides,signed_terms,latex_signature,permute_latex,gre_signature,permute_gre,paired_rows)
from math_tokenizer.eqgpt.corpus_conversion import convert
from scripts.prepare_downstream_corpus import invariant_support,gre_expression

@pytest.mark.parametrize('source',[
 r'u_t=\nu u_{xx}-\mu u_{xxx}+\gamma u_{xxxxx}-\varepsilon u^2u_x+\epsilon u(1-u)',
 r'u_t+u^4u_x+u_{xx}+\partial_{xx}(u^3)+u_{xxx}+u_{xxxx}=u^2',
 r'u_t=\frac{u_{xxx}}{u^3}-9\frac{u_x u_{xx}}{u^4}+12\frac{u_x^3}{u^5}',
 r'u_t= -u_{xx} - 2u^2u_x',
])
def test_original_latex_equivalence_and_signed_substrings(source):
 base=convert(source);key=invariant_support(gre_expression(base['gre_ids']))
 for i in range(1,5):
  transformed=permute_latex(source,random.Random(i),i)
  assert latex_signature(transformed)==latex_signature(source)
  assert invariant_support(gre_expression(convert(transformed)['gre_ids']))==key
 assert permute_latex(source,random.Random(1),0)==source


def test_subtraction_stays_attached_and_nested_expression_intact():
 text=r'u_t=-a u_{xx}+\frac{u_x}{1+u^2}-b u(1-u)'
 a,b=equation_sides(text)
 assert len(signed_terms(b))==3
 for i in range(5):assert latex_signature(permute_latex(text,random.Random(i),1))==latex_signature(text)


def test_gre_numerator_and_denominator_never_swap():
 ids=[5,6,2,8,3,9,4,7,4,19,1]
 for i in range(10):
  out=permute_gre(ids,random.Random(i),1)
  assert gre_signature(out)==gre_signature(ids)
  assert out[0]==5 and out[-1]==1 and out.count(1)==1 and out.count(5)==1


def test_seed_is_source_stable_and_zero_is_original():
 a=dict(source_id='a',original_latex=r'u_t+u u_x=u_{xx}',gre_ids=[5,6,2,7,3,8,2,9,1])
 b=dict(a,source_id='b')
 first=list(paired_rows([a],4));second=[x for x in paired_rows([b,a],4) if x['source_id']=='a']
 assert first==second
 assert first[0]['augmented_latex']==a['original_latex']
 assert first[0]['augmented_gre_ids']==a['gre_ids']
 assert len(first)==4

@pytest.mark.parametrize('bad',[r'u_t+u_x',r'u_t=u_x=0',r'u_t=(u_x'])
def test_ambiguous_equation_rejected(bad):
 with pytest.raises(ValueError):equation_sides(bad)
