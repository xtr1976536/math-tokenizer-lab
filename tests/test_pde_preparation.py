import pytest
from math_tokenizer.pde_protocol import split_corpus
from math_tokenizer.eqgpt.corpus_conversion import convert, ConversionError
from scripts.prepare_downstream_corpus import invariant_support, gre_expression, words_to_ids, TARGETS
from math_tokenizer.eqgpt.compact_alphabet import train_compact,CompactAlphabet
from math_tokenizer.eqgpt.latex_tokenizers import prefix_model


def test_group_connections_and_full_retention():
    rows=[dict(id=str(i),latex='u_t=0',source_group='s'+str(i),template_sha256='t'+str(i),canonical_sha256='c'+str(i)) for i in range(40)]
    rows[1]['source_group']=rows[0]['source_group']
    rows[2]['template_sha256']=rows[1]['template_sha256']
    rows[3]['canonical_sha256']=rows[2]['canonical_sha256']
    parts,manifest=split_corpus(rows)
    assert sum(map(len,parts.values()))==40
    assert any({'0','1','2','3'} <= {r['id'] for r in rr} for rr in parts.values())
    assert split_corpus(rows)[1]==manifest


def test_ranked_prefix_equals_independent_smaller_training():
    texts=[r'u_t+u u_x=0',r'u_t-u_{xx}=0',r'u_t+u_{xxx}=0']*3
    alphabet=sorted(set(''.join(texts).encode()))
    small=len(alphabet)+3+9
    parent=train_compact('ast_bpe',texts,small+12,alphabet)
    clipped=CompactAlphabet('ast_bpe',alphabet,prefix_model('ast_bpe',parent.internal,small+256-len(alphabet)))
    independent=train_compact('ast_bpe',texts,small,alphabet)
    assert clipped.vocab==independent.vocab
    assert clipped.internal.merges==independent.internal.merges
    for text in texts+[r'u_{xx}+u u_x=0']:
        assert clipped.encode(text)==independent.encode(text)


@pytest.mark.parametrize('text,case',[
    (r'u_t+u u_x-0.1u_{xx}=0','Burgers_equation'),
    (r'\partial_t v+v\partial_x v=\nu\partial_x^2 v','Burgers_equation'),
    (r'\frac{\partial^2 u}{\partial t^2}=c\frac{\partial^2 u}{\partial x^2}','Wave_equation'),
    (r'u_t+u_y+u_{yy}=0','Convection_diffusion_equation'),
    (r'u_t-u_{xx}+u-u^3=0','Chaffee_Infante_equation'),
    (r'u_t+\frac{u_x}{x}+u_{xx}=0','PDE_divide'),
    (r'u_t+u_x+u_{tx}=0','Eq_6_2_12'),
    (r'u_t+\frac{1}{2}\partial_x(u^2)+u_{xxx}=0','KdV_equation'),
    (r'u_{tt}-u_{xx}+5u=0','KG_equation'),
])
def test_target_equivalence(text,case):
    result=convert(text)
    assert invariant_support(gre_expression(result['gre_ids']))==invariant_support(gre_expression(words_to_ids(TARGETS[case])))
    assert result['support']==result['reconstructed_support']


@pytest.mark.parametrize('text',[
    r'\partial_t u+\cos(u)=0',
    r'\partial_t u+u_{xxxxxx}=0',
    r'\partial_t u+a(x)u_x=0',
    r'\partial_t u+\Delta u=0',
    r'\begin{cases}u_t=v_x\\v_t=u_x\end{cases}',
    r'\frac{\partial^2 u}{\partial x}=u',
    r'u_t+1=0',
    r'u_t+f(u)=0',
])
def test_unsupported_terms_never_disappear(text):
    with pytest.raises(ConversionError):convert(text)


def test_product_rule_expansion_and_non_target():
    a=convert(r'u_t+\partial_x(u\partial_x u)=0')
    b=convert(r'u_t+u_x^2+u u_{xx}=0')
    assert a['support']==b['support']
    assert invariant_support(gre_expression(a['gre_ids'])) not in [invariant_support(gre_expression(words_to_ids(w))) for w in TARGETS.values()]


def test_domain_annotation_preserved():
    result=convert(r'u_t=u_{xx}\quad\text{in }\Omega')
    assert result['domain_annotation']
    assert result['roundtrip_verified']

@pytest.mark.parametrize('text',[
    r'-\partial_x(\rho\partial_x\phi)=u',
    r'u_t+uu_x-\epsilon u_{xx}=P(D)(u^2+\frac{1}{2}u_x^2)',
    r'u_t+i u_{xx}=u u_x',
])
def test_unknown_fields_operators_and_complex_coefficients_not_erased(text):
    with pytest.raises(ConversionError):convert(text)


def test_scalar_coefficients_combined_before_stripping():
    a=convert(r'\partial_t v=d_1\partial_x^2 v+r_1(1-\varepsilon)v')
    assert a['gre_words'].count('u')==1
    assert len(a['coefficients'])==3


def test_grouped_derivative_and_unbraced_fraction():
    a=convert(r'u_t+\frac12(u^2)_x+u_{xxx}=0')
    assert invariant_support(gre_expression(a['gre_ids']))==invariant_support(gre_expression(words_to_ids(TARGETS['KdV_equation'])))


def test_explicit_field_arguments_preserve_derivative_dependencies():
    assert convert(r'\partial_t u(x,t)=\partial_x^2 u(x,t)')['roundtrip_verified']
    with pytest.raises(ConversionError):convert(r'\partial_t u(t)=\partial_x^2 u(t)')


def test_other_scalar_field_renaming_and_coupled_field_rejection():
    result=convert(r'\partial_t g+g\partial_x g=\partial_x^2 g')
    assert result['field_mapping']=={'g':'u'}
    with pytest.raises(ConversionError):convert(r'\partial_t u+\partial_x v=0')

@pytest.mark.parametrize('text',[
 r'v_t+\alpha v_{xxx}+(uv)_x=0',
 r'\partial_tu=\partial_x^2u+u(1-u-av)',
 r'\partial_x(\rho u^2+p)+\partial_y(\rho u v)=0',
])
def test_coupled_undifferentiated_fields_not_scalar_coefficients(text):
    with pytest.raises(ConversionError):convert(text)

from math_tokenizer.eqgpt.paper_normalization import review_row,spatial_dimension

def test_system_components_do_not_become_whole_system_conversions():
    row={'latex':r'\begin{cases}u_t=u_{xx}\\u(x,0)=f(x)\end{cases}'}
    items=review_row(row)
    assert len(items)==2
    assert items[0]['status']=='converted' and items[0]['component_of_system']
    assert items[1]['status']=='unresolved'


def test_laplacian_dimension_must_have_evidence():
    row={'latex':r'\Delta u+u=0'}
    assert review_row(row)[0]['status']=='unresolved'
    row['title']='Eigenfunctions in two-dimensional domains'
    item=review_row(row)[0]
    assert item['status']=='converted'
    assert 'uxx' in item['gre_words'] and 'uyy' in item['gre_words']
    row['context_before']=r'in $\mathbb{R}^3$'
    assert spatial_dimension(row)[0] is None


def test_negative_constant_powers_fit_unchanged_gre():
    result=convert(r'u_t=\frac{u_{xxx}}{u^3}-\frac{u_x u_{xx}}{u^4}')
    assert all(0<=i<57 for i in result['gre_ids'])
    assert result['support']==result['reconstructed_support']


def test_generalized_pde_integer_instances_are_distinct():
    a=convert(r'u_t+\partial_x(u^2)+\partial_{xxx}(u^2)=0')
    b=convert(r'u_t+\partial_x(u^3)+\partial_{xxx}(u^2)=0')
    assert a['support']!=b['support']


def test_context_variable_coefficients_never_certified_constant():
    row={'latex':r'u_t=u_{xx}+A u_x', 'context_after':'where the functions A are real-analytic in the domain.'}
    assert review_row(row)[0]['reason']=='variable_coefficient_context'
