import json
import subprocess
from math_tokenizer.corpus import ROOT
from math_tokenizer.quality_v6 import candidates, normalize, build_macros, formula_gate, classify, expand_macros,library_topics,TARGET_CATEGORIES


def test_source_positions_and_macro_expansion_preserve_spacing():
    text = r'''\newcommand{\norm}[1]{\left\lVert #1\right\rVert}
\begin{document}
% $invalid$
\begin{align} \norm{x+y} &= \frac{a}{b}\, \label{eq:one} \\ z &= x^2+y^2 \end{align}
\end{document}'''
    found = list(candidates(text))
    assert len(found) == 1
    c = found[0]
    assert text[slice(*c['source_span'])] == c['raw_latex']
    macros, ambiguous = build_macros([('main.tex', text, 'utf-8', text.encode())])
    result, operations, reason = normalize(c, macros, ambiguous)
    assert reason is None and r'\norm' not in result
    assert r'\,' in result and r'\label' not in result
    assert result.startswith(r'\begin{aligned}')
    assert formula_gate(result) is None


def test_ambiguous_and_recursive_macros_are_quarantined():
    members = [('a', r'\newcommand{\foo}{x}', '', b''), ('b', r'\newcommand{\foo}{y}', '', b'')]
    macros, ambiguous = build_macros(members)
    assert expand_macros(r'\foo+1=2', macros, ambiguous)[2].startswith('ambiguous_macro')
    assert expand_macros(r'\foo+1=2', {r'\foo': (0, r'\foo')}, set())[2] == 'recursive_macro'


def test_parameter_expansion_does_not_invent_a_mathematical_group():
    macros={r'\sq':(1,'#1^2'),r'\ratio':(2,r'\frac{#1}{#2}')}
    assert expand_macros(r'\sq{x+y}=z',macros,set())[0]=='x+y^2=z'
    assert expand_macros(r'\ratio{x+y}{z+w}=1',macros,set())[0]==r'\frac{x+y}{z+w}=1'
    rows=[{'id':'faithful','latex':'x+y^2=z','source_macro_reference':{'macro_definitions':r'\def\sq#1{#1^2}','unexpanded_latex':r'\sq{x+y}=z'}},
          {'id':'changed','latex':'{x+y}^2=z','source_macro_reference':{'macro_definitions':r'\def\sq#1{#1^2}','unexpanded_latex':r'\sq{x+y}=z'}}]
    p=subprocess.run(['node',str(ROOT/'scripts/latex_reference_v6.js')],input=''.join(json.dumps(r)+'\n' for r in rows),text=True,capture_output=True,check=True)
    faithful,changed=map(json.loads,p.stdout.splitlines())
    assert faithful['source_macro_fidelity'] and not changed['source_macro_fidelity']


def test_spacing_command_not_truncated_and_partial_not_always_pde():
    c = {'working_latex': r'\frac{a}{b}\,', 'layout_wrapper': None}
    assert normalize(c, {}, set())[0].endswith(r'\,')
    assert classify(r'\partial\Omega=\Gamma', 'geometry_topology', ['math.DG'])[0] == 'geometry_topology'
    assert classify(r'\frac{\partial u}{\partial t}=\Delta u', 'analysis', ['math.AP'])[0] == 'pde'
    assert classify(r'\partial_t\beta|_{t=0}', 'pde', ['math.AP'])[0] == 'analysis'
    assert classify(r'U=\frac{\partial}{\partial y_1}', 'geometry_topology', ['math.DG'])[0] == 'geometry_topology'
    assert 'integral_formula' in classify(r'\int_0^1 x^2 dx = 1/3', 'analysis', ['math.CA'])[1]
    assert 'sum_or_product' in classify(r'\sum_{i=1}^n x_i = y', 'algebra', ['math.RA'])[1]
    assert formula_gate(r'\frac{a}{b}=') == 'incomplete_relation_or_operator'
    assert formula_gate(r'\frac{}{b}=x') == 'empty_required_math_argument'


def test_independent_parser_rejects_damage_and_has_utf8_spans():
    rows = [{'id': 'good', 'latex': r'\frac{x^2}{y}+α=β'}, {'id': 'bad', 'latex': r'\frac{x}{'}]
    p = subprocess.run(['node', str(ROOT/'scripts/latex_reference_v6.js')], input=''.join(json.dumps(r)+'\n' for r in rows), text=True, capture_output=True, check=True)
    good, bad = [json.loads(line) for line in p.stdout.splitlines()]
    assert good['valid'] and not bad['valid']
    assert good['nodes'] > 0
    assert all(0 <= a < b <= len(rows[0]['latex'].encode()) for a, b in good['spans'])


def test_mixed_library_topics_are_not_silently_analysis():
    assert library_topics('pygrim/formulas/gcd.py')[0]=='number_theory'
    assert library_topics('pygrim/formulas/totient.py')[0]=='number_theory'
    assert 'geometry_topology' in library_topics('pygrim/formulas/modular_j.py')[1]
    assert {'math.OA','math.QA','math.SG','math.MG','math.CT','math.KT'}<=TARGET_CATEGORIES


def test_actual_extreme_training_has_only_verified_source_subjects():
    from math_tokenizer.benchmark_v6 import load_split
    for name in ('transfer_train','transfer_dev'):
        rows=load_split(name)
        assert rows
        assert all(r['source_repo']=='arxiv.org' and not set(r['subject_categories'])&TARGET_CATEGORIES for r in rows)
