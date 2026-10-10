import itertools,json,random
from pathlib import Path
import pytest
from math_tokenizer.ast import compatible
from math_tokenizer.structure_v2 import structure,complexity
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.bpe import MathBPE
from math_tokenizer.rl_v2 import LatticeTokenizer,RLConfig
from math_tokenizer.metrics_v2 import token_bytes,measure
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer,TikToken,ByteTokenizer
from math_tokenizer.corpus_v2 import tex_members,load_simple_macros,expand_aliases

TEXTS=[r'\frac{\partial u}{\partial t}=\sum_{i=1}^{n}u_i',r'\int_0^1 f(x) dx=1',r'x\in\mathbb{R}',r'\left|x\right|<1',r'(x+y)^2=x^2+2xy+y^2','α+β=γ']*8

def test_structure_levels_and_no_fake_ast():
    assert structure('x+y=z').kind=='semantic_ast'
    assert structure(r'\sum_{i=1}^n x_i=1').kind=='latex_syntax_tree'
    assert structure(r'\input{evil}').kind=='byte_fallback'
    assert structure(r'\sqrt{').kind=='byte_fallback'


def test_complexity_fixed_rule():
    assert complexity('x+1')['class']=='simple'
    assert complexity(r'\frac{\sum_{i=0}^{n} \frac{\partial^{2} u_i}{\partial x^{2}}}{\sqrt{1+x^{2}}}'+ '+x'*60)['class']=='complex'
    assert complexity('x+1')==complexity('x+1')


def test_incremental_trainer_vs_scan():
    m=FastBPE().train(TEXTS,320)
    slow=MathBPE().train(TEXTS,320)
    assert m.merges==slow.merges
    for text in TEXTS:assert m.encode(text)==slow.encode(text)


def test_hybrid_optimized_and_fuzz():
    rng=random.Random(33)
    texts=TEXTS+[''.join(rng.choice('ab12+{} α') for _ in range(rng.randint(0,70))) for _ in range(80)]
    model=FastBPE(True).train(texts,320)
    for text in texts:
        ids=model.encode(text);assert ids==model.encode_reference(text);assert model.decode(ids)==text
        s=structure(text);offset=0
        if s.kind!='byte_fallback':
            for id in ids:
                end=offset+len(model.vocab[id]);assert compatible(offset,end,s.spans);offset=end


def exhaustive_min_tokens(model,text):
    graph=model.graph(text)
    def search(pos):
        if pos==len(graph):return 0
        return min(1+search(int(end)) for end in graph[pos][0])
    return search(0)


def test_lattice_dp_bruteforce():
    base=FastBPE().train(['abcabc','abac','abab'],280)
    m=LatticeTokenizer(base)
    for text in ['abac','abcab','abab','aaaa']:
        ids=m.encode(text);assert len(ids)==exhaustive_min_tokens(m,text);assert m.decode(ids)==text

@pytest.mark.parametrize('estimator,gamma',[('naive',1.),('relative',1.),('rloo',1.),('rloo',.99)])
def test_rl_determinism_and_save(estimator,gamma,tmp_path):
    base=FastBPE().train(TEXTS,288)
    config=RLConfig(estimator=estimator,gamma=gamma,epochs=2,max_train_formulas=16,samples=3)
    a=LatticeTokenizer(base,mode='policy_map',config=config).train_policy(TEXTS,17)
    b=LatticeTokenizer(base,mode='policy_map',config=config).train_policy(TEXTS,17)
    assert a.theta.tolist()==b.theta.tolist()
    assert abs(a.theta[0]-config.initial_length_prior)>1e-6
    a.save(tmp_path/'m.json');c=LatticeTokenizer.load(tmp_path/'m.json')
    for text in TEXTS+['中文+🙂=2','']:
        assert a.decode(a.encode(text))==text
        assert c.encode(text)==a.encode(text)
    assert all(x['reward_scope']=='train_only' for x in a.training_log)

@pytest.mark.parametrize('kind',['bytes','hf','sp_bpe','sp_unigram','gpt2','cl100k_base','o200k_base'])
def test_byte_payload_mapping_and_common_metrics(kind,tmp_path):
    if kind=='bytes':m=ByteTokenizer()
    elif kind=='hf':m=HFTokenizer.train(TEXTS,384,tmp_path/'hf.json')
    elif kind.startswith('sp_'):m=SentencePieceTokenizer.train(TEXTS,384,kind[3:],tmp_path/kind)
    else:m=TikToken(kind)
    for text in TEXTS+['  中文🙂 + x\n','αβγ1234567']:
        assert b''.join(token_bytes(m,id) for id in m.encode(text))==text.encode()
    records=[dict(id=str(i),latex=text,split_group=f'g{i//2}',domain='pde',origin='human',complexity={'class':'simple'}) for i,text in enumerate(TEXTS)]
    summary,rows=measure(m,records,kind,0,upstream=False)
    assert summary['roundtrip_rate']==1 and 0<summary['renyi_efficiency']['2.0']<=1
    assert len(rows)==len(TEXTS)


def test_safe_zero_argument_alias_expansion():
    definitions=r'\newcommand{\R}{\mathbb{R}}\newcommand{\al}{\alpha}\def\be{\beta}\newcommand{\x}[1]{#1}'
    macros=load_simple_macros(definitions)
    assert macros=={r'\al':r'\alpha',r'\be':r'\beta'}
    assert expand_aliases(r'\al+\be=\x',macros)==r'\alpha+\beta=\x'

@pytest.mark.parametrize('text',[r'\begin{pmatrix}a&b\\c&d\end{pmatrix}^{2}=A',r'\operatorname{rank}(A)=2',r'\operatorname{Var}(X)=1'])
def test_complete_matrix_and_known_operators(text):
    from math_tokenizer.validation_v2 import validate
    assert validate(text) is None
    assert structure(text).kind=='latex_syntax_tree'

@pytest.mark.parametrize('text',[r'\begin{verbatim}evil\end{verbatim}',r'\operatorname{where}(x)=1',r'x=1\\y=2'])
def test_v2_prose_and_unknown_env_reject(text):
    from math_tokenizer.validation_v2 import validate
    assert validate(text) is not None


def test_reinforce_score_gradient_matches_exact_expectation():
    import numpy as np
    base=FastBPE().train(['ab']*10,257)
    model=LatticeTokenizer(base,mode='policy_map',config=RLConfig(initial_length_prior=.4))
    graph=model.graph('ab');ends,ids,x,cost=graph[0]
    p=model.policy(x)
    # Here actions terminate in one token or force the one-byte second token.
    rewards=-np.array([1 if end==2 else 2 for end in ends],dtype=float)
    exact=np.sum(p[:,None]*rewards[:,None]*(x-p@x),axis=0)
    original=model.theta.copy();eps=1e-5;finite=[]
    for j in range(len(original)):
        model.theta=original.copy();model.theta[j]+=eps;plus=model.policy(x)@rewards
        model.theta=original.copy();model.theta[j]-=eps;minus=model.policy(x)@rewards
        finite.append((plus-minus)/(2*eps))
    model.theta=original
    assert np.allclose(exact,finite,atol=1e-8)
    rng=np.random.default_rng(23);estimates=[]
    for _ in range(2000):
        paths=[model.sample(graph,rng) for _ in range(4)]
        returns=np.array([-sum(item[3] for item in path) for path in paths])
        estimate=np.zeros_like(original)
        for k,path in enumerate(paths):
            baseline=(returns.sum()-returns[k])/3
            estimate+=(returns[k]-baseline)*path[0][4]/4
        estimates.append(estimate)
    assert np.allclose(np.mean(estimates,axis=0),exact,atol=.015)

@pytest.mark.parametrize('structural,rl',[(False,False),(True,False),(False,True),(True,True)])
def test_legacy_incremental_exact_semantics(structural,rl):
    from math_tokenizer.legacy_fast import LegacyFastBPE,gains
    from math_tokenizer.bpe import Sequence
    texts=TEXTS+['aaaaa','aaaabaaaa','abababa']*4
    a=LegacyFastBPE(structural,rl).train(texts,288,17)
    b=MathBPE(structural,rl).train(texts,288,17)
    assert a.merges==b.merges
    assert a.policy.tolist()==b.policy.tolist()
    if rl:assert a.training_log==b.training_log
    for text in texts:
        seq=Sequence.make(text,structural);gg=gains(seq)
        for pair in seq.counts():assert gg[pair]==seq.gain(pair)


def test_official_renyi_crosscheck_and_no_cross_formula_pairs():
    from math_tokenizer.metrics_v2 import upstream_information
    import numpy as np
    result=upstream_information('toy',[[0,1,1],[0,1]],['x+x','x+x'],256)
    official=result['renyi_efficiency']['per_tokenizer']['toy']['renyi_2.0']['overall']
    expected=-np.log2((2/5)**2+(3/5)**2)/np.log2(256)
    assert abs(official-expected)<1e-12
