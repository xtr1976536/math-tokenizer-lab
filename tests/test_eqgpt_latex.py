"""Four-audit evidence: representation, budgets, no-SAT generation, numerical bridge."""
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import pytest
import torch
from math_tokenizer.eqgpt.author import CODE,DICTIONARY,ROOT
from math_tokenizer.eqgpt.latex_codec import *
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter,train_tokenizers,select_capacity,load_tokenizers
from math_tokenizer.eqgpt.latex_generation import generate
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator,select_top,merge_top
from math_tokenizer.eqgpt.author import numerical_namespace


def test_all_atoms_and_source_formulas_roundtrip_mathematically():
    for i,node in ATOM_TREES.items():assert latex_to_structure(render(node)+'=0')==[5,i,1]
    frame=pd.read_excel(CODE/'PDE_dataset_0725.xlsx');d=DICTIONARY['word2id']
    for raw in frame['Terms']:
        seq=[5]+[d[w] for w in raw.split(',')]+[1]
        text=structure_to_latex(seq);parsed=latex_to_structure(text)
        assert key(LatexParser(text).parse())==key(LatexParser(structure_to_latex(parsed)).parse())


def test_parser_accepts_math_equivalence_not_target_and_rejects_bad_or_unsupported():
    assert latex_to_structure(r'u_{xx}+u_{t}=0')==[5,9,2,6,1]
    assert latex_to_structure(r'u_{x^2}+u_t=0')==[5,9,2,6,1]
    assert latex_to_structure(r'\partial_{x}{(u\cdot u_x)}+u_t=0')==[5,16,2,6,1]
    for text in [r'\frac{u}{',r'u_x+=0','',r'u_t}=0']:
        with pytest.raises(LatexError):latex_to_structure(text)
    for text in [r'\cos{u}=0',r'u_{xxxxxx}=0',r'u_t=1',r'\input{secret}']:
        with pytest.raises(LatexError):latex_to_structure(text)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    from math_tokenizer.eqgpt.data import augment_training
    p=tmp_path_factory.mktemp('latex_tokenizers');s,_=augment_training('KdV equation',p)
    texts=[structure_to_latex(x) for x in s]
    adapters,reports=train_tokenizers(texts,p/'fixed')
    return texts,adapters,reports,p


def test_fixed_total_ids_whole_formula_and_roundtrip_reloaded(trained):
    texts,adapters,reports,p=trained;loaded=load_tokenizers(p/'fixed')
    assert len({r['training_strings_sha256'] for r in reports.values()})==1
    for n,a in adapters.items():
        assert a.vocab_size==1024 and a.pad==0 and a.eos==1 and a.bos==2
        for t in sorted(set(texts)):
            ids=a.sequence(t)
            assert loaded[n].sequence(t)==ids
            assert a.decode(ids[1:-1])==t
            assert a.bytes(ids[1:-1])==t.encode()


def test_auto_selection_train_only_groups_actual_size_and_mdl_choice(trained):
    texts,adapters,reports,p=trained
    for n in adapters:
        selected,r=select_capacity(n,texts,candidates=(384,512))
        assert selected==min(r['measurements'],key=lambda m:(m['mdl_bits'],m['actual_total_ids']))['requested_total_ids']
        assert r['fit_structures']+r['heldout_structures']>100
        assert r['fit_sha256']!=r['heldout_sha256']


class TinyPayload:
    vocab=[b'x',b'\xff'];capacity=2
    def encode(self,t):return [0]*len(t)
    def decode(self,ids):return b''.join(self.vocab[i] for i in ids).decode()

class TinyModel(torch.nn.Module):
    def __init__(self,chosen=3):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.decoder=type('D',(),{'pos_emb':type('P',(),{'num_embeddings':32})()})();self.chosen=chosen
    def forward(self,x):
        logits=torch.full((x.numel(),5),-1000.)
        logits[:,self.chosen]=1000.;return logits,None


def test_exact400_attempts_no_semantic_or_control_masks():
    a=LatexAdapter('bpe',TinyPayload())
    for selected,reason in [(1,'eos'),(0,'unexpected_control'),(2,'unexpected_control'),(3,'token_limit'),(4,'token_limit')]:
        model=TinyModel(selected)
        r=generate(model,a,attempts=400,max_tokens=4,batch_size=16)
        assert len(r)==400 and [x['attempt'] for x in r]==list(range(400))
        assert all(x['stop_reason']==reason for x in r)
        assert all(x['ids'][0]==selected for x in r)
        if selected==4:assert all(x.get('decode_error') for x in r)


def test_unconstrained_sampling_repeat_and_byte_budget():
    a=LatexAdapter('soft_ast_bpe',TinyPayload());m=TinyModel(3)
    x=generate(m,a,400,16,max_bytes=3);y=generate(m,a,400,16,max_bytes=3)
    assert x==y and all(r['stop_reason']=='byte_limit' and len(r['ids'])==3 for r in x)


def test_reward_bridge_actual_author_derivatives_and_variable_check_after_generation():
    class Analytic(torch.nn.Module):
        def forward(self,xt):return torch.exp(-2*xt[:,1:2])*torch.sin(xt[:,0:1])
    grid=torch.stack([torch.linspace(.2,1,16),torch.linspace(.1,.8,16)],1).requires_grad_()
    e=RewardEvaluator(Analytic(),grid,'KdV_equation',['x','t'],'cpu',.2)
    a=e.evaluate(dict(stopped=True,text=r'u_t+u_{xx}=0'))
    assert a['status']=='evaluated' and abs(a['reward']-(1-.2*np.log10(2)))<1e-6
    assert e.evaluate(dict(stopped=True,text=r'u_{yy}+u_t=0'))['status']=='variable_or_numeric_unavailable'
    assert e.evaluate(dict(stopped=False,text=r'u_t+u_{xx}=0'))['status']=='incomplete'
    assert e.evaluate(dict(stopped=True,text=r'u_t+=0'))['status']!='evaluated'


def test_top10_matches_author_and_no_invalid_refill():
    env=numerical_namespace('cpu')
    records=[dict(reward=float(i%11),status='evaluated',text=str(i)) for i in range(400)]
    scores=torch.tensor([r['reward'] for r in records]);idx,_=env['find_min_no_repeat'](scores,samples=400)
    assert select_top(records,env)==[records[i] for i in idx]
    assert select_top([dict(status='syntax_error',reward=None)]*400,env)==[]
    first=select_top(records,env);second=[dict(reward=20.,status='evaluated',text='new')]
    merged=merge_top(first,second)
    assert len(merged)==len(first)==10 and merged[0]['reward']==20.


def test_generation_never_calls_author_sat_interface(trained):
    texts,adapters,_,_=trained
    class Static(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.decoder=type('D',(),{'pos_emb':type('P',(),{'num_embeddings':32})()})()
        def forward(self,x):
            out=torch.full((x.numel(),1024),-1000.);out[:,1]=1000.;return out,None
        def step(self,*a):raise AssertionError('SAT must not run')
    with patch('math_tokenizer.eqgpt.models.valid_actions',side_effect=AssertionError('SAT must not run')):
        r=generate(Static(),adapters['ast_bpe'],400,4)
        assert len(r)==400


def test_fraction_denominator_grouping_is_preserved_for_numerical_evaluator():
    assert latex_to_structure(r'\frac{x}{u\cdot u}=0')==[5,22,4,7,4,7,1]
    assert latex_to_structure(r'\frac{x}{\frac{u}{y}}=0')==[5,22,4,7,3,34,1]


def test_no_evaluable_candidates_skip_updates_without_refill(tmp_path):
    from math_tokenizer.eqgpt.latex_experiment import search_latex
    a=LatexAdapter('bpe',TinyPayload());model=TinyModel(3)
    class Unused(torch.nn.Module):
        def forward(self,x):raise AssertionError('Incomplete output must not be evaluated')
    grid=torch.zeros(2,2,requires_grad=True)
    before=model.weight.detach().clone()
    history=search_latex(model,a,Unused(),grid,'KdV_equation',['x','t'],tmp_path,'cpu',525,
        attempts=400,cycles=2,finetune_epochs=5,max_tokens=4)
    report=json.loads((tmp_path/'search.json').read_text())
    assert len(report['attempts'])==800
    assert all(r['continuation_examples']==0 and r['continuation_epochs']==0 for r in history)
    assert all(r['before_sha256']==r['after_sha256'] for r in history)
    assert torch.equal(before,model.weight)
