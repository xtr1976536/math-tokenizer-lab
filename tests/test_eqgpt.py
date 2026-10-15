"""Numerical and protocol audits for the author-code comparison."""
import ast
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch

from math_tokenizer.eqgpt.author import (ARCHITECTURE,CODE,DICTIONARY,definitions,
    namespace,model_namespace,seed_all,tensor_hash,verify_source,numerical_namespace)
from math_tokenizer.eqgpt.data import augment_training,canonical,case_environment
from math_tokenizer.eqgpt.models import action_log_scores,build_models,valid_actions
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter,train_adapters,common_positions

@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    folder=tmp_path_factory.mktemp('eqgpt')
    seqs,info=augment_training('KdV equation',folder,525)
    adapters=train_adapters(seqs,folder/'tokenizers')
    return seqs,info,adapters


def test_original_sources_and_architecture():
    p=verify_source();assert p['revision']=='dc35fc0ac860ff224293f01d2a62204042095127'
    # Verify our documented constants against the ORIGINAL source assignments.
    tree=ast.parse((CODE/'gpt_model.py').read_text())
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
            for t in node.targets:
                if isinstance(t,ast.Name):values[t.id]=node.value.value
    for key,val in ARCHITECTURE.items():assert values[key]==val
    assert ARCHITECTURE['n_layers']==6 and ARCHITECTURE['d_ff']==2048


def test_training_target_removed_augmentation_same_and_deterministic(prepared,tmp_path):
    seqs,info,adapters=prepared
    other,other_info=augment_training('KdV equation',tmp_path,525)
    assert seqs==other and info==other_info
    assert info['training_sequences']==32*info['training_formulas']
    target=canonical(['ut','+','uxxx','+','u','*','ux'])
    assert all(canonical([DICTIONARY['id2word'][i] for i in seq])!=target for seq in seqs)
    assert info['author_augmentation_interior_eos']>0
    # Every arm sees the same entire sequence, including original EOS placement.
    for name,adapter in adapters.items():
        for sequence in seqs:
            assert adapter.decode_sequence(adapter.sequence(sequence))==sequence


def test_action_space_constraints_and_vocab_capacity(prepared):
    _,_,adapters=prepared
    for name,a in adapters.items():
        assert a.vocab_size==(57 if name=='eqgpt' else 1024)
        assert set(a.codewords)==set(range(1,57))-{5}
    E=numerical_namespace('cpu');mask=E['get_mask_invalid'](['x','t'])
    assert valid_actions([5],mask)==[i for i in range(6,57) if mask[i]!=0]
    assert valid_actions([5,6],mask)==[1,2,3,4]
    assert DICTIONARY['word2id']['uy'] not in valid_actions([5],mask)


def test_codeword_probability_matches_bruteforce_and_single_token_author():
    class Simple(torch.nn.Module):
        def forward(self,x):
            logits=torch.stack([torch.arange(9,dtype=torch.float32)+(int(sum(row))%3)*torch.arange(9).float()/10 for row in x])
            return logits[:,None,:].expand(-1,x.shape[1],-1).reshape(-1,9),None
    core=Simple();prefix=[2,7];codes={6:[3,4],7:[3,5],8:[6],9:[3,7,8]}
    scores=action_log_scores(core,prefix,codes,list(codes),'cpu')
    for action,code in codes.items():
        seq=prefix.copy();v=0.
        for t in code:
            logits,_=core(torch.tensor([seq]));v+=float(logits[-1].log_softmax(-1)[t]);seq.append(t)
        assert abs(float(scores[action])-v)<2e-6
    one={i:[i] for i in range(1,9)}
    probs=action_log_scores(core,prefix,one,list(one),'cpu')[1:9].softmax(-1)
    logits,_=core(torch.tensor([prefix]));reference=logits[-1,1:9].softmax(-1)
    assert torch.allclose(probs,reference,atol=1e-7)


def test_training_gradients_output_dimensions_and_shared_initialization(prepared):
    _,_,adapters=prepared;torch.set_num_threads(2)
    models,metadata=build_models(adapters,common_positions(adapters),525,'cpu')
    assert len({v['backbone_initial_sha256'] for v in metadata.values()})==1
    for name,m in models.items():
        a=adapters[name];m.train()
        sequence=a.sequence([5,6,2,9,1]);x=torch.tensor([sequence[:-1]])
        y=torch.tensor(sequence[1:]);before=m.core.decoder.layers[0].dec_self_attn.W_Q.weight.detach().clone()
        opt=torch.optim.Adam(m.parameters(),lr=1e-4)
        logits,_=m(x);loss=torch.nn.functional.cross_entropy(logits,y,ignore_index=0)
        assert logits.shape==(len(sequence)-1,a.vocab_size)
        loss.backward();opt.step();assert not torch.equal(before,m.core.decoder.layers[0].dec_self_attn.W_Q.weight)
        assert all(p.requires_grad for p in m.parameters())


def test_author_optimizer_binding_failure_and_common_repair():
    source=(CODE/'continue_train_GPT.py').read_text()
    assert 'optimizer = optim.Adam(model_Q.parameters(), lr=1e-5)' in source
    assert 'train_loss = train_step(model, data_loader, optimizer' in source
    # Reproduce the published issue independently, then demonstrate repair.
    a=torch.nn.Linear(1,1);b=torch.nn.Linear(1,1);x=torch.ones(1,1)
    opt=torch.optim.Adam(a.parameters());before=b.weight.detach().clone()
    b(x).sum().backward();opt.step();assert torch.equal(before,b.weight)
    opt=torch.optim.Adam(b.parameters());opt.zero_grad();b(x).sum().backward();opt.step()
    assert not torch.equal(before,b.weight)


def test_author_reward_and_derivatives_on_analytic_solution():
    class Analytic(torch.nn.Module):
        def forward(self,xt):
            x,t=xt[:,0:1],xt[:,1:2]
            # Diffusion solution: u_t=2u_xx. Distinct columns avoid the author's
            # exact duplicate-column deletion, which is intentionally retained.
            return torch.exp(-2*t)*torch.sin(x)
    E=numerical_namespace('cpu');grid=torch.stack([torch.linspace(.2,1,16),torch.linspace(.1,.8,16)],1).requires_grad_()
    net=Analytic();ut=E['calculate_terms']('ut',net,grid,['x','t']);uxx=E['calculate_terms']('uxx',net,grid,['x','t'])
    assert np.allclose(ut.reshape(-1),2*uxx.reshape(-1),rtol=1e-5,atol=1e-6)
    sentence=[6,2,9,1];mask=E['get_mask_invalid'](['x','t'])
    reward,_,_=E['calculate_reward'](sentence,net,grid,{},mask,['x','t'],.2)
    assert abs(reward-(1-.2*np.log10(2)))<1e-6
    coefficients=E['get_coefficients']([6,2,9,1],net,grid,['x','t'],0)
    assert np.allclose(coefficients,[2],atol=1e-5)


def test_cases_resolve_actual_author_data():
    for name in ['KdV_equation','Burgers_equation','Wave_equation','Eq_6_2_12','Possion_x_y']:
        env=case_environment(name,32,0,'cpu',525)
        assert np.asarray(env['un']).size>0 and env['Delete_equation_name']


def test_loaded_author_model_matches_whole_original_source_forward_and_sampling():
    from math_tokenizer.eqgpt.data import working_directory
    original={'__name__':'author_direct_test'}
    with working_directory(CODE):
        exec(compile((CODE/'gpt_model.py').read_text(),str(CODE/'gpt_model.py'),'exec'),original)
    original['device']=torch.device('cpu');seed_all(525)
    reference=original['GPT']()
    loaded=model_namespace(57,50,'cpu')['GPT']();loaded.load_state_dict(reference.state_dict())
    data=torch.tensor([[5,6,2,9],[5,7,3,8]])
    a,_=reference(data);b,_=loaded(data);assert torch.equal(a,b)
    mask=numerical_namespace('cpu')['get_mask_invalid'](['x','t'])
    seed_all(14);x,p=reference.step([5],mask)
    seed_all(14);y,q=loaded.step([5],mask)
    assert int(x)==int(y) and torch.equal(p,q)


def test_recovery_classifier_correct_redundant_incorrect_and_laplacian():
    from math_tokenizer.eqgpt.evaluation import classify
    d=DICTIONARY['word2id']
    def ids(words):return [5]+[d[w] for w in words.split(',')]+[1]
    assert classify(ids('ut,+,u,*,ux,+,uxxx'),'KdV_equation')['status']=='correct'
    assert classify(ids('ut,+,u,*,ux,+,uxxx,+,u'),'KdV_equation')['status']=='redundant'
    assert classify(ids('ut,+,uxx'),'KdV_equation')['status']=='incorrect'
    assert classify(ids('ut,+,u,*,ux,+,u,*,uy,+,Laplace(u)'),'Burgers_2D')['status']=='correct'


def test_coefficient_alignment_permutation_scale_and_mismatch():
    from math_tokenizer.eqgpt.evaluation import coefficient_error
    d=DICTIONARY['word2id'];seq=[5]+[d[w] for w in 'uxxx,+,u,*,ux,+,ut'.split(',')]+[1]
    # uxxx = -400 u ux -400 ut has the same scale as true KdV.
    assert coefficient_error(seq,[-400.,-400.],'KdV_equation')<1e-12
    assert coefficient_error([5,6,2,9,1],[-1.],'KdV_equation') is None
