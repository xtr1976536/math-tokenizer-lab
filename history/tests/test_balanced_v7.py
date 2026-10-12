"""Weighted lattice optimum and terminal-RL score estimator behavior."""
import math
import numpy as np
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.frontend_v6 import tree_ranges
from math_tokenizer.ast import compatible
from math_tokenizer.rl_transfer_v7 import TransferConfig,rollout,features


def test_weighted_native_path_matches_independent_exhaustive_small_lattice():
    vocab=[bytes([b]) for b in range(256)]+[b'x+',b'+y',b'x+y',b'y*z',b'+y*',b'x+y*z']
    text='x+y*z';raw=text.encode();spans=tree_ranges(text).tolist()
    def recurse(a):
        if a==len(raw):return [(0.,[])]
        result=[]
        for id,p in enumerate(vocab):
            if raw[a:a+len(p)]!=p:continue
            end=a+len(p);edge=1+penalty*(not compatible(a,end,spans))
            for cost,ids in recurse(end):result.append((edge+cost,[id]+ids))
        return result
    for penalty in (0.,.25,1.5,4.):
        model=BalancedAST(vocab,penalty);ids=model.encode(text)
        assert abs(model.objective(text,ids)-min(cost for cost,_ in recurse(0)))<1e-10
        assert model.decode(ids)==text


def test_balanced_known_command_mask_and_serialization(tmp_path):
    vocab=[bytes([b]) for b in range(256)]+[b'\\alpha',b'alpha+x',b'\\alpha+x']
    model=BalancedAST(vocab,.75);path=tmp_path/'balanced.json';model.save(path);loaded=BalancedAST.load(path)
    for text in [r'\alpha+x',r'\unknownABC{x}','α+😊=1','']:
        ids=model.encode(text);assert loaded.encode(text)==ids and loaded.decode(ids)==text
    assert b''.join(model.vocab[i] for i in model.encode(r'\alpha+x'))==b'\\alpha+x'


def test_calibrated_risk_native_cost_matches_independent_span_objective():
    vocab=[bytes([b]) for b in range(256)]+[b'x+',b'+y',b'x+y',b'y*z',b'+y*']
    for risk in [[0.]*8,[.1,.4,.9,.3,.2,.8,0.,0.]]:
        model=BalancedAST(vocab,1.5,risk)
        for text in ['x+y*z',r'\frac{x+y}{z}+x']:
            ids,cost=model.encode_with_cost(text)
            assert abs(cost-model.objective(text,ids))<1e-10
            assert model.decode(ids)==text


def test_lexicographic_decoder_never_trades_a_token_for_tree_cost():
    vocab=[bytes([b]) for b in range(256)]+[b'x+',b'+y',b'y*z',b'x+y',b'+y*',b'x+y*z']
    text='x+y*z';model=BalancedAST(vocab,0.,lexicographic=True)
    ids,cost=model.encode_with_cost(text)
    assert len(ids)==1 and cost==model.objective(text,ids)
    without_whole=BalancedAST(vocab[:-1],0.,lexicographic=True)
    ids=without_whole.encode(text)
    assert len(ids)==2
    # Independent exhaustive primary/secondary objective over every byte edge.
    from math_tokenizer.measure_v5 import cross_count
    spans=tree_ranges(text).tolist()
    def search(a):
        if a==len(text):return [(0,0)]
        options=[]
        for piece in vocab[:-1]:
            if text.encode()[a:a+len(piece)]!=piece:continue
            e=a+len(piece);cross=int(not compatible(a,e,spans))
            options.extend((n+1,c+cross) for n,c in search(e))
        return options
    assert (len(ids),without_whole.objective(text,ids))==min(search(0))


def test_fit_only_transfer_rollout_ignores_all_reward_reference_fields():
    atoms=[bytes([b]) for b in range(256)]+[b'\\alpha']
    fit=[{'id':str(i),'latex':r'\alpha+x_i=y','split_group':f'p:{i}'} for i in range(4)]
    config=TransferConfig(training_capacity=265,candidates=4)
    theta=np.arange(24)/100
    a,_,_=rollout(fit,atoms,265,theta,config)
    changed=[dict(r,reference={'spans':[[0,2]]},reward=100000) for r in fit]
    b,_,_=rollout(changed,atoms,265,theta,config)
    assert a.vocab==b.vocab and a.merges==b.merges
    for text in [r'\alpha+x_i=y','😊+x=1']:
        assert a.decode(a.encode(text))==text


def test_terminal_RL_updates_and_does_not_access_dev_or_test_ids():
    from math_tokenizer.rl_transfer_v7 import train_policy
    atoms=[bytes([b]) for b in range(256)]+[b'\\frac']
    fit=[{'id':f'f{i}','latex':r'\frac{x_i^2}{y}+x=y' if i%2 else r'\frac{x}{y_i}+y=x','split_group':f'fit:{i}'} for i in range(8)]
    reward=[{'id':f'r{i}','latex':r'\frac{x_i}{y}+x=x','split_group':f'reward:{i}',
             'reference':{'spans':[[6,9],[11,12]]}} for i in range(6)]
    config=TransferConfig(epochs=2,samples=3,training_capacity=272,fit_task_limit=8,reward_task_limit=6,candidates=4)
    record,_=train_policy(fit,reward,atoms,config,17)
    assert record['policy_change_norm']>0
    assert all(set(h['fit_ids'])<={r['id'] for r in fit} and set(h['reward_ids'])<={r['id'] for r in reward} for h in record['history'])
    assert all(h['updated'] for h in record['history'])
    assert len(record['policy'])==24
