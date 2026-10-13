import itertools
import numpy as np
import pytest
from math_tokenizer.signal_stage2 import CONTEXTS,ByteSignals,SignalTokenizer,TokenCodePredictor,contexts
from math_tokenizer.autonomous_stage2 import (PolicyConfig,PatchPredictor,boundary_gradient,
    probabilities,sample_boundaries,reward_to_go)
from math_tokenizer.predictive_stage2 import PredictiveTokenizer
from math_tokenizer.learned_units_stage2 import LearnedUnitBPE,discover_units


def rows(texts):return [{'id':str(i),'latex':s,'split_group':str(i%2)} for i,s in enumerate(texts)]


def test_native_signal_dp_is_global_optimum_and_reference_agrees(tmp_path):
    vocab=[bytes([b]) for b in range(256)]+[b'ab',b'bc',b'abc',b'aa',b'aba','α'.encode()]
    rng=np.random.default_rng(17);model=SignalTokenizer(vocab,rng.normal(0,.3,CONTEXTS))
    def paths(raw,a=0):
        if a==len(raw):return [[]]
        return [[i]+tail for i,p in enumerate(vocab) if raw.startswith(p,a) for tail in paths(raw,a+len(p))]
    for text in ['','abc','ababa','αab','\x00a','\\unknown{😊}',*[''.join(rng.choice(list('abc'),6)) for _ in range(12)]]:
        ids,cost=model.encode_with_cost(text);reference,value=model.encode_reference(text)
        assert model.decode(ids)==text and ids==reference and cost==pytest.approx(value)
        if len(text.encode())<=7:
            def objective(path):
                ends=np.cumsum([len(vocab[i]) for i in path]);cs=contexts(text.encode())
                return len(path)+sum(model.costs[cs[e]] for e in ends if e<len(text.encode()))
            assert cost==pytest.approx(min(map(objective,paths(text.encode()))))
    path=tmp_path/'model.npz';model.save(path);loaded=SignalTokenizer.load(path)
    assert loaded.encode('αabc')==model.encode('αabc')
    with pytest.raises(ValueError):SignalTokenizer(vocab,np.full(CONTEXTS,np.nan))


def test_causal_boundary_and_predictor_do_not_see_future():
    logits=np.linspace(-2,2,CONTEXTS);left=b'abcdXYZ';right=b'abcd123'
    a=sample_boundaries(left,logits,np.random.default_rng(9));b=sample_boundaries(right,logits,np.random.default_rng(9))
    assert np.array_equal(a[0][:5],b[0][:5]) and np.array_equal(a[3][:5],b[3][:5])
    signals=ByteSignals.fit(rows(['abcdef','abcxyz','bcdefa']))
    predictor=PatchPredictor(signals.base,PolicyConfig()).fit(rows(['abcdef','abcxyz']),logits,11)
    leftbits,_=predictor.score(left,a[0]);rightbits,_=predictor.score(right,b[0])
    assert np.array_equal(leftbits[:4],rightbits[:4])


def test_loo_reward_to_go_gradient_matches_exact_enumeration():
    raw=b'abc';cs=contexts(raw);theta=np.zeros(CONTEXTS);theta[cs[1]]=.3;theta[cs[2]]=-.4
    def allpaths(t):
        p=probabilities(t[cs]);result=[]
        for x,y in itertools.product((0,1),repeat=2):
            cut=np.array([1,x,y,1],bool);active=np.array([0,1,1,0],bool)
            # Includes interaction: the first action changes a later reward.
            reward=np.array([0.,.2*x,1.3*y+.7*x*y])
            result.append((p[1]**x*(1-p[1])**(1-x)*p[2]**y*(1-p[2])**(1-y),
                {'contexts':cs,'cuts':cut,'active':active,'p':p,'reward':reward}))
        return result
    expected=np.zeros(CONTEXTS)
    for w1,s1 in allpaths(theta):
        for w2,s2 in allpaths(theta):expected+=w1*w2*boundary_gradient([s1,s2])
    def utility(t):return sum(w*s['reward'].sum()/len(raw) for w,s in allpaths(t))
    for c in cs[1:3]:
        plus=theta.copy();minus=theta.copy();plus[c]+=1e-5;minus[c]-=1e-5
        assert expected[c]==pytest.approx((utility(plus)-utility(minus))/2e-5,abs=1e-7)
    assert np.allclose(reward_to_go([1,2,3]),[6,5,3])


def test_token_code_is_normalized_and_has_eos():
    predictor=TokenCodePredictor([[0,1],[0],[1,0]],256)
    assert np.allclose(np.exp2(-predictor.bits).sum(axis=1),1.)
    rr=rows(['ab','a']);score=predictor.score([[0,1],[0]],rr)
    assert score['bits'][0]==pytest.approx(predictor.bits[predictor.bos,0]+predictor.bits[0,1]+predictor.bits[1,predictor.eos])


def test_unknown_signal_context_is_neutral_and_model_rejects_missing_bytes():
    signal=ByteSignals.fit(rows(['ababab']))
    assert np.array_equal(signal.features[signal.mass==0],np.zeros((int((signal.mass==0).sum()),3)))
    assert len(contexts(b''))==1 and len(contexts(b'a'))==2
    with pytest.raises(ValueError):SignalTokenizer([b'a'])


def test_predictive_native_global_path_and_serialization(tmp_path):
    vocab=[bytes([b]) for b in range(256)]+[b'ab',b'bc',b'abc',b'aa',b'aba']
    code=TokenCodePredictor([[256,ord('c')],[258],[ord('a'),257],[260]],len(vocab))
    model=PredictiveTokenizer(vocab,code.bits,2.,np.random.default_rng(7).normal(0,.1,CONTEXTS))
    for text in ('','abc','ababa','aabc','α😊','\x00abc'):
        ids,cost=model.encode_with_cost(text);ref,value=model.encode_reference(text)
        assert cost==pytest.approx(value) and model.decode(ids)==text
        x=[len(vocab)+1]+ids+[len(vocab)]
        expected=sum(code.bits[a,b] for a,b in zip(x,x[1:]))+2*len(ids)
        pos=0;cs=contexts(text.encode())
        for id in ids:
            pos+=len(vocab[id])
            if pos<len(text.encode()):expected+=model.costs[cs[pos]]
        assert cost==pytest.approx(expected)
    path=tmp_path/'model.npz';model.save(path);loaded=PredictiveTokenizer.load(path)
    assert loaded.encode('abc')==model.encode('abc')


def test_data_discovered_units_are_lossless_without_command_rules(tmp_path):
    fit=rows(['abc+def','abc+def','abc+xyz','α+abc'])
    signals=ByteSignals.fit(fit)
    atoms,info=discover_units(fit,signals.features[:,0],signals.mass,300)
    assert len(atoms)<=300 and info['fit_ids']==[r['id'] for r in fit]
    model=LearnedUnitBPE.train(fit,atoms,300)
    assert len(model.vocab)<=300
    for text in ('','abc+def',r'\NovelCommand{α😊}','\x00abc'):
        assert model.decode(model.encode(text))==text
    path=tmp_path/'model.json';model.save(path);loaded=LearnedUnitBPE.load(path)
    assert loaded.encode(r'\NovelCommand{α😊}')==model.encode(r'\NovelCommand{α😊}')
