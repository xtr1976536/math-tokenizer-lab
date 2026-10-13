import itertools
import numpy as np
import pytest
pytest.importorskip('torch')
from math_tokenizer.joint_boundary_stage2 import ContextActor,actor_features,actor_gradient,expected_rate,calibrate


def test_long_actor_prefix_causality_and_save_load(tmp_path):
    actor=ContextActor(4,np.random.default_rng(3).normal(size=(4,257)))
    a=b'abcdXYZ';b=b'abcd123'
    assert np.array_equal(actor_features(a)[:5],actor_features(b)[:5])
    assert np.array_equal(actor.logits(a)[:5],actor.logits(b)[:5])
    assert np.array_equal(actor.sample(a,np.random.default_rng(1))[0][:5],actor.sample(b,np.random.default_rng(1))[0][:5])
    path=tmp_path/'actor.npz';actor.save(path);loaded=ContextActor.load(path)
    assert np.array_equal(actor.logits(a),loaded.logits(a))


def test_long_actor_gradient_matches_enumerated_expectation():
    actor=ContextActor(2);actor.weights[1,ord('a')]=.3;raw=b'abc'
    def paths(policy):
        p=1/(1+np.exp(-policy.logits(raw)));result=[]
        for x,y in itertools.product((0,1),repeat=2):
            probability=p[1]**x*(1-p[1])**(1-x)*p[2]**y*(1-p[2])**(1-y)
            result.append((probability,{'cut':np.array([1,x,y,1],bool),'active':np.array([0,1,1,0],bool),'p':p,
                'reward':np.array([0.,.2*x,1.3*y+.7*x*y])}))
        return result
    gradient=np.zeros_like(actor.weights)
    for w,a in paths(actor):
        for v,b in paths(actor):gradient+=w*v*actor_gradient(raw,[a,b],actor)
    def objective(policy):return sum(w*r['reward'].sum()/len(raw) for w,r in paths(policy))
    for position,byte in ((1,97),(1,98),(0,256),(0,97)):
        initial=actor.weights[position,byte];actor.weights[position,byte]=initial+1e-5;plus=objective(actor)
        actor.weights[position,byte]=initial-1e-5;minus=objective(actor);actor.weights[position,byte]=initial
        assert gradient[position,byte]==pytest.approx((plus-minus)/2e-5,abs=1e-7)


def test_expected_rate_calibration_uses_only_past_and_handles_cap():
    actor=ContextActor(4);rows=[{'latex':'abcdef'},{'latex':'xyz'}]
    calibrated=calibrate(rows,actor)
    assert expected_rate(rows,calibrated)==pytest.approx(.4,abs=1e-6)
    assert expected_rate([{'latex':'abc'}],actor,2)==pytest.approx(.58)
