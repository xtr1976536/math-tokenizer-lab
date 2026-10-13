import numpy as np
import pytest
torch=pytest.importorskip('torch')
from math_tokenizer.neural_stage2 import NeuralConfig,NeuralPatchPredictor,patch_features
from math_tokenizer.autonomous_stage2 import PolicyConfig,sample_boundaries
from math_tokenizer.signal_stage2 import CONTEXTS


def test_patch_features_and_neural_score_are_causal(tmp_path):
    torch.manual_seed(17);model=NeuralPatchPredictor(NeuralConfig(steps=2),PolicyConfig())
    logits=np.full(CONTEXTS,-.5);left=b'abcdXYZ';right=b'abcd123'
    cut=sample_boundaries(left,logits,np.random.default_rng(7))[0]
    a,e=patch_features(left,cut);b,f=patch_features(right,cut)
    assert np.array_equal(a[:5],b[:5]) and np.array_equal(e[:5],f[:5])
    aa,_=model.score(left,cut);bb,_=model.score(right,cut)
    assert np.array_equal(aa[:4],bb[:4])
    assert a.shape==(7,12) and np.all(a[0]==256)


def test_neural_predictor_training_normalization_serialization(tmp_path):
    rows=[{'id':str(i),'latex':s} for i,s in enumerate(['abcdef','abcxyz','α+x'])]
    model=NeuralPatchPredictor.fit(rows,np.full(CONTEXTS,-.5),17,NeuralConfig(steps=2,batch_targets=16))
    raw=b'abc';cuts=np.array([1,0,1,1],bool);before=model.score(raw,cuts)
    path=tmp_path/'teacher.pt';model.save(path);loaded=NeuralPatchPredictor.load(path)
    assert np.array_equal(before[0],loaded.score(raw,cuts)[0])
    x,e=patch_features(raw,cuts)
    with torch.no_grad():assert torch.allclose(torch.softmax(model.late(torch.from_numpy(x)),dim=-1).sum(dim=-1),torch.ones(len(raw)))
    assert model.memory_summary()['teacher_byte_targets']==32
