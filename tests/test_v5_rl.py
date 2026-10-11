import json,random
from pathlib import Path
from dataclasses import replace
import pytest,numpy as np
from math_tokenizer.rl_vocab_v5 import RLVocab,VocabRLConfig,internal_split
from math_tokenizer.bpe import MathBPE,Sequence
from math_tokenizer.metrics_v2 import token_bytes

TEXTS=[r'\alpha+\beta=\frac{x}{y}',r'(x+y)^2=x^2+2xy+y^2','α+β=γ']*20
RECORDS=[dict(id=str(i),latex=t,split_group=f'paper{i//6}',domain='analysis') for i,t in enumerate(TEXTS)]

@pytest.mark.parametrize('method',['frequency','random','reward_greedy','rl'])
def test_rl_vocab_byte_coverage_and_official_decode(method,tmp_path):
 config=VocabRLConfig(updates=4,fit_limit=100,reward_limit=30)
 m=RLVocab(config).train(RECORDS,288,17,method);p=tmp_path/'model.json';m.save(p);copy=RLVocab.load(p)
 fit,reward=internal_split(RECORDS,config);assert {r['split_group'] for r in fit}.isdisjoint({r['split_group'] for r in reward})
 assert set(m.fit_ids).isdisjoint(m.reward_ids);assert len(m.vocab)<=288
 for text in TEXTS+['中文🙂+1','\n  x+y \t','']:
  ids=m.encode(text);assert m.decode(ids)==text;assert copy.encode(text)==ids;assert m.decode_bytes(ids)==text.encode()
  assert b''.join(token_bytes(m,id) for id in ids)==text.encode()
 assert all(l['scope']=='training_internal_document_holdout' for l in m.logs)


def test_expected_bandit_score_gradient_finite_difference():
 x=np.array([[1.,0],[0.,1.],[1.,1.]]);theta=np.array([.3,-.2]);reward=np.array([.2,-.1,.4]);q=np.exp(x@theta);q/=q.sum();grad=(q*(reward-q@reward))@x
 finite=[]
 for k in range(2):
  plus=theta.copy();plus[k]+=1e-5;p=np.exp(x@plus);p/=p.sum()
  minus=theta.copy();minus[k]-=1e-5;m=np.exp(x@minus);m/=m.sum();finite.append((p@reward-m@reward)/2e-5)
 assert np.allclose(grad,finite,atol=1e-8)


def test_multimetric_reward_is_not_length_only():
 model=RLVocab(VocabRLConfig(updates=4,fit_limit=100,reward_limit=30)).train(RECORDS,288,17)
 assert any(abs(r['reward']['heldout_code_gain'])>1e-10 for r in model.logs)
 assert any(abs(r['reward']['boundary_cut_reduction'])>1e-10 for r in model.logs)
 assert any(abs(r['reward']['whitespace_gain_consistency'])>1e-10 for r in model.logs)
 assert any(abs(r['reward']['total']-r['reward']['compression_gain'])>1e-10 for r in model.logs)
 assert any(abs(t)>1e-8 for t in model.theta)


def test_streaming_tree_crossing_equivalent():
 from math_tokenizer.measure_v5 import cross_count
 from math_tokenizer.ast import compatible
 rng=random.Random(43)
 for _ in range(200):
  n=30;spans=[]
  for _ in range(20):
   a=rng.randrange(n);b=rng.randrange(a+1,n+1);spans.append((a,b))
  boundaries=sorted({0,n}|{rng.randrange(1,n) for _ in range(15)})
  assert cross_count(boundaries,spans)==sum(not compatible(a,b,spans) for a,b in zip(boundaries,boundaries[1:]))


def test_v5_metrics_match_v2_and_json_serialization(tmp_path):
 from math_tokenizer.measure_v5 import measure as m5
 from math_tokenizer.metrics_v2 import measure as m2
 config=VocabRLConfig(updates=4,fit_limit=100,reward_limit=30)
 model=RLVocab(config).train(RECORDS,288)
 records=[dict(r,domain='analysis',origin='ai_generated',complexity={'class':'simple'}) for r in RECORDS]
 a,rows=m5(model,records,'rl',17,repeats=1);b,_=m2(model,records,'rl',17,repeats=1,upstream=False)
 for key in ['tokens_per_formula','command_integrity_rate','number_integrity_rate','lexeme_preservation_rate','hybrid_tree_crossing_rate','utf8_complete_token_rate']:
  assert abs(a[key]-b[key])<1e-12
 json.dumps(a);json.dumps(rows)


def test_official_deployment_equals_python_ranked_merge_reference():
 config=VocabRLConfig(updates=4,fit_limit=100,reward_limit=30)
 model=RLVocab(config).train(RECORDS,320)
 for text in TEXTS+['x+x=0',r'\unknown{x}+\alpha','中文🙂+1']:
  sequence=Sequence.make(text,False)
  for left,right,new in model.merges:sequence.merge((left,right),new)
  assert sequence.ids==model.encode(text)


def test_rl_has_no_curated_atoms_or_structure_inference():
 cfg=VocabRLConfig(updates=4,fit_limit=100,reward_limit=30)
 m=RLVocab(cfg).train(RECORDS,288)
 assert m.vocab[:256]==[bytes([b]) for b in range(256)]
 assert len(m.vocab)==256+len({c for a,b,c in m.merges if c>=256})
 # Every nonbyte vocabulary entry is learned through a recorded merge action.
 assert all(any(c==i for a,b,c in m.merges) for i in range(256,len(m.vocab)))
 assert not hasattr(m,'command_atoms') and not hasattr(m,'structural')
