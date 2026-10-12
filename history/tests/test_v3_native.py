import json,random
from pathlib import Path
import pytest,numpy as np
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.rl_v2 import LatticeTokenizer,RLConfig
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.structure_v2 import structure
from math_tokenizer.ast import compatible
from math_tokenizer.vocab_v3 import VocabConfig,refine,optimize_swaps

TEXTS=[r'\frac{\partial u}{\partial t}=\sum_{i=1}^{n}u_i',r'\int_0^1 f(x) dx=1',r'x\in\mathbb{R}',r'\left|x\right|<1',r'(x+y)^2=x^2+2xy+y^2','α+β=γ']*4

@pytest.mark.parametrize('structural',[False,True])
@pytest.mark.parametrize('mode',['shortest','policy_map'])
def test_native_exact_python_policy(structural,mode,tmp_path):
 base=FastBPE(structural).train(TEXTS,320)
 slow=LatticeTokenizer(base,structural,mode,RLConfig(epochs=2,max_train_formulas=16)).train_policy(TEXTS)
 native=NativeTokenizer.from_v2(slow)
 rng=random.Random(17)
 fuzz=[''.join(rng.choice('ab{}+^_12 α') for _ in range(rng.randrange(0,70))) for _ in range(100)]
 for text in TEXTS+fuzz+['中文🙂+1','',r'\input{no}']:
  assert native.encode(text)==slow.encode(text)
  assert native.decode(native.encode(text))==text
 native.save(tmp_path/'m.json');restored=NativeTokenizer.load(tmp_path/'m.json')
 for text in TEXTS:assert restored.encode(text)==native.encode(text)


def test_native_matches_and_atomic_structure():
 base=AtomicASTBPE(True).train(TEXTS,320,command_budget=32)
 native=base.decoder()
 for text in TEXTS:
  raw=text.encode();starts,ends,ids=native.matches(text)
  for a,b,id in zip(starts,ends,ids):assert raw[a:b]==native.vocab[id]
  s=structure(text);offset=0
  for id in native.encode(text):
   end=offset+len(native.vocab[id]);assert compatible(offset,end,s.spans);offset=end
  assert native.decode(native.encode(text))==text
 assert len(native.vocab)<=320
 assert all(t in native.vocab for t in base.command_atoms)


def test_refinement_and_real_rl_swap_scope():
 records=[dict(id=str(i),latex=text,split_group=f'g{i//4}') for i,text in enumerate(TEXTS)]
 config=VocabConfig(seed_multiplier=2,pruning_rounds=2,policy_rounds=2,policy_samples=4,reward_formulas=8)
 model,original,mandatory,fit,reward=refine(records,280,config)
 assert {r['split_group'] for r in fit}.isdisjoint({r['split_group'] for r in reward})
 result=optimize_swaps(model,original,mandatory,fit,reward,config)
 assert len(result.vocab)<=280
 assert len(result.vocab)==len(set(result.vocab))
 for text in TEXTS:assert result.decode(result.encode(text))==text
 logs=result.training_log[-1]['logs']
 assert all(l['reward_scope']=='train_internal_document_holdout' for l in logs)
 costs=[l['incumbent_train_reward_cost'] for l in logs]
 assert costs==sorted(costs,reverse=True)


def test_native_crossmask_matches_reference_on_arbitrary_spans():
 rng=random.Random(812)
 for _ in range(60):
  raw=b'abcdefghij';spans=[]
  for j in range(8):
   a=rng.randrange(len(raw));b=rng.randrange(a+1,len(raw)+1);spans.append((a,b))
  a=rng.randrange(len(raw)-1);b=rng.randrange(a+2,len(raw)+1)
  vocab=[bytes([k]) for k in range(256)]+[raw[a:b]]
  model=NativeTokenizer(vocab,True)
  supplied=np.array(spans,dtype=np.int32).reshape(-1,2);safe=np.ones(len(raw)+1,dtype=np.uint8);interior=np.zeros(len(raw)+1,dtype=np.uint8)
  encoded=model.encode_prepared(raw,supplied,safe,interior)
  assert (256 in encoded)==compatible(a,b,tuple(spans))
  assert model.decode_bytes(encoded)==raw

@pytest.mark.parametrize('kwargs',[{'theta':[0.]*8},{'frequency':[1.]},{'frequency':[-1.]*256},{'mode':'unsupported'},{'token_cost':[0.]}])
def test_native_invalid_model_tables_rejected(kwargs):
 with pytest.raises(ValueError):NativeTokenizer([bytes([b]) for b in range(256)],**kwargs)


def test_native_missing_fallback_or_duplicate_rejected():
 with pytest.raises(ValueError):NativeTokenizer([b'a'])
 with pytest.raises(ValueError):NativeTokenizer([bytes([b]) for b in range(256)]+[b'a'])
