import random,json
from pathlib import Path
import pytest,numpy as np
from math_tokenizer.ast_v3 import AtomicASTBPE
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.rl_v4 import MaskedPolicy,MaskedConfig
from math_tokenizer.structure_v3 import structure_fast
from math_tokenizer.ast import compatible

TEXTS=[r'\frac{\partial u}{\partial t}=\sum_{i=1}^{n}u_i',r'\int_0^1 f(x)dx=1',r'(x+y)^2=x^2+2xy+y^2','α+β=γ']*8

@pytest.mark.parametrize('mode',['shortest','policy_map','greedy'])
def test_hard_tree_atoms_and_unknown_escape(mode,tmp_path):
 base=AtomicASTBPE(True).train(TEXTS,384,command_budget=200)
 m=ConstrainedTokenizer(base.vocab,True,'commands',mode)
 for text in TEXTS+['中文🙂+x',r'\unknown{x}=1','']:
  ids=m.encode(text);assert m.decode(ids)==text
  raw,spans,safe,interior,allowed,exceptions,s=m.prepare(text);pos=0
  for id in ids:
   end=pos+len(m.vocab[id]);assert allowed[pos] and allowed[end]
   if s.kind!='byte_fallback':assert compatible(pos,end,s.spans)
   pos=end
 m.save(tmp_path/'m.json');loaded=ConstrainedTokenizer.load(tmp_path/'m.json')
 for text in TEXTS:assert loaded.encode(text)==m.encode(text)


def test_exact_policy_gradient_matches_finite_difference():
 vocab=[bytes([b]) for b in range(256)]+[b'ab',b'bc',b'abc']
 policy=MaskedPolicy(vocab,MaskedConfig(hard_tree=False,epochs=0,entropy=0))
 policy.model.theta[:]=[.2,.1,-.1,.0,0,.05,.1]
 graph=policy.graph('abcab');value,gradient=policy.expected_gradient(graph);theta=policy.model.theta.copy();finite=[]
 for k in range(7):
  policy.model.theta=theta.copy();policy.model.theta[k]+=1e-5;plus=policy.expected_gradient(graph)[0]
  policy.model.theta=theta.copy();policy.model.theta[k]-=1e-5;minus=policy.expected_gradient(graph)[0]
  finite.append(-(plus-minus)/(2e-5)/len(graph))
 policy.model.theta=theta
 assert np.allclose(gradient,finite,atol=1e-8)


def test_expected_rl_learns_from_zero_prior_and_deterministic_seed(tmp_path):
 base=AtomicASTBPE(True).train(TEXTS,384,command_budget=160)
 config=MaskedConfig(epochs=4,max_train_formulas=8,initial_length_prior=0)
 a=MaskedPolicy(base.vocab,config).train(TEXTS,17);b=MaskedPolicy(base.vocab,config).train(TEXTS,17)
 assert a.theta.tolist()==b.theta.tolist() and abs(a.theta[0])>1e-6
 logs=a.training_log[0]['epochs'];assert logs[-1]['train_expected_tokens']<logs[0]['train_expected_tokens']
 assert a.training_log[0]['teacher_labels'] is False
 assert all(x['scope']=='train_only' for x in logs)
 for text in TEXTS:assert a.decode(a.encode(text))==text


def test_no_split_command_number_only_when_covered():
 vocab=[bytes([b]) for b in range(256)]+[b'\\alpha',b'12345']
 m=ConstrainedTokenizer(vocab,False,'commands_numbers','shortest')
 for text in [r'\alpha+12345',r'\unseen+99999']:
  raw,spans,safe,interior,allowed,exceptions,s=m.prepare(text);ids=m.encode(text);pos=0
  for id in ids:assert allowed[pos];pos+=len(m.vocab[id]);assert allowed[pos]
  assert m.decode(ids)==text
 assert any(t=='uncovered_command' for a,b,t in m.prepare(r'\unseen')[5])
 assert not m.prepare(r'\alpha+12345')[5]


def test_greedy_actual_forward_policy_equals_python_local_actions():
 base=AtomicASTBPE(True).train(TEXTS,384,command_budget=200)
 trainer=MaskedPolicy(base.vocab,MaskedConfig(epochs=2,infer='greedy',max_train_formulas=8));model=trainer.train(TEXTS,17)
 for text in TEXTS:
  graph=trainer.graph(text);pos=0;expected=[]
  while pos<len(graph):
   ends,ids,x,cost=graph[pos];action=int(np.argmax(x@model.theta));expected.append(int(ids[action]));pos=int(ends[action])
  assert model.encode(text)==expected


def test_policy_map_global_path_matches_exhaustive_small_graph():
 vocab=[bytes([b]) for b in range(256)]+[b'ab',b'bc',b'abc']
 trainer=MaskedPolicy(vocab,MaskedConfig(epochs=0,hard_tree=False,infer='policy_map',entropy=0));trainer.model.theta[:]=[.4,0,.2,0,0,.1,-.1]
 graph=trainer.graph('abcab')
 def visit(pos,score,ids):
  if pos==len(graph):return [(score,ids)]
  ends,tokens,x,cost=graph[pos];logit=x@trainer.model.theta;logit-=logit.max();p=np.exp(logit);p/=p.sum();out=[]
  for end,id,prob in zip(ends,tokens,p):out+=visit(int(end),score-np.log(prob),ids+[int(id)])
  return out
 choices=visit(0,0,[]);minimum=min(score for score,ids in choices);encoded=trainer.model.encode('abcab')
 assert any(ids==encoded and abs(score-minimum)<1e-10 for score,ids in choices)


def test_finite_difference_expected_rl_with_atom_mask():
 vocab=[bytes([b]) for b in range(256)]+[b'\\alpha',b'\\alpha+x',b'x+x',b'+x']
 trainer=MaskedPolicy(vocab,MaskedConfig(hard_tree=False,epochs=0,entropy=0));graph=trainer.graph(r'\alpha+x+x');theta=np.array([.2,0,.1,0,0,.1,.05]);trainer.model.theta=theta.copy();cost,g=trainer.expected_gradient(graph)
 finite=[]
 for k in range(7):
  trainer.model.theta=theta.copy();trainer.model.theta[k]+=1e-5;a=trainer.expected_gradient(graph)[0]
  trainer.model.theta=theta.copy();trainer.model.theta[k]-=1e-5;b=trainer.expected_gradient(graph)[0]
  finite.append(-(a-b)/2e-5/len(graph))
 assert np.allclose(finite,g,atol=1e-8)
