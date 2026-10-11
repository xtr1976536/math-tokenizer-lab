"""Masked-policy RL with exact on-policy expected-return policy gradient.

Finite segmentation DAG permits Rao-Blackwellization of REINFORCE: backward
expected costs + forward state occupancy compute its full expectation. This is
RL policy optimization, not shortest-path supervision or manual label fitting.
Optional trajectory REINFORCE comparison remains v2/v3. Gamma1 exact objective.
"""
from dataclasses import dataclass,asdict
import hashlib,time
import numpy as np
from .constrained_v4 import ConstrainedTokenizer
from .ast import compatible

@dataclass(frozen=True)
class MaskedConfig:
 epochs:int=24
 lr:float=.06
 max_train_formulas:int=768
 initial_length_prior:float=0.
 entropy:float=.005
 atomic:str='commands'
 hard_tree:bool=True
 infer:str='greedy'
 rare_cost:float=0.

class MaskedPolicy:
 def __init__(self,vocab,config):
  self.config=config;self.model=ConstrainedTokenizer(vocab,config.hard_tree,config.atomic,config.infer)
  self.model.theta[:]=0;self.model.theta[0]=config.initial_length_prior
 def graph(self,text):
  model=self.model;raw,spans,safe,interior,allowed,exceptions,s=model.prepare(text);n=len(raw);a,b,ids=model.matches(text)
  options=[[] for _ in range(n)];longest=np.ones(n+1)
  for start,end,id in zip(a,b,ids):longest[start]=max(longest[start],end-start)
  for start,end,id in zip(a,b,ids):
   start,end,id=int(start),int(end),int(id)
   if not allowed[start] or not allowed[end]:continue
   crossing=bool(s.spans) and not compatible(start,end,s.spans)
   if model.structural and crossing:continue
   length=end-start;x=[np.log(length),model.logfreq[id]/8,float(safe[end]),-float(interior[end]),-float(crossing),length/max(1,n),np.log(longest[end])]
   options[start].append((end,id,x,1.+self.config.rare_cost/max(1.,model.frequency[id])))
  graph=[]
  for opts in options:
   if not opts:graph.append(None);continue
   graph.append((np.array([x[0] for x in opts],dtype=int),np.array([x[1] for x in opts],dtype=int),np.array([x[2] for x in opts],dtype=float),np.array([x[3] for x in opts],dtype=float)))
  return graph
 def expected_gradient(self,graph):
  n=len(graph);values=np.zeros(n+1);probabilities=[None]*n;advantages=[None]*n;entropygrads=[None]*n
  for pos in range(n-1,-1,-1):
   if graph[pos] is None:continue
   ends,ids,x,cost=graph[pos];logits=x@self.model.theta;logits-=logits.max();p=np.exp(logits);p/=p.sum();q=cost+values[ends];values[pos]=p@q;probabilities[pos]=p;advantages[pos]=values[pos]-q
   logp=np.log(np.maximum(p,1e-30));entropygrads[pos]=-(p*(logp+1))@(x-p@x)
  occupancy=np.zeros(n+1);occupancy[0]=1;grad=np.zeros(7)
  for pos,g in enumerate(graph):
   if g is None or not occupancy[pos]:continue
   ends,ids,x,cost=g;p=probabilities[pos];weight=occupancy[pos]
   grad+=weight*((p*advantages[pos])@x+self.config.entropy*entropygrads[pos]);np.add.at(occupancy,ends,weight*p)
  return values[0],grad/max(1,n)
 def train(self,texts,seed=17):
  start=time.perf_counter();rng=np.random.default_rng(seed);selected=sorted(set(texts),key=lambda t:hashlib.sha256(t.encode()).digest())[:self.config.max_train_formulas]
  counts=np.ones(len(self.model.vocab))
  for text in selected:
   a,b,ids=self.model.matches(text);np.add.at(counts,ids,1)
  self.model.frequency=counts;self.model.logfreq=np.log(counts);graphs=[self.graph(t) for t in selected]
  m=np.zeros(7);v=np.zeros(7);updates=0;logs=[]
  for epoch in range(self.config.epochs):
   order=rng.permutation(len(graphs));costs=[];batch=[]
   for idx in order:
    cost,grad=self.expected_gradient(graphs[idx]);costs.append(cost);batch.append(grad)
    if len(batch)==16 or idx==order[-1]:
     grad=np.mean(batch,axis=0);norm=np.linalg.norm(grad)
     if norm>1:grad/=norm
     updates+=1;m=.9*m+.1*grad;v=.999*v+.001*grad*grad
     self.model.theta+=self.config.lr*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8);batch=[]
   logs.append({'epoch':epoch,'train_expected_tokens':float(np.mean(costs)),'theta':self.model.theta.tolist(),'estimator':'exact_expected_on_policy_REINFORCE_DAG','scope':'train_only','formulas':len(selected)})
  self.model.train_seconds=time.perf_counter()-start;self.model.training_log=[{'config':asdict(self.config),'seed':seed,'epochs':logs,'method':'masked_expected_policy_gradient','teacher_labels':False,'shortest_path_targets':False}];return self.model
