"""RL-guided BPE vocabulary construction, deployment uses official Rust BPE.

Policy selects merge actions; no parser, manual command vocabulary or hard AST
mask. Rewards use only training-internal heldout paper groups. Compression alone
is an ablation; composite utility uses heldout code cost, rare-token exposure and
lexical byte-cut rate (generic lexical units inferred from raw string). All
weights remain explicit designer choices, not claimed fully human-free.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict
import hashlib,heapq,json,math,re,time
import numpy as np
from tokenizers import Tokenizer,models,pre_tokenizers,decoders
from .bpe import Sequence
from .corpus import ROOT

LEX=re.compile(r'\\[A-Za-z]+|\\[^A-Za-z]|\d+(?:\.\d+)?|[^\s]')

@dataclass(frozen=True)
class VocabRLConfig:
 objective:str='composite'
 candidates:int=12
 updates:int=24
 learning_rate:float=.08
 exploration:float=.05
 mdl_weight:float=.4
 boundary_weight:float=.4
 rare_weight:float=.15
 robustness_weight:float=.25
 reward_limit:int=384
 fit_limit:int=6000
 min_pair:int=2
 initial_frequency_prior:float=0.
 feature_mode:str='statistics'
 estimator:str='expected'

class Indexed:
 """Exact replacement/boundary statistics, updated only on affected formulas.

 Raw adjacent frequency and nonoverlapping replacement gain are distinct for
 repeated pairs (a,a). Maintaining both avoids rescanning reward strings for
 each candidate. A versioned heap preserves the old frequency/lexical ordering.
 """
 def __init__(self,texts,cuts=None,sequences=None):
  self.seqs=sequences if sequences is not None else [Sequence.make(t,False) for t in texts];self.counts=Counter();self.gains=Counter();self.boundaries=Counter();self.index=defaultdict(set);self.local=[];self.cuts=cuts
  if cuts is not None and len(cuts)!=len(self.seqs):raise ValueError('cut/formula count mismatch')
  for i,s in enumerate(self.seqs):
   c,g,b=self.scan(s,None if cuts is None else cuts[i]);self.local.append((c,g,b));self.counts.update(c);self.gains.update(g);self.boundaries.update(b)
   for pair in c:self.index[pair].add(i)
  self.generation=Counter();self.heap=[(-c,p,0) for p,c in self.counts.items() if c>0];heapq.heapify(self.heap)
 @staticmethod
 def scan(seq,cuts=None):
  raw=Counter();gain=Counter();boundary=Counter();last_self={}
  for i,pair in enumerate(zip(seq.ids,seq.ids[1:])):
   raw[pair]+=1
   if pair[0]==pair[1]:
    if last_self.get(pair,-2)==i-1:continue
    last_self[pair]=i
   gain[pair]+=1
   if cuts is not None and seq.ends[i] in cuts:boundary[pair]+=1
  return raw,gain,boundary
 def top(self,k,min_pair=2):
  held=[];pairs=[]
  while self.heap and len(pairs)<k:
   item=heapq.heappop(self.heap);negative,pair,version=item
   if self.generation[pair]!=version or self.counts[pair]!=-negative:continue
   held.append(item)
   if -negative<min_pair:break
   pairs.append(pair)
  for item in held:heapq.heappush(self.heap,item)
  return pairs
 def clone(self):
  """Independent mutable environment, share only immutable source counters."""
  obj=object.__new__(type(self))
  obj.seqs=[Sequence(s.ids.copy(),s.starts.copy(),s.ends.copy(),s.spans) for s in self.seqs]
  obj.counts=self.counts.copy();obj.gains=self.gains.copy();obj.boundaries=self.boundaries.copy()
  obj.index=defaultdict(set,{p:indices.copy() for p,indices in self.index.items()})
  obj.local=self.local.copy();obj.cuts=self.cuts
  obj.generation=self.generation.copy();obj.heap=self.heap.copy();return obj
 def merge(self,pair,new):
  changed={}
  for i in sorted(self.index[pair].copy()):
   old,oldgain,oldboundary=self.local[i];s=self.seqs[i];s.merge(pair,new)
   current,gain,boundary=self.scan(s,None if self.cuts is None else self.cuts[i]);self.local[i]=(current,gain,boundary)
   for p in old.keys()|current.keys():
    if p not in changed:changed[p]=self.counts[p]
    self.counts[p]+=current[p]-old[p];self.gains[p]+=gain[p]-oldgain[p];self.boundaries[p]+=boundary[p]-oldboundary[p]
    if not current[p]:self.index[p].discard(i)
    elif not old[p]:self.index[p].add(i)
  for p,oldcount in changed.items():
   if self.counts[p]==oldcount:continue
   self.generation[p]+=1
   if self.counts[p]>0:heapq.heappush(self.heap,(-self.counts[p],p,self.generation[p]))
  if len(self.heap)>4*max(1,len(self.counts)):
   self.heap=[(-count,p,self.generation[p]) for p,count in self.counts.items() if count>0];heapq.heapify(self.heap)


def internal_split(records,config):
 groups=sorted({r['split_group'] for r in records},key=lambda g:hashlib.sha256(('v5-reward:'+g).encode()).digest())
 reward_groups=set(groups[:max(1,len(groups)//5)])
 fit=[r for r in records if r['split_group'] not in reward_groups];reward=[r for r in records if r['split_group'] in reward_groups]
 fit=sorted(fit,key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:config.fit_limit]
 reward=sorted(reward,key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:config.reward_limit]
 return fit,reward


def bytealphabet():
 bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1));cs=bs[:];n=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+n);n+=1
 return {b:chr(c) for b,c in zip(bs,cs)}

class MergeSession:
 """Shared v5 merge environment for local and whole-trajectory RL."""
 def __init__(self,model,records):
  self.model=model;self.config=model.config
  self.fit,self.reward=internal_split(records,self.config)
  if not self.fit or not self.reward:raise ValueError('fit and reward require separate nonempty paper groups')
  self.size={payload:i for i,payload in enumerate(model.vocab)}
  factory=getattr(model,'indexed_factory',Indexed)
  self.perturbed=factory([re.sub(r'\s+','',r['latex']) for r in self.reward]);self.perturbed_total=sum(len(s.ids) for s in self.perturbed.seqs)
  self.bytes_count=sum(len(r['latex'].encode()) for r in self.reward);cuts=[]
  for row in self.reward:
   text=row['latex'];offset=[0]
   for c in text:offset.append(offset[-1]+len(c.encode()))
   cuts.append({j for m in LEX.finditer(text) for j in range(offset[m.start()]+1,offset[m.end()])})
  self.f=factory([r['latex'] for r in self.fit]);self.h=factory([r['latex'] for r in self.reward],cuts)
  self.fitusage=Counter(id for s in self.f.seqs for id in s.ids);self.totalfit=sum(self.fitusage.values());self.totalheld=sum(len(s.ids) for s in self.h.seqs)
 def candidates(self,frequency_only=False):
  pairs=self.f.top(1 if frequency_only else self.config.candidates,self.config.min_pair)
  features=[];rewards=[];components=[];vsize=len(self.model.vocab)+1
  for pair in pairs:
   payload=self.model.vocab[pair[0]]+self.model.vocab[pair[1]];fitgain=self.f.gains[pair];heldgain=self.h.gains[pair];boundarygain=self.h.boundaries[pair]
   pa=(self.fitusage[pair[0]]+1)/(self.totalfit+vsize);pb=(self.fitusage[pair[1]]+1)/(self.totalfit+vsize);pn=(fitgain+1)/(self.totalfit-fitgain+vsize)
   mdl_gain=heldgain*math.log2(pn/(pa*pb))-8*len(payload)*.01
   compression=heldgain/max(1,self.totalheld);mdl=mdl_gain/max(1,self.bytes_count);boundary=boundarygain/max(1,self.totalheld)
   rare=compression*(1 if fitgain<5 else 0)
   perturb_gain=self.perturbed.gains[pair]/max(1,self.perturbed_total)
   robustness=min(compression,perturb_gain)-abs(compression-perturb_gain)
   composite=compression+self.config.mdl_weight*mdl+self.config.boundary_weight*boundary-self.config.rare_weight*rare+self.config.robustness_weight*robustness
   objective=compression if self.config.objective=='length' else composite
   rewards.append(objective);components.append({'compression_gain':compression,'heldout_code_gain':mdl,'boundary_cut_reduction':boundary,'rare_exposure':rare,'whitespace_gain_consistency':robustness,'total':objective})
   feature=[math.log1p(fitgain)/10,math.log(len(payload))/4,float(payload.startswith(b'\\')),float(payload.isdigit()),float(any(c>=128 for c in payload)),float(any(c in payload for c in b'{}_^'))]
   if self.config.feature_mode=='statistics':feature += [compression*1000,mdl*1000,boundary*1000,rare*1000,robustness*1000]
   features.append(feature)
  return pairs,np.asarray(features),np.asarray(rewards),components
 def commit(self,pair,component,policy,best_reward,method):
  model=self.model;payload=model.vocab[pair[0]]+model.vocab[pair[1]];new=self.size.get(payload)
  if new is None:new=len(model.vocab);self.size[payload]=new;model.vocab.append(payload)
  step=len(model.merges);model.merges.append((pair[0],pair[1],new))
  heldgain=self.h.gains[pair];fitgain=self.f.gains[pair]
  self.fitusage[pair[0]]-=fitgain;self.fitusage[pair[1]]-=fitgain;self.fitusage[new]+=fitgain;self.totalfit-=fitgain;self.totalheld-=heldgain
  self.perturbed_total-=self.perturbed.gains[pair]
  self.f.merge(pair,new);self.h.merge(pair,new);self.perturbed.merge(pair,new)
  model.logs.append({'step':step,'action_pair':list(pair),'reward':component,'best_candidate_reward':float(best_reward),'policy':policy.tolist(),'scope':'training_internal_document_holdout','method':method})
 def finish(self,capacity,seed,method,seconds):
  model=self.model;model.capacity=capacity;model.seed=seed;model.method=method;model.train_seconds=seconds
  model.fit_ids=[r['id'] for r in self.fit];model.reward_ids=[r['id'] for r in self.reward]
  assert len(model.vocab)<=capacity and len(model.vocab)==len(set(model.vocab))
  model.build_official();return model


class RLVocab:
 def __init__(self,config=None):self.config=config or VocabRLConfig();self.vocab=[bytes([b]) for b in range(256)];self.merges=[];self.theta=np.zeros(11 if self.config.feature_mode=='statistics' else 6);self.logs=[]
 def train(self,records,capacity=512,seed=17,method='rl'):
  if method not in ('rl','frequency','random','reward_greedy'):raise ValueError('unsupported merge-selection method')
  start=time.perf_counter();rng=np.random.default_rng(seed)
  self.vocab=[bytes([b]) for b in range(256)];self.merges=[];self.logs=[]
  self.theta=np.zeros(11 if self.config.feature_mode=='statistics' else 6);self.theta[0]=self.config.initial_frequency_prior
  if capacity<256:raise ValueError('complete byte fallback requires capacity >=256')
  session=MergeSession(self,records)
  for step in range(capacity-256):
   candidates,x,r,components=session.candidates(method=='frequency')
   if not candidates:break
   if method=='frequency':choice=0
   elif method=='reward_greedy':choice=int(np.argmax(r))
   elif method=='random':choice=int(rng.integers(len(candidates)))
   else:
    for _ in range(self.config.updates):
     logits=x@self.theta;logits-=logits.max();p=np.exp(logits);p/=p.sum();p=(1-self.config.exploration)*p+self.config.exploration/len(p)
     baseline=float(p@r)
     if self.config.estimator=='expected':
      # Exact current-bandit score-function expectation; exploration mixture's
      # derivative remains (1-epsilon) times pure policy derivative.
      q=np.exp(x@self.theta-np.max(x@self.theta));q/=q.sum()
      gradient=(1-self.config.exploration)*(q*(r-float(q@r)))@x
      norm=float(np.linalg.norm(gradient))
      if norm>1:gradient/=norm
      self.theta+=self.config.learning_rate*1000*gradient;continue
     action=int(rng.choice(len(p),p=p))
     # Pure-softmax score adjusted for explicit exploration mixture.
     q=np.exp(x@self.theta-np.max(x@self.theta));q/=q.sum();score=(1-self.config.exploration)*q[action]/p[action]*(x[action]-q@x)
     self.theta+=self.config.learning_rate*1000*(r[action]-baseline)*score
    choice=int(np.argmax(x@self.theta))
   session.commit(candidates[choice],components[choice],self.theta,np.max(r),method)
  return session.finish(capacity,seed,method,time.perf_counter()-start)
 def clear_cache(self):
  if hasattr(self.tokenizer.model,'clear_cache'):self.tokenizer.model.clear_cache()
 def build_official(self):
  map=bytealphabet();spelling=lambda payload:''.join(map[b] for b in payload)
  vocab={spelling(v):i for i,v in enumerate(self.vocab)};merges=[(spelling(self.vocab[a]),spelling(self.vocab[b])) for a,b,c in self.merges]
  tokenizer=Tokenizer(models.BPE(vocab=vocab,merges=merges));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False);tokenizer.decoder=decoders.ByteLevel();self.tokenizer=tokenizer
 def encode(self,text):return self.tokenizer.encode(text,add_special_tokens=False).ids
 def encode_batch(self,texts):return [e.ids for e in self.tokenizer.encode_batch(texts,add_special_tokens=False)]
 def decode(self,ids):return self.tokenizer.decode(ids,skip_special_tokens=False)
 def decode_bytes(self,ids):return b''.join(self.vocab[id] for id in ids)
 def save(self,path):
  self.tokenizer.save(str(path.with_suffix('.tokenizer.json')))
  path.write_text(json.dumps({'format':'rl-vocab-v5','config':asdict(self.config),'vocab_hex':[v.hex() for v in self.vocab],'merges':self.merges,
    'policy':self.theta.tolist(),'logs':self.logs,'capacity':self.capacity,'seed':self.seed,'method':self.method,'train_seconds':self.train_seconds,'fit_ids':self.fit_ids,'reward_ids':self.reward_ids},indent=2))
 @classmethod
 def load(cls,path):
  d=json.loads(path.read_text());obj=cls(VocabRLConfig(**d['config']));obj.vocab=[bytes.fromhex(t) for t in d['vocab_hex']];obj.merges=[tuple(m) for m in d['merges']];obj.theta=np.array(d['policy']);obj.logs=d['logs'];obj.capacity=d['capacity'];obj.seed=d['seed'];obj.method=d['method'];obj.train_seconds=d['train_seconds'];obj.fit_ids=d['fit_ids'];obj.reward_ids=d['reward_ids'];obj.build_official();return obj
