"""RL-guided BPE vocabulary construction, deployment uses official Rust BPE.

Policy selects merge actions; no parser, manual command vocabulary or hard AST
mask. Rewards use only training-internal heldout paper groups. Compression alone
is an ablation; composite utility uses heldout code cost, rare-token exposure and
lexical byte-cut rate (generic lexical units inferred from raw string). All
weights remain explicit designer choices, not claimed fully human-free.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict
import hashlib,json,math,re,time
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
 def __init__(self,texts):
  self.seqs=[Sequence.make(t,False) for t in texts];self.counts=Counter();self.index=defaultdict(set);self.local=[]
  for i,s in enumerate(self.seqs):
   c=s.counts();self.local.append(c);self.counts.update(c)
   for pair in c:self.index[pair].add(i)
 def merge(self,pair,new):
  for i in sorted(self.index[pair].copy()):
   old=self.local[i];s=self.seqs[i];s.merge(pair,new);current=s.counts();self.local[i]=current
   for p in old.keys()|current.keys():
    self.counts[p]+=current[p]-old[p]
    if not current[p]:self.index[p].discard(i)
    elif not old[p]:self.index[p].add(i)


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

class RLVocab:
 def __init__(self,config=None):self.config=config or VocabRLConfig();self.vocab=[bytes([b]) for b in range(256)];self.merges=[];self.theta=np.zeros(11 if self.config.feature_mode=='statistics' else 6);self.logs=[]
 def train(self,records,capacity=512,seed=17,method='rl'):
  start=time.perf_counter();rng=np.random.default_rng(seed);self.theta[0]=self.config.initial_frequency_prior
  fit,reward=internal_split(records,self.config);f=Indexed([r['latex'] for r in fit]);h=Indexed([r['latex'] for r in reward]);size={bytes([b]):b for b in range(256)}
  perturbed=Indexed([re.sub(r'\s+','',r['latex']) for r in reward]);perturbed_total=sum(len(seq.ids) for seq in perturbed.seqs)
  bytes_count=sum(len(r['latex'].encode()) for r in reward);cuts=[]
  for r in reward:
   text=r['latex'];offset=[0]
   for c in text:offset.append(offset[-1]+len(c.encode()))
   cuts.append({j for m in LEX.finditer(text) for j in range(offset[m.start()]+1,offset[m.end()])})
  # Fit unigram predictive coding cost with Dirichlet smoothing. Used only as
  # intrinsic heldout code-cost proxy, never called LM BPB or semantic truth.
  fitusage=Counter(id for s in f.seqs for id in s.ids);totalfit=sum(fitusage.values());totalheld=sum(len(s.ids) for s in h.seqs)
  for step in range(capacity-256):
   candidates=sorted((p for p,c in f.counts.items() if c>=self.config.min_pair),key=lambda p:(-f.counts[p],p))[:self.config.candidates if method!='frequency' else 1]
   if not candidates:break
   features=[];rewards=[];components=[];vsize=len(self.vocab)+1
   for pair in candidates:
    payload=self.vocab[pair[0]]+self.vocab[pair[1]];fitgain=sum(f.seqs[i].gain(pair) for i in f.index[pair]);heldgain=0;boundarygain=0
    for i in h.index.get(pair,()):
     s=h.seqs[i];k=0
     while k<len(s.ids)-1:
      if (s.ids[k],s.ids[k+1])==pair:
       heldgain+=1;boundarygain+=s.ends[k] in cuts[i];k+=2
      else:k+=1
    # New-token predicted probability estimated from fit only; MDL overhead
    # charged in payload bytes per reward-snapshot byte.
    pa=(fitusage[pair[0]]+1)/(totalfit+vsize);pb=(fitusage[pair[1]]+1)/(totalfit+vsize);pn=(fitgain+1)/(totalfit-fitgain+vsize)
    mdl_gain=heldgain*math.log2(pn/(pa*pb))-8*len(payload)*.01
    compression=heldgain/max(1,totalheld);mdl=mdl_gain/max(1,bytes_count);boundary=boundarygain/max(1,totalheld)
    rare=(heldgain/max(1,totalheld))*(1 if fitgain<5 else 0)
    perturb_gain=sum(perturbed.seqs[i].gain(pair) for i in perturbed.index.get(pair,()))/max(1,perturbed_total)
    robustness=min(compression,perturb_gain)-abs(compression-perturb_gain)
    composite=compression+self.config.mdl_weight*mdl+self.config.boundary_weight*boundary-self.config.rare_weight*rare+self.config.robustness_weight*robustness
    objective=compression if self.config.objective=='length' else composite
    rewards.append(objective);components.append({'compression_gain':compression,'heldout_code_gain':mdl,'boundary_cut_reduction':boundary,'rare_exposure':rare,'whitespace_gain_consistency':robustness,'total':objective})
    feature=[math.log1p(fitgain)/10,math.log(len(payload))/4,float(payload.startswith(b'\\')),
      float(payload.isdigit()),float(any(c>=128 for c in payload)),float(any(c in payload for c in b'{}_^'))]
    if self.config.feature_mode=='statistics':feature += [compression*1000,mdl*1000,boundary*1000,rare*1000,robustness*1000]
    features.append(feature)
   x=np.asarray(features);r=np.asarray(rewards)
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
   pair=candidates[choice];payload=self.vocab[pair[0]]+self.vocab[pair[1]]
   new=size.get(payload)
   if new is None:new=len(self.vocab);size[payload]=new;self.vocab.append(payload)
   self.merges.append((pair[0],pair[1],new));heldgain=sum(h.seqs[i].gain(pair) for i in h.index.get(pair,()))
   fitgain=sum(f.seqs[i].gain(pair) for i in f.index.get(pair,()))
   fitusage[pair[0]]-=fitgain;fitusage[pair[1]]-=fitgain;fitusage[new]+=fitgain;totalfit-=fitgain;totalheld-=heldgain
   perturbed_gain=sum(perturbed.seqs[i].gain(pair) for i in perturbed.index.get(pair,()));perturbed_total-=perturbed_gain
   f.merge(pair,new);h.merge(pair,new);perturbed.merge(pair,new)
   self.logs.append({'step':step,'action_pair':list(pair),'reward':components[choice],'best_candidate_reward':float(np.max(r)),
     'policy':self.theta.tolist(),'scope':'training_internal_document_holdout','method':method})
  self.train_seconds=time.perf_counter()-start;self.capacity=capacity;self.seed=seed;self.method=method
  self.fit_ids=[r['id'] for r in fit];self.reward_ids=[r['id'] for r in reward];assert len(self.vocab)<=capacity and len(self.vocab)==len(set(self.vocab))
  self.build_official();return self
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
