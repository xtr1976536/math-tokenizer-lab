"""v5 branch: terminal RL learns primitive merge barriers, Rust trains BPE.

Every action specifies an atomic symbol to isolate before BPE fitting. The
policy is trained with independent-mask RLOO, never by replacing it with best
sample search. At inference both barriers and vocabulary are frozen. No AST
parser or policy runs online, only the lexical codec and official Rust BPE.
"""
from collections import Counter
from dataclasses import asdict,dataclass
import json
import time
import re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,Regex
from .rl_route_v6 import AtomCodec,derive_atoms,ReferenceReward
from .rl_vocab_v5 import VocabRLConfig,internal_split
from .trajectory_v5 import TrajectoryConfig,rloo_gradient


@dataclass(frozen=True)
class MaskConfig:
    epochs:int=12
    samples:int=4
    learning_rate:float=.3
    candidates:int=48
    initial_probability:float=.10
    atom_budget:int=384
    tree_weight:float=.25


class FrozenBarrierV5:
    def __init__(self,atoms,tokenizer,codec=None):
        self.atoms=list(atoms);self.codec=codec or AtomCodec(atoms)
        if tokenizer.token_to_id('!') is None:
            # Pure alias migration preserves the stored merge order and IDs.
            data=json.loads(tokenizer.to_str());aliases={chr(0xE000+i):self.codec.symbol_map[i] for i in range(len(atoms))}
            convert=lambda text:''.join(aliases[c] for c in text)
            data['model']['vocab']={convert(s):id for s,id in data['model']['vocab'].items()}
            data['model']['merges']=[[convert(a),convert(b)] for a,b in data['model']['merges']]
            if data['pre_tokenizer'] is not None:
                pattern=data['pre_tokenizer']['pattern']['Regex']
                data['pre_tokenizer']['pattern']['Regex']=re.sub('[\ue000-\uf8ff]',lambda m:re.escape(aliases[m[0]]),pattern)
            tokenizer=Tokenizer.from_str(json.dumps(data))
        self.tokenizer=tokenizer
        self.vocab=[]
        for id in range(tokenizer.get_vocab_size()):
            self.vocab.append(b''.join(atoms[self.codec.symbol_reverse[c]] for c in tokenizer.id_to_token(id)))
        self.capacity=len(self.vocab);self.train_seconds=0.
    def encode(self,text):return self.tokenizer.encode(self.codec.symbols(text),add_special_tokens=False).ids
    def decode_bytes(self,ids):return b''.join(self.vocab[id] for id in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()
    def clear_cache(self):
        if hasattr(self.tokenizer.model,'clear_cache'):self.tokenizer.model.clear_cache()
    def save(self,path):
        self.tokenizer.save(str(path.with_suffix('.tokenizer.json')))
        d={'format':'v5-rl-merge-barrier-v6','symbol_alphabet':'compact-byte-alias-v1','atom_hex':[p.hex() for p in self.atoms],'capacity':self.capacity,'train_seconds':self.train_seconds,
           'mask':getattr(self,'mask',[]),'method':getattr(self,'method','barrier_control'),'fit_ids':getattr(self,'fit_ids',[]),'reward_ids':getattr(self,'reward_ids',[]),
           'mask_candidates':getattr(self,'mask_candidates',[]),'policy':getattr(self,'policy',[]),'initial_policy':getattr(self,'initial_policy',[]),
           'history':getattr(self,'history',[]),'mask_config':getattr(self,'mask_config',{}),'initial_terminal':getattr(self,'initial_terminal',None),'final_terminal':getattr(self,'final_terminal',None)}
        path.write_text(json.dumps(d,indent=2)+'\n')
    @classmethod
    def load(cls,path):
        path=Path(path)
        d=json.loads(path.read_text());obj=cls([bytes.fromhex(p) for p in d['atom_hex']],Tokenizer.from_file(str(path.with_suffix('.tokenizer.json'))))
        for k,v in d.items():
            if k not in ('format','atom_hex'):setattr(obj,k,v)
        return obj


def fit_barriers(atoms,symbols,barriers,capacity,codec=None,symbols_are_legacy=False):
    codec=codec or AtomCodec(atoms)
    if not symbols_are_legacy:symbols=[''.join(chr(0xE000+codec.symbol_reverse[c]) for c in text) for text in symbols]
    t=Tokenizer(models.BPE())
    if barriers:t.pre_tokenizer=pre_tokenizers.Split(Regex('['+''.join(chr(0xE000+id) for id in sorted(barriers))+']'),behavior='isolated')
    t.train_from_iterator(symbols,trainers.BpeTrainer(vocab_size=capacity,min_frequency=2,show_progress=False,
        initial_alphabet=[chr(0xE000+i) for i in range(len(atoms))],special_tokens=[]))
    model=FrozenBarrierV5(atoms,t,codec);model.mask=sorted(barriers);assert len(model.vocab)<=capacity
    return model


class BarrierPolicyV5:
    def __init__(self,config=None,mask=None):self.config=config or VocabRLConfig(fit_limit=1000000,reward_limit=512);self.mask_config=mask or MaskConfig()
    def train(self,records,capacity=1024,seed=17,update=True,progress=None):
        start=time.perf_counter();fit,reward=internal_split(records,self.config);mc=self.mask_config
        atoms=derive_atoms(fit,capacity,mc.atom_budget);codec=AtomCodec(atoms);symbols=[codec.symbols_ctypes(r['latex']) for r in fit]
        counts=Counter(ord(c)-0xE000 for text in symbols for c in text)
        candidates=[id for id,count in sorted(counts.items(),key=lambda x:(-x[1],x[0])) if count>=2][:mc.candidates]
        theta=np.full(len(candidates),np.log(mc.initial_probability/(1-mc.initial_probability)));initial=theta.copy()
        objective=ReferenceReward(fit,reward,TrajectoryConfig(tree_weight=mc.tree_weight))
        base=fit_barriers(atoms,symbols,[],capacity,codec,True);initial_stats=objective.score(base);anchor=max(initial_stats['utility'],1e-8)
        rng=np.random.default_rng(seed);first=np.zeros_like(theta);second=np.zeros_like(theta);history=[];best=None;best_value=float('inf')
        for epoch in range(mc.epochs):
            p=1/(1+np.exp(-np.clip(theta,-12,12)));scores=[];returns=[];statistics=[]
            for sample in range(mc.samples):
                action=rng.random(len(p))<p;barriers=[id for id,a in zip(candidates,action) if a]
                model=fit_barriers(atoms,symbols,barriers,capacity,codec,True);stats=objective.score(model)
                scores.append(action.astype(float)-p);returns.append((initial_stats['utility']-stats['utility'])/anchor)
                statistics.append(stats|{'mask':barriers})
                if stats['utility']<best_value:best,best_value=model,stats['utility']
                if progress:progress(epoch,sample,stats)
            gradient,advantages=rloo_gradient(scores,returns)
            if update:
                first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
                theta+=mc.learning_rate*(first/(1-.9**(epoch+1)))/(np.sqrt(second/(1-.999**(epoch+1)))+1e-8)
                theta=np.clip(theta,-12,12)
            history.append({'epoch':epoch,'samples':statistics,'gradient_norm':float(np.linalg.norm(gradient)),'advantages':advantages.tolist(),'policy':theta.tolist(),'updates_enabled':update})
        barriers=[id for id,value in zip(candidates,theta) if value>=0]
        final=fit_barriers(atoms,symbols,barriers,capacity,codec,True);final.method='rl_learned_merge_barriers' if update else 'barrier_no_policy_update'
        final.fit_ids=[r['id'] for r in fit];final.reward_ids=[r['id'] for r in reward];final.train_seconds=time.perf_counter()-start
        final.policy=theta.tolist();final.initial_policy=initial.tolist();final.mask_candidates=candidates;final.history=history;final.mask_config=asdict(mc)
        final.initial_terminal=initial_stats;final.final_terminal=objective.score(final)
        self.model=final;self.best_sample=best;self.frequency=base
        for control in (best,base):
            control.fit_ids=final.fit_ids;control.reward_ids=final.reward_ids;control.mask_candidates=candidates;control.mask_config=asdict(mc)
        best.method='barrier_same_budget_search_control';best.train_seconds=final.train_seconds
        base.method='atomic_official_frequency_control'
        self.training_seconds=final.train_seconds;return final
