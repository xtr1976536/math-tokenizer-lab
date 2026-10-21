"""v5 continuation: grammar atoms, terminal RLOO, unchanged Rust BPE.

The lexical prior is explicit. It is never credited to policy learning.
Matching atoms/frequency and fixed-policy search isolate RL's contribution.
"""
from collections import Counter
from dataclasses import asdict
import json
import time
from types import SimpleNamespace
import numpy as np
from tokenizers import Tokenizer, models
from .bpe import Sequence
from .frontend_v6 import library, lexical_ranges, python_bridge
from .native_v3 import NativeTokenizer
from .rl_vocab_v5 import Indexed, RLVocab, internal_split,bytealphabet
from .trajectory_v5 import TrajectoryVocab, TerminalReward


class ReferenceReward(TerminalReward):
    def __init__(self,fit,reward,config):
        super().__init__(fit,reward,config)
        self.structures=[]
        for row in reward:
            raw=row['latex'].encode()
            lex=tuple((int(a),int(b),raw[a:b].decode()) for a,b in lexical_ranges(row['latex']) if raw[a:b].strip())
            self.structures.append(SimpleNamespace(kind='independent_katex_syntax_ast',spans=row['reference']['spans'],lexemes=lex))


class AtomCodec:
    def __init__(self,atoms):
        self.atoms=list(atoms)
        if len(atoms)>4096:raise ValueError('primitive alphabet exceeds BMP PUA allocation')
        # Trie is constructed once, never per formula. IDs are lexical base IDs.
        trie=NativeTokenizer(atoms,False,'shortest')
        self.children,self.terminal=trie.children,trie.terminal
        self.training_indices={}
        self.bridge=python_bridge()
        self.symbol_map=bytealphabet()|{i:chr(0xE000+i-256) for i in range(256,len(atoms))}
        self.symbol_reverse={v:k for k,v in self.symbol_map.items()}

    def symbols(self,text):
        return self.bridge.symbols(text,self.children,self.terminal)

    def symbols_ctypes(self,text):
        raw=text.encode();out=np.empty(max(1,len(raw)*3),dtype=np.uint8)
        count=library().mt_symbols(raw,len(raw),self.children.ctypes.data,self.terminal.ctypes.data,out.ctypes.data)
        return out[:count].tobytes().decode()

    def indexed(self,texts,cuts=None):
        key=(tuple(texts),None if cuts is None else tuple(frozenset(c) for c in cuts))
        if key in self.training_indices:return self.training_indices[key].clone()
        seqs=[]
        for text in texts:
            ids=[self.symbol_reverse[c] for c in self.symbols(text)];starts=[];ends=[];position=0
            for id in ids:starts.append(position);position+=len(self.atoms[id]);ends.append(position)
            assert position==len(text.encode())
            seqs.append(Sequence(ids,starts,ends,()))
        template=Indexed(texts,cuts,sequences=seqs);self.training_indices[key]=template
        return template.clone()


def derive_atoms(fit,capacity,atom_budget=256):
    counts=Counter()
    for row in fit:
        raw=row['latex'].encode()
        for a,b in lexical_ranges(row['latex']):
            payload=raw[a:b]
            if len(payload)>1 and (payload.startswith(b'\\') or payload[0]>=128):counts[payload]+=1
    chosen=[p for p,c in sorted(counts.items(),key=lambda x:(-x[1],x[0]))][:min(atom_budget,capacity-256)]
    return [bytes([i]) for i in range(256)]+chosen


class AtomicVocab(RLVocab):
    def __init__(self,config,atoms,codec=None):
        super().__init__(config);self.atoms=list(atoms);self.vocab=list(atoms)
        self.codec=codec or AtomCodec(atoms);self.indexed_factory=self.codec.indexed
        self.spellings=[self.codec.symbol_map[i] for i in range(len(atoms))]

    def build_official(self):
        spelling=list(self.spellings[:len(self.atoms)])
        for a,b,new in self.merges:
            value=spelling[a]+spelling[b]
            if new==len(spelling):spelling.append(value)
            elif spelling[new]!=value:raise ValueError('inconsistent atomic merge spelling')
        self.spellings=spelling
        self.tokenizer=Tokenizer(models.BPE(vocab={s:i for i,s in enumerate(spelling)},merges=[(spelling[a],spelling[b]) for a,b,new in self.merges]))

    def encode(self,text):return self.tokenizer.encode(self.codec.symbols(text),add_special_tokens=False).ids
    def decode(self,ids):return self.decode_bytes(ids).decode()

    def save(self,path):
        super().save(path);d=json.loads(path.read_text());d.update(format='v5-atomic-bpe-v6',atom_hex=[p.hex() for p in self.atoms]);path.write_text(json.dumps(d,indent=2))

    @classmethod
    def load(cls,path):
        from .rl_vocab_v5 import VocabRLConfig
        d=json.loads(path.read_text());obj=cls(VocabRLConfig(**d['config']),[bytes.fromhex(p) for p in d['atom_hex']])
        obj.vocab=[bytes.fromhex(p) for p in d['vocab_hex']];obj.merges=[tuple(m) for m in d['merges']];obj.theta=np.array(d['policy'])
        for key in ('logs','capacity','seed','method','train_seconds','fit_ids','reward_ids'):setattr(obj,key,d[key])
        obj.build_official();return obj


class AtomicTrajectoryV5(TrajectoryVocab):
    terminal_reward_class=ReferenceReward

    def __init__(self,config=None,trajectory=None,atom_budget=256):
        super().__init__(config,trajectory);self.atom_budget=atom_budget

    def make_rollout_model(self):return AtomicVocab(self.config,self.atoms,self.training_codec)

    def build_official(self):
        model=AtomicVocab(self.config,self.atoms);model.vocab=self.vocab;model.merges=self.merges;model.build_official()
        self.tokenizer=model.tokenizer;self.codec=model.codec;self.spellings=model.spellings

    def encode(self,text):return self.tokenizer.encode(self.codec.symbols(text),add_special_tokens=False).ids
    def decode(self,ids):return self.decode_bytes(ids).decode()

    def train(self,records,capacity=1024,seed=17,initial_theta=None,update=True,progress=None):
        fit,reward=internal_split(records,self.config)
        self.atoms=derive_atoms(fit,capacity,self.atom_budget)
        self.training_codec=AtomCodec(self.atoms)
        return super().train(records,capacity,seed,initial_theta,update,progress)

    def save(self,path):
        super().save(path);d=json.loads(path.read_text());d.update(format='v5-atomic-trajectory-v6',atom_hex=[p.hex() for p in self.atoms],atom_budget=self.atom_budget,
            lexical_prior='train_derived_complete_commands_and_unicode; all primitive IDs count',
            online_backend='native lexical codec plus unchanged official Rust BPE',reward_reference='independent pinned KaTeX syntax AST')
        path.write_text(json.dumps(d,indent=2))

    @classmethod
    def load(cls,path):
        from .rl_vocab_v5 import VocabRLConfig
        from .trajectory_v5 import TrajectoryConfig
        d=json.loads(path.read_text());obj=cls(VocabRLConfig(**d['config']),TrajectoryConfig(**d['trajectory_config']),d['atom_budget'])
        obj.atoms=[bytes.fromhex(p) for p in d['atom_hex']];obj.vocab=[bytes.fromhex(p) for p in d['vocab_hex']];obj.merges=[tuple(m) for m in d['merges']]
        obj.theta=np.array(d['policy']);obj.initial_theta=np.array(d['initial_policy']);obj.history=d['training_history']
        obj.initial_stats=d['initial_terminal_stats'];obj.final_stats=d['final_terminal_stats']
        for key in ('logs','capacity','seed','method','train_seconds','fit_ids','reward_ids'):setattr(obj,key,d[key])
        obj.build_official();return obj
