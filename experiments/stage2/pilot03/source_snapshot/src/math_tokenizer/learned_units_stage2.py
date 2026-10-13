"""Data-discovered byte units + original HF BPE; no mathematical lexicon.

Primitive candidates are maximal runs between a fit-learned boundary threshold.
Their entire byte payloads are ranked by observed byte savings; this frequency
criterion is disclosed as a vocabulary-construction prior, not RL reward.
"""
from collections import Counter
import json,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models,trainers
from .signal_stage2 import SignalTokenizer,contexts,TokenCodePredictor,paper_token_rate


def discover_units(fit,table,mass,capacity,quantile=.6,max_bytes=32,unit_budget=384):
    seen=np.flatnonzero(mass>0);order=seen[np.argsort(table[seen],kind='stable')]
    cumulative=np.cumsum(mass[order]);cut=table[order[np.searchsorted(cumulative,quantile*cumulative[-1])]]
    counts=Counter()
    for row in fit:
        raw=row['latex'].encode();cs=contexts(raw);start=0
        for i in range(1,len(raw)+1):
            if i==len(raw) or table[cs[i]]>cut or i-start>=max_bytes:
                piece=raw[start:i]
                if len(piece)>1:counts[piece]+=1
                start=i
    candidates=sorted((p for p,n in counts.items() if n>=2),key=lambda p:(-counts[p]*(len(p)-1),p))
    chosen=candidates[:min(unit_budget,capacity-256)]
    return [bytes([i]) for i in range(256)]+chosen,{'quantile':quantile,'threshold':float(cut),
        'observed_candidates':len(candidates),'chosen_units':len(chosen),'criterion':'frequency times byte savings; no lexical class',
        'fit_ids':[r['id'] for r in fit]}


class LearnedUnitBPE:
    def __init__(self,atoms,tokenizer):
        self.atoms=list(atoms);self.codec=SignalTokenizer(atoms);self.tokenizer=tokenizer
        self.vocab=[]
        for i in range(tokenizer.get_vocab_size()):
            self.vocab.append(b''.join(atoms[ord(c)-0xE000] for c in tokenizer.id_to_token(i)))
        self.capacity=len(self.vocab);self.train_seconds=0.;self.training_log=[]
    def symbols(self,text):return ''.join(chr(0xE000+i) for i in self.codec.encode(text))
    @classmethod
    def train(cls,fit,atoms,capacity):
        started=time.perf_counter();codec=SignalTokenizer(atoms);tokenizer=Tokenizer(models.BPE())
        tokenizer.train_from_iterator((''.join(chr(0xE000+i) for i in codec.encode(r['latex'])) for r in fit),
            trainers.BpeTrainer(vocab_size=capacity,min_frequency=2,show_progress=False,
                                initial_alphabet=[chr(0xE000+i) for i in range(len(atoms))],special_tokens=[]))
        obj=cls(atoms,tokenizer);obj.train_seconds=time.perf_counter()-started;return obj
    def encode(self,text):return self.tokenizer.encode(self.symbols(text),add_special_tokens=False).ids
    def decode_bytes(self,ids):return b''.join(self.vocab[i] for i in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()
    def clear_cache(self):
        if hasattr(self.tokenizer.model,'clear_cache'):self.tokenizer.model.clear_cache()
    def save(self,path):
        Path(path).write_text(json.dumps({'format':'learned-units-stage2','atom_hex':[p.hex() for p in self.atoms],
            'tokenizer':json.loads(self.tokenizer.to_str()),'train_seconds':self.train_seconds,'training_log':self.training_log},indent=2)+'\n')
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text());obj=cls([bytes.fromhex(p) for p in d['atom_hex']],Tokenizer.from_str(json.dumps(d['tokenizer'])))
        obj.train_seconds=d['train_seconds'];obj.training_log=d['training_log'];return obj


def select_units(fit,reward,table,mass,capacity,reference_rate):
    best=None;logs=[]
    for quantile in (.4,.6,.8):
        atoms,info=discover_units(fit,table,mass,capacity,quantile)
        model=LearnedUnitBPE.train(fit,atoms,capacity)
        fitenc=[model.encode(r['latex']) for r in fit];rr=[model.encode(r['latex']) for r in reward]
        rate=paper_token_rate(rr,reward);code=TokenCodePredictor(fitenc,len(model.vocab)).score(rr,reward)['paper_macro_token_bigram_code_bits_per_byte']
        violation=max(0.,rate/(reference_rate*1.01)-1.)
        info.update(paper_tokens_per_byte=rate,paper_code_bpb=code,violation=violation)
        logs.append(info);key=(violation>0,violation,code,quantile)
        if best is None or key<best[0]:best=(key,model,info)
    best[1].training_log=[{'selection':'fit/reward prediction code + byte compression guard; no math quality labels',
                           'candidates':logs,'selected':best[2],'reward_ids':[r['id'] for r in reward]}]
    return best[1],logs
