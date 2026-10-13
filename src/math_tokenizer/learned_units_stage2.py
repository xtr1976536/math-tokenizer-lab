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


def discover_context_classes(fit,capacity,classes=8,unit_budget=384):
    """Cluster bytes by observed neighbor distributions, without character types.

    A generic high-confidence class transition can attach a one-byte prefix to
    the following run. This is an explicit unsupervised morphology architecture,
    not a claim that those attachment decisions were learned through RL.
    """
    counts=np.zeros((256,256));freq=np.zeros(256)
    for row in fit:
        x=np.frombuffer(row['latex'].encode(),dtype=np.uint8)
        np.add.at(freq,x,1.)
        if len(x)>1:np.add.at(counts,(x[:-1],x[1:]),1.)
    outgoing=(counts+.05)/(counts.sum(axis=1,keepdims=True)+12.8)
    incoming=(counts.T+.05)/(counts.sum(axis=0)[:,None]+12.8)
    features=np.sqrt(np.c_[outgoing,incoming]);seen=np.flatnonzero(freq>=4)
    # Deterministic farthest-center initialization, no linguistic seeds.
    centers=[features[seen[np.argmax(freq[seen])]]]
    for _ in range(classes-1):
        distance=np.min(np.sum((features[seen,None,:]-np.array(centers)[None,:,:])**2,axis=2),axis=1)
        centers.append(features[seen[np.argmax(distance)]])
    centers=np.array(centers)
    for _ in range(40):
        labels=np.argmin(np.sum((features[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        new=centers.copy()
        for k in range(classes):
            selected=seen[labels[seen]==k]
            if len(selected):new[k]=features[selected].mean(axis=0)
        if np.max(np.abs(new-centers))<1e-8:break
        centers=new
    nextclass=np.zeros((256,classes))
    for b in range(256):
        for k in range(classes):nextclass[b,k]=counts[b,labels==k].sum()
    nextclass/=np.maximum(1,nextclass.sum(axis=1,keepdims=True))
    units=Counter()
    for row in fit:
        raw=row['latex'].encode();i=0
        while i<len(raw):
            start=i;k=int(labels[raw[i]]);i+=1
            while i<len(raw) and labels[raw[i]]==k and i-start<32:i+=1
            piece=raw[start:i]
            if len(piece)>1:units[piece]+=1
            if start>0 and len(piece)>1 and labels[raw[start-1]]!=k and nextclass[raw[start-1],k]>=.8:
                units[raw[start-1:i]]+=1
    candidates=sorted((p for p,n in units.items() if n>=2),key=lambda p:(-units[p]*(len(p)-1),p))
    atoms=[bytes([i]) for i in range(256)]+candidates[:min(unit_budget,capacity-256)]
    info={'classes':classes,'labels':labels.tolist(),'fit_ids':[r['id'] for r in fit],
          'cluster_training':'neighbor-distribution k-means; no byte character classes or command table',
          'attachment_prior':'one-byte prefix when observed next-cluster probability >=0.8',
          'chosen_units':len(atoms)-256,'candidate_count':len(candidates)}
    return atoms,info


def select_context_units(fit,reward,capacity,reference_rate):
    best=None;logs=[]
    for classes in (4,8,12):
        atoms,info=discover_context_classes(fit,capacity,classes)
        model=LearnedUnitBPE.train(fit,atoms,capacity)
        enc=[model.encode(r['latex']) for r in fit];rr=[model.encode(r['latex']) for r in reward]
        rate=paper_token_rate(rr,reward);code=TokenCodePredictor(enc,len(model.vocab)).score(rr,reward)['paper_macro_token_bigram_code_bits_per_byte']
        violation=max(0.,rate/(reference_rate*1.01)-1.)
        info.update(paper_tokens_per_byte=rate,paper_code_bpb=code,violation=violation);logs.append(info)
        key=(violation>0,violation,code,classes)
        if best is None or key<best[0]:best=(key,model,info)
    best[1].training_log=[{'candidates':logs,'selected':best[2],'selection':'training reward code with compression guard'}]
    return best[1],logs
