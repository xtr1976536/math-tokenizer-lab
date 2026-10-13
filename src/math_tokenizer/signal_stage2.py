"""Fit-only byte signals, compiled dictionary inference, prediction-code probes.

All signal lookup keys are prior raw bytes. No syntax/category metadata is used.
The dictionary DP may inspect its input as ordinary offline tokenizers do; it
must not be used as a causal boundary policy inside a byte language model.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sysconfig
from pathlib import Path
import numpy as np
from .corpus import ROOT

CONTEXTS=257*257
_bridge=None


def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    source=ROOT/'native/signal_stage2.c';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('signal-stage2-'+digest+sysconfig.get_config_var('EXT_SUFFIX'))
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        cmd=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(tmp),'-lm']
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,dest)
    spec=importlib.util.spec_from_file_location('_signal_stage2',dest)
    _bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


def contexts(raw):
    """At position i, use bytes strictly before i; EOS context at position n."""
    x=np.frombuffer(raw,dtype=np.uint8).astype(np.int32)
    previous=np.r_[256,x];older=np.r_[256,256,x[:-1]] if len(x) else np.array([256])
    return older*257+previous


class ByteSignals:
    names=('entropy','surprise','distribution_change')

    @classmethod
    def fit(cls,rows,beta=8.):
        if beta<=0:raise ValueError('positive smoothing required')
        counts1=np.zeros((257,256),dtype=np.float64);sparse=defaultdict(Counter)
        for r in rows:
            raw=r['latex'].encode();cs=contexts(raw)
            for c,b in zip(cs,raw):
                counts1[c%257,b]+=1;sparse[int(c)][b]+=1
        base=(counts1+.5)/(counts1.sum(axis=1,keepdims=True)+128.)
        features=np.zeros((CONTEXTS,3));mass=np.zeros(CONTEXTS)
        for c,count in sparse.items():
            total=sum(count.values());p=beta*base[c%257].copy()
            for b,n in count.items():p[b]+=n
            p/=total+beta;q=base[c%257];mid=(p+q)/2
            features[c,0]=-np.sum(p*np.log2(p))
            a,b=divmod(c,257)
            features[c,1]=-np.log2(base[a,b]) if b<256 else 0.
            features[c,2]=.5*np.sum(p*np.log2(p/mid)+q*np.log2(q/mid))
            mass[c]=total
        weights=mass/max(1,mass.sum());mean=weights@features
        scale=np.sqrt(weights@((features-mean)**2));scale=np.maximum(scale,1e-8)
        normalized=(features-mean)/scale;normalized[mass==0]=0.
        obj=cls();obj.features=normalized;obj.mass=mass;obj.base=base;obj.mean=mean;obj.scale=scale
        obj.fit_ids=[r['id'] for r in rows];obj.beta=beta
        return obj

    def save(self,path):
        np.savez_compressed(path,features=self.features,mass=self.mass,base=self.base,
                            mean=self.mean,scale=self.scale,fit_ids=np.array(self.fit_ids),beta=self.beta)


class SignalTokenizer:
    def __init__(self,vocab,costs=None):
        self.vocab=list(vocab)
        if len(set(self.vocab))!=len(self.vocab) or any(not isinstance(p,bytes) or not p for p in self.vocab):
            raise ValueError('unique nonempty byte vocabulary required')
        if not all(bytes([b]) in self.vocab for b in range(256)):raise ValueError('all byte fallbacks required')
        self.costs=np.ascontiguousarray(np.zeros(CONTEXTS) if costs is None else costs,dtype=np.float64)
        if self.costs.shape!=(CONTEXTS,) or not np.isfinite(self.costs).all():raise ValueError('invalid signal table')
        children=[{}];terminals=[-1]
        for id,piece in enumerate(self.vocab):
            node=0
            for b in piece:
                if b not in children[node]:children[node][b]=len(children);children.append({});terminals.append(-1)
                node=children[node][b]
            terminals[node]=id
        self.children=np.full((len(children),256),-1,dtype=np.int32)
        for i,row in enumerate(children):
            for b,v in row.items():self.children[i,b]=v
        self.terminal=np.array(terminals,dtype=np.int32);self.maxlen=max(map(len,self.vocab))
        self.capacity=len(self.vocab);self.train_seconds=0.;self.training_log=[];self.native=bridge()

    def encode(self,text):return self.native.encode(text,self.children,self.terminal,self.costs,self.maxlen)
    def encode_with_cost(self,text):return self.native.encode(text,self.children,self.terminal,self.costs,self.maxlen,True)
    def decode_bytes(self,ids):return b''.join(self.vocab[i] for i in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()
    def encode_reference(self,text):
        raw=text.encode();cs=contexts(raw);best=[float('inf')]*(len(raw)+1);path=[None]*(len(raw)+1);best[-1]=0.
        byfirst=defaultdict(list)
        for id,p in enumerate(self.vocab):byfirst[p[0]].append((id,p))
        for a in range(len(raw)-1,-1,-1):
            # Match native increasing-length traversal and deterministic ties.
            for id,p in sorted(byfirst[raw[a]],key=lambda item:len(item[1])):
                if raw.startswith(p,a):
                    end=a+len(p);value=1.+(self.costs[cs[end]] if end<len(raw) else 0.)+best[end]
                    if value<best[a]-1e-12:best[a]=value;path[a]=(end,id)
        ids=[];a=0
        while a<len(raw):a,id=path[a];ids.append(id)
        return ids,best[0]
    def save(self,path):
        path=Path(path);np.savez_compressed(path,vocab=np.array([p.hex() for p in self.vocab]),costs=self.costs,
            source_sha256=hashlib.sha256((ROOT/'native/signal_stage2.c').read_bytes()).hexdigest(),
            train_seconds=self.train_seconds,training_log=json.dumps(self.training_log))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['source_sha256'])!=hashlib.sha256((ROOT/'native/signal_stage2.c').read_bytes()).hexdigest():
                raise ValueError('native checkpoint source mismatch')
            obj=cls([bytes.fromhex(p) for p in d['vocab']],d['costs']);obj.train_seconds=float(d['train_seconds'])
            obj.training_log=json.loads(str(d['training_log']));return obj


class TokenCodePredictor:
    """Normalized token Markov distribution with EOS; selected-path code only."""
    def __init__(self,encodings,vocab_size,beta=16.):
        self.vocab_size=vocab_size;self.eos=vocab_size;self.bos=vocab_size+1;self.beta=beta
        counts=np.zeros((vocab_size+2,vocab_size+1),dtype=np.float64)
        for ids in encodings:
            xs=np.array([self.bos]+ids+[self.eos],dtype=np.int32)
            np.add.at(counts,(xs[:-1],xs[1:]),1.)
        uni=counts.sum(axis=0)+.5;uni/=uni.sum()
        self.bits=-np.log2((counts+beta*uni)/(counts.sum(axis=1,keepdims=True)+beta))
    def score(self,encodings,rows):
        result=[];papers=defaultdict(lambda:[0.,0])
        for ids,row in zip(encodings,rows):
            x=np.array([self.bos]+ids+[self.eos]);bits=float(self.bits[x[:-1],x[1:]].sum())
            result.append(bits);papers[row['split_group']][0]+=bits;papers[row['split_group']][1]+=len(row['latex'].encode())
        return {'token_bigram_code_bits_per_byte':sum(result)/max(1,sum(len(r['latex'].encode()) for r in rows)),
                'paper_macro_token_bigram_code_bits_per_byte':float(np.mean([b/n for b,n in papers.values()])),
                'bits':result,'predictor_parameters':int(self.bits.size),
                'interpretation':'chosen token sequence code, including EOS; not marginal raw-string LM NLL'}


def paper_token_rate(encodings,rows):
    groups=defaultdict(lambda:[0,0])
    for ids,r in zip(encodings,rows):groups[r['split_group']][0]+=len(ids);groups[r['split_group']][1]+=len(r['latex'].encode())
    return float(np.mean([t/b for t,b in groups.values()]))


def select_strength(vocab,table,fit,reward,reference_rate,strengths=(0.,.1,.25,.5,1.)):
    """Selection reads prediction code + rate only; no structural metric."""
    evaluations=[];best=None
    for strength in strengths:
        model=SignalTokenizer(vocab,-strength*np.tanh(table))
        fitids=[model.encode(r['latex']) for r in fit];ids=[model.encode(r['latex']) for r in reward]
        score=TokenCodePredictor(fitids,len(vocab)).score(ids,reward)
        rate=paper_token_rate(ids,reward);violation=max(0.,rate/(reference_rate*1.01)-1.)
        key=(violation>0,violation,score['paper_macro_token_bigram_code_bits_per_byte'],strength)
        evaluation={'strength':strength,'paper_tokens_per_byte':rate,'compression_guard_violation':violation,
                    'paper_prediction_code_bpb':score['paper_macro_token_bigram_code_bits_per_byte']}
        evaluations.append(evaluation)
        if best is None or key<best[0]:best=(key,model,evaluation)
    best[1].training_log=[{'selection':'training-internal token-code subject to compression guard',
                            'reward_ids':[r['id'] for r in reward],'candidates':evaluations,'selected':best[2]}]
    return best[1],evaluations
