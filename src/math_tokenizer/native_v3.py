"""Portable C/ctypes inference. Rebuilds from checked-in source automatically.

Exact-v2 native mode is a speed optimization, NOT an algorithmic improvement.
Structure parser unchanged for comparable speed and structural metrics.
"""
from pathlib import Path
import ctypes,hashlib,subprocess,platform,json,time
import numpy as np
from .corpus import ROOT
from .structure_v3 import structure_fast as structure

_lib=None

def library():
    global _lib
    if _lib is not None:return _lib
    source=ROOT/'native/decoder.c';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    path=ROOT/'.native_cache'/f'decoder-{platform.machine()}-{digest}.{"dylib" if platform.system()=="Darwin" else "so"}'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        subprocess.run(['cc','-O3','-fPIC','-shared',str(source),'-o',str(path),'-lm'],check=True,capture_output=True,text=True)
    _lib=ctypes.CDLL(str(path))
    _lib.mt_decode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,
         ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
         ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,
         ctypes.c_void_p,ctypes.c_void_p]
    _lib.mt_decode.restype=ctypes.c_int
    _lib.mt_matches.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,
                             ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    _lib.mt_matches.restype=ctypes.c_int
    return _lib

class NativeTokenizer:
    def __init__(self,vocab,structural=False,mode='shortest',theta=None,frequency=None,splitfee=0.,crossfee=0.,token_cost=None):
        self.vocab=list(vocab)
        if not self.vocab or any(not isinstance(t,bytes) or not t for t in self.vocab):raise ValueError('vocabulary must contain nonempty byte payloads')
        if len(set(self.vocab))!=len(self.vocab):raise ValueError('duplicate vocabulary payloads')
        if not all(bytes([b]) in self.vocab for b in range(256)):raise ValueError('complete byte fallback required')
        if mode not in ('shortest','policy_map','policy_structure'):raise ValueError('unsupported native mode')
        self.structural=structural;self.mode=mode;self.splitfee=splitfee;self.crossfee=crossfee
        self.theta=np.array(theta if theta is not None else [0.]*7,dtype=np.float64)
        self.frequency=np.array(frequency if frequency is not None else np.ones(len(vocab)),dtype=np.float64)
        if not 1<=len(self.theta)<=7 or not np.isfinite(self.theta).all():raise ValueError('invalid policy dimension/values')
        if len(self.frequency)!=len(vocab) or not np.isfinite(self.frequency).all() or np.any(self.frequency<=0):raise ValueError('invalid frequency table')
        self.logfreq=np.log(self.frequency);self.token_cost=np.array(token_cost if token_cost is not None else np.zeros(len(vocab)),dtype=np.float64)
        if len(self.token_cost)!=len(vocab) or not np.isfinite(self.token_cost).all():raise ValueError('invalid token costs')
        self.maxlen=max(map(len,vocab));children=[];terminal=[]
        children.append({});terminal.append(-1)
        for id,token in enumerate(vocab):
            node=0
            for b in token:
                if b not in children[node]:children[node][b]=len(children);children.append({});terminal.append(-1)
                node=children[node][b]
            terminal[node]=id
        self.children=np.full((len(children),256),-1,dtype=np.int32)
        for i,row in enumerate(children):
            for b,target in row.items():self.children[i,b]=target
        self.terminal=np.array(terminal,dtype=np.int32);self.train_seconds=0.;self.capacity=len(vocab);self.training_log=[]
        library()
    @classmethod
    def from_v2(cls,model):
        obj=cls(model.vocab,model.structural,model.mode,model.theta,model.frequency,model.config.structure_weight,model.config.structure_weight)
        obj.train_seconds=model.train_seconds;return obj

    def encode(self,text):
        raw=text.encode();n=len(raw)
        if not n:return []
        s=structure(text)
        spans=np.array(s.spans,dtype=np.int32).reshape(-1,2) if s.kind!='byte_fallback' else np.empty((0,2),dtype=np.int32)
        safe=np.zeros(n+1,dtype=np.uint8);safe[0]=safe[n]=1;interior=np.zeros(n+1,dtype=np.uint8)
        for a,b,t in s.lexemes:
            safe[a]=safe[b]=1
            if t.startswith('\\') or t[0:1].isdigit():interior[a+1:b]=1
        return self.encode_prepared(raw,spans,safe,interior)

    def encode_prepared(self,raw,spans,safe,interior):
        n=len(raw);out=np.empty(n,dtype=np.int32)
        count=library().mt_decode(raw,n,self.children.ctypes.data,self.terminal.ctypes.data,self.maxlen,
             spans.ctypes.data,len(spans),int(self.structural),safe.ctypes.data,interior.ctypes.data,self.logfreq.ctypes.data,
             self.theta.ctypes.data,len(self.theta),3 if self.mode=='policy_structure' else 1 if self.mode=='policy_map' else 0,self.splitfee,self.crossfee,0.,
             self.token_cost.ctypes.data if self.mode not in ('policy_map','policy_structure') else None,out.ctypes.data)
        if count<0:raise RuntimeError('native decoder error '+str(count))
        return out[:count].tolist()
    def matches(self,text):
        raw=text.encode();cap=max(1,len(raw)*self.maxlen);a=np.empty(cap,dtype=np.int32);b=np.empty(cap,dtype=np.int32);ids=np.empty(cap,dtype=np.int32)
        n=library().mt_matches(raw,len(raw),self.children.ctypes.data,self.terminal.ctypes.data,self.maxlen,a.ctypes.data,b.ctypes.data,ids.ctypes.data,cap)
        if n<0:raise RuntimeError('native match buffer insufficient')
        return a[:n],b[:n],ids[:n]
    def decode_bytes(self,ids):return b''.join(self.vocab[i] for i in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()
    def save(self,path):
        d={'format':'native-math-v3','vocab_hex':[v.hex() for v in self.vocab],'structural':self.structural,'mode':self.mode,
           'theta':self.theta.tolist(),'frequency':self.frequency.tolist(),'splitfee':self.splitfee,'crossfee':self.crossfee,'token_cost':self.token_cost.tolist(),
           'capacity':self.capacity,'train_seconds':self.train_seconds,'training_log':self.training_log,'native_source_sha256':hashlib.sha256((ROOT/'native/decoder.c').read_bytes()).hexdigest()}
        path.write_text(json.dumps(d,indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text());obj=cls([bytes.fromhex(x) for x in d['vocab_hex']],d['structural'],d['mode'],d['theta'],d['frequency'],d['splitfee'],d['crossfee'],d['token_cost'])
        obj.train_seconds=d['train_seconds'];obj.capacity=d['capacity'];obj.training_log=d['training_log'];return obj
