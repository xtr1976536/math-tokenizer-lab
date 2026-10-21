"""Mathematical byte-level adaptation of the AST-BPE hard merge principle.

SQL AST-BPE (Müller and Vlachos, 2019) restricts adjacent merges to neighboring
siblings. Here lexical leaves have byte children and the fixed mathematical
source hierarchy is represented by intervals: partial crossings are forbidden.
No learned primitive reservations; inference applies ranked training merges.
"""
import hashlib,heapq,importlib.util,json,os,platform,subprocess,sysconfig,time
import numpy as np
from .corpus import ROOT
from .atoms_v4 import GrammarBPE,FastSpanSequence
from .frontend_v6 import structure_native
from .ast import compatible

_bridge=None
def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    paths=[ROOT/'native/hard_ranked_2026.c',ROOT/'native/frontend_v6.c']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()[:16]
    target=ROOT/'.native_cache'/('hard-ranked-'+digest+sysconfig.get_config_var('EXT_SUFFIX'))
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
        cmd=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(paths[0]),'-o',str(tmp)]
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,target)
    spec=importlib.util.spec_from_file_location('_hard_ranked_2026',target)
    _bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge);return _bridge

class HardASTBPE:
    def __init__(self,vocab,merges,train_seconds=0):
        self.vocab=vocab;self.merges=merges;self.capacity=len(vocab);self.train_seconds=train_seconds
        size=1
        while size<max(1,2*len(merges)):size*=2
        self.table=np.full((size,4),-1,dtype=np.int32);self.rules={}
        for rank,(a,b,c) in enumerate(merges):
            assert vocab[a]+vocab[b]==vocab[c]
            if (a,b) in self.rules:continue
            self.rules[(a,b)]=(rank,c);i=((a*2654435761)^(b*2246822519))&(size-1)
            while self.table[i,0]>=0:i=(i+1)&(size-1)
            self.table[i]=[a,b,c,rank]
        self.backend=bridge()
    @classmethod
    def train(cls,texts,capacity=1024):
        started=time.perf_counter()
        trainer=GrammarBPE(True).train(texts,capacity,atom_budget=0,subtree_budget=0,structure_fn=structure_native)
        return cls(trainer.vocab,trainer.merges,time.perf_counter()-started)
    def encode(self,text):return self.backend.encode(text,self.table)
    def encode_reference(self,text):
        raw=text.encode();spans=structure_native(text).spans;n=len(raw)
        ids=list(raw);starts=list(range(n));ends=list(range(1,n+1))
        following=list(range(1,n))+[-1] if n else [];previous=list(range(-1,n-1));version=[0]*n;heap=[]
        def offer(l,r):
            if l>=0 and r>=0 and (ids[l],ids[r]) in self.rules:
                rank,new=self.rules[(ids[l],ids[r])];heapq.heappush(heap,(rank,l,r,version[l],version[r],new))
        for i in range(n-1):offer(i,i+1)
        while heap:
            rank,l,r,vl,vr,new=heapq.heappop(heap)
            if following[l]!=r or version[l]!=vl or version[r]!=vr:continue
            if not compatible(starts[l],ends[r],spans):continue
            ids[l]=new;ends[l]=ends[r];version[l]+=1;version[r]+=1;following[l]=following[r];following[r]=-2
            if following[l]>=0:previous[following[l]]=l
            offer(previous[l],l);offer(l,following[l])
        return [ids[i] for i in range(n) if following[i]!=-2]
    def decode_bytes(self,ids):return b''.join(self.vocab[i] for i in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()
    def save(self,path):
        path.write_text(json.dumps({'format':'mathematical-hard-ast-ranked-2026','vocab_hex':[p.hex() for p in self.vocab],
             'merges':self.merges,'train_seconds':self.train_seconds,'capacity':self.capacity},indent=2)+'\n')
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text());return cls([bytes.fromhex(p) for p in d['vocab_hex']],d['merges'],d['train_seconds'])
