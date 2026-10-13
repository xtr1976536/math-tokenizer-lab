"""Exact learned token-transition signal in dictionary path inference.

Prediction code is learned on fit only. Alternating fitting is not global EM:
compression has a separate deployment constraint. This is unsupervised signal
learning, not RL or a mathematical world model.
"""
import hashlib,importlib.util,json,os,platform,subprocess,sysconfig
from pathlib import Path
import numpy as np
from .corpus import ROOT
from .signal_stage2 import SignalTokenizer,TokenCodePredictor,contexts,paper_token_rate

_bridge=None
def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    source=ROOT/'native/predictive_stage2.c';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('predictive-stage2-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        cmd=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(tmp),'-lm']
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,dest)
    spec=importlib.util.spec_from_file_location('_predictive_stage2',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


class PredictiveTokenizer(SignalTokenizer):
    def __init__(self,vocab,bits,rate_cost=0.,costs=None):
        super().__init__(vocab,costs);v=len(vocab)
        self.bits=np.ascontiguousarray(bits,dtype=np.float64);self.rate_cost=float(rate_cost)
        if self.bits.shape!=(v+2,v+1) or not np.isfinite(self.bits).all() or self.rate_cost<0 or not np.isfinite(self.rate_cost):
            raise ValueError('invalid predictive table or rate cost')
        self.lengths=np.array(list(map(len,vocab)),dtype=np.int32);self.predictive_native=bridge()
    def encode(self,text):return self.predictive_native.encode(text,self.children,self.terminal,self.lengths,self.bits,self.costs,self.maxlen,len(self.vocab),self.rate_cost)
    def encode_with_cost(self,text):return self.predictive_native.encode(text,self.children,self.terminal,self.lengths,self.bits,self.costs,self.maxlen,len(self.vocab),self.rate_cost,True)
    def encode_reference(self,text):
        raw=text.encode();v=len(self.vocab);cs=contexts(raw);states=[{} for _ in range(len(raw)+1)];states[0][v+1]=(0.,[])
        for a in range(len(raw)):
            for previous,(value,path) in list(states[a].items()):
                for id,p in enumerate(self.vocab):
                    if raw.startswith(p,a):
                        e=a+len(p);cost=value+self.bits[previous,id]+self.rate_cost+(self.costs[cs[e]] if e<len(raw) else 0.)
                        if id not in states[e] or cost<states[e][id][0]-1e-12:states[e][id]=(cost,path+[id])
        choices=[(cost+self.bits[id,v],path) for id,(cost,path) in states[-1].items()]
        value,path=min(choices,key=lambda x:x[0]);return path,value
    def save(self,path):
        np.savez_compressed(Path(path),vocab=np.array([p.hex() for p in self.vocab]),costs=self.costs,bits=self.bits,
            rate_cost=self.rate_cost,source_sha256=hashlib.sha256((ROOT/'native/predictive_stage2.c').read_bytes()).hexdigest(),
            train_seconds=self.train_seconds,training_log=json.dumps(self.training_log))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['source_sha256'])!=hashlib.sha256((ROOT/'native/predictive_stage2.c').read_bytes()).hexdigest():raise ValueError('native source mismatch')
            obj=cls([bytes.fromhex(p) for p in d['vocab']],d['bits'],float(d['rate_cost']),d['costs'])
            obj.train_seconds=float(d['train_seconds']);obj.training_log=json.loads(str(d['training_log']));return obj


def train_predictive(vocab,fit,reward,reference_rate,rounds=3):
    base=SignalTokenizer(vocab);enc=[base.encode(r['latex']) for r in fit];log=[]
    for iteration in range(rounds):
        predictor=TokenCodePredictor(enc,len(vocab));best=None;candidates=[]
        for rate_cost in (0.,1.,2.,4.,8.,16.):
            model=PredictiveTokenizer(vocab,predictor.bits,rate_cost)
            rr=[model.encode(r['latex']) for r in reward];rate=paper_token_rate(rr,reward)
            code=predictor.score(rr,reward)['paper_macro_token_bigram_code_bits_per_byte']
            violation=max(0.,rate/(reference_rate*1.01)-1.)
            key=(violation>0,violation,code,rate_cost)
            candidates.append({'rate_cost':rate_cost,'paper_tokens_per_byte':rate,'paper_code_bpb':code,'violation':violation})
            if best is None or key<best[0]:best=(key,model)
        model=best[1];enc=[model.encode(r['latex']) for r in fit]
        log.append({'round':iteration,'candidates':candidates,'selected_rate_cost':model.rate_cost,
                    'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward]})
    model.training_log=log;return model
