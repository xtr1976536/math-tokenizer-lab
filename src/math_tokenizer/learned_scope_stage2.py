"""Generic paired-byte law discovery, native scope signal, deployed-cost RL.

Any inherited math-atom alphabet is explicitly a mathematical lexical prior.
Pair identities are learned from fit; the stack algorithm is a human structural
architecture prior. No AST or evaluation quality label participates in learning.
"""
from collections import Counter
import hashlib,importlib.util,json,os,platform,subprocess,sysconfig,time
from pathlib import Path
import numpy as np
from .corpus import ROOT
from .signal_stage2 import SignalTokenizer,TokenCodePredictor,contexts,paper_token_rate,CONTEXTS

_bridge=None
def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    source=ROOT/'native/learned_scope_stage2.c';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('learned-scope-stage2-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        cmd=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(tmp),'-lm']
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,dest)
    spec=importlib.util.spec_from_file_location('_learned_scope_stage2',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


def discover_pairs(fit,min_support=32,min_confidence=.9):
    raws=[r['latex'].encode() for r in fit];counts=[Counter(raw) for raw in raws];globalcount=Counter()
    for count in counts:globalcount.update(count)
    symbols=sorted(b for b,n in globalcount.items() if n>=min_support);candidates=[]
    for a in symbols:
        for b in symbols:
            if a==b:continue
            support=[i for i,c in enumerate(counts) if c[a] and c[b]]
            if len(support)<min_support:continue
            equal=[i for i in support if counts[i][a]==counts[i][b]]
            if len(equal)/len(support)<min_confidence:continue
            good=0
            for i in equal:
                depth=0;valid=True
                for x in raws[i]:
                    if x==a:depth+=1
                    elif x==b:
                        depth-=1
                        if depth<0:valid=False;break
                good+=valid and depth==0
            confidence=good/len(support)
            if confidence>=min_confidence:candidates.append({'open':a,'close':b,'confidence':confidence,'support':len(support)})
    candidates.sort(key=lambda r:(-r['support']*r['confidence'],r['open'],r['close']))
    chosen=[];used=set()
    for c in candidates:
        if c['open'] not in used and c['close'] not in used:chosen.append(c);used.update((c['open'],c['close']))
    return [(r['open'],r['close']) for r in chosen],{'fit_ids':[r['id'] for r in fit],
            'candidates':candidates,'selected':chosen,'prior':'paired-byte count equality and prefix balance, no fixed character identities'}


class LearnedScopeTokenizer(SignalTokenizer):
    def __init__(self,vocab,atoms,pairs,scope_cost=0.,costs=None,token_costs=None):
        super().__init__(vocab,costs);self.atoms=list(atoms)
        if not set(atoms)<=set(vocab):raise ValueError('all protected primitives must be vocabulary entries')
        self.primitive=SignalTokenizer(atoms);self.pairs=np.array(pairs,dtype=np.int32).reshape(-1,2)
        if np.any(self.pairs<0) or np.any(self.pairs>255):raise ValueError('byte pairs required')
        self.scope_cost=float(scope_cost);self.token_costs=np.ascontiguousarray(np.zeros(len(vocab)) if token_costs is None else token_costs,dtype=np.float64)
        if self.token_costs.shape!=(len(vocab),) or not np.isfinite(self.token_costs).all() or not np.isfinite(scope_cost) or scope_cost<0:raise ValueError('invalid costs')
        self.scope_native=bridge()
    def encode(self,text):return self.scope_native.encode(text,self.children,self.terminal,self.primitive.children,self.primitive.terminal,
        self.pairs,self.costs,self.token_costs,self.maxlen,self.scope_cost)
    def encode_with_cost(self,text):return self.scope_native.encode(text,self.children,self.terminal,self.primitive.children,self.primitive.terminal,
        self.pairs,self.costs,self.token_costs,self.maxlen,self.scope_cost,True)
    def prepared_reference(self,text):
        raw=text.encode();allowed=np.ones(len(raw)+1,dtype=bool);a=0;units=sorted(self.atoms,key=len,reverse=True)
        while a<len(raw):
            p=next(p for p in units if raw.startswith(p,a));allowed[a+1:a+len(p)]=False;a+=len(p)
        marked=set()
        for x,y in self.pairs:
            stack=[]
            for i,b in enumerate(raw):
                if b==x:stack.append(i)
                elif b==y and stack:
                    start=stack.pop()
                    if i>start+1:marked.update((start+1,i))
        return raw,allowed,marked
    def encode_reference(self,text):
        raw,allowed,marked=self.prepared_reference(text);n=len(raw);best=[float('inf')]*(n+1);paths=[None]*(n+1);best[n]=0.;cs=contexts(raw)
        byfirst={b:[] for b in range(256)}
        for id,p in enumerate(self.vocab):byfirst[p[0]].append((id,p))
        for a in range(n-1,-1,-1):
            if not allowed[a]:continue
            for id,p in sorted(byfirst[raw[a]],key=lambda x:len(x[1])):
                end=a+len(p)
                if raw.startswith(p,a) and allowed[end]:
                    value=1+self.token_costs[id]+self.scope_cost*sum(a<x<end for x in marked)+(self.costs[cs[end]] if end<n else 0)+best[end]
                    if value<best[a]-1e-12:best[a]=value;paths[a]=(end,id)
        a=0;ids=[]
        while a<n:a,id=paths[a];ids.append(id)
        return ids,best[0]
    def save(self,path):
        np.savez_compressed(path,vocab=np.array([p.hex() for p in self.vocab]),atoms=np.array([p.hex() for p in self.atoms]),
          pairs=self.pairs,costs=self.costs,token_costs=self.token_costs,scope_cost=self.scope_cost,
          source_sha256=hashlib.sha256((ROOT/'native/learned_scope_stage2.c').read_bytes()).hexdigest(),
          training_log=json.dumps(self.training_log),train_seconds=self.train_seconds)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['source_sha256'])!=hashlib.sha256((ROOT/'native/learned_scope_stage2.c').read_bytes()).hexdigest():raise ValueError('source mismatch')
            obj=cls([bytes.fromhex(x) for x in d['vocab']],[bytes.fromhex(x) for x in d['atoms']],d['pairs'],float(d['scope_cost']),d['costs'],d['token_costs'])
            obj.training_log=json.loads(str(d['training_log']));obj.train_seconds=float(d['train_seconds']);return obj


def select_scope(vocab,atoms,pairs,fit,reward,rate_guard):
    logs=[];best=None
    for scope_cost in (0.,.1,.3,.6,1.):
        model=LearnedScopeTokenizer(vocab,atoms,pairs,scope_cost)
        fitids=[model.encode(r['latex']) for r in fit];ids=[model.encode(r['latex']) for r in reward]
        code=TokenCodePredictor(fitids,len(vocab)).score(ids,reward)['paper_macro_token_bigram_code_bits_per_byte'];rate=paper_token_rate(ids,reward)
        violation=max(0.,rate/rate_guard-1.);key=(violation>0,violation,code,scope_cost)
        logs.append({'scope_cost':scope_cost,'paper_code_bpb':code,'paper_tokens_per_byte':rate,'violation':violation})
        if best is None or key<best[0]:best=(key,model)
    best[1].training_log=[{'selection':'prediction code subject to compression budget, no quality labels','candidates':logs,
                          'reward_ids':[r['id'] for r in reward]}]
    return best[1],logs


def optimize_deployed_costs(base,fit,reward,seed,algorithm,rounds=16,samples=4,sigma=.25):
    """Parameter score-function RL optimizing the actual deployed MAP encoder.

    Every method shares frozen fit-only token-code teacher and total evaluations.
    This objective is a selected-path coding proxy, not raw-string LM likelihood.
    """
    if algorithm not in ('parameter_rl','random_search','cem_search','no_update'):raise ValueError('unknown optimizer')
    v=len(base.vocab);initial=np.zeros(v);mean=initial.copy();rng=np.random.default_rng(seed)
    predictor=TokenCodePredictor([base.encode(r['latex']) for r in fit],v)
    reference=paper_token_rate([base.encode(r['latex']) for r in reward],reward);lagrange=0.
    first=np.zeros(v);second=np.zeros(v);trace=[];pool=[];started=time.perf_counter()
    for iteration in range(rounds):
        center=initial if algorithm in ('random_search','no_update') else mean
        theta=[center+sigma*rng.normal(size=v) for _ in range(samples)] if algorithm!='no_update' else [center.copy() for _ in range(samples)]
        codes=[];rates=[]
        for t in theta:
            model=LearnedScopeTokenizer(base.vocab,base.atoms,base.pairs,base.scope_cost,base.costs,np.clip(t,-.8,.8))
            ids=[model.encode(r['latex']) for r in reward]
            code=predictor.score(ids,reward)['paper_macro_token_bigram_code_bits_per_byte'];rate=paper_token_rate(ids,reward)
            codes.append(code);rates.append(rate);pool.append((code,rate,t.copy()))
        utility=np.array(codes)+lagrange*(np.array(rates)-reference);advantages=-utility-(-utility).sum()/(samples-1)+(-utility)/(samples-1)
        gradient=np.mean(np.array([(t-center)/sigma**2 for t in theta])*advantages[:,None],axis=0)
        gradient/=max(1.,np.linalg.norm(gradient))
        if algorithm=='parameter_rl':
            first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
            mean+=.08*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8)
        elif algorithm=='cem_search':mean=.5*mean+.5*np.mean([theta[k] for k in np.argsort(utility)[:samples//2]],axis=0)
        lagrange=float(np.clip(lagrange+2*(np.mean(rates)-reference),0,20))
        trace.append({'round':iteration,'mean_code_bpb':float(np.mean(codes)),'mean_tokens_per_byte':float(np.mean(rates)),
                      'gradient_norm':float(np.linalg.norm(gradient)),'lagrange':lagrange})
    # Same incumbent criterion and initial candidate included for all algorithms.
    initialids=[base.encode(r['latex']) for r in reward]
    pool.append((predictor.score(initialids,reward)['paper_macro_token_bigram_code_bits_per_byte'],reference,initial))
    chosen=min(pool,key=lambda x:(x[1]>reference*1.01,max(0,x[1]/(reference*1.01)-1),x[0]))
    model=LearnedScopeTokenizer(base.vocab,base.atoms,base.pairs,base.scope_cost,base.costs,np.clip(chosen[2],-.8,.8))
    model.train_seconds=time.perf_counter()-started;model.training_log=[{'algorithm':algorithm,'seed':seed,'trace':trace,
        'oracle_evaluations':rounds*samples+1,'teacher_sha256':hashlib.sha256(predictor.bits.tobytes()).hexdigest(),
        'same_training_deployment_MAP':True,'selected_code_bpb':chosen[0],'selected_rate':chosen[1],
        'fit_ids':[r['id'] for r in fit],'reward_ids':[r['id'] for r in reward],
        'objective':'frozen fit-only selected-path token Markov code, compression guard; no mathematical quality labels'}]
    return model
