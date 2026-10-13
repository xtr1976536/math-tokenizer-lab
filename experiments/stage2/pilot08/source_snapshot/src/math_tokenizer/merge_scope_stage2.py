"""Statistically learned BPE merge trees plus learned paired-byte boundaries.

The merge hierarchy is a corpus-derived statistical tree, NOT a mathematical AST.
It is built with a native heap then used as a signal in the same inference call.
"""
import hashlib,heapq,importlib.util,json,os,platform,subprocess,sysconfig
from pathlib import Path
import numpy as np
from .corpus import ROOT
from .learned_scope_stage2 import LearnedScopeTokenizer
from .signal_stage2 import TokenCodePredictor,paper_token_rate,contexts

_bridge=None
def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    source=ROOT/'native/merge_scope_stage2.c';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('merge-scope-stage2-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        cmd=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(tmp),'-lm']
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,dest)
    spec=importlib.util.spec_from_file_location('_merge_scope_stage2',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


def extract_merges(tokenizer):
    d=json.loads(tokenizer.to_str())['model'];v=d['vocab'];return [(v[a],v[b],v[a+b],rank) for rank,(a,b) in enumerate(d['merges'])]


class MergeScopeTokenizer(LearnedScopeTokenizer):
    def __init__(self,vocab,atoms,pairs,merges,scope_cost=0.,tree_cost=0.,costs=None,token_costs=None):
        super().__init__(vocab,atoms,pairs,scope_cost,costs,token_costs)
        self.merges=[tuple(map(int,m)) for m in merges];self.tree_cost=float(tree_cost)
        if not np.isfinite(tree_cost) or tree_cost<0:raise ValueError('nonnegative merge-tree penalty required')
        size=1
        while size<max(1,len(merges)*2):size*=2
        self.merge_table=np.full((size,4),-1,dtype=np.int32);self.merge_lookup={}
        for a,b,new,rank in self.merges:
            if min(a,b,new,rank)<0 or max(a,b,new)>=len(vocab):raise ValueError('invalid merge IDs')
            if vocab[a]+vocab[b]!=vocab[new]:raise ValueError('merge payload mismatch')
            self.merge_lookup[(a,b)]=(rank,new);key=((a*2654435761)^(b*2246822519))&(size-1)
            while self.merge_table[key,0]>=0:key=(key+1)&(size-1)
            self.merge_table[key]=[a,b,new,rank]
        reverse={p:i for i,p in enumerate(vocab)};self.atom_ids=np.array([reverse[p] for p in atoms],dtype=np.int32)
        self.merge_native=bridge()
    def encode(self,text):return self.merge_native.encode(text,self.children,self.terminal,self.primitive.children,self.primitive.terminal,
        self.pairs,self.costs,self.token_costs,self.merge_table,self.atom_ids,self.maxlen,self.scope_cost,self.tree_cost)
    def encode_with_cost(self,text):return self.merge_native.encode(text,self.children,self.terminal,self.primitive.children,self.primitive.terminal,
        self.pairs,self.costs,self.token_costs,self.merge_table,self.atom_ids,self.maxlen,self.scope_cost,self.tree_cost,True)
    def reference_spans(self,text):
        raw,allowed,_=self.prepared_reference(text);n=len(raw);units=sorted(enumerate(self.atoms),key=lambda x:len(x[1]),reverse=True)
        ids=[];starts=[];ends=[];position=0
        while position<n:
            atom,p=next((i,p) for i,p in units if raw.startswith(p,position));ids.append(int(self.atom_ids[atom]));starts.append(position)
            position+=len(p);ends.append(position)
        size=len(ids);following=list(range(1,size))+[-1];previous=list(range(-1,size-1));version=[0]*size;heap=[];spans=[]
        def offer(l,r):
            if l>=0 and r>=0 and (ids[l],ids[r]) in self.merge_lookup:
                rank,new=self.merge_lookup[(ids[l],ids[r])];heapq.heappush(heap,(rank,l,r,version[l],version[r],new))
        for i in range(size-1):offer(i,i+1)
        while heap:
            rank,l,r,vl,vr,new=heapq.heappop(heap)
            if following[l]!=r or version[l]!=vl or version[r]!=vr:continue
            ids[l]=new;ends[l]=ends[r];version[l]+=1;version[r]+=1;following[l]=following[r];following[r]=-2
            if following[l]>=0:previous[following[l]]=l
            spans.append((starts[l],ends[l]));offer(previous[l],l);offer(l,following[l])
        marked=set()
        for x,y in self.pairs:
            stack=[]
            for i,b in enumerate(raw):
                if not (allowed[i] and allowed[i+1]):continue
                if b==x:stack.append(i)
                elif b==y and stack:
                    start=stack.pop()
                    if i>start+1:marked.update((start+1,i))
        final=[i for i in range(size) if following[i]!=-2]
        return raw,allowed,marked,spans,[ids[i] for i in final]
    def encode_reference(self,text):
        raw,allowed,marked,spans,_=self.reference_spans(text);n=len(raw);cs=contexts(raw)
        best=[float('inf')]*(n+1);paths=[None]*(n+1);best[n]=0.;byfirst={b:[] for b in range(256)}
        for id,p in enumerate(self.vocab):byfirst[p[0]].append((id,p))
        for a in range(n-1,-1,-1):
            if not allowed[a]:continue
            for id,p in sorted(byfirst[raw[a]],key=lambda x:len(x[1])):
                e=a+len(p)
                if raw.startswith(p,a) and allowed[e]:
                    cross=any(a<s<e<t or s<a<t<e for s,t in spans)
                    value=1+self.token_costs[id]+self.tree_cost*cross+self.scope_cost*sum(a<x<e for x in marked)+(self.costs[cs[e]] if e<n else 0)+best[e]
                    if value<best[a]-1e-12:best[a]=value;paths[a]=(e,id)
        a=0;ids=[]
        while a<n:a,id=paths[a];ids.append(id)
        return ids,best[0]
    def save(self,path):
        np.savez_compressed(path,vocab=np.array([p.hex() for p in self.vocab]),atoms=np.array([p.hex() for p in self.atoms]),
            pairs=self.pairs,merges=np.array(self.merges,dtype=np.int32).reshape(-1,4),tree_cost=self.tree_cost,scope_cost=self.scope_cost,
            costs=self.costs,token_costs=self.token_costs,train_seconds=self.train_seconds,training_log=json.dumps(self.training_log),
            source_sha256=hashlib.sha256((ROOT/'native/merge_scope_stage2.c').read_bytes()).hexdigest())
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['source_sha256'])!=hashlib.sha256((ROOT/'native/merge_scope_stage2.c').read_bytes()).hexdigest():raise ValueError('source mismatch')
            obj=cls([bytes.fromhex(x) for x in d['vocab']],[bytes.fromhex(x) for x in d['atoms']],d['pairs'],d['merges'],
                float(d['scope_cost']),float(d['tree_cost']),d['costs'],d['token_costs'])
            obj.train_seconds=float(d['train_seconds']);obj.training_log=json.loads(str(d['training_log']));return obj


def select_merge_scope(vocab,atoms,pairs,merges,fit,reward,rate_guard,neural_predictor=None):
    best=None;logs=[]
    for tree in (0.,.1,.3,.6):
        for scope in (0.,.1,.3,.6):
            model=MergeScopeTokenizer(vocab,atoms,pairs,merges,scope,tree)
            fitids=[model.encode(r['latex']) for r in fit];ids=[model.encode(r['latex']) for r in reward]
            code=TokenCodePredictor(fitids,len(vocab)).score(ids,reward)['paper_macro_token_bigram_code_bits_per_byte'];rate=paper_token_rate(ids,reward)
            if neural_predictor is None:feedback=code
            else:
                papers={}
                for row,path in zip(reward,ids):
                    raw=row['latex'].encode();cuts=np.zeros(len(raw)+1,dtype=bool);cuts[0]=cuts[-1]=True;pos=0
                    for id in path:pos+=len(vocab[id]);cuts[pos]=True
                    bits,_=neural_predictor.score(raw,cuts);b,n=papers.get(row['split_group'],(0.,0));papers[row['split_group']]=(b+float(bits.sum()),n+len(raw))
                feedback=float(np.mean([b/n for b,n in papers.values()]))
            violation=max(0.,rate/rate_guard-1.);key=(violation>0,violation,feedback,tree,scope)
            logs.append({'tree_cost':tree,'scope_cost':scope,'code_bpb':code,'feedback':feedback,'rate':rate,'violation':violation})
            if best is None or key<best[0]:best=(key,model)
    best[1].training_log=[{'selection':'neural conditional patch code' if neural_predictor else 'token code',
          'candidates':logs,'reward_ids':[r['id'] for r in reward],
          'causality_note':'dictionary segmentation sees full input; neural code conditional on segmentation is not unconditional raw-string likelihood'}]
    return best[1],logs
