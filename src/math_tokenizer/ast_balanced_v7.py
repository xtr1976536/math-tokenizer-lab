"""AST family: exact finite-penalty lattice instead of rigid legal merges.

The penalty is dev-selected and explicitly reported; not a semantic guarantee.
Dictionary changes and decoder changes have distinct same-vocabulary controls.
"""
import hashlib,importlib.util,json,os,platform,subprocess,sysconfig
from .corpus import ROOT
from .ast_route_v6 import NativeASTV4

_bridge=None


def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    sources=[ROOT/'native/balanced_v7.c',ROOT/'native/frontend_v6.c']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('balanced-v7-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        temporary=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        command=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(sources[0]),'-o',str(temporary),'-lm']
        if platform.system()=='Darwin':command+=['-undefined','dynamic_lookup']
        subprocess.run(command,check=True,capture_output=True);os.replace(temporary,dest)
    spec=importlib.util.spec_from_file_location('_balanced_v7',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


class BalancedAST(NativeASTV4):
    def __init__(self,vocab,penalty=.5,risk_weights=None,lexicographic=False):
        import math
        if not math.isfinite(penalty) or penalty<0:raise ValueError('penalty must be finite and nonnegative')
        if risk_weights is not None and (len(risk_weights)!=8 or any(not math.isfinite(w) or w<0 for w in risk_weights)):
            raise ValueError('invalid calibrated structural risk weights')
        super().__init__(vocab,False);self.penalty=float(penalty);self.risk_weights=risk_weights;self.lexicographic=bool(lexicographic);self.balanced_bridge=bridge()
    def encode(self,text):return self.balanced_bridge.encode(text,self.children,self.terminal,self.maxlen,self.penalty,False,self.risk_weights,self.lexicographic)
    def encode_with_cost(self,text):return self.balanced_bridge.encode(text,self.children,self.terminal,self.maxlen,self.penalty,True,self.risk_weights,self.lexicographic)
    def objective(self,text,ids=None):
        from .frontend_v6 import tree_ranges
        from .measure_v5 import cross_count
        ids=self.encode(text) if ids is None else ids;bounds=[0]
        for id in ids:bounds.append(bounds[-1]+len(self.vocab[id]))
        if self.risk_weights is not None:
            raw=text.encode();spans=tree_ranges(text).tolist();cost=0.
            for a,b in zip(bounds,bounds[1:]):
                risk=0.
                for s,e in spans:
                    if a<s<b<e or s<a<e<b:
                        bucket=(0 if raw[s:s+1]==b'{' else 1 if raw[s:s+1]==b'\\' else 2)*2+int(e-s>8)
                        risk=max(risk,self.risk_weights[bucket])
                cost+=risk
            return cost if self.lexicographic else len(ids)+self.penalty*cost
        crossing= cross_count(bounds,tree_ranges(text).tolist())
        return crossing if self.lexicographic else len(ids)+self.penalty*crossing
    def save(self,path):
        NativeASTV4.save(self,path);d=json.loads(path.read_text());d.update(format='ast-v4-balanced-lattice-v7',penalty=self.penalty,
            objective='token count + penalty*native-source-tree partial-crossing tokens',
            risk_weights=self.risk_weights,lexicographic=self.lexicographic,balanced_source_sha256=hashlib.sha256((ROOT/'native/balanced_v7.c').read_bytes()).hexdigest());path.write_text(json.dumps(d,indent=2)+'\n')
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text())
        assert d['balanced_source_sha256']==hashlib.sha256((ROOT/'native/balanced_v7.c').read_bytes()).hexdigest()
        assert d['frontend_source_sha256']==hashlib.sha256((ROOT/'native/frontend_v6.c').read_bytes()).hexdigest()
        obj=cls([bytes.fromhex(p) for p in d['vocab_hex']],d['penalty'],d.get('risk_weights'),d.get('lexicographic',False))
        for key in ('capacity','train_seconds','training_log'):setattr(obj,key,d[key])
        return obj
