"""AST-family dictionary path with native parameter-boundary distortion."""
import hashlib, importlib.util, json, math, os, platform, subprocess, sysconfig
from .corpus import ROOT
from .ast_route_v6 import NativeASTV4
_bridge=None


def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    sources=[ROOT/'native/boundary_v8.c',ROOT/'native/frontend_v6.c']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('boundary-v8-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        cmd=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(sources[0]),'-o',str(tmp),'-lm']
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,dest)
    spec=importlib.util.spec_from_file_location('_boundary_v8',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


def brace_boundaries(text):
    from .frontend_v6 import reference_lex
    raw=text.encode();stack=[];positions=set()
    for a,b in reference_lex(text):
        if raw[a:b]==b'{':stack.append(a)
        elif raw[a:b]==b'}' and stack:
            start=stack.pop()
            if a>start+1:positions.update((start+1,a))
    return positions-{0,len(raw)}


class BoundaryAST(NativeASTV4):
    def __init__(self,vocab,tree_penalty=.25,boundary_penalty=.15):
        if any(not math.isfinite(v) or v<0 for v in (tree_penalty,boundary_penalty)):raise ValueError('finite nonnegative weights required')
        super().__init__(vocab,False)
        self.tree_penalty=float(tree_penalty);self.boundary_penalty=float(boundary_penalty);self.boundary_bridge=bridge()
    def encode(self,text):return self.boundary_bridge.encode(text,self.children,self.terminal,self.maxlen,self.tree_penalty,self.boundary_penalty)
    def encode_with_cost(self,text):return self.boundary_bridge.encode(text,self.children,self.terminal,self.maxlen,self.tree_penalty,self.boundary_penalty,True)
    def objective(self,text,ids=None):
        from .frontend_v6 import tree_ranges
        from .measure_v5 import cross_count
        ids=self.encode(text) if ids is None else ids;bounds=[0]
        for i in ids:bounds.append(bounds[-1]+len(self.vocab[i]))
        return len(ids)+self.tree_penalty*cross_count(bounds,tree_ranges(text).tolist())+self.boundary_penalty*len(brace_boundaries(text)-set(bounds))
    def save(self,path):
        super().save(path);d=json.loads(path.read_text());d.update(format='ast-v4-boundary-lattice-v8',tree_penalty=self.tree_penalty,
            boundary_penalty=self.boundary_penalty,boundary_source_sha256=hashlib.sha256((ROOT/'native/boundary_v8.c').read_bytes()).hexdigest())
        path.write_text(json.dumps(d,indent=2)+'\n')
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text())
        assert d['boundary_source_sha256']==hashlib.sha256((ROOT/'native/boundary_v8.c').read_bytes()).hexdigest()
        assert d['frontend_source_sha256']==hashlib.sha256((ROOT/'native/frontend_v6.c').read_bytes()).hexdigest()
        obj=cls([bytes.fromhex(p) for p in d['vocab_hex']],d['tree_penalty'],d['boundary_penalty'])
        for k in ('capacity','train_seconds','training_log'):setattr(obj,k,d[k])
        return obj
