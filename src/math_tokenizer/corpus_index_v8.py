"""Exact C++ corpus index, same count/gain/tie semantics as v7 Indexed."""
import hashlib,importlib.util,os,platform,subprocess,sysconfig
from .corpus import ROOT
_bridge=None


def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    source=ROOT/'native/corpus_index_v8.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('corpus-index-v8-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        cmd=['c++','-O3','-std=c++17','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(tmp)]
        if platform.system()=='Darwin':cmd+=['-undefined','dynamic_lookup']
        subprocess.run(cmd,check=True,capture_output=True);os.replace(tmp,dest)
    spec=importlib.util.spec_from_file_location('_corpus_index_v8',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


class CorpusFitIndex:
    def __init__(self,texts,sequences=None):
        if sequences is None:raise ValueError('atomic ID sequences required')
        self.bridge=bridge();self.state=self.bridge.init([s.ids for s in sequences]);self.gains={}
    def top(self,k,min_pair=2):
        rows=self.bridge.top(self.state,k,min_pair);self.gains={(a,b):gain for a,b,gain in rows}
        return [(a,b) for a,b,_ in rows]
    def merge(self,pair,new):self.bridge.merge(self.state,*pair,new)
    def snapshot(self):return self.bridge.snapshot(self.state)
