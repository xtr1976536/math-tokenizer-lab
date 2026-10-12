from collections import Counter
import heapq,hashlib,importlib.util,os,platform,subprocess,sysconfig
from .corpus import ROOT
from .rl_vocab_v5 import Indexed

_bridge=None


def bridge():
    global _bridge
    if _bridge is not None:return _bridge
    source=ROOT/'native/fit_index_v7.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest=ROOT/'.native_cache'/('fit-index-v7-'+digest+sysconfig.get_config_var('EXT_SUFFIX'));dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        temporary=dest.with_name(dest.name+'.'+str(os.getpid())+'.tmp')
        command=['c++','-O3','-std=c++17','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(temporary)]
        if platform.system()=='Darwin':command+=['-undefined','dynamic_lookup']
        subprocess.run(command,check=True,capture_output=True);os.replace(temporary,dest)
    spec=importlib.util.spec_from_file_location('_fit_index_v7',dest);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge)
    return _bridge


class NativeFitIndexed(Indexed):
    @staticmethod
    def scan(seq,cuts=None):
        if cuts is not None:raise ValueError('fit-only index cannot measure reward cuts')
        raw,gain=bridge().scan(seq.ids)
        return Counter(raw),Counter(gain),Counter()

    def merge(self,pair,new):
        changed={}
        for i in sorted(self.index[pair].copy()):
            old,oldgain,_=self.local[i];seq=self.seqs[i]
            ids,starts,ends,raw,gain=bridge().merge(seq.ids,seq.starts,seq.ends,*pair,new)
            seq.ids,seq.starts,seq.ends=ids,starts,ends;current=Counter(raw);gain=Counter(gain)
            self.local[i]=(current,gain,Counter())
            for p in old.keys()|current.keys():
                if p not in changed:changed[p]=self.counts[p]
                self.counts[p]+=current[p]-old[p];self.gains[p]+=gain[p]-oldgain[p]
                if not current[p]:self.index[p].discard(i)
                elif not old[p]:self.index[p].add(i)
        for p,count in changed.items():
            if self.counts[p]==count:continue
            self.generation[p]+=1
            if self.counts[p]>0:heapq.heappush(self.heap,(-self.counts[p],p,self.generation[p]))
        if len(self.heap)>4*max(1,len(self.counts)):
            self.heap=[(-c,p,self.generation[p]) for p,c in self.counts.items() if c>0];heapq.heapify(self.heap)
