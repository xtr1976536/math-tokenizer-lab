"""AST-v4 continuation: native source tree and shortest constrained encoding."""
import json
import time
import hashlib
from pathlib import Path
import numpy as np
from .atoms_v4 import GrammarBPE
from .constrained_v4 import ConstrainedTokenizer, library4
from .frontend_v6 import library, tree_ranges, structure_native, python_bridge
from .corpus import ROOT


class NativeASTV4(ConstrainedTokenizer):
    def __init__(self,vocab,hard_tree=True):
        super().__init__(vocab,hard_tree,'commands','shortest')
        library()
        self.bridge=python_bridge()

    @classmethod
    def train(cls,texts,capacity,atom_budget=256,subtree_budget=0):
        start=time.perf_counter()
        trainer=GrammarBPE(True).train(texts,capacity,atom_budget=atom_budget,subtree_budget=subtree_budget,structure_fn=structure_native)
        obj=cls(trainer.vocab);obj.capacity=capacity;obj.train_seconds=time.perf_counter()-start
        obj.training_log=trainer.training_log;return obj

    def encode(self,text):
        return self.bridge.ast_encode(text,self.children,self.terminal,self.maxlen,int(self.structural))

    def encode_ctypes(self,text):
        raw=text.encode();n=len(raw)
        if not n:return []
        allowed=np.empty(n+1,dtype=np.uint8)
        library().mt_allowed(raw,n,self.children.ctypes.data,self.terminal.ctypes.data,allowed.ctypes.data)
        spans=tree_ranges(text);out=np.empty(n,dtype=np.int32)
        count=library4().mt_shortest(raw,n,self.children.ctypes.data,self.terminal.ctypes.data,self.maxlen,spans.ctypes.data,len(spans),int(self.structural),allowed.ctypes.data,out.ctypes.data)
        if count<0:raise RuntimeError('native AST constraint graph disconnected')
        return out[:count].tolist()

    def save(self,path):
        super().save(path);d=json.loads(path.read_text());d['format']='ast-v4-native-binding-v6';d['frontend']='native_supported_arithmetic_ast_and_binding_cst_fallback';d['frontend_source_sha256']=hashlib.sha256((ROOT/'native/frontend_v6.c').read_bytes()).hexdigest();path.write_text(json.dumps(d,indent=2))

    @classmethod
    def load(cls,path):
        path=Path(path)
        d=json.loads(path.read_text());obj=cls([bytes.fromhex(p) for p in d['vocab_hex']],d['structural'])
        if d.get('frontend_source_sha256') and d['frontend_source_sha256']!=hashlib.sha256((ROOT/'native/frontend_v6.c').read_bytes()).hexdigest():raise ValueError('native AST checkpoint frontend source changed; retrain or use pinned source')
        for key in ('capacity','train_seconds','training_log'):setattr(obj,key,d[key])
        return obj
