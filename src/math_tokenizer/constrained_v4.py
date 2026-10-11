"""Grammar-derived action masks and native compiled constrained RL policy."""
import ctypes,hashlib,json,platform,subprocess
from pathlib import Path
import numpy as np
from .corpus import ROOT
from .native_v3 import NativeTokenizer
from .structure_v3 import structure_fast

_lib=None

def library4():
 global _lib
 if _lib is not None:return _lib
 source=ROOT/'native/decoder_v4.c';sha=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
 dest=ROOT/'.native_cache'/f'v4-{platform.machine()}-{sha}.{"dylib" if platform.system()=="Darwin" else "so"}'
 dest.parent.mkdir(exist_ok=True)
 if not dest.exists():subprocess.run(['cc','-O3','-fPIC','-shared',str(source),'-o',str(dest),'-lm'],check=True,capture_output=True,text=True)
 lib=ctypes.CDLL(str(dest));lib.mt_decode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_void_p,ctypes.c_void_p];lib.mt_decode.restype=ctypes.c_int;_lib=lib;return lib

class ConstrainedTokenizer(NativeTokenizer):
 def __init__(self,vocab,hard_tree=True,atomic='commands',mode='policy_map',theta=None,frequency=None):
  super().__init__(vocab,hard_tree,'policy_map' if mode=='greedy' else mode,theta,frequency)
  self.inference=mode;self.atomic=atomic;self.payload_set=set(vocab);self.atom_exceptions=0
  library4()
 def prepare(self,text):
  raw=text.encode();n=len(raw);s=structure_fast(text);spans=np.array(s.spans,dtype=np.int32).reshape(-1,2) if s.kind!='byte_fallback' else np.empty((0,2),dtype=np.int32)
  safe=np.zeros(n+1,dtype=np.uint8);safe[0]=safe[n]=1;interior=np.zeros(n+1,dtype=np.uint8);allowed=np.ones(n+1,dtype=np.uint8);exceptions=[]
  # Unicode codepoint atomism is grammar-independent, but a complete codepoint
  # must exist in vocab for a one-ID encoding; otherwise explicit byte escape.
  if not text.isascii():
   offset=0
   for char in text:
    b=char.encode();end=offset+len(b)
    if len(b)>1 and b in self.payload_set:allowed[offset+1:end]=0
    elif len(b)>1:exceptions.append((offset,end,'uncovered_unicode'))
    offset=end
  for a,b,t in s.lexemes:
   safe[a]=safe[b]=1
   command=t.startswith('\\');numeric=t[0:1].isdigit()
   if command or numeric:interior[a+1:b]=1
   protect=command or (self.atomic=='commands_numbers' and numeric)
   if protect and b-a>1:
    if raw[a:b] in self.payload_set:allowed[a+1:b]=0
    else:exceptions.append((a,b,'uncovered_command' if command else 'uncovered_number'))
  return raw,spans,safe,interior,allowed,exceptions,s
 def encode(self,text):
  if not text:return []
  raw,spans,safe,interior,allowed,exceptions,s=self.prepare(text);return self.encode_prepared4(raw,spans,safe,interior,allowed)
 def encode_prepared4(self,raw,spans,safe,interior,allowed):
  out=np.empty(len(raw),dtype=np.int32);mode=4 if self.inference=='greedy' else 1 if self.inference=='policy_map' else 0
  count=library4().mt_decode(raw,len(raw),self.children.ctypes.data,self.terminal.ctypes.data,self.maxlen,spans.ctypes.data,len(spans),int(self.structural),safe.ctypes.data,interior.ctypes.data,allowed.ctypes.data,self.logfreq.ctypes.data,self.theta.ctypes.data,len(self.theta),mode,0.,0.,0.,None,out.ctypes.data)
  if count<0:raise RuntimeError('constraint graph disconnected:'+str(count))
  return out[:count].tolist()
 def save(self,path):
  super().save(path);d=json.loads(path.read_text());d['format']='masked-rl-math-v4';d['atomic']=self.atomic;d['inference']=self.inference;d['native_source_sha256']=hashlib.sha256((ROOT/'native/decoder_v4.c').read_bytes()).hexdigest();path.write_text(json.dumps(d,indent=2))
 @classmethod
 def load(cls,path):
  d=json.loads(path.read_text());obj=cls([bytes.fromhex(t) for t in d['vocab_hex']],d['structural'],d['atomic'],d['inference'],d['theta'],d['frequency']);obj.train_seconds=d['train_seconds'];obj.training_log=d['training_log'];obj.capacity=d['capacity'];return obj
