"""Native lexical partition / binding CST with a Python reference lexer."""
import ctypes
import hashlib
import platform
import re
import subprocess
import importlib.util
import sysconfig
import os
import numpy as np
from .corpus import ROOT
from .structure_v2 import Structure

_lib=None
_bridge=None


def python_bridge():
    global _bridge
    if _bridge is not None:return _bridge
    sources=[ROOT/'native'/name for name in ('python_frontend_v6.c','frontend_v6.c','decoder_v4.c','compact_codec_v6.c')]
    key=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
    target=ROOT/'.native_cache'/('py-v6-'+key+sysconfig.get_config_var('EXT_SUFFIX'))
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        temporary=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
        command=['cc','-O3','-fPIC','-shared','-I'+sysconfig.get_paths()['include'],str(sources[0]),'-o',str(temporary),'-lm']
        if platform.system()=='Darwin':command.extend(['-undefined','dynamic_lookup'])
        subprocess.run(command,check=True,capture_output=True)
        os.replace(temporary,target)
    spec=importlib.util.spec_from_file_location('_math_frontend_v6',target);_bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(_bridge);return _bridge


def library():
    global _lib
    if _lib is not None:return _lib
    source=ROOT/'native/frontend_v6.c';sha=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target=ROOT/'.native_cache'/f'frontend-v6-{platform.machine()}-{sha}.{"dylib" if platform.system()=="Darwin" else "so"}'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():subprocess.run(['cc','-O3','-fPIC','-shared',str(source),'-o',str(target)],check=True,capture_output=True)
    _lib=ctypes.CDLL(str(target))
    _lib.mt_lex.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];_lib.mt_lex.restype=ctypes.c_int
    _lib.mt_tree.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];_lib.mt_tree.restype=ctypes.c_int
    for name in ('mt_allowed','mt_symbols'):
        getattr(_lib,name).argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        getattr(_lib,name).restype=ctypes.c_int
    return _lib


def lexical_ranges(text):
    raw=text.encode();out=np.empty((max(1,len(raw)),2),dtype=np.int32)
    count=library().mt_lex(raw,len(raw),out.ctypes.data)
    return out[:count]


def tree_ranges(text):
    raw=text.encode();out=np.empty((max(1,8*len(raw)),2),dtype=np.int32)
    count=library().mt_tree(raw,len(raw),out.ctypes.data)
    if count<0:raise MemoryError('native CST allocation failed')
    return out[:count]


def structure_native(text):
    raw=text.encode();lex=tuple((int(a),int(b),raw[a:b].decode()) for a,b in lexical_ranges(text) if raw[a:b].strip())
    spans=tuple(sorted(set(map(tuple,tree_ranges(text).tolist()))))
    return Structure('native_hybrid_expression_ast_cst',spans,lex,0,'supported arithmetic AST; other syntax uses binding CST, not universal semantic AST')


def reference_lex(text):
    offsets=[0]
    for c in text:offsets.append(offsets[-1]+len(c.encode()))
    return [(offsets[m.start()],offsets[m.end()]) for m in re.finditer(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[^\x00-\x7f]|[\s\S]',text)]
