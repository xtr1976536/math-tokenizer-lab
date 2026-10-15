"""Read-only environment, provenance, data, and CUDA readiness checks."""
import argparse,hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path
import torch
from math_tokenizer.eqgpt.author import verify_source,ROOT
from math_tokenizer.eqgpt.data import data_root

p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
expected={'torch':'2.3.1','numpy':'1.26.4','pandas':'2.2.2','scipy':'1.13.1','matplotlib':'3.8.4','sympy':'1.12.1','tokenizers':'0.20.3'}
versions={name:importlib.metadata.version(name) for name in expected}
if any(versions[k].split('+')[0]!=v for k,v in expected.items()):raise RuntimeError('Environment does not match requirements-eqgpt.lock.txt: '+str(versions))
source=verify_source();missing=[]
for entry in source['data_files']:
 path=data_root()/entry['path']
 if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:missing.append(entry['path'])
if missing:raise RuntimeError('Missing or modified author data: '+str(missing))
report=dict(python=sys.version,versions=versions,author_revision=source['revision'],author_files_verified=len(source['files']),
            data_files_verified=len(source['data_files']),cuda_available=torch.cuda.is_available(),
            require_cuda=a.require_cuda,cpu_checks_passed=True)
if a.require_cuda:
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required for formal training')
 capability=torch.cuda.get_device_capability()
 if capability[0]>=10:raise RuntimeError('Blackwell GPU requires newer PyTorch; select an Ada/Ampere GPU for the pinned author environment')
 x=torch.randn(128,128,device='cuda',requires_grad=True);x.square().mean().backward();torch.cuda.synchronize()
 report.update(gpu=torch.cuda.get_device_name(),capability=list(capability),cuda_runtime=torch.version.cuda,
               vram_bytes=torch.cuda.get_device_properties(0).total_memory,cuda_backward_passed=True)
try:
 from math_tokenizer.hard_ast_2026 import HardASTBPE
 from math_tokenizer.ast_boundary_v8 import BoundaryAST
 m=HardASTBPE.train([r'\frac{x}{y}+u_t'],280)
 assert m.decode(m.encode(r'\frac{x}{y}+u_t'))==r'\frac{x}{y}+u_t'
 report['native_compiler_check']=True
except Exception as err:raise RuntimeError('C compiler/native tokenizer check failed') from err
if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
