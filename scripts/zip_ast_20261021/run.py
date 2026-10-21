"""Isolated ZIP implementation, freshly fitted vocabularies, common byte metrics."""
from pathlib import Path
import argparse,bisect,csv,gzip,hashlib,json,os,platform,re,sys,time,zipfile
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/zip_ast_20261021'
ZIP=Path('/Users/xiao/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_nncsrgfc2h8e22_5a50/temp/drag/eqgpt_shared8_v8_pde_vocab896_pos50_cleanup_100epoch(2).zip')
SIZES=[320,384,448,512,640,768,896,1024]
BASELINES=['Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base']
METRICS=['fragmentation_pct','cmd_pct','cross_preservation_pct','num_pct']
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def h(s):return sha(s.encode())
def extract():
 vendor=OUT/'vendor';files={}
 if not ZIP.exists():
  frozen=json.loads((OUT/'zip_provenance.json').read_text())
  for name,digest in frozen['files'].items():assert sha((vendor/name).read_bytes())==digest
  return vendor
 with zipfile.ZipFile(ZIP) as z:
  prefix=z.namelist()[0].split('/')[0]+'/'
  for n in z.namelist():
   rel=n.removeprefix(prefix)
   if (rel.startswith('src/math_tokenizer/') and rel.endswith('.py')) or (rel.startswith('native/') and rel.endswith(('.c','.h'))) or rel=='LICENSE':
    p=vendor/rel;assert p.resolve().is_relative_to(vendor.resolve());raw=z.read(n);files[rel]=sha(raw)
    if p.exists():assert p.read_bytes()==raw,'Vendor source modified'
    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 write(OUT/'zip_provenance.json',dict(zip_sha256=sha(ZIP.read_bytes()),zip_name=ZIP.name,files=files,policy='Only Python/C implementation sources and license extracted; no packaged vocabulary or training data loaded'))
 return vendor

def protocols(rows):
 def split(pool,name,seed,train_filter=lambda r:True,test_filter=lambda r:True):
  counts=defaultdict(int)
  for r in pool:counts[r['split_group']]+=1
  groups=sorted(counts,key=lambda g:(-counts[g],h(f'{seed}:{g}')))
  bins={'train':set(),'test':set(),'dev':set()};totals={k:0 for k in bins};weights={'train':.7,'test':.2,'dev':.1}
  for g in groups:
   k=min(bins,key=lambda k:(totals[k]/weights[k],h(f'{seed}:{g}:{k}')))
   bins[k].add(g);totals[k]+=counts[g]
  testgroups=bins['test'];devgroups=bins['dev']
  if name.startswith('pde_'):
   from itertools import combinations
   # Small PDE catalog: choose genuinely different, size-balanced paper holdouts.
   testgroups=set(min(combinations(groups,4),key=lambda gs:(abs(sum(counts[g] for g in gs)-len(pool)*.2),h(str(seed)+':'+','.join(sorted(gs))))))
   remain=[g for g in groups if g not in testgroups]
   devgroups=set(min(combinations(remain,2),key=lambda gs:(abs(sum(counts[g] for g in gs)-len(pool)*.1),h(str(seed)+':dev:'+','.join(sorted(gs))))))
  test=[r for r in pool if r['split_group'] in testgroups and test_filter(r)]
  dev=[r for r in pool if r['split_group'] in devgroups and train_filter(r)]
  train=[r for r in pool if r['split_group'] not in testgroups|devgroups and train_filter(r)]
  return clean(name,seed,train,dev,test)
 def clean(name,seed,train,dev,test):
  # Held-out test is never used to FIT vocabulary; fingerprints only purge overlap.
  blocked={k:{r[k] for r in test+dev} for k in ('canonical_sha256','template_sha256')}
  fit=[r for r in train if all(r[k] not in blocked[k] for k in blocked)]
  testkeys={k:{r[k] for r in test} for k in blocked}
  dev=[r for r in dev if all(r[k] not in testkeys[k] for k in testkeys)]
  assert fit and test
  return dict(name=name,seed=seed,train=fit,dev=dev,test=test,purged_train=len(train)-len(fit))
 out=[];pde=[r for r in rows if r['formula_partition']=='pde']
 for seed in (17,29,43):out.append(split(pde,'pde_id_'+str(seed),seed))
 for seed in (17,29,43):out.append(split(rows,'benchmark_id_'+str(seed),seed))
 for domain in ('algebra','geometry_topology','number_theory'):
  test=[r for r in rows if r['domain']==domain];held={r['split_group'] for r in test}
  rest=[r for r in rows if r['split_group'] not in held and r['domain']!=domain]
  groups=sorted({r['split_group'] for r in rest},key=lambda g:h('ood-dev:'+g));dg=set(groups[:max(1,len(groups)//10)])
  out.append(clean('benchmark_ood_'+domain,0,[r for r in rest if r['split_group'] not in dg],[r for r in rest if r['split_group'] in dg],test))
 for seed in (17,29,43):out.append(split(rows,'benchmark_simple_to_complex_'+str(seed),seed,lambda r:r['complexity']['class']=='simple',lambda r:r['complexity']['class']=='complex'))
 return out

def split_audit(p):
 result={}
 for a,b in [('train','test'),('train','dev'),('dev','test')]:
  for key in ('id','split_group','canonical_sha256','template_sha256'):
   n=len({r[key] for r in p[a]}&{r[key] for r in p[b]});assert n==0,(p['name'],a,b,key,n);result[a+'_'+b+'_'+key]=n
 if '_ood_' in p['name']:assert not {r['domain'] for r in p['test']}&{r['domain'] for r in p['train']}
 if 'simple_to_complex' in p['name']:assert {r['complexity']['class'] for r in p['train']}=={'simple'} and {r['complexity']['class'] for r in p['test']}=={'complex'}
 return result

def lex(text):
 offs=[0]
 for c in text:offs.append(offs[-1]+len(c.encode()))
 return [(offs[m.start()],offs[m.end()],m.group()) for m in re.finditer(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[^\x00-\x7f]|[\s\S]',text)]

def cross_fast(bounds,spans):
 starts=defaultdict(list);ends=defaultdict(list)
 for a,b in spans:starts[a].append(b);ends[b].append(a)
 sk=sorted(starts);ek=sorted(ends);total=0
 for a,b in zip(bounds,bounds[1:]):
  bad=any(max(starts[s])>b for s in sk[bisect.bisect_right(sk,a):bisect.bisect_left(sk,b)])
  if not bad:bad=any(min(ends[e])<a for e in ek[bisect.bisect_right(ek,a):bisect.bisect_left(ek,b)])
  total+=bad
 return total

def per_formula(model,r,token_bytes):
 text=r['latex'];raw=text.encode();ids=model.encode(text);pieces=[token_bytes(model,int(i)) for i in ids]
 assert all(pieces) and b''.join(pieces)==raw,(r['id'],'payload roundtrip')
 assert model.decode(ids)==text,(r['id'],'decode roundtrip')
 bounds=[0]
 for piece in pieces:bounds.append(bounds[-1]+len(piece))
 internal=bounds[1:-1];cmd=cmdok=num=numok=0
 for a,b,t in lex(text):
  intact=bisect.bisect_right(internal,a)==bisect.bisect_left(internal,b)
  if re.fullmatch(r'\\[A-Za-z]+',t):cmd+=1;cmdok+=intact
  if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',t):num+=1;numok+=intact
 crossing=cross_fast(bounds,r['reference']['spans'])
 return dict(id=r['id'],group=r['split_group'],domain=r['domain'],complexity=r['complexity']['class'],bytes=len(raw),tokens=len(ids),cuts=len(ids)-1,possible_cuts=max(0,len(raw)-1),commands=cmd,commands_intact=cmdok,numbers=num,numbers_intact=numok,crossing_tokens=crossing)

def summarize(rr):
 def ratio(a,b):
  d=sum(r[b] for r in rr);return 100*sum(r[a] for r in rr)/d if d else None
 return dict(fragmentation_pct=ratio('cuts','possible_cuts'),cmd_pct=ratio('commands_intact','commands'),cross_preservation_pct=100-ratio('crossing_tokens','tokens'),num_pct=ratio('numbers_intact','numbers'))

def bootstrap(rr,seed=20261021):
 groups=defaultdict(list)
 for r in rr:groups[r['group']].append(r)
 # Cluster bootstrap; no treating formula duplicates as independent papers.
 keys=sorted(groups);cols=['cuts','possible_cuts','commands_intact','commands','crossing_tokens','tokens','numbers_intact','numbers']
 a=np.array([[sum(r[c] for r in groups[g]) for c in cols] for g in keys],float);rng=np.random.default_rng(seed);out=[]
 for i in range(500):
  v=a[rng.integers(0,len(a),len(a))].sum(axis=0);out.append([100*v[0]/v[1] if v[1] else np.nan,100*v[2]/v[3] if v[3] else np.nan,100*(1-v[4]/v[5]),100*v[6]/v[7] if v[7] else np.nan])
 z=np.array(out);return {m:np.nanquantile(z[:,i],[.025,.975]).tolist() if np.isfinite(z[:,i]).any() else None for i,m in enumerate(METRICS)}

def evaluate(model,rows,name,size,folder,token_bytes):
 target=folder/f'{name.replace("/","--")}_{size or "official"}.json'
 if target.exists():return json.loads(target.read_text())
 rr=[per_formula(model,r,token_bytes) for r in rows]
 # Formula order and metric denominators must be exactly identical across methods.
 result=dict(method=name,requested_total_ids=size,actual_total_ids=(len(model.vocab)+3 if hasattr(model,'vocab') else model.vocab_size),formulas=len(rr),papers=len({r['group'] for r in rr}),test_ids_sha256=h('\n'.join(r['id'] for r in rows)),**summarize(rr),ci95=bootstrap(rr),denominators={k:sum(r[k] for r in rr) for k in ('bytes','tokens','possible_cuts','commands','numbers')})
 with gzip.open(target.with_suffix('.jsonl.gz'),'wt') as f:
  for r in rr:f.write(json.dumps(r)+'\n')
 write(target,result);token_bytes.cache_clear();return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only');ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
 from math_tokenizer.active_benchmark import read_catalog
 rows=read_catalog();ps=protocols(rows);vendor=extract()
 protocol=dict(version='zip-ast-four-metrics-v1',sizes=SIZES,controls=3,methods=['ast_bpe','soft_ast_bpe']+BASELINES,metrics={'fragmentation_pct':'100*sum(T-1)/sum(UTF8_bytes-1),lower better; number of cuts, NOT semantic damage','cmd_pct':'100*intact ASCII-letter LaTeX commands / all such commands','cross_preservation_pct':'100*(1-partially-crossing tokens/all tokens), independent frozen KaTeX byte spans','num_pct':'100*intact [0-9]+(?:\\.[0-9]+)? lexemes / all such lexemes'},evaluation='Same exact formula strings, no special tokens, no truncation, no normalization, aggregate ratio of counts; 500 source-cluster bootstrap replicates; missing command/number denominators are null, not100',training='Each size independently fitted from scratch; no packaged learned vocabulary; dev unused; ZIP weights fixed .25,.30',pretrained='Official frozen full vocabulary, not size matched; previous exposure unknown; no claim of controlled training-data advantage',catalog_sha256=sha((ROOT/'data/benchmark/metadata/formulas.jsonl.gz').read_bytes()),scripts_sha256=sha(Path(__file__).read_bytes()),environment=dict(python=platform.python_version(),platform=platform.platform()),splits=[{**{k:v for k,v in p.items() if k not in ('train','dev','test')},**{k:[r['id'] for r in p[k]] for k in ('train','dev','test')},'audit':split_audit(p)} for p in ps])
 frozen=OUT/'protocol.json'
 if frozen.exists():
  old=json.loads(frozen.read_text());assert old['splits']==protocol['splits'] and old['catalog_sha256']==protocol['catalog_sha256']
 else:write(frozen,protocol)
 print('Protocols',[(p['name'],len(p['train']),len(p['dev']),len(p['test'])) for p in ps],flush=True)
 if args.prepare_only:return
 # Load the existing mature baselines BEFORE importing the isolated ZIP package.
 from math_tokenizer.final_models import load_final
 from math_tokenizer.metrics_v2 import token_bytes
 baselines={name:load_final(name) for name in BASELINES}
 for module in list(sys.modules):
  if module=='math_tokenizer' or module.startswith('math_tokenizer.'):del sys.modules[module]
 sys.path.insert(0,str(vendor/'src'))
 from math_tokenizer.hard_ast_2026 import HardASTBPE
 from math_tokenizer.ast_boundary_v8 import BoundaryAST
 from math_tokenizer.atomic_bpe_v7 import AtomicBPE
 from math_tokenizer.rl_route_v6 import derive_atoms
 import ast
 source=vendor/'src/math_tokenizer/eqgpt/latex_tokenizers.py'
 node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_one')
 namespace=dict(HardASTBPE=HardASTBPE,BoundaryAST=BoundaryAST,AtomicBPE=AtomicBPE,derive_atoms=derive_atoms)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
 train_one=namespace['train_one']
 assert Path(sys.modules['math_tokenizer.hard_ast_2026'].__file__).is_relative_to(vendor)
 for p in ps:
  if args.only and not any(p['name'].startswith(x) for x in args.only.split(',')):continue
  folder=OUT/p['name'];folder.mkdir(exist_ok=True);md=folder/'models';md.mkdir(exist_ok=True)
  print('BEGIN',p['name'],flush=True)
  for name,model in baselines.items():evaluate(model,p['test'],name,None,folder,token_bytes)
  for size in SIZES:
   for name in ('ast_bpe','soft_ast_bpe'):
    checkpoint=md/f'{name}_{size}.json';st=time.perf_counter()
    if checkpoint.exists():model=(HardASTBPE if name=='ast_bpe' else BoundaryAST).load(checkpoint)
    else:
     model=train_one(name,[r['latex'] for r in p['train']],size);model.save(checkpoint)
    result=evaluate(model,p['test'],name,size,folder,token_bytes)
    write(md/f'{name}_{size}.provenance.json',dict(train_ids_sha256=h('\n'.join(r['id'] for r in p['train'])),train_strings_sha256=h(json.dumps([r['latex'] for r in p['train']],ensure_ascii=False)),checkpoint_sha256=sha(checkpoint.read_bytes()),fit_from='fresh train_one on this protocol training strings; never packaged vocab',actual_total_ids=len(model.vocab)+3,requested_total_ids=size))
    print('DONE',p['name'],name,size,'actual',result['actual_total_ids'],'sec',round(time.perf_counter()-st,2),*[round(result[m],3) if result[m] is not None else None for m in METRICS],flush=True)
 print('Selected protocols complete',flush=True)
if __name__=='__main__':main()
