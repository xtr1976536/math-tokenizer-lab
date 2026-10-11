"""Versioned formula benchmark with new-paper, domain and extreme transfer splits.

Old release unchanged. New test files are selected/frozen by provenance only;
model code never loads them until dev selection freezes. Exact and renamed
syntax duplicates vs previously published corpus are removed from new test.
"""
import hashlib,json,re,gzip
from pathlib import Path
from collections import Counter,defaultdict
from .corpus import ROOT,canonical
from .corpus_archive_v5 import ensure_master
ensure_master()
from .corpus_v2 import tex_members,extract_math,load_simple_macros,expand_aliases,clean_formula
from .validation_v2 import validate
from .structure_v3 import structure_fast
from .structure_v2 import complexity

RESEARCH_OOD={'algebra','geometry_topology','number_theory','combinatorics','logic'}
TARGET_CATEGORIES={'math.AG','math.DG','math.GT','math.AT','math.GR','math.RA','math.RT','math.AC','math.NT','math.CO','math.LO'}

def template_key(text):
 # Keep operator/command text. Collapse scalar symbol names and numeric values;
 # deliberately conservative syntax duplicate key, not semantic equivalence.
 tokens=re.findall(r'\\[A-Za-z]+|\\[^A-Za-z]|\d+(?:\.\d+)?|[^\s]',text)
 seen={};out=[]
 for t in tokens:
  if re.fullmatch(r'[A-Za-z]',t):
   if t not in seen:seen[t]=str(len(seen))
   out.append('S'+seen[t])
  elif t[0:1].isdigit():out.append('N')
  else:out.append(t)
 return ''.join(out)

def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def build():
 release=ROOT/'data/releases/v5';old=[json.loads(x) for x in (ROOT/'data/releases/v2/curated/corpus.jsonl').read_text().splitlines()]
 sources=json.loads((release/'arxiv_sources.json').read_text());new=[];rejections=[]
 for source in sources:
  raw=(ROOT/source['local']).read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256']
  members=list(tex_members(raw));aliases={}
  for name,text in members:aliases.update(load_simple_macros(text))
  for name,text in members:
   for formula,line in extract_math(text):
    original=formula;formula=expand_aliases(formula,aliases);reason=validate(formula)
    if reason:
     rejections.append({'paper':source['id'],'file':name,'line':line,'reason':reason,'latex':formula});continue
    # Known malformed TeX escapes/control parameter placeholders are rejected.
    if re.search(r'\\\d|[\x00-\x08\x0b\x0c\x0e-\x1f]',formula):
     rejections.append({'paper':source['id'],'file':name,'line':line,'reason':'control_or_numeric_command','latex':formula});continue
    paper=source['domain'];domain=paper
    if paper=='pde' and not re.search(r'\\(?:partial|nabla|Delta)\b',formula):domain='analysis'
    if re.search(r'\\begin\{(?:p|b|B|v|V|small)?matrix\}|\\det\b|\\operatorname\{(?:rank|spec|det|tr)\}',formula):domain='linear_algebra'
    h=hashlib.sha256(canonical(formula).encode()).hexdigest();group='arxiv:'+source['id']
    id=hashlib.sha256((group+source['version']+name+str(line)+formula).encode()).hexdigest()[:24];s=structure_fast(formula)
    new.append({'id':id,'latex':formula,'raw_latex':original if original!=formula else None,'canonical_sha256':h,
     'template_sha256':hashlib.sha256(template_key(formula).encode()).hexdigest(),'domain':domain,'paper_domain':paper,
     'subject_categories':source['categories'],'primary_category':source['primary_category'],'source_id':group,'source_group':group,'split_group':group,
     'source_revision':source['version'],'source_sha256':source['sha256'],'source_path':name,'location':f'{name}:line:{line}',
     'source_repo':'arxiv.org','title':source['title'],'authors':source['authors'],'source_url':source['source_url'],'abstract_url':source['abstract_url'],
     'version_date':source['version_date'],'license':source['license'],'license_url':source['license_url'],'origin':'human','origin_basis':source['origin_basis'],
     'origin_evidence':source['evidence'],'latex_author_origin':'historical_human_source','academic_level':'research','structure_kind':s.kind,
     'ast_supported':s.kind=='semantic_ast','complexity':complexity(formula),
     'transformation':'fixed-v5 source extraction; only safe symbol macro aliases expanded; no TeX execution','corpus_parent':'arxiv_v5_new'})
 oldhash={r['canonical_sha256'] for r in old};oldtemplates={hashlib.sha256(template_key(r['latex']).encode()).hexdigest() for r in old}
 unique={}
 for r in new:
  if r['canonical_sha256'] in oldhash:
   rejections.append({'id':r['id'],'reason':'duplicate_previous_release','latex':r['latex']});continue
  if r['canonical_sha256'] in unique:
   unique[r['canonical_sha256']].setdefault('duplicate_provenance',[]).append({k:v for k,v in r.items() if k!='duplicate_provenance'})
  else:unique[r['canonical_sha256']]=r
 new=list(unique.values());bydomain=defaultdict(list)
 for source in sources:bydomain[source['domain']].append('arxiv:'+source['id'])
 assignments={}
 for domain,groups in sorted(bydomain.items()):
  groups=sorted(groups,key=lambda g:hashlib.sha256(('v5-paper-split:'+g).encode()).digest());n=len(groups)
  nt=max(1,round(.25*n)) if n>=2 else 1;nd=max(1,round(.15*n)) if n>=4 else 0
  for i,g in enumerate(groups):assignments[g]='new_test' if i<nt else 'new_dev' if i<nt+nd else 'new_train'
 accepted=[]
 for r in new:
  split=assignments[r['split_group']]
  if any(assignments[d['split_group']]!=split for d in r.get('duplicate_provenance',[])):
   rejections.append({'id':r['id'],'reason':'cross_partition_duplicate','latex':r['latex']});continue
  if split=='new_test' and r['template_sha256'] in oldtemplates:
   rejections.append({'id':r['id'],'reason':'test_syntax_template_seen_in_old_release','latex':r['latex']});continue
  r['split']=split;accepted.append(r)
 # Fresh test additionally excludes templates seen in new train/dev papers.
 new_fit_templates={r['template_sha256'] for r in accepted if r['split']!='new_test'}
 filtered=[]
 for r in accepted:
  if r['split']=='new_test' and r['template_sha256'] in new_fit_templates:
   rejections.append({'id':r['id'],'reason':'test_syntax_template_seen_in_new_fit','latex':r['latex']});continue
  filtered.append(r)
 accepted=filtered;accepted.sort(key=lambda r:r['id']);rows=old+accepted
 # Preserve old split boundaries and avoid taking old test into new training.
 train=[dict(r,benchmark_split='train') for r in old if r['split']=='train']+[dict(r,benchmark_split='train') for r in accepted if r['split']=='new_train']
 dev=[dict(r,benchmark_split='dev') for r in old if r['split']=='dev']+[dict(r,benchmark_split='dev') for r in accepted if r['split']=='new_dev']
 fresh=[dict(r,benchmark_split='fresh_paper_test') for r in accepted if r['split']=='new_test']
 # Generalization: train PDE/ODE/analysis papers only, remove any mixed target
 # domain document entirely. Targets truly unseen by this specialized tokenizer.
 source_categories={'arxiv:'+s['id']:set(s['categories']) for s in json.loads((ROOT/'data/arxiv_sources.json').read_text())+sources}
 excluded_groups={r['split_group'] for r in train+dev if r.get('paper_domain',r['domain']) in RESEARCH_OOD or r['domain'] in RESEARCH_OOD
   or bool(source_categories.get(r['source_group'],set())&TARGET_CATEGORIES)}
 transfer_train=[r for r in train if r['domain'] in {'pde','ode','analysis'} and r['split_group'] not in excluded_groups]
 transfer_dev=[r for r in dev if r['domain'] in {'pde','ode','analysis'} and r['split_group'] not in excluded_groups]
 extreme=[r for r in fresh if r.get('paper_domain',r['domain']) in RESEARCH_OOD and r['domain'] in RESEARCH_OOD]
 out=release/'curated';out.mkdir(parents=True,exist_ok=True)
 write(out/'corpus.jsonl',rows);write(out/'new_arxiv.jsonl',accepted)
 for name,rs in [('train',train),('dev',dev),('fresh_paper_test',fresh),('transfer_train',transfer_train),('transfer_dev',transfer_dev),('extreme_domain_test',extreme)]:write(out/'splits'/f'{name}.jsonl',rs)
 buckets=defaultdict(list)
 for r in rows:buckets[(r['domain'],r['origin'],r['complexity']['class'])].append(r)
 for (domain,origin,complexityclass),rs in sorted(buckets.items()):
  write(out/'by_domain'/domain/origin/f'{complexityclass}.jsonl',rs)
  path=release/'formula_only'/domain/origin/f'{complexityclass}.jsonl';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r['latex'],ensure_ascii=False)+'\n' for r in rs))
 write(release/'quarantine/new_rejections.jsonl',rejections)
 summary={'release':'v5','count':len(rows),'old_count':len(old),'new_count':len(accepted),'new_papers_downloaded':len(sources),
  'new_papers_accepted':len({r['source_group'] for r in accepted}),'origins':dict(Counter(r['origin'] for r in rows)),
  'domains':dict(Counter(r['domain'] for r in rows)),'complexity':dict(Counter(r['complexity']['class'] for r in rows)),
  'splits':{name:{'formulas':len(rs),'papers':len({r['split_group'] for r in rs}),'domains':dict(Counter(r['domain'] for r in rs))} for name,rs in [('train',train),('dev',dev),('fresh_paper_test',fresh),('transfer_train',transfer_train),('transfer_dev',transfer_dev),('extreme_domain_test',extreme)]},
  'template_test_exclusion':'exact lexical shape after single-symbol renaming and number collapse vs entire prior v2 release and new train/dev; not semantic dedup proof',
  'corpus_sha256':hashlib.sha256((out/'corpus.jsonl').read_bytes()).hexdigest(),'new_test_sha256':hashlib.sha256((out/'splits/fresh_paper_test.jsonl').read_bytes()).hexdigest(),
  'specialized_excluded_categories':sorted(TARGET_CATEGORIES),
  'scope':'old v2 preserved separately; new test not old exposed test; extreme has train-subject exclusion; quality algorithmic with spot-check not universal expert verification'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));return summary
if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
