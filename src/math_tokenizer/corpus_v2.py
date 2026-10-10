"""Expanded, versioned research-level formula corpus; v1 snapshot stays immutable."""
import gzip,hashlib,io,json,re,tarfile
from pathlib import Path
from collections import Counter,defaultdict
from .corpus import ROOT,latex_math,rst_math,canonical
from .validation_v2 import validate,MATRIX_ENVS
from .structure_v2 import complexity,structure


def tex_members(data):
    # Read archives in memory, never extract paths or run TeX.
    try:
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:*') as archive:
            for member in sorted(archive.getmembers(),key=lambda m:m.name):
                if member.isfile() and member.name.endswith(('.tex','.ltx')) and member.size<=4_000_000:
                    stream=archive.extractfile(member)
                    yield member.name,stream.read().decode('utf-8',errors='replace')
        return
    except tarfile.ReadError:pass
    if data.startswith(b'\x1f\x8b'):data=gzip.decompress(data)
    if not data.startswith(b'%PDF'):
        yield 'main.tex',data.decode('utf-8',errors='replace')


def clean_formula(text):
    # Remove only equation metadata and alignment layout, never semantic text.
    text=re.sub(r'\\(?:label|tag)\{[^{}]*\}','',text)
    text=re.sub(r'\\(?:nonumber|notag)\b','',text)
    return text.strip().rstrip('.,;').strip()

def extract_math(text):
    # Preserve source line numbers by replacing comments with equal-length spaces.
    text=re.sub(r'(?m)(?<!\\)%[^\n]*',lambda m:' '*len(m.group()),text)
    text=re.sub(r'\\begin\{(?:verbatim|lstlisting|minted)\}.*?\\end\{(?:verbatim|lstlisting|minted)\}',lambda m:'\n'*m.group().count('\n'),text,flags=re.S)
    for formula,line in latex_math(text):
        # Avoid deleting rows of systems, matrices or cases. Single-row layout
        # can lose its & alignment marker without losing symbols.
        has_matrix=any(r'\begin{'+name+'}' in formula for name in MATRIX_ENVS)
        if not has_matrix and (r'\\' in formula or r'\begin' in formula):continue
        yield clean_formula(formula if has_matrix else formula.replace('&','')),line
    for m in re.finditer(r'\\begin\{(?:displaymath|math)\}(.*?)\\end\{(?:displaymath|math)\}',text,re.S):
        yield clean_formula(m[1]),1+text.count('\n',0,m.start())


def load_simple_macros(text):
    # Expand ONLY zero-argument symbol aliases with known safe standard meaning.
    # Parameterized macros need a real macro parser and are quarantined.
    from .corpus import ALLOWED
    macros={}
    pattern=re.compile(r'\\(?:newcommand|renewcommand|providecommand)\s*(?:\{(\\[A-Za-z]+)\}|(\\[A-Za-z]+))\s*\{([^{}]+)\}')
    for m in pattern.finditer(text):
        name=m[1] or m[2];body=m[3]
        if re.fullmatch(r'\\[A-Za-z]+',body) and body[1:] in ALLOWED:macros[name]=body
    for m in re.finditer(r'\\def\s*(\\[A-Za-z]+)\s*\{([^{}]+)\}',text):
        if re.fullmatch(r'\\[A-Za-z]+',m[2]) and m[2][1:] in ALLOWED:macros[m[1]]=m[2]
    return macros

def expand_aliases(text,macros):
    return re.sub(r'\\[A-Za-z]+',lambda m:macros.get(m[0],m[0]),text)


def build():
    records=[];rejected=[]
    base=ROOT/'data/releases/v1/curated/corpus.jsonl'
    for line in base.read_text().splitlines():
        r=json.loads(line);r['corpus_parent']='v1'
        if validate(r['latex']) is None: records.append(r)
        else: rejected.append({'source_id':r['source_id'],'location':r['location'],'reason':'v2_revalidation:'+validate(r['latex']),'latex':r['latex']})
    def add(text,meta):
        reason=validate(text)
        if reason:
            rejected.append({'source_id':meta['source_id'],'location':meta['location'],'reason':reason,'latex':text});return
        if re.search(r'\\begin\{(?:p|b|B|v|V|small)?matrix\}|\\(?:det)\b|\\operatorname\{(?:det|tr|rank|spec)\}',text):
            meta['domain']='linear_algebra';meta['domain_basis']='explicit_matrix_or_linear_operator'
        record=dict(meta,latex=text,canonical_sha256=hashlib.sha256(canonical(text).encode()).hexdigest(),
            id=hashlib.sha256((meta['source_id']+meta['location']+text).encode()).hexdigest()[:24])
        records.append(record)
    # Expand existing model-produced sources beyond v1's document cap.
    registry=json.loads((ROOT/'data/source_registry.json').read_text())
    for source in registry:
        if source['id'].startswith('openai:'):
            path=ROOT/source['local'];text=path.read_text(errors='replace');macros=load_simple_macros(text)
            group=re.sub(r'-(September|October)-\d+-2026$','',source['group'])
            for formula,line in extract_math(text):
                raw=formula;formula=expand_aliases(formula,macros)
                pd=source['domain'];is_pde=pd=='pde' and bool(re.search(r'\\partial|\\nabla|\\Delta',formula)) and '=' in formula
                add(formula,dict(source_id=source['id'],source_revision=source['revision'],source_sha256=source['sha256'],source_repo=source['repo'],
                    origin='possible_mixed',origin_evidence=source['evidence'],latex_author_origin='possible_mixed',license=source['license'],
                    domain='pde' if is_pde else 'analysis' if pd=='pde' else pd,subdomain='research_paper_formula',source_path=source['path'],
                    source_group=group,split_group=group,location=f'line:{line}',transformation='math extraction; layout metadata removed; audited zero-arg aliases expanded',
                    raw_latex=raw if raw!=formula else None,academic_level='research'))
    fungrim_path=ROOT/'data/raw/fungrim_formulas.jsonl'
    if fungrim_path.exists():
        source=json.loads((ROOT/'data/fungrim_source.json').read_text())
        for line in fungrim_path.read_text().splitlines():
            row=json.loads(line)
            add(clean_formula(row['latex']),dict(source_id='fungrim:'+row['entry_id'],source_revision=row['revision'],source_sha256=source['export_sha256'],
                source_repo=row['repo'],source_path=row['source_path'],source_group=row['source_group'],split_group=row['source_group'],
                location=f"{row['source_path']}:line:{row['source_line']}:entry:{row['entry_id']}",
                origin=row['origin'],origin_evidence=row['origin_evidence'],latex_author_origin='official_symbolic_renderer',license='MIT',domain=row['domain'],
                subdomain='special_function_formula_library',semantic_expression=row['semantic_expression'],academic_level=row['academic_level'],transformation=row['transformation']))
    arxiv=json.loads((ROOT/'data/arxiv_sources.json').read_text())
    for source in arxiv:
        data=(ROOT/source['local']).read_bytes();assert hashlib.sha256(data).hexdigest()==source['sha256']
        members=list(tex_members(data));aliases={}
        for path,text in members:aliases.update(load_simple_macros(text))
        for path,text in members:
            for formula,line in extract_math(text):
                raw=formula;formula=expand_aliases(formula,aliases)
                pd=source['domain'];is_pde=pd=='pde' and bool(re.search(r'\\partial|\\nabla|\\Delta',formula))
                group='arxiv:'+source['id']
                add(formula,dict(source_id=group,source_revision=source['version'],source_sha256=source['sha256'],source_repo='arxiv.org',
                    source_url=source['source_url'],abstract_url=source['abstract_url'],title=source['title'],authors=source['authors'],
                    origin='human',origin_basis=source['origin_basis'],origin_evidence=source['evidence'],latex_author_origin='historical_human_source',
                    version_date=source['version_date'],license=source['license'],license_url=source['license_url'],
                    domain='pde' if is_pde else 'analysis' if pd=='pde' else pd,subdomain='research_paper_formula',source_path=path,
                    source_group=group,split_group=group,location=f'{path}:line:{line}',transformation='source math extraction; no TeX execution; audited zero-argument symbol aliases expanded',
                    raw_latex=raw if raw!=formula else None,academic_level='research'))
    # Duplicate provenance retained without converting human/unknown into AI by
    # inference. Mixed-origin duplicate membership is stated explicitly.
    unique={}
    for r in records:
        h=r['canonical_sha256']
        if h in unique:
            old=unique[h]
            # Same source repeated from v1 extraction does not add new evidence.
            if r['source_id']!=old['source_id'] or r['source_group']!=old['source_group']:
                old.setdefault('duplicate_provenance',[]).append({k:v for k,v in r.items() if k!='duplicate_provenance'})
            old['origin_memberships']=sorted(set(old.get('origin_memberships',[old['origin']]))|{r['origin']})
        else:unique[h]=r
    records=list(unique.values())
    # v2 split defined and frozen by group hash; v1 stays separately evaluable.
    def split(g):
        n=int(hashlib.sha256(('mathlab-v2:'+g).encode()).hexdigest()[:8],16)%100
        return 'train' if n<70 else 'dev' if n<85 else 'test'
    # Group stratification uses provenance/domain only, never tokenizer scores.
    group_rows=defaultdict(list)
    for r in records:group_rows[r['source_group']].append(r)
    strata=defaultdict(list)
    for group,rs in group_rows.items():
        primary=Counter(r['domain'] for r in rs).most_common(1)[0][0]
        origin=Counter(r['origin'] for r in rs).most_common(1)[0][0]
        strata[(origin,primary)].append(group)
    assignments={}
    for key,groups in sorted(strata.items()):
        ordered=sorted(groups,key=lambda g:hashlib.sha256(('mathlab-v2-stratified:'+g).encode()).digest())
        n=len(ordered)
        ntest=max(1,round(.15*n)) if n>=2 else 0
        ndev=max(1,round(.15*n)) if n>=3 else 0
        for i,group in enumerate(ordered):assignments[group]='test' if i<ntest else 'dev' if i<ntest+ndev else 'train'
    def split(g):
        return assignments.get(g,'train')
    clean=[]
    for r in records:
        r['split_group']=r['source_group'];r['split']=split(r['split_group'])
        if any(split(d['source_group'])!=r['split'] for d in r.get('duplicate_provenance',[])):
            rejected.append({'source_id':r['source_id'],'location':r['location'],'reason':'canonical_duplicate_crosses_split','latex':r['latex']});continue
        r['complexity']=complexity(r['latex']);s=structure(r['latex']);r['structure_kind']=s.kind
        r['ast_supported']=s.kind=='semantic_ast';r['academic_level']=r.get('academic_level','university_or_research_source')
        clean.append(r)
    clean.sort(key=lambda r:r['id'])
    out=ROOT/'data/releases/v2/curated';out.mkdir(parents=True,exist_ok=True)
    def write(path,rows,formula_only=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r['latex'] if formula_only else r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    for path in out.rglob('*.jsonl'):path.unlink()
    write(out/'corpus.jsonl',clean)
    for s in ('train','dev','test'):write(out/'splits'/f'{s}.jsonl',[r for r in clean if r['split']==s])
    domains=sorted({r['domain'] for r in clean})
    for d in domains:
        for o in ('human','possible_mixed','ai_generated','unknown'):
            for c in ('simple','complex'):
                rows=[r for r in clean if r['domain']==d and r['origin']==o and r['complexity']['class']==c]
                write(out/'by_domain'/d/o/f'{c}.jsonl',rows)
                write(out.parent/'formula_only'/d/o/f'{c}.jsonl',rows,True)
    write(out/'by_complexity/simple.jsonl',[r for r in clean if r['complexity']['class']=='simple'])
    write(out/'by_complexity/complex.jsonl',[r for r in clean if r['complexity']['class']=='complex'])
    write(out.parent/'quarantine/rejections.jsonl',rejected)
    summary={'release':'v2','count':len(clean),'rejected':len(rejected),'arxiv_papers_downloaded':len(arxiv),
        'arxiv_papers_with_accepted_formulas':len({r['source_id'] for r in clean if r['source_id'].startswith('arxiv:')}),
        'split_counts':dict(Counter(r['split'] for r in clean)),
        'origin_counts':dict(Counter(r['origin'] for r in clean)),
        'domain_counts':dict(Counter(r['domain'] for r in clean)),
        'complexity_counts':dict(Counter(r['complexity']['class'] for r in clean)),
        'structure_counts':dict(Counter(r['structure_kind'] for r in clean)),
        'domain_origin_complexity':dict(Counter(r['domain']+'/'+r['origin']+'/'+r['complexity']['class'] for r in clean)),
        'split_domain':{s:dict(Counter(r['domain'] for r in clean if r['split']==s)) for s in ('train','dev','test')},
        'split_group_counts':{s:len({r['split_group'] for r in clean if r['split']==s}) for s in ('train','dev','test')},
        'split_policy':'document/family stratified by dominant domain+origin; pre-model fixed hashes; >=3 groups get each split; >=2 groups get train/test',
        'group_assignments':assignments,
        'corpus_sha256':hashlib.sha256((out/'corpus.jsonl').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));return summary
if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
