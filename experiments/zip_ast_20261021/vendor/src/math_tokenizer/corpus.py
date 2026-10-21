"""Conservative formula-only extraction, provenance, quarantine and grouped splits."""
import hashlib,json,re,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import openpyxl
from .ast import parse,shape,ParseError, SYMBOLS,FUNCTIONS,DECORATORS,BINARY
ROOT=Path(__file__).resolve().parents[2]
# src/math_tokenizer -> project
ROOT=Path(__file__).resolve().parents[2]

EXTRA=set(('sum prod int iint iiint oint lim limsup liminf sup inf bigcup bigcap '
           'left right middle big Big bigg Bigg langle rangle lvert rvert Vert vert '
           'dots ldots cdots vdots ddots quad qquad colon mapsto iff implies '
           'bmod pmod mod operatorname binom dbinom choose mathscr mathfrak '
           'mathop limits nolimits not neg land lor wedge vee setminus backslash '
           'qquad displaystyle textstyle scriptstyle scriptscriptstyle '
           'lbrace rbrace lceil rceil lfloor rfloor perp parallel lesssim gtrsim '
           'prec succ preceq succeq simeq cong propto asymp hookrightarrow '
           'longrightarrow longleftrightarrow leftrightarrow otimes oplus '
           'bullet circ star ast dagger ddots prime top bot').split())
ALLOWED=SYMBOLS|FUNCTIONS|DECORATORS|EXTRA|{x[1:] for x in BINARY if x.startswith('\\')}|{'frac','dfrac','tfrac','sqrt'}

def canonical(text):
    return re.sub(r'\s+','',text)

def validate(text):
    if not isinstance(text,str) or not 3<=len(text.encode())<=2048:
        return 'length'
    if any(c in text for c in ('$','%','#','`')):
        return 'delimiter_comment_or_parameter'
    if '\\begin' in text or '\\end' in text:
        return 'environment_needs_manual_review'
    if '\\text' in text or '\\label' in text or '\\tag' in text:
        return 'prose_or_metadata_command'
    depth=0
    for i,c in enumerate(text):
        if i and text[i-1]=='\\':continue
        if c=='{':depth+=1
        if c=='}':depth-=1
        if depth<0:return 'unbalanced_braces'
    if depth:return 'unbalanced_braces'
    commands=re.findall(r'\\([A-Za-z]+)',text)
    unknown=sorted(set(commands)-ALLOWED)
    if unknown:return 'unknown_macro:'+','.join(unknown)
    plain=re.sub(r'\\[A-Za-z]+|\\[^A-Za-z]','',text)
    if re.search(r'[A-Za-z]{3,}',plain):return 'possible_prose_or_multiword_identifier'
    if not re.search(r'[=+*/^_<>]|\\(?:frac|partial|sum|int|nabla|sqrt|in|to)',text):
        return 'isolated_symbol_or_label'
    if re.search(r'[\u4e00-\u9fff]',text):return 'prose'
    # Lexical acceptance is deliberately broader than the semantic AST subset.
    # Mathematical truth is not claimed by a syntax filter.
    return None

SUFFIX='xyzts'
def derivative(expr,variables):
    n=len(variables)
    numerator=r'\partial'+('^{'+str(n)+'}' if n>1 else '')+' '+expr
    denominator=' '.join(r'\partial '+v+('^{'+str(variables.count(v))+'}' if variables.count(v)>1 else '') for v in dict.fromkeys(variables))
    return r'\frac{'+numerator+'}{'+denominator+'}'

def eqterm(term):
    if term in ('+','*','/'):
        return {'+':'+','*':r'\cdot','/':'/'}[term]
    simple={'sinx':r'\sin(x)','sint':r'\sin(t)','(x+y)':'(x+y)',
            'Laplace(u)':r'\Delta u','Laplace(utt)':r'\Delta '+derivative('u','tt'),
            'BiLaplace(u)':r'\Delta^{2} u',
            '(uxx+ux/x)^2':'('+derivative('u','xx')+'+'+derivative('u','x')+'/x)^{2}'}
    if term in simple:return simple[term]
    # Explicit composite derivative mapping; interpretation is recorded.
    composite={'uux':'u '+derivative('u','x'),'u^3':'u^{3}','u^4':'u^{4}',
               '1/u':'1/u','u^-2*ux':'u^{-2} '+derivative('u','x'),
               'u(u^2)xx':'u '+derivative('(u^{2})','xx')}
    match=re.fullmatch(r'\((.*)\)([xyzt]+)',term)
    if match:
        if match[1] not in composite:raise ValueError('unmapped composite '+term)
        return derivative('('+composite[match[1]]+')',match[2])
    match=re.fullmatch(r'([uvpw])([xyzt]+)(?:\^(\d+))?',term)
    if match:
        out=derivative(match[1],match[2])
        return '('+out+')^{'+match[3]+'}' if match[3] else out
    match=re.fullmatch(r'(sin|sinh|exp|sqrt)\(([uvxyzt]|-[xyzt])\)',term)
    if match:return '\\'+match[1]+('{'+match[2]+'}' if match[1]=='sqrt' else '('+match[2]+')')
    if re.fullmatch(r'[uvpwxyzt](?:\^\d+)?',term):
        return re.sub(r'\^(\d+)',r'^{\1}',term)
    raise ValueError('unmapped term '+term)

def latex_math(text):
    # Comments are removed before extraction; no TeX execution occurs.
    text=re.sub(r'(?m)(?<!\\)%[^\n]*','',text)
    pattern=re.compile(r'\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\\)\$\$(.*?)\$\$|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$',re.S)
    for m in pattern.finditer(text):
        formula=next(x for x in m.groups() if x is not None)
        yield formula,1+text.count('\n',0,m.start())
    # Single-equation environments; aligned prose/system environments quarantine.
    for m in re.finditer(r'\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\1\}',text,re.S):
        yield m[2],1+text.count('\n',0,m.start())

def rst_math(text):
    for m in re.finditer(r':math:`([^`]+)`',text):
        yield m[1],1+text.count('\n',0,m.start())
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.strip()=='.. math::':
            block=[];j=i+1
            while j<len(lines) and (not lines[j].strip() or lines[j].startswith(' ')):
                if lines[j].strip().startswith(':'):break
                block.append(lines[j].strip());j+=1
            if any(block):yield '\n'.join(block).strip(),i+1

class UnionFind:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def build():
    sources=json.loads((ROOT/'data/source_registry.json').read_text())
    records=[];rejections=[]
    def add(formula,meta):
        formula=formula.strip()
        reason=validate(formula)
        if reason:
            rejections.append(dict(meta,formula=formula,reason=reason));return
        record=dict(meta,latex=formula,canonical_sha256=hashlib.sha256(canonical(formula).encode()).hexdigest())
        record['id']=hashlib.sha256((meta['source_id']+str(meta['location'])+formula).encode()).hexdigest()[:24]
        try:
            parse(formula);record['ast_supported']=True
        except (ParseError,RecursionError):record['ast_supported']=False
        records.append(record)
    for source in sources:
        base=dict(source_id=source['id'],source_revision=source['revision'],source_sha256=source['sha256'],
                  source_repo=source['repo'],origin=source['origin'],origin_evidence=source['evidence'],license=source['license'])
        if source['id']=='eqgpt':
            w=openpyxl.load_workbook(ROOT/source['local'],data_only=True)
            for sheet in w:
                for rowno,row in enumerate(sheet.values,1):
                    if rowno==1 or not isinstance(row[1],str):continue
                    raw=row[1]
                    meta=dict(base,location=f'{sheet.title}:row:{rowno}',domain='pde',subdomain='coefficient_free_pde_structure',
                              source_path=source['path'],source_group='eqgpt:'+canonical(raw),raw_structure=raw,
                              transformation='AI-assisted deterministic mapping of GRE terms to coefficient-free differential expression; no equality/coefficient invented',
                              latex_author_origin='ai_assisted_conversion',equation_name=str(row[0]))
                    try:
                        formula=' '.join(eqterm(x.strip()) for x in raw.split(','))
                        if not re.search(r'\\partial|\\Delta',formula):raise ValueError('not a PDE derivative structure')
                        # Group is invariant to reorderings of top-level additive terms.
                        meta['source_group']='eqgpt:'+ '|'.join(sorted(raw.split(',+,')))
                        add(formula,meta)
                    except ValueError as exc:
                        rejections.append(dict(meta,reason=str(exc),formula=raw))
        elif source['id'].startswith('openai:'):
            text=(ROOT/source['local']).read_text(errors='replace')
            for formula,line in latex_math(text):
                # PDE papers include norms/algebraic statements. Only explicit
                # partial/spatial differential equations enter the PDE bucket.
                paper_domain=source['domain']
                is_pde=paper_domain=='pde' and bool(re.search(r'\\partial|\\nabla|\\Delta',formula)) and '=' in formula
                domain='pde' if is_pde else ('analysis' if paper_domain=='pde' else paper_domain)
                # Files from one related manuscript family cannot cross splits.
                group=re.sub(r'-(September|October)-\d+-2026$','',source['group'])
                add(formula,dict(base,location=f'line:{line}',source_path=source['path'],domain=domain,subdomain='paper_formula',source_group=group,transformation='math delimiter extraction; outer whitespace stripped',latex_author_origin='possible_mixed'))
        else:
            with tarfile.open(ROOT/source['local']) as archive:
                for member in sorted(archive.getmembers(),key=lambda x:x.name):
                    if '/doc/src/' not in member.name or not member.name.endswith('.rst'):continue
                    text=archive.extractfile(member).read().decode(errors='replace')
                    path=member.name.split('/',1)[1]
                    domain=('linear_algebra' if 'matric' in path else 'combinatorics' if 'combinator' in path else
                            'probability' if 'stats' in path else 'ode' if '/ode.' in path else
                            'number_theory' if 'diophantine' in path or 'ntheory' in path else
                            'algebra' if 'polys' in path else 'analysis')
                    for formula,line in rst_math(text):
                        pde='/pde.' in path and bool(re.search(r'\\partial|\\nabla|\\Delta',formula))
                        add(formula,dict(base,location=f'line:{line}',source_path=path,domain='pde' if pde else domain,subdomain='documentation_formula',source_group='sympy:'+path,transformation='RST math role/directive extraction',latex_author_origin='unknown'))
    # Explicit assistant-authored synthetic examples: template-generated, not
    # independent LLM samples. Families, seeds and generation recipe recorded.
    ai_base=dict(source_id='assistant-synthetic-v1',source_revision='2026-10-10-v1',source_sha256=None,
                 source_repo='this_repository',origin='ai_generated',license='CC0-1.0',
                 origin_evidence='All templates and formula text authored by Codex in this task; deterministic expansion, no human editing.',
                 transformation='assistant-authored templates; deterministic parameter expansion',latex_author_origin='ai_generated',source_path='src/math_tokenizer/corpus.py')
    templates=[('pde',r'\frac{\partial u}{\partial t}=a \frac{\partial^{2} u}{\partial x^{2}}+b u^{2}'),
      ('pde',r'\frac{\partial^{2} v}{\partial t^{2}}=a \Delta v-b v'),
      ('algebra',r'(x+a)^{2}=x^{2}+2 a x+a^{2}'),
      ('algebra',r'\frac{x^{2}-a^{2}}{x-a}=x+a'),
      ('linear_algebra',r'\det(A B)=\det(A)\det(B)'),
      ('linear_algebra',r'\lambda^{2}-a\lambda+b=0'),
      ('analysis',r'\frac{d}{d x}\sin(a x)=a\cos(a x)'),
      ('analysis',r'\int_{0}^{a} x^{2} d x=\frac{a^{3}}{3}'),
      ('probability',r'\mathbb{E}(X^{2})=\mathbb{E}(X)^{2}+\mathrm{Var}(X)'),
      ('probability',r'p(x)=\frac{1}{a}\exp(-x/a)'),
      ('number_theory',r'x^{2}+y^{2}=a^{2}'),
      ('number_theory',r'\gcd(a,b)=\gcd(b,a)'),
      ('combinatorics',r'\binom{n}{2}=\frac{n(n-1)}{2}'),
      ('combinatorics',r'2^{n}=\sum_{k=0}^{n}\binom{n}{k}'),
      ('ode',r'\frac{d y}{d t}=a y-b y^{2}'),
      ('ode',r'y(t)=a\exp(-b t)'),
      ('logic',r'(A\land B)\implies A'),
      ('logic',r'A\lor\neg A')]
    for i,(domain,template) in enumerate(templates):
        for a in (2,3,5,7):
            # Substitute scalar a only outside command names, avoiding corruption.
            formula=re.sub(r'(?<![A-Za-z\\])a(?![A-Za-z])',str(a),template)
            add(formula,dict(ai_base,location=f'template:{i}:a:{a}',domain=domain,subdomain='synthetic_template',source_group=f'assistant-template:{i}'))
    # Canonical duplicates are one sample with all provenance retained. Origin
    # disagreement is never resolved by silently choosing a preferred source.
    by_hash={};uf=UnionFind()
    for r in records:
        g=r['source_group'];uf.find(g)
        if r['canonical_sha256'] in by_hash:
            old=by_hash[r['canonical_sha256']]
            old.setdefault('duplicate_provenance',[]).append(r)
            if old['origin']!=r['origin']:old['origin']='possible_mixed'
        else:by_hash[r['canonical_sha256']]=r
    records=list(by_hash.values())
    for r in records:r['split_group']=r['source_group']
    groups=sorted({r['split_group'] for r in records})
    # Fixed deterministic document/group hash split, independent of method.
    def split(g):
        v=int(hashlib.sha256(('mathlab-v1:'+g).encode()).hexdigest()[:8],16)%100
        return 'train' if v<70 else 'dev' if v<85 else 'test'
    for r in records:r['split']=split(r['split_group'])
    # A duplicate with provenance in different partitions is quarantined ONLY
    # at formula level, avoiding a giant component through ubiquitous x=0.
    unique=[]
    for r in records:
        aliases=r.get('duplicate_provenance',[])
        if any(split(a['source_group'])!=r['split'] for a in aliases):
            rejections.append(dict(r,reason='duplicate_spans_partitions'));continue
        unique.append(r)
    records=unique
    # Initial pilot: reproducible hash sampling <=120 formulae per document.
    # EqGPT structures and small synthetic families are retained completely.
    sampled=[];doc_buckets=defaultdict(list)
    for r in records:doc_buckets[r['source_group']].append(r)
    for group,rows in sorted(doc_buckets.items()):
        sampled.extend(sorted(rows,key=lambda r:r['id'])[:120])
    records=sampled
    records.sort(key=lambda r:r['id'])
    out=ROOT/'data/curated';out.mkdir(exist_ok=True,parents=True)
    for old in out.rglob('*.jsonl'):old.unlink()
    def write(path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    write(out/'corpus.jsonl',records)
    for s in ('train','dev','test'):write(out/'splits'/f'{s}.jsonl',[r for r in records if r['split']==s])
    buckets=defaultdict(list)
    for r in records:buckets[(r['domain'],r['origin'])].append(r)
    domains=('pde','ode','algebra','linear_algebra','analysis','number_theory','combinatorics','probability','logic','mathematical_physics')
    for d in domains:
        for o in ('human','possible_mixed','ai_generated','unknown'):
            write(out/'by_domain'/d/f'{o}.jsonl',buckets[(d,o)])
    write(ROOT/'data/quarantine/rejections.jsonl',rejections)
    summary={'count':len(records),'rejected':len(rejections),'split_counts':dict(Counter(r['split'] for r in records)),
             'domain_origin':{f'{d}/{o}':len(rs) for (d,o),rs in sorted(buckets.items())},
             'ast_coverage':sum(r['ast_supported'] for r in records)/max(1,len(records)),
             'group_counts':dict(Counter(split(g) for g in groups)),
             'corpus_sha256':hashlib.sha256((out/'corpus.jsonl').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
