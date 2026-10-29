"""Source-backed GRE benchmark distinct from the author's 221 structures.

Parameterized families are explicitly marked, not passed off as independent
papers. Zero-coefficient ablations and coefficient resamplings do not inflate
structure counts. All candidates, including duplicates/rejections, are saved.
"""
from collections import Counter
import csv,gzip,itertools,json,re
from functools import lru_cache
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from math_tokenizer.corpus import ROOT
from math_tokenizer.pde_protocol import digest
from math_tokenizer.eqgpt.corpus_conversion import convert,ConversionError
from scripts.prepare_downstream_corpus import (invariant_support,gre_expression,words_to_ids,TARGETS,check_author_atoms)

OUT=ROOT/'data/formulas/gre_external'


@lru_cache(maxsize=None)
def snapshot(arxiv,location):
    path=OUT/'sources'/f'{arxiv}.html'
    soup=BeautifulSoup(path.read_text(),'html.parser')
    tag=soup.find(id=location)
    if tag is None:raise ValueError('Missing source location '+arxiv+' '+location)
    return dict(arxiv=arxiv,source_group='arxiv:'+arxiv,title=(soup.select_one('h1.ltx_title').get_text(' ',strip=True) if soup.select_one('h1.ltx_title') else (soup.title.get_text(' ',strip=True) if soup.title else arxiv)),
                source_url=f'https://arxiv.org/html/{arxiv}v1#{location}',source_revision='v1',
                source_file=str(path.relative_to(ROOT)),source_sha256=digest(path.read_bytes()),
                source_location=location,source_math=[m.get('alttext') for m in tag.select('math[alttext]')],
                source_context=tag.parent.get_text(' ',strip=True)[:1800])


def candidates():
    output=[]
    def add(arxiv,location,latex,family,parameters=None,rules=None):
        record=dict(snapshot(arxiv,location))
        record.update(latex=latex,family=family,parameters=parameters or {},
                      origin='declared_parameter_instance' if parameters else 'source_equation',
                      transformations=rules or ['notation normalized to project scalar field u'],
                      id=digest(arxiv+location+latex)[:24])
        output.append(record)
    # Source formula, including full KBF terms; initial condition retained in source_math.
    add('2604.17606','S1.E1',r'u_t=\nu u_{xx}-\mu u_{xxx}+\gamma u_{xxxxx}-\varepsilon u^2 u_x+\epsilon u(1-u)',
        'fifth_order_KdV_Burgers_Fisher',rules=['source field y -> u','source initial condition retained as metadata, not a PDE term'])
    add('2402.07789','S1.E1',r'u_t+\alpha u u_x-\mu u_{xx}+\beta u_{xxx}=r u(1-u)','KdV_Burgers_Fisher')
    add('2402.03788','S1.E1',r'u_t+\beta_1u_{xx}+\beta_2u_{xxxx}+\delta u_{xxx}+u u_x=0','dispersive_KS')
    add('2402.03788','S1.E2',r'u_t+u_{xx}+k u_{xxxx}+u u_x=0','KS',rules=['extract PDE row; boundary and initial conditions retained in source_math'])
    # Paper explicitly studies arbitrary real m; retain only bounded integer examples.
    for m in range(1,7):
        add('1112.5707','S1.E1',rf'u_t+u^{m}u_x+\alpha u_{{xx}}+\beta u_{{xxx}}+\gamma u_{{xxxx}}=0',
            'generalized_KS',{'m':m},['integer instance of source power family; no coefficient ablation'])
    # Source m,n family; original paper explicitly excludes n=0. All here positive.
    for m,n in itertools.product(range(1,6),repeat=2):
        add('1904.01291','S1.E1',rf'u_t+\partial_x(u^{m})+\partial_{{xxx}}(u^{n})=0',
            'Rosenau_Hyman_Kmn',{'m':m,'n':n},['integer instance m,n in 1..5','grouped source derivative notation -> partial notation'])
    # CSS equation (1.2) sign used exactly as body, not inconsistent abstract sign.
    for ell,p in itertools.product(range(2,7),range(1,5)):
        add('1904.01291','S1.E2',rf'u_t+u^{ell-2}u_x-\alpha {p}\partial_x(u^{p-1}u_x^2)+2\alpha\partial_{{xx}}(u^{p}u_x)=0',
            'Cooper_Shepard_Sodano',{'l':ell,'p':p},['integer instance l=2..6,p=1..4','equation (1.2) sign retained; alpha nonzero'])
    for k in range(4,9):
        add('2308.08541','S1.EGx2',rf'\partial_t u+\partial_{{xxx}}u+\mu u^{k}\partial_xu=0',
            'supercritical_gKdV',{'k':k,'mu':1},['k in source range >=4; focusing mu=1','extract PDE row; initial condition retained'])
    # Cahn-Hilliard W is explicitly supplied in equation (1.2), with W'=u^3-u.
    add('1202.1735','S1.E1',r'u_t=\partial_{xx}(u^3)-u_{xx}-\varepsilon^2 u_{xxxx}','Cahn_Hilliard_1D',
        rules=['source explicitly one-dimensional','substitute displayed potential W(xi)=(1-xi^2)^2/4, equation S1.E2','roman u typography -> u','epsilon is the source constant scaling parameter; move it outside the derivative exactly'])
    # Additional explicit rational equations, no family instantiation or truncation.
    add('1904.01291','S5.E5',r'u_t=\frac{u_{xxx}}{u^3}-9\frac{u_x u_{xx}}{u^4}+12\frac{u_x^3}{u^5}+\frac{2}{3}\lambda_1\frac{u_x}{u^3}+\frac12\lambda_2\frac{u_x}{u^2}-c u_x','rational_integrable_5_5')
    add('1904.01291','S5.E6',r'u_t=\frac{a u_{xxx}}{u^6}-12\frac{a u_x u_{xx}}{u^7}+21\frac{a u_x^3}{u^8}+\beta_1\frac{u_x}{u^4}+\beta_2\frac{u_x}{u^3}+\beta_3u_x','rational_integrable_5_6')
    add('1904.01291','S5.E9',r'u_t=\frac{a u_{xxx}}{u^3}-6\frac{a u_x u_{xx}}{u^4}+6\frac{a u_x^3}{u^5}+\beta_1\frac{u_x}{u^3}+\beta_2u u_x+\beta_3u_x','rational_integrable_5_9')
    # Scholarly publisher source checked in browser; exact source formula and
    # DOI retained independently of the blocked terminal HTML download.
    for name,nonlinearity in [('quadratic_cubic',r'\nu u^2-u^3'),('cubic_quintic',r'\nu u^3-u^5')]:
        output.append(dict(id=digest('JDE_2018_SH_'+name)[:24],family='Swift_Hohenberg_'+name,
            source_group='doi:10.1016/j.jde.2018.07.064',
            title='Existence and stability of spatially localized patterns',
            source_url='https://www.sciencedirect.com/science/article/pii/S0022039618304285',
            source_location='equation (1.1)' if name=='quadratic_cubic' else 'equation (1.2)',
            source_math=[r'U_t=-(1+\Delta)^2U-\mu U+'+nonlinearity.replace('u','U')],
            source_access='publisher rendered equation verified through web retrieval; terminal request returned 403; open manuscript saved from NSF repository',
            source_file='data/formulas/gre_external/sources/Makrides_Sandstede_JDE2019.pdf',
            source_sha256=digest((OUT/'sources/Makrides_Sandstede_JDE2019.pdf').read_bytes()),
            repository_url='https://par.nsf.gov/servlets/purl/10099356',
            latex=rf'u_t=-u-2u_{{xx}}-u_{{xxxx}}-\mu u+{nonlinearity}',
            origin='declared_dimension_instance',parameters={'space_dimension':1},
            transformations=['publisher explicitly allows U(x,t) as well as U(x,y,t); choose 1D branch','expand -(1+partial_x^2)^2u exactly','U -> u']))
    return output


def polynomial_family_candidates():
    """Instances of source equation (4), not fabricated additions of terms.

    Independent functions are fixed to f=u^(p+1)/(p+1), phi=u^q,
    g=u^r. Positive integer degrees are predeclared; alpha=beta=gamma=1.
    Equation (4) explicitly permits these functions and constants.
    """
    base=dict(snapshot('2402.03788','S1.E4'))
    output=[]
    for p,q,r in itertools.product(range(1,19),range(2,12),range(1,12)):
        latex=rf'u_t+u^{p}u_x+u_{{xx}}+\partial_{{xx}}(u^{q})+u_{{xxx}}+u_{{xxxx}}=u^{r}'
        record=dict(base,latex=latex,family='generalized_KS_polynomial_flux_diffusion_reaction',
                    parameters={'p':p,'q':q,'r':r,'alpha':1,'beta':1,'gamma':1},
                    function_choices={'f':f'u^({p+1})/{p+1}','phi':f'u^{q}','g':f'u^{r}'},
                    origin='declared_function_family_instance',
                    transformations=['instantiate independent source functions in equation (4)',
                       'chain rule d_x[u^(p+1)/(p+1)] = u^p*u_x',
                       'nonzero constants alpha=beta=gamma=1; no coefficient ablation'],
                    family_scope='1 source equation; degree grid p=1..18,q=2..11,r=1..11; not independent publications',
                    id=digest('2402.03788:S1.E4:'+latex)[:24])
        output.append(record)
    return output


def kawahara_family_candidates():
    """Integer powers of f(u) in the explicitly stated generalized Kawahara class."""
    base=dict(snapshot('1810.02863','S0.E1'))
    out=[]
    for p in range(1,31):
        latex=rf'u_t+u_{{xxxxx}}+u_{{xxx}}+u^{p}u_x=0'
        out.append(dict(base,latex=latex,family='generalized_Kawahara_power_flux',
            parameters={'p':p,'a':1,'b':1},origin='declared_function_family_instance',
            transformations=['source u_t=a(t)u_5x+b(t)u_xxx+f(u)u_x instantiated with constant a=b=1','f(u)=u^p, p=1..30'],
            family_scope='one source equation; integer powers p=1..30; not independent publications',
            id=digest('1810.02863:S0.E1:'+latex)[:24]))
    return out


def eqgpt_221_candidates():
    """Retain the 221 source GRE rows as LaTeX records after target removal."""
    import pandas as pd
    from math_tokenizer.eqgpt.author import DICTIONARY
    from math_tokenizer.eqgpt.latex_codec import structure_to_latex
    path=ROOT/'eqgpt_original/code/PDE_dataset_0725.xlsx';source_sha=digest(path.read_bytes())
    target_keys={invariant_support(gre_expression(words_to_ids(words))) for words in TARGETS.values()}
    df=pd.read_excel(path);out=[]
    for _,row in df.iterrows():
        words=[x for x in str(row['Terms']).split(',') if x]
        ids=[5]+[DICTIONARY['word2id'][w] for w in words]+[1]
        key=invariant_support(gre_expression(ids))
        latex=structure_to_latex(ids)
        out.append(dict(id=digest('eqgpt221:'+str(row['Equation name'])+str(row['Terms']))[:24],
          equation_name=str(row['Equation name']),source_group='eqgpt_original_221',source_file=str(path.relative_to(ROOT)),source_sha256=source_sha,
          source_url='https://github.com/xtr1976536/math-tokenizer-lab',latex=latex,original_latex=latex,
          gre_ids=ids,gre_words=[DICTIONARY['id2word'][i] for i in ids],family='eqgpt_handbook_221',
          origin='eqgpt_original_221_after_target_exclusion',excluded_target=key in target_keys,
          transformations=['GRE handbook row rendered as equivalent LaTeX; coefficient-free source structure retained']))
    return out

def reference221():
    df=pd.read_excel(ROOT/'eqgpt_original/code/PDE_dataset_0725.xlsx')
    keys={};fail=[]
    for _,row in df.iterrows():
        try:
            key=invariant_support(gre_expression(words_to_ids(row['Terms'].split(','))))
            keys.setdefault(key,[]).append(str(row['Equation name']))
        except Exception as exc:fail.append(dict(name=str(row['Equation name']),error=str(exc)))
    assert len(df)==221 and not fail
    return keys


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--include-polynomial-families',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    original=list(map(json.loads,gzip.open(ROOT/'data/formulas/pde/pde_equations.jsonl.gz','rt')))
    originals={r['source_group'] for r in original}
    oldkeys=reference221()
    original_components=list(map(json.loads,(ROOT/'data/formulas/pde1500_gre_review/components.jsonl').read_text().splitlines()))
    original_keys={invariant_support(gre_expression(r['gre_ids'])) for r in original_components if 'gre_ids' in r and r['status']!='unresolved'}
    targets={name:invariant_support(gre_expression(words_to_ids(words))) for name,words in TARGETS.items()}
    records=candidates()
    if args.include_polynomial_families:
        records+=polynomial_family_candidates()
        records+=kawahara_family_candidates()
        records+=eqgpt_221_candidates()
    seen={};eligible=[]
    for index,r in enumerate(records,1):
        if index%50==0:print('CHECK',index,'/',len(records),'accepted',len(eligible),flush=True)
        try:
            result=convert(r['latex']);r['conversion']=result
            key=invariant_support(gre_expression(result['gre_ids']))
            r['handbook_matches']=oldkeys.get(key,[])
            r['original1500_source_overlap']=r['source_group'] in originals
            r['original1500_known_structure_overlap']=key in original_keys
            r['matched_targets']=[name for name,k in targets.items() if k==key]
            r['gre_length']=len(result['gre_ids']);r['fits_current_50_positions']=len(result['gre_ids'])<=50
            if r['handbook_matches']:r['status']='excluded_original221_structure'
            elif r['matched_targets']:r['status']='excluded_target_structure'
            elif r['original1500_source_overlap'] or r['original1500_known_structure_overlap']:r['status']='excluded_original1500_overlap'
            elif key in seen:r['status']='duplicate_structure';r['duplicate_of']=seen[key]
            else:
                seen[key]=r['id'];r['status']='eligible';eligible.append(r)
        except ConversionError as exc:r.update(status='conversion_unresolved',reason=exc.code,detail=str(exc))
    used={i for r in eligible for i in r['conversion']['gre_ids'] if i>=6}
    numerical=check_author_atoms(used)
    for r in eligible:
        bad=[i for i in r['conversion']['gre_ids'] if i>=6 and numerical[i]['status']!='PASS']
        if bad:r.update(status='numerical_check_failed',bad_atoms=bad)
    eligible=[r for r in eligible if r['status']=='eligible']
    from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
    hard=CompactAlphabet.load(ROOT/'experiments/upstream_811/latex_ascii/512.model.json')
    for r in eligible:
        for text in (r['latex'],r['conversion']['normalized_latex']):
            assert hard.decode(hard.encode(text))==text
    for filename,items in [('candidate_records.jsonl',records),('benchmark.jsonl',eligible),('benchmark_maxpos50.jsonl',[r for r in eligible if r['fits_current_50_positions']])]:
        with (OUT/filename).open('w') as f:
            for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (OUT/'benchmark.csv').open('w',newline='') as f:
        fields=['id','family','origin','parameters','source_group','source_url','source_location','latex','gre_length','fits_current_50_positions']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(eligible)
    manifest=dict(candidate_records=len(records),status_counts=dict(Counter(r['status'] for r in records)),
        benchmark_unique_structures=len(eligible),fits_current_50_positions=sum(r['fits_current_50_positions'] for r in eligible),
        source_groups=len({r['source_group'] for r in eligible}),families=dict(Counter(r['family'] for r in eligible)),
        explicit_source_equations=sum(r['origin']=='source_equation' for r in eligible),
        declared_family_or_dimension_instances=sum(r['origin']!='source_equation' for r in eligible),
        dictionary_size=57,no_oov=all(all(0<=i<57 for i in r['conversion']['gre_ids']) for r in eligible),
        original221_all_rows_compared=221,original221_unique_expanded_supports=len(oldkeys),
        deduplication='coefficient-free exact expanded support under x/y/z permutations; no arbitrary nonlinear transformations',
        numeric_atom_audit=numerical,
        limitations=['Original1500 overlap checked by source and all successfully converted components; unresolved mathematical equivalences are not exhaustively ruled out',
          'Parameterized instances do not count as independently sourced PDE families; split by source/family to prevent leakage',
          'Only bounded integer family instances sampled; no fabricated combinations or zero-coefficient ablations',
          'Size may include explicitly declared independent-function family instances; not a claim of that many independent published equations; downstream model and augmentation not changed',
          'EqWorld database endpoints searched but direct TLS/timeout failures prevented verified new formula inclusion'],
        files_sha256={f:digest((OUT/f).read_bytes()) for f in ['candidate_records.jsonl','benchmark.jsonl','benchmark_maxpos50.jsonl','benchmark.csv']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='numeric_atom_audit'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
