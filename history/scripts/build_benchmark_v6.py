"""Source-first strict release. Run before fitting/choosing any v6 tokenizer."""
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import time

from math_tokenizer.corpus import ROOT, canonical
from math_tokenizer.quality_v6 import source_members, build_macros, candidates, normalize, formula_gate, classify, template_key, TARGET_CATEGORIES,source_macro_reference,library_topics
from math_tokenizer.structure_v2 import complexity

RELEASE = ROOT/'data/releases/v6'
RAW = ROOT/'data/raw/benchmark_v6'
OOD = {'algebra', 'geometry_topology', 'number_theory', 'combinatorics', 'logic'}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def jsonlines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))


def build():
    RAW.mkdir(parents=True, exist_ok=True)
    new_sources = json.loads((RELEASE/'arxiv_sources.json').read_text())
    sources = json.loads((ROOT/'data/arxiv_sources.json').read_text()) + json.loads((ROOT/'data/releases/v5/arxiv_sources.json').read_text()) + new_sources
    assert len({s['id'] for s in sources}) == len(sources)
    fresh_groups = {'arxiv:'+s['id'] for s in new_sources}
    by_domain = defaultdict(list)
    for s in sources:by_domain[s['domain']].append(s)
    assignment = {}
    for domain, ss in sorted(by_domain.items()):
        fresh = sorted([s for s in ss if 'arxiv:'+s['id'] in fresh_groups], key=lambda s:digest('v6-new-paper:'+s['id']))
        # Half the newly collected papers are holdout; all old papers have been exposed.
        for i,s in enumerate(fresh):assignment['arxiv:'+s['id']] = 'fresh_test' if i < max(1,len(fresh)//2) else 'dev' if i == len(fresh)//2 and len(fresh)>=4 else 'train'
        old = sorted([s for s in ss if 'arxiv:'+s['id'] not in fresh_groups], key=lambda s:digest('v6-old-paper:'+s['id']))
        for i,s in enumerate(old):assignment['arxiv:'+s['id']] = 'dev' if i % 7 == 0 else 'train'
    rows, rejected, member_manifest = [], [], []
    for index, source in enumerate(sources):
        assert source['license_verified_on_version_page'] and source['version_date'][:4] < '2020'
        raw = (ROOT/source['local']).read_bytes()
        assert digest(raw) == source['sha256']
        members = list(source_members(raw)); macros, ambiguous = build_macros(members)
        for name, text, encoding, member_raw in members:
            member_manifest.append({'paper': source['id'], 'member':name, 'sha256':digest(member_raw), 'encoding':encoding, 'bytes':len(member_raw)})
            if text is None:
                rejected.append({'paper':source['id'],'file':name,'reason':encoding});continue
            # Included style files provide macro evidence, never formula examples.
            if not name.endswith(('.tex','.ltx')):continue
            for c in candidates(text):
                formula, operations, reason = normalize(c, macros, ambiguous)
                reason = reason or formula_gate(formula)
                id = digest('arxiv:'+source['id']+source['version']+name+str(c['source_span'])+c['raw_latex'])[:24]
                if reason:
                    rejected.append({'id':id,'paper':source['id'],'file':name,'line':c['line'],'reason':reason});continue
                domain, types = classify(formula,source['domain'],source['categories'])
                group = 'arxiv:'+source['id']
                rows.append({'id':id,'latex':formula,'raw_latex':c['raw_latex'], 'transformations':operations,
                    'source_span_characters':c['source_span'],'source_member_sha256':digest(member_raw),'raw_latex_sha256':digest(c['raw_latex']),
                    'source_encoding':encoding,'source_path':name,'location':f"{name}:line:{c['line']}",'delimiter':c['delimiter'],
                    'canonical_sha256':digest(canonical(formula)), 'template_sha256':digest(template_key(formula)),
                    'domain':domain,'paper_domain':source['domain'],'formula_types':types,'subject_categories':source['categories'],
                    'primary_category':source.get('primary_category',source['categories'][0]),'origin':'human',
                    'origin_basis':source['origin_basis'],'origin_evidence':source['evidence'], 'academic_level':'research_source_with_nontrivial_syntax_gate',
                    'source_id':group,'source_group':group,'split_group':group,'split':assignment[group],
                    'source_revision':source['version'],'source_sha256':source['sha256'],'source_repo':'arxiv.org',
                    'title':source['title'],'authors':source['authors'],'source_url':source['source_url'],'abstract_url':source['abstract_url'],
                    'version_date':source['version_date'],'license':source['license'],'license_url':source['license_url'],
                    'complexity':complexity(formula), 'new_source':group in fresh_groups})
                used=[name for op in operations if isinstance(op,dict) for name in op.get('source_macro_expansion',[])]
                if used:rows[-1]['source_macro_reference']=source_macro_reference(c,macros,ambiguous,used)
        if index % 20 == 0:print('extracted',index+1,'papers',len(rows),'candidate formulas',flush=True)
    # Non-arXiv provenance and origins remain explicit. EqGPT is a structural
    # supplement, simple assistant templates are not the strict main benchmark.
    for line in (ROOT/'data/releases/v5/curated/corpus.jsonl').open():
        old = json.loads(line)
        if old['source_repo']=='arxiv.org':continue
        reason = formula_gate(old['latex'])
        if old.get('source_repo','').lower().find('eqgpt')>=0 or old.get('subdomain')=='coefficient_free_pde_structure':reason='pde_structure_supplement_not_complete_equation'
        if old['origin'] in ('ai','ai_generated'):reason='simple_ai_template_supplement_not_research_benchmark'
        if reason:
            rejected.append({'id':old['id'],'source':old['source_id'],'reason':reason});continue
        r = dict(old); r['split'] = 'dev' if int(digest('v6-other:'+old['split_group'])[:8],16)%7==0 else 'train'
        r.setdefault('source_url', 'https://github.com/'+r['source_repo']+'/blob/'+r['source_revision']+'/'+r['source_path'])
        r['complexity']=complexity(r['latex'])
        r['paper_domain'] = old.get('paper_domain',old['domain']);r['subject_categories'] = old.get('subject_categories',[])
        if r['source_repo']=='fredrik-johansson/fungrim':
            import sys
            if str(ROOT/'vendor/fungrim') not in sys.path:sys.path.insert(0,str(ROOT/'vendor/fungrim'))
            import pygrim,pygrim.formulas
            r['paper_domain'],r['source_topic_tags']=library_topics(r['source_path'])
            r['classification_basis']='versioned library topic map, filename and source entry; overlapping topic tags explicit'
            entry=pygrim.formulas.entries_dict[r['source_id'].split(':',1)[1]]
            r['source_assumptions_latex']=[a.latex() for argument in entry.args() if argument.head()==pygrim.Assumptions for a in argument.args()]
            r['source_variables']=[str(a) for argument in entry.args() if argument.head()==pygrim.Variables for a in argument.args()]
        r['domain'],r['formula_types']=classify(r['latex'],r['paper_domain'],r['subject_categories'])
        r['canonical_sha256']=digest(canonical(r['latex']));r['template_sha256']=digest(template_key(r['latex']))
        r['new_source']=False;rows.append(r)
    input_path = RAW/'parser_input.jsonl';output_path=RAW/'parser_output.jsonl'
    jsonlines(input_path,({'id':r['id'],'latex':r['latex'],'source_macro_reference':r.get('source_macro_reference')} for r in rows))
    with output_path.open('w') as output:
        subprocess.run(['node',str(ROOT/'scripts/latex_reference_v6.js'),str(input_path)],stdout=output,check=True)
    parsed = {r['id']:r for r in map(json.loads,output_path.open())}
    accepted=[]
    for r in rows:
        ref=parsed[r['id']]
        if not ref['valid']:
            rejected.append({'id':r['id'],'source':r['source_id'],'reason':'independent_katex_syntax','detail':ref['reason']});continue
        if ref.get('source_macro_fidelity') is False:
            rejected.append({'id':r['id'],'source':r['source_id'],'reason':'independent_source_macro_rendering_mismatch'});continue
        r['reference']={k:v for k,v in ref.items() if k not in ('id','valid')};accepted.append(r)
    # Cross-partition rendered/whitespace duplicates are removed, not assigned
    # a preferred copy that could silently move holdout formulas into fit.
    partitions=defaultdict(set)
    for r in accepted:
        for key in (r['canonical_sha256'],r['reference']['mathml_sha256']):partitions[key].add(r['split'])
    old_templates=set()
    for line in (ROOT/'data/releases/v5/curated/corpus.jsonl').open():old_templates.add(digest(template_key(json.loads(line)['latex'])))
    fit_templates={r['template_sha256'] for r in accepted if r['split']!='fresh_test'}
    seen=set();curated=[]
    for r in sorted(accepted,key=lambda r:r['id']):
        exact=r['canonical_sha256'];mathml=r['reference']['mathml_sha256']
        if len(partitions[exact])>1 or len(partitions[mathml])>1:reason='cross_partition_equivalent_formula'
        elif r['split']=='fresh_test' and r['template_sha256'] in old_templates|fit_templates:reason='fresh_template_previously_exposed'
        elif exact in seen:reason='same_partition_exact_duplicate'
        else:reason=None
        if reason:rejected.append({'id':r['id'],'source':r['source_id'],'reason':reason});continue
        seen.add(exact);curated.append(r)
    splits={name:[] for name in ('train','dev','fresh_test','transfer_train','transfer_dev','extreme_domain_test')}
    for r in curated:
        splits[r['split']].append(r['id'])
        excluded=bool(set(r['subject_categories'])&TARGET_CATEGORIES) or r['paper_domain'] in OOD
        if r['split'] in ('train','dev') and r['source_repo']=='arxiv.org' and not excluded and r['paper_domain'] in {'pde','ode','analysis'}:splits['transfer_'+r['split']].append(r['id'])
        if r['split']=='fresh_test' and excluded:splits['extreme_domain_test'].append(r['id'])
    curated.sort(key=lambda r:r['id']);master=RELEASE/'curated/corpus.jsonl';jsonlines(master,curated)
    with (master.with_suffix('.jsonl.gz')).open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:zipped.write(master.read_bytes())
    write(RELEASE/'splits.json',splits);write(RELEASE/'source_members.json',member_manifest)
    jsonlines(RELEASE/'quarantine/rejections.jsonl',rejected)
    buckets=defaultdict(list)
    for r in curated:buckets[(r['domain'],r['origin'],r['complexity']['class'])].append(r['latex'])
    domains=sorted({r['domain'] for r in curated})
    for domain in domains:
        for origin in ('human','possible_mixed','unknown','ai_generated'):
            for c in ('simple','complex'):jsonlines(RELEASE/'formula_only'/domain/origin/(c+'.jsonl'),buckets[(domain,origin,c)])
    summary={'release':'v6','formulas':len(curated),'papers_downloaded':len(sources),'new_papers':len(new_sources),
        'papers_with_accepted_formulas':len({r['source_group'] for r in curated if r['source_repo']=='arxiv.org'}),
        'new_paper_formulas':sum(r['new_source'] for r in curated),'origins':dict(Counter(r['origin'] for r in curated)),
        'domains':dict(Counter(r['domain'] for r in curated)),'formula_types':dict(Counter(t for r in curated for t in r['formula_types'])),
        'complexity':dict(Counter(r['complexity']['class'] for r in curated)),'splits':{k:len(v) for k,v in splits.items()},
        'rejections':dict(Counter(r['reason'] for r in rejected)),'corpus_sha256':digest(master.read_bytes()),
        'splits_sha256':digest((RELEASE/'splits.json').read_bytes()),'katex_version':'0.16.25',
        'semantic_theorem_verification':False,'human_origin_is_historical_inference':True,
        'old_v5_test_is_now_development_data':True,'fresh_test_is_new_paper_and_template_disjoint':True,
        'built_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(RELEASE/'manifest.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':build()
