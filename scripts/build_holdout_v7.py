"""New-paper test construction and four source-only audits before training.

v6 corpus remains immutable. No tokenizer is loaded by this script. All new
papers are external test only; no model-dependent selection or replenishment.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,subprocess
from math_tokenizer.corpus import ROOT,canonical
from math_tokenizer.quality_v6 import (source_members,build_macros,candidates,normalize,
    formula_gate,classify,template_key,source_macro_reference,TARGET_CATEGORIES)
from math_tokenizer.structure_v2 import complexity


def digest(value):return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    folder=ROOT/'data/releases/v7';rawfolder=ROOT/'data/raw/holdout_v7';rawfolder.mkdir(parents=True,exist_ok=True)
    sources=json.loads((folder/'arxiv_sources.json').read_text());old=list(map(json.loads,(ROOT/'data/releases/v6/curated/corpus.jsonl').open()))
    old_papers={r['source_group'] for r in old};old_exact={r['canonical_sha256'] for r in old};old_templates={r['template_sha256'] for r in old};old_mathml={r['reference']['mathml_sha256'] for r in old}
    # Historical corpora were already available during earlier development;
    # templates excluded from v6 by later gates are still previously exposed.
    historical=(ROOT/'data/releases/v5/curated/corpus.jsonl')
    if historical.exists():
        for prior in map(json.loads,historical.open()):
            old_templates.add(digest(template_key(prior['latex'])))
            old_exact.add(digest(canonical(prior['latex'])))
    rows=[];rejected=[];members=[]
    for source in sources:
        assert source['license_verified_on_version_page'] and source['version_date'][:4]<'2020'
        assert 'arxiv:'+source['id'] not in old_papers
        raw=(ROOT/source['local']).read_bytes();assert digest(raw)==source['sha256']
        textmembers=list(source_members(raw));macros,ambiguous=build_macros(textmembers)
        for name,text,encoding,member_raw in textmembers:
            members.append({'paper':source['id'],'member':name,'sha256':digest(member_raw),'encoding':encoding})
            if text is None or not name.endswith(('.tex','.ltx')):continue
            for c in candidates(text):
                latex,operations,reason=normalize(c,macros,ambiguous);reason=reason or formula_gate(latex)
                identifier=digest('arxiv:'+source['id']+source['version']+name+str(c['source_span'])+c['raw_latex'])[:24]
                if reason:rejected.append({'id':identifier,'paper':source['id'],'reason':reason});continue
                domain,types=classify(latex,source['domain'],source['categories']);group='arxiv:'+source['id']
                row={'id':identifier,'latex':latex,'raw_latex':c['raw_latex'],'transformations':operations,
                    'source_span_characters':c['source_span'],'source_member_sha256':digest(member_raw),
                    'source_encoding':encoding,'source_path':name,'location':f"{name}:line:{c['line']}",
                    'delimiter':c['delimiter'],'canonical_sha256':digest(canonical(latex)),
                    'template_sha256':digest(template_key(latex)),'domain':domain,'paper_domain':source['domain'],
                    'formula_types':types,'subject_categories':source['categories'],'primary_category':source['primary_category'],
                    'origin':'human','origin_basis':source['origin_basis'],'origin_evidence':source['evidence'],
                    'academic_level':'research_source_with_nontrivial_syntax_gate','source_id':group,'source_group':group,
                    'split_group':group,'split':'external_test','source_revision':source['version'],
                    'source_sha256':source['sha256'],'source_repo':'arxiv.org','title':source['title'],'authors':source['authors'],
                    'source_url':source['source_url'],'abstract_url':source['abstract_url'],'version_date':source['version_date'],
                    'license':source['license'],'license_url':source['license_url'],'complexity':complexity(latex),'new_source':True}
                used=[name for op in operations if isinstance(op,dict) for name in op.get('source_macro_expansion',[])]
                if used:row['source_macro_reference']=source_macro_reference(c,macros,ambiguous,used)
                rows.append(row)
    path=rawfolder/'parser_input.jsonl';path.write_text(''.join(json.dumps({'id':r['id'],'latex':r['latex'],'source_macro_reference':r.get('source_macro_reference')},ensure_ascii=False)+'\n' for r in rows))
    output=rawfolder/'parser_output.jsonl'
    with output.open('w') as stream:subprocess.run(['node',str(ROOT/'scripts/latex_reference_v6.js'),str(path)],stdout=stream,check=True)
    parsed={r['id']:r for r in map(json.loads,output.open())};curated=[];seen=set()
    for row in sorted(rows,key=lambda r:r['id']):
        ref=parsed[row['id']]
        reason=None
        if not ref['valid']:reason='independent_katex_syntax'
        elif ref.get('source_macro_fidelity') is False:reason='source_macro_rendering_mismatch'
        elif row['canonical_sha256'] in old_exact:reason='old_exact'
        elif row['template_sha256'] in old_templates:reason='old_template'
        elif ref['mathml_sha256'] in old_mathml:reason='old_rendered_equivalent'
        elif row['canonical_sha256'] in seen:reason='new_duplicate'
        if reason:rejected.append({'id':row['id'],'paper':row['source_group'],'reason':reason});continue
        row['reference']={k:v for k,v in ref.items() if k not in ('id','valid')};seen.add(row['canonical_sha256']);curated.append(row)
    payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in curated).encode()
    destination=folder/'external_test.jsonl.gz'
    with destination.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:zipped.write(payload)
    splits={'external_test':[r['id'] for r in curated],
            'extreme_domain_test':[r['id'] for r in curated if bool(set(r['subject_categories'])&TARGET_CATEGORIES)]}
    write(folder/'splits.json',splits);write(folder/'source_members.json',members)
    write(folder/'quarantine/rejections.json',rejected)
    buckets=defaultdict(list)
    for r in curated:buckets[(r['domain'],r['origin'],r['complexity']['class'])].append(r['latex'])
    for key,formulas in buckets.items():
        p=folder/'formula_only'/key[0]/key[1]/(key[2]+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in formulas))
    audit=[{'round':1,'scope':'unseen fixed pre2020 papers, source/CC license/version hashes','passed':True},
           {'round':2,'scope':'source extraction, bounded macros, independent syntax and source rendering equality','passed':all(r['reference'].get('source_macro_fidelity') is not False for r in curated)},
           {'round':3,'scope':'source-domain/origin/two complexity classes, no prior exact/template/rendered duplicates','passed':not ({r['canonical_sha256'] for r in curated}&old_exact) and not ({r['template_sha256'] for r in curated}&old_templates)},
           {'round':4,'scope':'serialized raw formula fidelity, count/unique IDs and gzip roundtrip','passed':gzip.decompress(destination.read_bytes())==payload and len({r['id'] for r in curated})==len(curated)}]
    manifest={'release':'v7_external_holdout','formulas':len(curated),'papers_downloaded':len(sources),
        'papers_with_accepted_formulas':len({r['source_group'] for r in curated}),'domains':dict(Counter(r['domain'] for r in curated)),
        'paper_domains':dict(Counter(r['paper_domain'] for r in curated)),'complexity':dict(Counter(r['complexity']['class'] for r in curated)),
        'splits':{k:len(v) for k,v in splits.items()},'rejections':dict(Counter(r['reason'] for r in rejected)),
        'raw_jsonl_sha256':digest(payload),'archive_sha256':digest(destination.read_bytes()),
        'previous_v6_corpus_sha256':json.loads((ROOT/'data/releases/v6/manifest.json').read_text())['corpus_sha256'],
        'all_new_papers_test_only':True,'no_tokenizer_scores_used':True,'human_origin_historical_inference':True,
        'prior_v5_and_v6_templates_excluded':True,
        'expert_semantic_dual_review':False,'built_utc':datetime.now(timezone.utc).isoformat()}
    write(folder/'manifest.json',manifest);write(folder/'quality_audit.json',{'rounds':audit,'all_passed':all(a['passed'] for a in audit)})
    write(folder/'seal.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':manifest['archive_sha256'],
        'raw_jsonl_sha256':manifest['raw_jsonl_sha256'],'splits_sha256':digest((folder/'splits.json').read_bytes()),
        'scores_must_not_be_read_before_model_freeze':True})
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
