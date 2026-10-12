"""Four distinct data reviews; no tokenizer or model scores consulted."""
from collections import Counter, defaultdict
import hashlib
import json
import random
import subprocess
from math_tokenizer.benchmark_v6 import RELEASE, master_path
from math_tokenizer.corpus import ROOT
from math_tokenizer.quality_v6 import source_members, candidates, normalize, build_macros, formula_gate, classify, TARGET_CATEGORIES, template_key
from math_tokenizer.structure_v2 import complexity


def sha(value):return hashlib.sha256(value).hexdigest()


def main():
    rows=list(map(json.loads,master_path().open()));byid={r['id']:r for r in rows}
    splits=json.loads((RELEASE/'splits.json').read_text())
    sources={ 'arxiv:'+s['id']:s for path in ('data/arxiv_sources.json','data/releases/v5/arxiv_sources.json','data/releases/v6/arxiv_sources.json') for s in json.loads((ROOT/path).read_text())}
    logs=[]
    # Round 1: source archive/member hashes, licensing and raw substring audit.
    cached={};verified=0
    for r in rows:
        if r['source_repo']!='arxiv.org':
            assert r['license'] and r['source_url'] and r['source_sha256'];continue
        source=sources[r['source_group']]
        if r['source_group'] not in cached:
            data=(ROOT/source['local']).read_bytes();assert sha(data)==source['sha256']
            assert source['license_verified_on_version_page'] and source['version_date'][:4]<'2020'
            members=list(source_members(data));cached[r['source_group']]={n:(t,raw) for n,t,e,raw in members}
        text,raw=cached[r['source_group']][r['source_path']]
        assert sha(raw)==r['source_member_sha256']
        assert text[slice(*r['source_span_characters'])]==r['raw_latex']
        assert sha(r['raw_latex'].encode())==r['raw_latex_sha256'];verified+=1
    logs.append({'round':1,'review':'license_pinned_source_member_and_exact_raw_span','verified_arxiv_formulas':verified,'passed':True})
    # Round 2: independent KaTeX rerun over every accepted formula.
    raw=ROOT/'data/raw/benchmark_v6';inputpath=raw/'audit_input.jsonl';output=raw/'audit_output.jsonl'
    inputpath.write_text(''.join(json.dumps({'id':r['id'],'latex':r['latex'],'source_macro_reference':r.get('source_macro_reference')},ensure_ascii=False)+'\n' for r in rows))
    with output.open('w') as f:subprocess.run(['node',str(ROOT/'scripts/latex_reference_v6.js'),str(inputpath)],stdout=f,check=True)
    for reference in map(json.loads,output.open()):
        r=byid[reference['id']];assert reference['valid'] and formula_gate(r['latex']) is None
        assert reference['mathml_sha256']==r['reference']['mathml_sha256']
        assert reference.get('source_macro_fidelity') is not False
        assert reference['spans']==r['reference']['spans']
        assert all(0<=a<b<=len(r['latex'].encode()) for a,b in reference['spans'])
    logs.append({'round':2,'review':'whole_corpus_independent_syntax_and_utf8_interval_replay','formulas':len(rows),'passed':True})
    # Round 3: labels, complexity, partitions, exact/rendered/template leakage.
    partitions=defaultdict(set)
    for r in rows:
        assert (r['domain'],r['formula_types'])==classify(r['latex'],r['paper_domain'],r['subject_categories'])
        assert r['complexity']==complexity(r['latex'])
        assert r['origin'] in {'human','unknown','possible_mixed'}
        for key in (r['source_group'],r['canonical_sha256'],r['reference']['mathml_sha256']):partitions[key].add(r['split'])
    assert all(len(v)==1 for v in partitions.values())
    old={sha(template_key(json.loads(line)['latex']).encode()) for line in (ROOT/'data/releases/v5/curated/corpus.jsonl').open()}
    fit={r['template_sha256'] for r in rows if r['split']!='fresh_test'}
    assert not {byid[id]['template_sha256'] for id in splits['fresh_test']} & (old|fit)
    assert all(byid[id]['new_source'] for id in splits['fresh_test'])
    for name in ('transfer_train','transfer_dev'):
        assert all(byid[id]['source_repo']=='arxiv.org' and not set(byid[id]['subject_categories'])&TARGET_CATEGORIES and byid[id]['paper_domain'] in {'pde','ode','analysis'} for id in splits[name])
    assert set(splits['extreme_domain_test'])<=set(splits['fresh_test'])
    logs.append({'round':3,'review':'multiaxis_classification_complexity_and_document_template_leakage','passed':True})
    # Round 4: stratified fixed source-normalization spot replay and pure exports.
    buckets=defaultdict(list)
    for r in rows:buckets[(r['domain'],r['origin'],r['complexity']['class'])].append(r)
    sample=[r for key,rs in sorted(buckets.items()) for r in sorted(rs,key=lambda r:sha(('quality-v6:'+r['id']).encode()))[:4]]
    macros={};count=0
    for r in sample:
        if r['source_repo']!='arxiv.org':continue
        group=r['source_group'];source=sources[group]
        if group not in macros:macros[group]=build_macros(list(source_members((ROOT/source['local']).read_bytes())))
        text=cached[group][r['source_path']][0]
        matches=[c for c in candidates(text) if c['source_span']==r['source_span_characters']]
        assert len(matches)==1 and normalize(matches[0],*macros[group])[0]==r['latex'];count+=1
    for key,rs in buckets.items():
        domain,origin,c=key;contents=list(map(json.loads,(RELEASE/'formula_only'/domain/origin/(c+'.jsonl')).open()))
        assert contents==[r['latex'] for r in rs] and all(isinstance(x,str) for x in contents)
    logs.append({'round':4,'review':'stratified_source_transformation_replay_and_formula_only_exports','sample_formulas':len(sample),'raw_source_replays':count,'passed':True})
    report={'rounds':logs,'all_passed':True,'corpus_sha256':sha(master_path().read_bytes()),
        'limitations':['Automated and agent source reviews; no claim of independent human expert annotation.','Syntax and traceable source fidelity do not certify mathematical truth.','Pre2020 human origin is a historical inference, not author attestation.','Research provenance and a nontrivial syntax gate do not certify educational difficulty.'],
        'quality_sample_ids':[r['id'] for r in sample]}
    (RELEASE/'quality_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
