"""Merge external GRE benchmark with newly certified components from the original 1500.

The merged release keeps provenance and coefficient-free structure deduplication;
no original row is silently converted into an external source record.
"""
import json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from scripts.prepare_downstream_corpus import invariant_support,gre_expression
from scripts.report_gre_benchmark import reference221


def main():
    ext=ROOT/'data/formulas/gre_external'; review=ROOT/'data/formulas/pde1500_gre_review';out=ROOT/'data/formulas/gre_benchmark_merged';out.mkdir(exist_ok=True)
    external=list(map(json.loads,(ext/'benchmark.jsonl').read_text().splitlines()))
    original=list(map(json.loads,(review/'eligible_components.jsonl').read_text().splitlines()))
    old=reference221();seen={}
    merged=[];counts={'external':0,'original1500_component':0,'duplicate_original1500_component':0,'original1500_target_excluded':0}
    for r in external:
        key=invariant_support(gre_expression(r['conversion']['gre_ids']));seen[key]=r['id'];r=dict(r);r['benchmark_origin']='external_paper';merged.append(r);counts['external']+=1
    for r in original:
        key=invariant_support(gre_expression(r['gre_ids']))
        if r['status']=='target_excluded':counts['original1500_target_excluded']+=1;continue
        if key in old or key in seen:
            counts['duplicate_original1500_component']+=1;continue
        r=dict(r);r['benchmark_origin']='original1500_certified_component';r['merged_id']=r['parent_id']+':component:'+str(r['component_index']);seen[key]=r['merged_id'];merged.append(r);counts['original1500_component']+=1
    with (out/'benchmark.jsonl').open('w') as f:
        for r in merged:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(status='PASS',merged_structures=len(merged),external_structures=len(external),original1500_certified_added=counts['original1500_component'],counts=counts,deduplication='coefficient-free expanded support; original 221 and external/original overlaps excluded',external_source_manifest=str((ext/'manifest.json').relative_to(ROOT)),original_review_audit=str((review/'audit.json').relative_to(ROOT))),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(merged_structures=len(merged),counts=counts),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
