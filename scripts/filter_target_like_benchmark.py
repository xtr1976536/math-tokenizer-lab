"""Remove exact/highly target-like structures before downstream preparation.

Policy: exclude exact target support (already absent) and candidates whose
coefficient-free support contains a target support with <=2 additional terms.
The latter catches direct target variants while retaining broad families with
many additional operators.
"""
import json
from collections import Counter
from pathlib import Path
from math_tokenizer.corpus import ROOT
from scripts.prepare_downstream_corpus import TARGETS,words_to_ids,invariant_support,gre_expression


def main():
    source=ROOT/'data/formulas/gre_external/benchmark.jsonl';out=ROOT/'data/formulas/gre_benchmark_filtered';out.mkdir(exist_ok=True)
    rows=list(map(json.loads,source.read_text().splitlines()))
    targets={name:set(invariant_support(gre_expression(words_to_ids(words)))) for name,words in TARGETS.items()}
    keep=[];excluded=[]
    for row in rows:
        support=set(row['conversion']['support']);matches=[]
        for name,key in targets.items():
            if key<=support:
                matches.append(dict(case=name,target_terms=len(key),candidate_terms=len(support),extra_terms=len(support-key),
                                    overlap_fraction=len(key)/len(support)))
        close=[m for m in matches if m['extra_terms']<=2]
        if close:
            row=dict(row);row['filter_status']='excluded_target_like';row['target_like_matches']=close;excluded.append(row)
        else:
            row=dict(row);row['filter_status']='retained';row['target_like_matches']=matches;keep.append(row)
    (out/'benchmark.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in keep))
    (out/'excluded_target_like.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded))
    manifest=dict(status='PASS',input_records=len(rows),retained_records=len(keep),excluded_records=len(excluded),excluded_by_case=dict(Counter(m['case'] for r in excluded for m in r['target_like_matches']),),policy='exact coefficient-free expanded support; subset with at most two extra terms; no coefficient-only variants retained',source=str(source.relative_to(ROOT)),old_benchmark_untouched=True)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
