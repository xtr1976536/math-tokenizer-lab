"""Create unreviewed independent-annotation material; never claim human review."""
from collections import defaultdict
import argparse,hashlib,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v7 import read_external
from math_tokenizer.independent_v7 import source_scopes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=['v7','v8'],default='v7');args=parser.parse_args()
    rows=read_external(args.release);buckets=defaultdict(list)
    for r in rows:buckets[(r['paper_domain'],r['complexity']['class'])].append(r)
    chosen=[]
    for key,group in sorted(buckets.items()):
        chosen.extend(sorted(group,key=lambda r:hashlib.sha256((args.release+'-review:'+r['id']).encode()).digest())[:8])
    records=[]
    for row in chosen:
        records.append({'id':row['id'],'latex':row['latex'],'raw_latex':row['raw_latex'],
            'source_url':row['source_url'],'source_location':row['location'],'paper_domain':row['paper_domain'],
            'complexity_proxy':row['complexity'],'transformations':row['transformations'],
            'automated_pylatexenc_scope_spans':source_scopes(row['latex']),
            'automated_katex_spans':row['reference']['spans'],
            'reviewer_1':{'source_fidelity':None,'scope_boundaries':None,'domain_agreement':None,'notes':None},
            'reviewer_2':{'source_fidelity':None,'scope_boundaries':None,'domain_agreement':None,'notes':None},
            'adjudication':None})
    folder=ROOT/f'data/releases/{args.release}/review_materials';folder.mkdir(exist_ok=True)
    (folder/'unreviewed_samples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    (folder/'status.json').write_text(json.dumps({'samples':len(records),'human_reviews_completed':0,
        'expert_gold':False,'purpose':'Source/domain/scope double-review material only; expert answers deliberately blank, not evaluated as ground truth.',
        'sampling':'Up to8 hash-chosen samples per source-domain/proxy-complexity class, no tokenizer score selection.'},indent=2)+'\n')
    print('unreviewed independent annotation samples',len(records),flush=True)


if __name__=='__main__':main()
