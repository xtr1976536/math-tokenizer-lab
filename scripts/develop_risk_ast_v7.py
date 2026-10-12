"""Source-node reliability calibration on fit, finite-risk lattice on dev.

This is agreement with training KaTeX, not independent mathematical gold.
Small typed categories are an explicit parser prior, with no hand-ranked risks.
"""
import argparse,json
from collections import Counter
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.frontend_v6 import tree_ranges
from math_tokenizer.current_models import load_current
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def bucket(raw,a,b):return (0 if raw[a:a+1]==b'{' else 1 if raw[a:a+1]==b'\\' else 2)*2+int(b-a>8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];dev=[byid[id] for id in scope['dev_ids']]
    good=Counter();count=Counter()
    for row in fit:
        raw=row['latex'].encode();gold=set(map(tuple,row['reference']['spans']))
        for s,e in tree_ranges(row['latex']).tolist():
            k=bucket(raw,s,e);count[k]+=1;good[k]+=(s,e) in gold
    risk=[(good[k]+1)/(count[k]+2) if count[k] else 0. for k in range(8)]
    calibration={'fit_id_count':len(fit),'risk_weights':risk,'counts':dict(count),'matching_spans':dict(good),
                 'reference':'training-only KaTeX span agreement','manual_prior':'brace/command/other start and length>8 categories','expert_gold':False}
    (folder/'ast_node_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    results=[]
    for source in ('ast_v4','v5'):
        base=load_current(source,a.scenario,a.capacity)
        for penalty in (.75,1.5,3.,6.):
            name=f'ast_risk_{source}_p{penalty}';model=BalancedAST(base.vocab,penalty,risk);model.train_seconds=base.train_seconds
            model.training_log=[{'dictionary_source':source,'fit_only_node_agreement':calibration}]
            model.save(folder/'models'/f'{name}.json')
            m,_=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit));m.update(phase='development',dev_selection_utility=selection_utility(m),capacity=a.capacity)
            results.append(m);(folder/'risk_ast_dev.json').write_text(json.dumps(results,indent=2)+'\n')
            print(a.scenario,a.capacity,name,round(m['document_macro_tokens_per_byte'],6),round(m['independent_tree_crossing'],4),round(m['dev_selection_utility'],6),flush=True)


if __name__=='__main__':main()
