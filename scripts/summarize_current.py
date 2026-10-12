"""Seven-system report; retained metrics, shared calibration, explicit ranks."""
from collections import defaultdict
import csv,json
import numpy as np
from scipy.stats import rankdata
from math_tokenizer.corpus import ROOT

WEIGHTS={'document_macro_tokens_per_byte':.30,'encode_mib_s':.25,'independent_tree_crossing':.20,
         'command_integrity':.10,'number_integrity':.05,'fit_predictive_unigram_bits_per_byte':.10}
LOWER={'document_macro_tokens_per_byte','independent_tree_crossing','fit_predictive_unigram_bits_per_byte'}


def main():
    out=ROOT/'experiments/current';assert json.loads((out/'three_round_audit.json').read_text())['all_passed']
    metrics=json.loads((out/'metrics.json').read_text());boards={};csvrows=[]
    for capacity in (1024,2048):
        for scenario,suite in [('broad','external_test'),('specialized','extreme_domain_test')]:
            selected=[dict(m) for m in metrics if m['suite']==suite and (m['scenario']=='pretrained' or (m['capacity']==capacity and m['scenario']==scenario))]
            assert len(selected)==7 and len({m['test_id_sha256'] for m in selected})==1
            data={}
            for m in selected:
                if m['scenario']=='pretrained':m.update(m['fit_calibrations'][scenario]);m.update(m['additional_by_fit_scope'][scenario])
                data[m['method']]=m
            names=sorted(data);scores={n:0. for n in names};permetric={}
            for metric,weight in WEIGHTS.items():
                x=[data[n][metric]*(1 if metric in LOWER else -1) for n in names]
                permetric[metric]=dict(zip(names,map(int,rankdata(x,method='min'))))
                for n,rank in zip(names,rankdata(x,method='average')):scores[n]+=weight*(rank-1)/(len(names)-1)
            label=f'{capacity}/{scenario}';order=sorted(names,key=lambda n:(scores[n],n))
            boards[label]={'metrics':data,'scores':scores,'ranking':order,'metric_ranks':permetric}
            for name,m in data.items():csvrows.append({'capacity_budget':capacity,'evaluation_scenario':scenario,'method':name,'composite_rank':order.index(name)+1,
                **{k:v for k,v in m.items() if isinstance(v,(str,int,float,bool)) or v is None}})
    combined={}
    for c in (1024,2048):
        score={n:np.mean([boards[f'{c}/{s}']['scores'][n] for s in ('broad','specialized')]) for n in boards[f'{c}/broad']['scores']}
        combined[str(c)]={'scores':score,'ranking':sorted(score,key=lambda n:(score[n],n))}
    (out/'rankings.json').write_text(json.dumps({'weights':WEIGHTS,'boards':boards,'combined':combined,
        'selection':'Retrospective restriction of frozen v8 to seven user-selected methods; original scores unchanged',
        'pretrained_capacity':'Full official vocabulary, not equal-budget causal comparison'},indent=2)+'\n')
    fields=sorted(set().union(*(r.keys() for r in csvrows)))
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
    print('Seven-system combined rankings',combined,flush=True)


if __name__=='__main__':main()
