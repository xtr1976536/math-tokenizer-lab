"""Paper-cluster estimates, honest controlled rankings and weight sensitivity."""
from collections import defaultdict,Counter
import gzip,json,itertools,math
import numpy as np
from scipy.stats import rankdata
from math_tokenizer.corpus import ROOT

OUT=ROOT/'results/v6/full'
LOWER={'document_macro_tokens_per_byte','independent_tree_crossing','fit_predictive_unigram_bits_per_byte'}


def aggregate_rows(rows):
    totals=Counter();groups=defaultdict(lambda:[0,0])
    for row in rows:
        for k in ('tokens','bytes','commands','commands_intact','numbers','numbers_intact','lexemes','lexemes_intact','operators','operators_isolated','boundary_tp','boundary_pred','boundary_ref'):totals[k]+=row[k]
        groups[row['group']][0]+=row['tokens'];groups[row['group']][1]+=row['bytes']
        totals['crossing']+=row['crossing_tokens'] or 0
    return {'formulas':len(rows),'tokens_per_formula':totals['tokens']/len(rows),'tokens_per_byte':totals['tokens']/totals['bytes'],
       'document_macro_tokens_per_byte':float(np.mean([t/b for t,b in groups.values()])),
       'command_integrity':totals['commands_intact']/totals['commands'] if totals['commands'] else None,
       'number_integrity':totals['numbers_intact']/totals['numbers'] if totals['numbers'] else None,
       'independent_tree_crossing':totals['crossing']/totals['tokens'],
       'roundtrip_rate':sum(r['roundtrip'] for r in rows)/len(rows)}


def family(name):return 'v5' if name in {'v5_trajectory','v5_structural'} else name


def rankings(metrics,weights):
    grouped=defaultdict(list)
    for m in metrics:
        correct_suite='fresh_test' if m['scenario']=='broad' else 'extreme_domain_test'
        if m['group']=='matched_primary' and m['suite']==correct_suite:grouped[(m['capacity'],m['scenario'],family(m['method']))].append(m)
    means={key:{metric:float(np.mean([m[metric] for m in values])) for metric in weights} for key,values in grouped.items()}
    boards={};combined={}
    for capacity in (1024,2048):
        scores=defaultdict(list)
        for scenario in ('broad','specialized'):
            names=sorted(key[2] for key in means if key[:2]==(capacity,scenario))
            eligible=[name for name in names if all(m['rank_eligible_lossless'] for m in grouped[(capacity,scenario,name)])]
            score={name:0. for name in eligible}
            for metric,weight in weights.items():
                values=np.array([means[(capacity,scenario,name)][metric] for name in eligible]);values=values if metric in LOWER else -values
                percentile=(rankdata(values,method='average')-1)/max(1,len(eligible)-1)
                for name,value in zip(eligible,percentile):score[name]+=weight*value
            primary=sorted(eligible,key=lambda name:means[(capacity,scenario,name)]['document_macro_tokens_per_byte'])
            overall=sorted(eligible,key=lambda name:(score[name],name))
            pareto=[]
            for name in eligible:
                point=np.array([means[(capacity,scenario,name)][metric]*(1 if metric in LOWER else -1) for metric in weights])
                dominated=False
                for other in eligible:
                    if other==name:continue
                    p=np.array([means[(capacity,scenario,other)][metric]*(1 if metric in LOWER else -1) for metric in weights])
                    if np.all(p<=point) and np.any(p<point):dominated=True;break
                if not dominated:pareto.append(name)
            boards[f'{capacity}/{scenario}']={'primary_ranking':primary,'comprehensive_ranking':overall,'scores':score,'pareto':pareto,
              'metrics':{name:means[(capacity,scenario,name)] for name in eligible},'excluded_nonlossless':[name for name in names if name not in eligible]}
            for name in eligible:scores[name].append(score[name])
        common={name:float(np.mean(values)) for name,values in scores.items() if len(values)==2}
        combined[str(capacity)]={'ranking':sorted(common,key=lambda name:(common[name],name)),'scores':common}
    return boards,combined


def paired_bootstrap(left,right,repeats=2000):
    assert {r['id'] for r in left}=={r['id'] for r in right};l={r['id']:r for r in left};rr={r['id']:r for r in right};groups=defaultdict(list)
    for id,row in l.items():groups[row['group']].append(id)
    names=sorted(groups);diff=[]
    for name in names:
        ids=groups[name];bytes=sum(l[id]['bytes'] for id in ids)
        diff.append((sum(l[id]['tokens'] for id in ids)-sum(rr[id]['tokens'] for id in ids))/bytes)
    rng=np.random.default_rng(8923);values=np.array(diff);draws=np.mean(values[rng.integers(0,len(values),(repeats,len(values)))],axis=1)
    return {'paper_macro_tokens_per_byte_difference':float(values.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'papers':len(values),'resamples':repeats,'negative_favors_left':True}


def main():
    metrics=json.loads((OUT/'metrics.json').read_text());protocol=json.loads((ROOT/'results/v6/evaluation_protocol.json').read_text());weights=protocol['comprehensive_weights']
    rows=defaultdict(list);by_metric={}
    for row in map(json.loads,gzip.open(OUT/'per_formula.jsonl.gz','rt')):rows[(row['key'],row['suite'])].append(row)
    grouped=[]
    for m in metrics:
        rs=rows[(m['key'],m['suite'])];by_metric[(m['capacity'],m['scenario'],m['suite'],m['method'],m['seed'])]=rs
        for column in ('domain','paper_domain','origin','complexity','group'):
            bucket=defaultdict(list)
            for row in rs:bucket[row[column]].append(row)
            for value,subset in bucket.items():grouped.append({'key':m['key'],'method':m['method'],'seed':m['seed'],'capacity':m['capacity'],'scenario':m['scenario'],'suite':m['suite'],'axis':column,'value':value,**aggregate_rows(subset)})
    (OUT/'grouped_metrics.json').write_text(json.dumps(grouped,indent=2)+'\n')
    boards,combined=rankings(metrics,weights);sensitivity=[]
    for speed in (.10,.20,.30,.40):
        alternate={k:v*(1-speed)/(1-weights['encode_mib_s']) for k,v in weights.items() if k!='encode_mib_s'};alternate['encode_mib_s']=speed
        _,rank=rankings(metrics,alternate);sensitivity.append({'speed_weight':speed,'ranking':rank})
    pairs={}
    for capacity in (1024,2048):
        for scenario,suite in [('broad','fresh_test'),('specialized','extreme_domain_test')]:
            selected=[m for m in metrics if m['capacity']==capacity and m['scenario']==scenario and m['suite']==suite and m['group']=='matched_primary' and family(m['method'])=='v5']
            targets=[('ast_v4','ast_v4_previous_dictionary'),('ast_v4','hf_bpe')]
            for m in selected:targets += [(m['method'],'v5_previous_local'),(m['method'],'v5_atomic_frequency')]
            if any(m['method']=='v5_previous_trajectory' and m['capacity']==capacity and m['scenario']==scenario for m in metrics):targets += [(m['method'],'v5_previous_trajectory') for m in selected]
            for left,right in targets:
                left_keys=[k for k in by_metric if k[:3]==(capacity,scenario,suite) and k[3]==left];right_keys=[k for k in by_metric if k[:3]==(capacity,scenario,suite) and k[3]==right]
                if not right_keys:continue
                for lk in left_keys:pairs[f'{capacity}/{scenario}/{left}-vs-{right}/{lk[4]}']=paired_bootstrap(by_metric[lk],by_metric[right_keys[0]])
    result={'weights':weights,'within_scenario':boards,'combined':combined,'weight_sensitivity':sensitivity,'paired_paper_bootstraps':pairs,
            'scope':'Matched training and ID budget, broad/extreme scenarios equal weight; fixed pretrained references not in causal algorithm ranking.',
            'uncertainty':'Paper-cluster CIs condition on the realized trained vocabulary; stochastic seed spread is separately reported; deterministic seeds are not independent replications.'}
    (OUT/'rankings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'combined':combined,'primary':{k:v['primary_ranking'] for k,v in boards.items()}},indent=2),flush=True)


if __name__=='__main__':main()
