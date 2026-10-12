"""Same-input direct boards, grouped effects, predeclared paper-cluster pairs."""
from collections import defaultdict
import csv,gzip,json,math
import numpy as np
from scipy.stats import rankdata
from math_tokenizer.corpus import ROOT
try:
    from scripts.summarize_full_v6 import aggregate_rows
except ModuleNotFoundError:
    from summarize_full_v6 import aggregate_rows

LOWER={'document_macro_tokens_per_byte','independent_tree_crossing','fit_predictive_unigram_bits_per_byte'}


def build_boards(metrics,weights):
    boards={}
    for capacity in (1024,2048):
        for scenario,suite,label in [('broad','external_test','broad'),('specialized','extreme_domain_test','specialized'),
                                     ('broad','extreme_domain_test','broad_extreme'),('broad','stress_test','broad_stress'),('specialized','stress_test','specialized_stress')]:
            eligible=[m for m in metrics if m['suite']==suite and (
                m['group']=='fixed_pretrained_reference' or m['group']=='main' and m['capacity']==capacity and m['scenario']==scenario)]
            assert len({(m['formulas'],m['test_id_sha256'],m['bytes']) for m in eligible})==1
            grouped=defaultdict(list)
            for original in eligible:
                m=dict(original)
                if m['scenario']=='pretrained':
                    m.update(m['fit_calibrations'][scenario]);m.update(m['additional_by_fit_scope'][scenario])
                grouped[m['method']].append(m)
            names=sorted(n for n,rs in grouped.items() if all(m['rank_eligible_lossless'] for m in rs));means={};seed_ranges={}
            for name,rs in grouped.items():
                means[name]={k:float(np.mean([r[k] for r in rs])) for k,v in rs[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
                means[name]['keys']=[r['key'] for r in rs];means[name]['model_group']=rs[0]['group'];means[name]['seed_count']=len(rs)
                seed_ranges[name]={k:{'min':min(r[k] for r in rs),'max':max(r[k] for r in rs)} for k in weights}
            score={n:0. for n in names};metric_rank={}
            for key,weight in weights.items():
                values=[means[n][key]*(1 if key in LOWER else -1) for n in names]
                ranks=rankdata(values,method='average');metric_rank[key]=dict(zip(names,map(int,rankdata(values,method='min'))))
                for n,r in zip(names,ranks):score[n]+=weight*(r-1)/max(1,len(names)-1)
            points={n:np.array([means[n][k]*(1 if k in LOWER else -1) for k in weights]) for n in names}
            pareto=[n for n in names if not any(np.all(points[o]<=points[n]) and np.any(points[o]<points[n]) for o in names if o!=n)]
            boards[f'{capacity}/{label}']={'suite':suite,'fit_scenario':scenario,'test_id_sha256':eligible[0]['test_id_sha256'],
                'metrics':means,'seed_ranges':seed_ranges,'scores':score,'metric_ranks':metric_rank,
                'compression_order':sorted(names,key=lambda n:(means[n]['document_macro_tokens_per_byte'],n)),
                'comprehensive_order':sorted(names,key=lambda n:(score[n],n)),'pareto':pareto,
                'excluded_nonlossless':sorted(set(grouped)-set(names))}
    combined={}
    for capacity in (1024,2048):
        a,b=boards[f'{capacity}/broad'],boards[f'{capacity}/specialized'];assert set(a['scores'])==set(b['scores'])
        scores={n:(a['scores'][n]+b['scores'][n])/2 for n in a['scores']}
        combined[str(capacity)]={'ranking':sorted(scores,key=lambda n:(scores[n],n)),'scores':scores}
    return boards,combined


def paired_effect(left_sets,right_sets,repeats=2000,family_size=1):
    # Seed averaging is per formula, then paper clustering. The CI conditions
    # on observed seed models and does not invent independent training samples.
    def means(sets):
        expected={r['id'] for r in sets[0]};assert all({r['id'] for r in rs}==expected for rs in sets)
        by=defaultdict(list)
        for rs in sets:
            for r in rs:by[r['id']].append(r)
        return {id:{'tokens':float(np.mean([r['tokens'] for r in vals])),'bytes':vals[0]['bytes'],'group':vals[0]['group']} for id,vals in by.items()}
    a,b=means(left_sets),means(right_sets);assert set(a)==set(b);groups=defaultdict(list)
    for id,r in a.items():groups[r['group']].append(id)
    differences=[]
    for group,ids in sorted(groups.items()):
        differences.append(sum(a[id]['tokens']-b[id]['tokens'] for id in ids)/sum(a[id]['bytes'] for id in ids))
    diff=np.array(differences);rng=np.random.default_rng(8923)
    samples=diff[rng.integers(0,len(diff),(repeats,len(diff)))].mean(axis=1)
    q=.05/(2*family_size)
    return {'paper_macro_tokens_per_byte_difference':float(diff.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
        'familywise_95_bonferroni_ci':np.quantile(samples,[q,1-q]).tolist(),'family_size':family_size,'papers':len(groups),
        'resamples':repeats,'bootstrap_tail_resolution':1/repeats,'adjusted_tail_probability':q,
        'negative_favors_left':True,'left_seed_count':len(left_sets),'right_seed_count':len(right_sets),
        'conditional_on_frozen_vocabularies':True,'warning':'Adjusted tail is estimated from finite bootstrap resamples; not proof of universal improvement.'}


def main():
    folder=ROOT/'results/v7/full';assert json.loads((folder/'fairness_audit.json').read_text())['all_passed']
    protocol=json.loads((folder/'evaluation_protocol.json').read_text());weights=protocol['comprehensive_weights'];metrics=json.loads((folder/'metrics.json').read_text())
    rows=defaultdict(list)
    for r in map(json.loads,gzip.open(folder/'per_formula.jsonl.gz','rt')):rows[(r['key'],r['suite'])].append(r)
    grouped=[]
    for m in metrics:
        rs=rows[(m['key'],m['suite'])]
        for axis in ('domain','paper_domain','complexity','origin','group'):
            buckets=defaultdict(list)
            for r in rs:buckets[r[axis]].append(r)
            for label,subset in buckets.items():grouped.append({'key':m['key'],'method':m['method'],'capacity':m['capacity'],'scenario':m['scenario'],'suite':m['suite'],
                'axis':axis,'value':label,**aggregate_rows(subset)})
    (folder/'grouped_metrics.json').write_text(json.dumps(grouped,indent=2)+'\n')
    boards,combined=build_boards(metrics,weights);targets=[]
    for capacity in (1024,2048):
        for scenario,suite in [('broad','external_test'),('specialized','extreme_domain_test')]:
            candidates=[m for m in metrics if m['capacity']==capacity and m['scenario']==scenario and m['suite']==suite]
            for left,right in [('ast_lexicographic','official_atomic_bpe'),('ast_lexicographic','lexical_shortest'),
                ('v5_terminal_transfer','v5_terminal_no_update'),('v5_terminal_transfer','v5_policy_search')]:
                l=[rows[(m['key'],suite)] for m in candidates if m['method']==left];r=[rows[(m['key'],suite)] for m in candidates if m['method']==right]
                if not l or not r:continue
                targets.append((f'{capacity}/{scenario}/{left}-vs-{right}',l,r))
    pairs={name:paired_effect(left,right,protocol['paper_bootstrap_resamples'],len(targets)) for name,left,right in targets}
    sensitivity=[]
    for speed in (.1,.2,.3,.4):
        alternate={k:v*(1-speed)/(1-weights['encode_mib_s']) for k,v in weights.items() if k!='encode_mib_s'};alternate['encode_mib_s']=speed
        _,ranking=build_boards(metrics,alternate);sensitivity.append({'speed_weight':speed,'combined':ranking})
    result={'weights':weights,'scope':'Practical methods at actual vocabulary/source; same prior/dictionary controls independently reported, not universal causal algorithm rank.',
        'within_scenario':boards,'combined':combined,'primary_paired_effects':pairs,'weight_sensitivity':sensitivity,
        'bootstrap_family_size':len(targets),'deterministic_seeds_not_treated_as_replicates':True,
        'downstream_tested':False}
    (folder/'rankings.json').write_text(json.dumps(result,indent=2)+'\n')
    csv_rows=[]
    for board,values in boards.items():
        for name,m in values['metrics'].items():
            csv_rows.append({'board':board,'method':name,'compression_rank':values['metric_ranks']['document_macro_tokens_per_byte'][name],
                'comprehensive_position':values['comprehensive_order'].index(name)+1,**{k:v for k,v in m.items() if k!='keys'}})
    fields=sorted(set().union(*(set(r) for r in csv_rows)))
    with (folder/'all_main_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
    print('Main combined rankings',json.dumps({k:v['ranking'] for k,v in combined.items()},indent=2),flush=True)


if __name__=='__main__':main()
