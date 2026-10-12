"""Paired paper-cluster effects for all six prespecified core count metrics.

Seeds are averaged within each formula, never repeated as extra formulas.
Ratios of count sums and paper macro compression have separate estimands.
Confidence intervals condition on the frozen realized vocabularies.
"""
from collections import defaultdict
import gzip,json
import numpy as np
from math_tokenizer.corpus import ROOT

SPECS={
 'document_macro_tokens_per_byte':('tokens','bytes','paper_macro'),
 'independent_tree_crossing':('crossing_tokens','tokens','ratio'),
 'command_integrity':('commands_intact','commands','ratio'),
 'number_integrity':('numbers_intact','numbers','ratio'),
 'independent_scope_boundary_recall':('scope_boundaries_preserved','scope_boundaries','ratio'),
 'fit_predictive_unigram_bits_per_byte':('predictive_bits','bytes','ratio')}


def main():
    folder=ROOT/'results/v8/full';protocol=json.loads((folder/'evaluation_protocol.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text());rows=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    # Compress into source-paper count sums instead of retaining all row dicts.
    fields=set(f for numerator,denominator,_ in SPECS.values() for f in (numerator,denominator))
    for r in map(json.loads,gzip.open(folder/'per_formula.jsonl.gz','rt')):
        if r['suite'] not in ('external_test','extreme_domain_test'):continue
        totals=rows[(r['key'],r['suite'])][r['group']]
        for field in fields:totals[field]+=r.get(field) or 0.
    comparisons=[]
    for capacity in (1024,2048):
        for scenario,suite in [('broad','external_test'),('specialized','extreme_domain_test')]:
            candidates=[m for m in metrics if m['capacity']==capacity and m['scenario']==scenario and m['suite']==suite]
            for left,right in protocol['primary_pairs_explicit']:
                l=[m for m in candidates if m['method']==left];r=[m for m in candidates if m['method']==right]
                if l and r:comparisons.append((capacity,scenario,suite,left,right,l,r))
    family=len(comparisons)*len(SPECS);result={}
    for capacity,scenario,suite,left,right,l,r in comparisons:
        papers=sorted(rows[(l[0]['key'],suite)])
        assert all(sorted(rows[(m['key'],suite)])==papers for m in l+r)
        draw=np.random.default_rng(8129).integers(0,len(papers),(200000,len(papers)))
        values={}
        for metric,(num,den,estimand) in SPECS.items():
            def counts(models):
                return np.array([[np.mean([rows[(m['key'],suite)][p][f] for m in models]) for f in (num,den)] for p in papers])
            a,b=counts(l),counts(r)
            if estimand=='paper_macro':
                samples=(a[:,0]/a[:,1]-b[:,0]/b[:,1])[draw].mean(axis=1)
                point=float(np.mean(a[:,0]/a[:,1]-b[:,0]/b[:,1]))
            else:
                asum=a[draw].sum(axis=1);bsum=b[draw].sum(axis=1)
                valid=(asum[:,1]>0)&(bsum[:,1]>0)
                samples=(asum[valid,0]/asum[valid,1]-bsum[valid,0]/bsum[valid,1])
                point=float(a[:,0].sum()/a[:,1].sum()-b[:,0].sum()/b[:,1].sum())
            q=.05/(2*family)
            values[metric]={'difference_left_minus_right':point,'ci95':np.quantile(samples,[.025,.975]).tolist(),
                'familywise_ci95':np.quantile(samples,[q,1-q]).tolist(),'estimand':estimand,
                'family_size':family,'valid_bootstrap_draws':len(samples),'left_seed_count':len(l),'right_seed_count':len(r)}
        result[f'{capacity}/{scenario}/{left}-vs-{right}']={'papers':len(papers),'metrics':values}
    (folder/'multimetric_paired_effects.json').write_text(json.dumps({'resamples':200000,'family_size':family,
        'comparisons':result,'conditional_on_frozen_vocabularies':True,'training_seed_variance_not_paper_variance':True},indent=2)+'\n')
    print('Multimetric pairs',len(comparisons),'family',family,flush=True)


if __name__=='__main__':main()
