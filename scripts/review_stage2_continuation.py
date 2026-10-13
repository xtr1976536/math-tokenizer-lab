"""Independent archive/snapshot replay and original conjunctive success gate."""
import gzip,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.learned_scope_stage2 import LearnedScopeTokenizer
from math_tokenizer.merge_scope_stage2 import MergeScopeTokenizer
from math_tokenizer.neural_stage2 import NeuralPatchPredictor
from math_tokenizer.neural_deployed_stage2 import neural_path_code
from math_tokenizer.signal_stage2 import SignalTokenizer
from run_stage2 import sha,write,balanced


def gates(metric,ref):
    failures=[];ratio=metric['document_macro_tokens_per_byte']/ref['document_macro_tokens_per_byte']
    if ratio>1.01:failures.append('compression')
    for k in ('command_integrity','number_integrity','independent_scope_boundary_recall','lexical_boundary_f1'):
        if metric[k] is not None and ref[k] is not None and metric[k]<ref[k]-.01:failures.append(k)
    if metric['independent_tree_crossing']>ref['independent_tree_crossing']+.01:failures.append('tree_crossing')
    if metric['encode_mib_s']<.9*ref['encode_mib_s']:failures.append('speed')
    if metric['roundtrip_rate']!=1 or metric['payload_fidelity_rate']!=1:failures.append('fidelity')
    biggain=(ratio<=.95 or metric['encode_mib_s']>=1.2*ref['encode_mib_s'] or
        metric['independent_scope_boundary_recall']>=ref['independent_scope_boundary_recall']+.05 or
        metric['independent_tree_crossing']<=ref['independent_tree_crossing']-.05)
    return {'failures':failures,'substantial_gain':bool(biggain),'passed':bool(not failures and biggain),
            'compression_regression_percent':100*(ratio-1),'encode_ratio':metric['encode_mib_s']/ref['encode_mib_s']}


def main():
    folder=ROOT/'experiments/stage2';by={r['id']:r for r in read_catalog()};review=[];decisions=[];totalrows=0
    for name in ('pilot05','pilot06','pilot07','pilot08','pilot09'):
        p=folder/name;manifest=json.loads((p/'manifest.json').read_text());audits=json.loads((p/'audits.json').read_text())
        assert len(audits)>=3 and all(a['passed'] for a in audits)
        for source,digest in manifest['source_sha256'].items():assert sha(p/'source_snapshot'/source)==digest
        assert manifest['catalog_sha256']==sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')
        if name=='pilot05':
            freeze=json.loads((p/'freeze.json').read_text())
            for path,digest in freeze['model_sha256'].items():assert sha(p/path)==digest
            rr=json.loads((p/'results.json').read_text());assert len(rr)==12
            for seed in (17,29,43):assert len({r['oracle_byte_targets'] for r in rr if r['seed']==seed})==1
            review.append({'pilot':name,'passed':True,'scope':'teacher/policy/source hashes; equal neural byte-target budgets; matched reward rates'})
            continue
        metrics=json.loads((p/'metrics.json').read_text());lookup={(m['capacity'],m['method'],m['suite']):m for m in metrics}
        groups=defaultdict(list)
        with gzip.open(p/'per_formula.jsonl.gz','rt') as f:
            for line in f:r=json.loads(line);groups[(r['capacity'],r['method'],r['suite'])].append(r)
        spotchecks=0
        for key,rr in groups.items():
            m=lookup[key];assert len(rr)==m['formulas'] and {r['id'] for r in rr}==set(manifest['scopes'][m['suite']])
            assert sum(r['tokens'] for r in rr)==m['tokens'] and sum(r['bytes'] for r in rr)==m['bytes']
            assert all(r['roundtrip'] and r['payload_exact'] for r in rr)
            papers=defaultdict(lambda:[0,0]);tp=total=0
            for r in rr:
                papers[r['group']][0]+=r['tokens'];papers[r['group']][1]+=r['bytes'];tp+=r.get('scope_boundaries_preserved',0);total+=r.get('scope_boundaries',0)
            assert abs(np.mean([t/b for t,b in papers.values()])-m['document_macro_tokens_per_byte'])<1e-12
            if total:assert abs(tp/total-m['independent_scope_boundary_recall'])<1e-12
            path=p/str(m['capacity'])/'models'/f"{m['method']}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as d:merge='merges' in d.files;protected='atoms' in d.files
                model=(MergeScopeTokenizer if merge else LearnedScopeTokenizer if protected else SignalTokenizer).load(path)
                for row in rr[:3]:
                    text=by[row['id']]['latex'];ids=model.encode(text)
                    assert len(ids)==row['tokens'] and model.decode(ids)==text;spotchecks+=1
            if m['suite']!='dev' and (m['method'].startswith('raw_') or m['method'].startswith('minimal_prior')):
                decisions.append({'pilot':name,'capacity':m['capacity'],'method':m['method'],'suite':m['suite'],
                                 **gates(m,lookup[(m['capacity'],'boundary_ast',m['suite'])])})
        for cap in manifest['capacities']:
            freeze=json.loads((p/str(cap)/'freeze.json').read_text());files=freeze.get('model_sha256',freeze.get('models'))
            assert all(sha(p/str(cap)/file)==digest for file,digest in files.items())
        if name=='pilot06':
            logs=json.loads((p/'oracle_log.json').read_text())
            for cap in manifest['capacities']:
                for family in ('raw','minimal_prior'):
                    for seed in (17,29,43):
                        picked=[r for r in logs if r['capacity']==cap and r['family']==family and r['seed']==seed]
                        assert len(picked)==4 and len({r['oracle_evaluations'] for r in picked})==1 and len({r['teacher_sha256'] for r in picked})==1
        n=sum(map(len,groups.values()));totalrows+=n
        review.append({'pilot':name,'passed':True,'result_rows':n,'metric_records':len(metrics),'public_replay_spotchecks':spotchecks,
                       'scope':'all row IDs/count/paper/scope aggregates and fidelity; frozen source/model/catalog hashes'})
    # Independently score actual pilot09 paths under the shared neural critic.
    p=folder/'pilot09';manifest=json.loads((p/'manifest.json').read_text());teacher=NeuralPatchPredictor.load(folder/'pilot05/models/teacher.pt')
    deployed=[]
    for cap in (1024,2048):
        for suite in ('dev','internal_eval','domain_ood'):
            rows=balanced([by[i] for i in manifest['scopes'][suite]],128)
            for seed in (17,29,43):
                for algorithm in ('parameter_rl','random_search','cem_search','no_update'):
                    name=f'minimal_prior_{algorithm}_{seed}';model=LearnedScopeTokenizer.load(p/str(cap)/'models'/f'{name}.npz')
                    score,rate=neural_path_code(model,rows,teacher)
                    deployed.append({'capacity':cap,'suite':suite,'seed':seed,'algorithm':algorithm,
                                     'conditional_neural_byte_code':score,'paper_tokens_per_byte':rate,'formulas':len(rows)})
    comparisons=[]
    for cap in (1024,2048):
        for suite in ('dev','internal_eval','domain_ood'):
            for seed in (17,29,43):
                a=next(r for r in deployed if r['capacity']==cap and r['suite']==suite and r['seed']==seed and r['algorithm']=='parameter_rl')
                for control in ('random_search','cem_search','no_update'):
                    b=next(r for r in deployed if r['capacity']==cap and r['suite']==suite and r['seed']==seed and r['algorithm']==control)
                    comparisons.append({'capacity':cap,'suite':suite,'seed':seed,'control':control,
                        'relative_improvement':1-a['conditional_neural_byte_code']/b['conditional_neural_byte_code']})
    successful=[]
    for pilot in ('pilot06','pilot07','pilot08','pilot09'):
        methods={d['method'] for d in decisions if d['pilot']==pilot}
        for method in methods:
            ds=[d for d in decisions if d['pilot']==pilot and d['method']==method]
            if len(ds)==4 and all(d['passed'] for d in ds):successful.append({'pilot':pilot,'method':method})
    output={'status':'exploratory completed; no joint stage2 success claim','review':review,'deployment_gates':decisions,
        'successful_both_capacity_methods':successful,'independent_formula_rows_checked':totalrows,
        'deployed_neural_scores':deployed,'deployed_neural_comparisons':comparisons,
        'limitations':['adaptive exposed internal scopes','three seeds, single machine','small local neural teacher is not full LM',
                       'conditional deployed reconstruction is not causal string likelihood','minimal-prior route retains math atom alphabet',
                       'some new model training-duration fields do not include full selection/teacher work']}
    write(folder/'continuation_review.json',output)
    print('Independent continuation review complete',totalrows,'formula rows; both-capacity successes:',successful)


if __name__=='__main__':main()
