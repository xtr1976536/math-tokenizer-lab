"""Long-context frozen/joint causal router experiment with crossed critics."""
import argparse,json,copy
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import numpy as np
import torch
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.signal_stage2 import CONTEXTS
from math_tokenizer.neural_stage2 import NeuralPatchPredictor,NeuralConfig
from math_tokenizer.autonomous_stage2 import PolicyConfig
from math_tokenizer.joint_boundary_stage2 import ContextActor,optimize,evaluate
from run_stage2 import balanced,write,sha,provenance_audit,paired_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/stage2/pilot10');a=p.parse_args();out=a.output.resolve()
    if (out/'manifest.json').exists():raise ValueError('new output required')
    if ROOT/'experiments/stage2' not in out.parents:raise ValueError('invalid output')
    out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    ids=json.loads((ROOT/'experiments/stage2/pilot04/run_manifest.json').read_text())['scopes'];by={r['id']:r for r in read_catalog()}
    scopes={k:[by[i] for i in v] for k,v in ids.items()};fit=balanced(scopes['fit'],768);reward=balanced(scopes['reward'],48)
    files=['docs/stage2_joint_protocol.md','src/math_tokenizer/joint_boundary_stage2.py','src/math_tokenizer/neural_stage2.py',
           'scripts/run_joint_boundary_stage2.py','scripts/run_stage2.py']
    for file in files:
        target=out/'source_snapshot'/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/file).read_bytes())
    write(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'scopes':ids,'fit_ids':[r['id'] for r in fit],
          'reward_ids':[r['id'] for r in reward],'source_sha256':{f:sha(ROOT/f) for f in files},'torch':torch.__version__,
          'catalog_sha256':sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),'status':'adaptive exploratory, no production promotion',
          'supervision':{'manual_boundaries':0,'math_lexicon':0,'AST_labels':0,'math_metric_reward':0},
          'architecture':'16 preceding-byte additive actor;16-prefix+8-previous neural critic;not recurrent/world model'})
    audits=[provenance_audit(scopes)];write(out/'audits.json',audits)
    teacher=NeuralPatchPredictor.fit(fit,np.zeros(CONTEXTS),373,NeuralConfig(prefix=16,previous=8),PolicyConfig())
    teacher.save(out/'models/initial_teacher.pt');trained=[]
    for joint in (False,True):
        for seed in (17,29,43):
            for algorithm in ('boundary_rl','random_search','cem_search','no_update'):
                actor,critic,metadata=optimize(fit,reward,teacher,seed,algorithm,joint)
                name=('joint' if joint else 'frozen')+'_'+algorithm+'_'+str(seed)
                actor.save(out/'models'/f'{name}_actor.npz');critic.save(out/'models'/f'{name}_critic.pt')
                write(out/f'{name}_training.json',metadata);trained.append((name,actor,critic,metadata))
                print('Trained',name,round(metadata['training_seconds'],2),'s',flush=True)
    freeze={'utc':datetime.now(timezone.utc).isoformat(),'models':{str(p.relative_to(out)):sha(p) for p in (out/'models').iterdir()}}
    write(out/'freeze.json',freeze)
    for joint in (False,True):
        for seed in (17,29,43):
            records=[r for n,a,t,r in trained if r['joint']==joint and r['seed']==seed]
            assert len({r['oracle_byte_targets'] for r in records})==1 and len({r['teacher_byte_targets'] for r in records})==1
    audits.append({'round':2,'passed':True,'scope':'causal actor and expected-gradient tests; same oracle and predictor target budgets within each condition; reward-only rate calibration'})
    results=[]
    for name,actor,critic,meta in trained:
        common=next(t for n,a,t,r in trained if r['joint']==meta['joint'] and r['algorithm']=='no_update' and r['seed']==meta['seed'])
        scores={}
        for suite in ('dev','internal_eval','domain_ood'):
            probe=balanced(scopes[suite],128)
            scores[suite]={k:evaluate(probe,actor,t,99173+meta['seed']) for k,t in [('own',critic),('initial',teacher),('common_no_update',common)]}
        scores['dev']['deterministic_own']=evaluate(balanced(scopes['dev'],128),actor,critic,99173+meta['seed'],deterministic=True)
        results.append({'name':name,**meta,'scores':scores})
    comparisons=[]
    for joint in (False,True):
        for seed in (17,29,43):
            rl=next(r for r in results if r['joint']==joint and r['seed']==seed and r['algorithm']=='boundary_rl')
            for control in ('random_search','cem_search','no_update'):
                other=next(r for r in results if r['joint']==joint and r['seed']==seed and r['algorithm']==control)
                for suite in ('dev','internal_eval','domain_ood'):
                    for critic in ('own','initial','common_no_update'):
                        x=rl['scores'][suite][critic];y=other['scores'][suite][critic]
                        comparisons.append({'joint':joint,'seed':seed,'control':control,'suite':suite,'critic':critic,
                            'relative_loss_improvement':1-x['paper_macro_causal_byte_bpb']/y['paper_macro_causal_byte_bpb'],
                            'rl_rate':x['boundary_rate'],'control_rate':y['boundary_rate'],**paired_interval(x['rows'],y['rows'])})
    for result in results:
        assert abs(result['calibrated_reward_rate']-.4)<1e-5
        for scope in result['scores'].values():
            for score in scope.values():
                groups=defaultdict(lambda:[0.,0])
                for row in score['rows']:groups[row['group']][0]+=row['bits'];groups[row['group']][1]+=row['bytes']
                assert abs(np.mean([b/n for b,n in groups.values()])-score['paper_macro_causal_byte_bpb'])<1e-12
    assert all(sha(out/p)==h for p,h in freeze['models'].items())
    audits.append({'round':3,'passed':True,'scope':'full own/cross-critic paper aggregation, frozen teacher/actor hashes, expected rate',
                   'policy_formula_score_rows':sum(len(s['rows']) for r in results for scope in r['scores'].values() for s in scope.values())})
    write(out/'results.json',results);write(out/'comparisons.json',comparisons);write(out/'audits.json',audits)
    write(out/'completion.json',{'status':'exploratory causal policies completed; not deployable fixed-vocabulary tokenizer',
                               'systems':len(trained),'comparisons':len(comparisons)})
    print('Long-context joint pilot complete.',flush=True)


if __name__=='__main__':main()
