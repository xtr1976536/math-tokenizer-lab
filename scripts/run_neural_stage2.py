"""Neural frozen-oracle autonomous boundary experiment, paper-disjoint scopes."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.signal_stage2 import ByteSignals,CONTEXTS
from math_tokenizer.autonomous_stage2 import PolicyConfig,evaluate_policy
from math_tokenizer.neural_stage2 import NeuralConfig,NeuralPatchPredictor,optimize_neural_policy
from run_stage2 import balanced,write,sha,paired_interval,provenance_audit
from calibrate_stage2 import expected_rate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/stage2/pilot05')
    p.add_argument('--rounds',type=int,default=32);p.add_argument('--teacher-steps',type=int,default=600)
    a=p.parse_args();out=a.output.resolve()
    if (out/'manifest.json').exists():raise ValueError('use a new output directory')
    if ROOT/'experiments/stage2' not in out.parents:raise ValueError('invalid output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    scopes=json.loads((ROOT/'experiments/stage2/pilot04/run_manifest.json').read_text())['scopes']
    by={r['id']:r for r in read_catalog()};rows={k:[by[i] for i in ids] for k,ids in scopes.items()}
    audit=[provenance_audit(rows)];fit=balanced(rows['fit'],768);reward=balanced(rows['reward'],48)
    policy=PolicyConfig(rounds=a.rounds);neural=NeuralConfig(steps=a.teacher_steps)
    sourcefiles=['src/math_tokenizer/neural_stage2.py','src/math_tokenizer/autonomous_stage2.py',
        'scripts/run_neural_stage2.py','scripts/calibrate_stage2.py','scripts/run_stage2.py','docs/stage2_next_protocol.md']
    for file in sourcefiles:
        dest=out/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/file).read_bytes())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scopes':scopes,'torch':torch.__version__,
        'config':asdict(neural),'policy_config':asdict(policy),'source_sha256':{s:sha(ROOT/s) for s in sourcefiles},
        'catalog_sha256':sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),'teacher_fit_ids':[r['id'] for r in fit],
        'reward_ids':[r['id'] for r in reward],'causality':'boundary sees previous bytes; patch features and target head see only <i',
        'human_intervention':{'boundary_labels':0,'command_inventory':0,'AST_labels':0,'math_metric_reward':0,
             'priors':['architecture','byte windows','smoothing-free neural CE','rate budget','optimizer settings','max patch']},
        'status':'adaptive internal neural pilot, not upstream architecture reproduction'}
    write(out/'manifest.json',manifest);write(out/'audits.json',audit)
    initial=np.full(CONTEXTS,np.log(.4/.6));teacher=NeuralPatchPredictor.fit(fit,initial,373,neural,policy)
    teacher.save(out/'models/teacher.pt');frozen=sha(out/'models/teacher.pt')
    print('Neural teacher trained',round(teacher.train_seconds,2),'s',teacher.memory_summary(),flush=True)
    signals=ByteSignals.fit(rows['fit']);results=[]
    for seed in (17,29,43):
        for algorithm in ('boundary_rl','random_search','cem_search','no_update'):
            result=optimize_neural_policy(fit,reward,signals,.4,seed,algorithm,teacher,policy)
            name=f'{algorithm}_{seed}';np.savez_compressed(out/'models'/f'{name}.npz',logits=result['logits'],initial=result['initial_logits'])
            metadata={k:v for k,v in result.items() if k not in ('logits','initial_logits')};write(out/f'{name}_training.json',metadata)
            results.append((result,metadata));print(name,'trained',round(result['training_seconds'],2),'s',flush=True)
    # Freeze all trained policies before any dev/eval prediction is computed.
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':{str(p.relative_to(out)):sha(p) for p in (out/'models').iterdir()}})
    # Check causal prefix and serialized predictor on a sample; unit tests check further cases.
    loaded=NeuralPatchPredictor.load(out/'models/teacher.pt');cuts=np.array([1,0,1,0,1],bool)
    assert np.array_equal(teacher.score(b'abcd',cuts)[0],loaded.score(b'abcd',cuts)[0])
    for seed in (17,29,43):
        rr=[r for r,m in results if r['seed']==seed];assert len({r['oracle_byte_targets'] for r in rr})==1
    audit.append({'round':2,'passed':True,'scope':'causal-feature tests; frozen teacher save/load; identical teacher and byte-target/trajectory budgets',
        'teacher_parameters':teacher.memory_summary(),'teacher_sha256':frozen})
    write(out/'audits.json',audit)
    evaluated=[]
    for result,metadata in results:
        logits=result['logits'];lo=-8.;hi=8.
        for _ in range(24):
            mid=(lo+hi)/2
            if expected_rate(reward,logits+mid,policy.max_patch)>.4:hi=mid
            else:lo=mid
        offset=(lo+hi)/2;calibrated=logits+offset
        scores={}
        for suite in ('dev','internal_eval','domain_ood'):
            probes=balanced(rows[suite],128)
            scores[suite]=evaluate_policy(probes,calibrated,teacher,99173+metadata['seed'],draws=4)
            scores[suite]['interpretation']='frozen causal neural patch predictor; matched training expected rate; not deployed tokenizer or full LM'
        evaluated.append({**metadata,'rate_offset':offset,'calibrated_expected_reward_rate':expected_rate(reward,calibrated,policy.max_patch),
                          'scores':scores})
    write(out/'results.json',evaluated);comparisons=[]
    for seed in (17,29,43):
        for suite in ('dev','internal_eval','domain_ood'):
            rl=next(r['scores'][suite] for r in evaluated if r['seed']==seed and r['algorithm']=='boundary_rl')
            for control in ('random_search','cem_search','no_update'):
                other=next(r['scores'][suite] for r in evaluated if r['seed']==seed and r['algorithm']==control)
                comparisons.append({'seed':seed,'suite':suite,'control':control,
                    'relative_loss_improvement':1-rl['paper_macro_causal_byte_prediction_bpb']/other['paper_macro_causal_byte_prediction_bpb'],
                    'rl_eval_rate':rl['boundary_rate'],'control_eval_rate':other['boundary_rate'],
                    **paired_interval(rl['rows'],other['rows'])})
    write(out/'comparisons.json',comparisons)
    for r in evaluated:
        assert abs(r['calibrated_expected_reward_rate']-.4)<1e-5
        for suite,score in r['scores'].items():
            groups={}
            for rr in score['rows']:
                b,n=groups.get(rr['group'],(0.,0));groups[rr['group']]=(b+rr['bits'],n+rr['bytes'])
            assert abs(np.mean([b/n for b,n in groups.values()])-score['paper_macro_causal_byte_prediction_bpb'])<1e-12
    assert sha(out/'models/teacher.pt')==frozen
    assert sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')==manifest['catalog_sha256']
    audit.append({'round':3,'passed':True,'scope':'paper aggregates, calibrated reward rate, immutable teacher and catalog',
                  'evaluated_formula_policy_rows':sum(len(s['rows']) for r in evaluated for s in r['scores'].values())})
    write(out/'audits.json',audit);write(out/'completion.json',{'status':'completed exploratory neural router pilot; not production success',
        'policies':len(results),'comparisons':len(comparisons),'audits':len(audit)})
    print('Neural pilot complete.',flush=True)


if __name__=='__main__':main()
