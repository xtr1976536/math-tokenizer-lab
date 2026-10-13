"""Verified polynomial environment, operation critic, deployed cost controls."""
import argparse,gzip,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.operation_data_stage2 import generate,coefficients,derivative,shift,OPERATIONS
from math_tokenizer.operation_critic_stage2 import OperationCritic,score_model,model_cuts
from math_tokenizer.operation_optimizer_stage2 import optimize
from math_tokenizer.learned_scope_stage2 import LearnedScopeTokenizer
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from math_tokenizer.final_models import load_final
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.independent_v7 import additional,source_scopes
from tokenizers import Tokenizer
import sentencepiece as spm
from run_stage2 import balanced,write,sha,timing,provenance_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/stage2/pilot11');a=p.parse_args();out=a.output.resolve()
    if (out/'manifest.json').exists():raise ValueError('new output directory required')
    if ROOT/'experiments/stage2' not in out.parents:raise ValueError('invalid output')
    out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    ids=json.loads((ROOT/'experiments/stage2/pilot04/run_manifest.json').read_text())['scopes'];by={r['id']:r for r in read_catalog()};scopes={k:[by[i] for i in v] for k,v in ids.items()}
    rows,datamanifest=generate();groups={k:{r['group'] for r in rows if r['scope']==k} for k in ('fit','dev','extreme')}
    assert not groups['fit']&groups['dev'] and not groups['fit']&groups['extreme']
    # Auxiliary sources split further by canonical source, not operation pairs.
    fitgroups=sorted(groups['fit']);rewardgroups=set(fitgroups[-16:]);teacherrows=[r for r in rows if r['scope']=='fit' and r['group'] not in rewardgroups]
    reward=[r for r in rows if r['scope']=='fit' and r['group'] in rewardgroups]
    for row in reward:row['scope']='reward'
    files=['docs/stage2_joint_protocol.md','scripts/run_operation_stage2.py','src/math_tokenizer/operation_data_stage2.py',
        'src/math_tokenizer/operation_critic_stage2.py','src/math_tokenizer/operation_optimizer_stage2.py','scripts/run_stage2.py']
    for file in files:
        target=out/'source_snapshot'/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/file).read_bytes())
    write(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'scopes':ids,
         'source_sha256':{f:sha(ROOT/f) for f in files},'catalog_sha256':sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),
         'auxiliary':datamanifest,'teacher_pair_ids':[r['id'] for r in teacherrows],'reward_pair_ids':[r['id'] for r in reward],
         'source_group_disjoint':True,'status':'narrow synthetic operation environment, not world-model or stage2 success',
         'natural_training':'base vocabulary still trained on natural fit4096; optimization uses auxiliary reward pairs only'})
    with gzip.open(out/'operation_pairs.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    audits=[provenance_audit(scopes),{'round':1,'passed':True,'scope':'all synthetic pairs independently coefficient verified; source-disjoint teacher/reward/dev/extreme; benchmark separate',
                'verified_pairs':len(rows),'teacher_pairs':len(teacherrows),'reward_pairs':len(reward)}]
    # Give true and same-length shuffled labels exactly the same byte target budget.
    teacher=OperationCritic.fit(teacherrows);shuffled=OperationCritic.fit(teacherrows,shuffled=True)
    assert teacher.training_byte_targets==shuffled.training_byte_targets
    teacher.save(out/'models/operation_teacher.pt');shuffled.save(out/'models/shuffled_teacher.pt')
    write(out/'critic_training.json',{'true':teacher.fit_log,'shuffled':shuffled.fit_log,'teacher_byte_targets':teacher.training_byte_targets,
       'shuffled_byte_targets':shuffled.training_byte_targets,'parameters':sum(p.numel() for p in teacher.parameters()),
       'shuffle':'permute targets only among exactly equal UTF8 lengths; same training byte budget'})
    print('Operation critics trained',teacher.train_seconds,shuffled.train_seconds,'seconds',flush=True)
    metrics=[];allrows=[];operation_scores=[];checks=0
    for capacity in (1024,2048):
        folder=out/str(capacity);modeldir=folder/'models';modeldir.mkdir(parents=True,exist_ok=True);prev=ROOT/f'experiments/stage2/pilot06/{capacity}/models'
        models={'byte_bpe':HFTokenizer(Tokenizer.from_file(str(prev/'byte_bpe.json'))),
                'math_atom_bpe':AtomicBPE.load(prev/'math_atom_bpe.json'),'boundary_ast':BoundaryAST.load(prev/'boundary_ast.json'),
                'unigram':SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(prev/'unigram.model')))}
        for method,name in [('Qwen2.5-0.5B','qwen25'),('meta-llama/Meta-Llama-3-8B','llama3'),('tiktoken_o200k_base','o200k')]:models[name]=load_final(method)
        base=LearnedScopeTokenizer.load(ROOT/f'experiments/stage2/pilot08/{capacity}/models/minimal_prior_distilled_signal.npz');models['minimal_prior_base']=base
        for seed in (17,29,43):
            for algorithm in ('parameter_rl','random_search','cem_search','no_update'):
                model=optimize(base,reward,teacher,seed,algorithm);name=f'minimal_prior_{algorithm}_{seed}'
                models[name]=model;model.save(modeldir/f'{name}.npz');write(folder/f'{algorithm}_{seed}_training.json',model.training_log)
                print('Operation-guided model trained',capacity,algorithm,seed,flush=True)
        base.save(modeldir/'minimal_prior_base.npz')
        probes=balanced(scopes['fit'],32)
        for name,m in models.items():
            if not isinstance(m,LearnedScopeTokenizer):continue
            for r in probes:
                ids,cost=m.encode_with_cost(r['latex']);ref,value=m.encode_reference(r['latex'])
                assert ids==ref and abs(cost-value)<1e-8 and m.decode(ids)==r['latex'];checks+=1
        write(folder/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'models':{str(p.relative_to(folder)):sha(p) for p in modeldir.iterdir()}})
        for scope in ('dev','extreme'):
            data=[r for r in rows if r['scope']==scope]
            for name,m in models.items():
                if name in ('qwen25','llama3','o200k'):continue
                operation_scores.append({'capacity':capacity,'method':name,'scope':scope,'critic':'true',**score_model(m,data,teacher)})
            operation_scores.append({'capacity':capacity,'method':'minimal_prior_base','scope':scope,'critic':'shuffled',**score_model(base,data,shuffled)})
        for suite in ('dev','internal_eval','domain_ood'):
            data=scopes[suite];cache={r['id']:source_scopes(r['latex']) for r in data};times=timing(models,data)
            for name,m in models.items():
                usage=fit_usage(m,scopes['fit']);summary,rr=measure(m,data,name,repeats=1,training_usage=usage)
                extra,sr=additional(m,data,usage,scope_cache=cache,probe_limit=128);summary.update(extra);summary.update(times[name]);summary.update(capacity=capacity,suite=suite);summary.pop('usage')
                scopeby={r['id']:r for r in sr}
                for r in rr:r.update(capacity=capacity,suite=suite,**{k:v for k,v in scopeby.get(r['id'],{}).items() if k!='id'})
                metrics.append(summary);allrows.extend(rr)
            print('Natural evaluation completed',capacity,suite,flush=True)
    audits.append({'round':2,'passed':True,'scope':'teacher/controls exact byte budget; generic native/reference checks; EOS and serialization unit tests',
                   'native_probes':checks,'operation_teacher_byte_targets':teacher.training_byte_targets})
    grouped=defaultdict(list)
    for r in allrows:grouped[(r['capacity'],r['method'],r['suite'])].append(r)
    for m in metrics:
        rr=grouped[(m['capacity'],m['method'],m['suite'])];assert len(rr)==m['formulas'] and sum(r['tokens'] for r in rr)==m['tokens']
        assert all(r['roundtrip'] and r['payload_exact'] for r in rr);groups=defaultdict(lambda:[0,0])
        for r in rr:groups[r['group']][0]+=r['tokens'];groups[r['group']][1]+=r['bytes']
        assert abs(np.mean([t/b for t,b in groups.values()])-m['document_macro_tokens_per_byte'])<1e-12
    for score in operation_scores:
        groups=defaultdict(lambda:[0.,0])
        for r in score['rows']:groups[r['group']][0]+=r['bits'];groups[r['group']][1]+=r['bytes']
        assert abs(np.mean([b/n for b,n in groups.values()])-score['group_macro_target_byte_bpb'])<1e-12
    assert sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')==json.loads((out/'manifest.json').read_text())['catalog_sha256']
    audits.append({'round':3,'passed':True,'scope':'natural row/paper/fidelity and operation source-macro aggregates; immutable natural benchmark',
                   'natural_result_rows':len(allrows),'operation_result_rows':sum(len(s['rows']) for s in operation_scores)})
    write(out/'metrics.json',metrics);write(out/'operation_scores.json',operation_scores);write(out/'audits.json',audits)
    with gzip.open(out/'per_formula.jsonl.gz','wt') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(out/'completion.json',{'status':'exploratory verified-operation feedback experiment completed; gate review required',
         'metrics':len(metrics),'natural_result_rows':len(allrows),'operation_score_records':len(operation_scores)})
    print('Operation-guided pilot complete.',flush=True)


if __name__=='__main__':main()
