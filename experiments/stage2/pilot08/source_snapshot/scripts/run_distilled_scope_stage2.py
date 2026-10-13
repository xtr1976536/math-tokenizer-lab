"""Statistical merge hierarchy, neural/token prediction signal; actual native DP."""
import argparse,gzip,json
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
import numpy as np
from tokenizers import Tokenizer
import sentencepiece as spm
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.final_models import load_final
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.signal_stage2 import paper_token_rate,TokenCodePredictor
from math_tokenizer.learned_scope_stage2 import LearnedScopeTokenizer
from math_tokenizer.merge_scope_stage2 import MergeScopeTokenizer,extract_merges
from math_tokenizer.distilled_scope_stage2 import select_distilled
from math_tokenizer.neural_stage2 import NeuralPatchPredictor
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.independent_v7 import additional,source_scopes
from run_stage2 import balanced,write,sha,timing,provenance_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/stage2/pilot08');a=p.parse_args();out=a.output.resolve()
    if (out/'manifest.json').exists():raise ValueError('use new output directory')
    if ROOT/'experiments/stage2' not in out.parents:raise ValueError('invalid output')
    out.mkdir(parents=True,exist_ok=True);old=ROOT/'experiments/stage2/pilot06';oldmanifest=json.loads((old/'manifest.json').read_text())
    by={r['id']:r for r in read_catalog()};scopes={k:[by[i] for i in ids] for k,ids in oldmanifest['scopes'].items()}
    fit=scopes['fit'];reward=balanced(scopes['reward'],128);teacherpath=ROOT/'experiments/stage2/pilot05/models/teacher.pt'
    teacher=NeuralPatchPredictor.load(teacherpath);pairlog=json.loads((old/'learned_pairs.json').read_text())
    pairs=[(r['open'],r['close']) for r in pairlog['selected']]
    sources=['scripts/run_distilled_scope_stage2.py','src/math_tokenizer/distilled_scope_stage2.py','src/math_tokenizer/merge_scope_stage2.py','native/merge_scope_stage2.c',
             'docs/stage2_next_protocol.md','src/math_tokenizer/neural_stage2.py','scripts/run_stage2.py']
    for source in sources:
        dest=out/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/source).read_bytes())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scopes':oldmanifest['scopes'],'reward_used_ids':[r['id'] for r in reward],
        'catalog_sha256':oldmanifest['catalog_sha256'],'capacities':[1024,2048],'teacher_sha256':sha(teacherpath),
        'source_sha256':{s:sha(ROOT/s) for s in sources},'status':'adaptive internal statistical hierarchy distillation experiment',
        'prior_accounting':oldmanifest['prior_accounting'],'hierarchy':'fit-only BPE merge hierarchy distilled per-ID, not mathematical AST',
        'neural_score':'conditional patch byte code; full-input dictionary segmentation is not a causal byte LM'}
    write(out/'manifest.json',manifest);audits=[provenance_audit(scopes)];allmetrics=[];allrows=[];checks=0
    for capacity in (1024,2048):
        folder=out/str(capacity);modeldir=folder/'models';modeldir.mkdir(parents=True,exist_ok=True);prev=old/str(capacity)/'models'
        byte=HFTokenizer(Tokenizer.from_file(str(prev/'byte_bpe.json')));atom=AtomicBPE.load(prev/'math_atom_bpe.json');ast=BoundaryAST.load(prev/'boundary_ast.json')
        models={'byte_bpe':byte,'math_atom_bpe':atom,'boundary_ast':ast,
                'unigram':SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(prev/'unigram.model')))}
        for method,name in [('Qwen2.5-0.5B','qwen25'),('meta-llama/Meta-Llama-3-8B','llama3'),('tiktoken_o200k_base','o200k')]:models[name]=load_final(method)
        families={'raw':([token_bytes(byte,i) for i in range(byte.vocab_size)],[bytes([i]) for i in range(256)],extract_merges(byte.tokenizer)),
                  'minimal_prior':(atom.vocab,atom.atoms,extract_merges(atom.tokenizer))}
        for family,(vocab,atoms,merges) in families.items():
            plain=LearnedScopeTokenizer(vocab,atoms,[],0.);reference=paper_token_rate([plain.encode(r['latex']) for r in reward],reward)
            models[family+'_no_signal']=LearnedScopeTokenizer(vocab,atoms,[],0.)
            teacher_model=MergeScopeTokenizer(vocab,atoms,pairs,merges)
            selected,logs=select_distilled(teacher_model,pairs,balanced(fit,768),reward,reference)
            models[family+'_distilled_signal']=selected;write(folder/f'{family}_distilled_selection.json',logs)
        print('Learned statistical hierarchy signal models',capacity,flush=True)
        probes=balanced(fit,32)+[{'latex':s} for s in ('','α😊',r'\Unknown{a{b}}','\x00')]
        for name,m in models.items():
            if not isinstance(m,LearnedScopeTokenizer):continue
            for r in probes:
                ids,cost=m.encode_with_cost(r['latex']);ref,value=m.encode_reference(r['latex'])
                assert ids==ref and abs(cost-value)<1e-8 and m.decode(ids)==r['latex'];checks+=1
            path=modeldir/f'{name}.npz';m.save(path);loaded=LearnedScopeTokenizer.load(path)
            assert all(loaded.encode(r['latex'])==m.encode(r['latex']) for r in probes)
        write(folder/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
           'models':{str(p.relative_to(folder)):sha(p) for p in modeldir.iterdir()},'selected_before_test':True})
        for suite in ('dev','internal_eval','domain_ood'):
            rows=scopes[suite];cache={r['id']:source_scopes(r['latex']) for r in rows};times=timing(models,rows)
            for name,m in models.items():
                usage=fit_usage(m,fit);summary,rr=measure(m,rows,name,repeats=1,training_usage=usage)
                extra,sr=additional(m,rows,usage,scope_cache=cache,probe_limit=128);summary.update(extra);summary.update(times[name]);summary.update(capacity=capacity,suite=suite)
                if summary['vocab_actual']<=2048:
                    fitids=[m.encode(r['latex']) for r in fit];ids=[m.encode(r['latex']) for r in rows]
                    code=TokenCodePredictor(fitids,summary['vocab_actual']).score(ids,rows);summary.update({k:v for k,v in code.items() if k!='bits'})
                    for r,bits in zip(rr,code['bits']):r['token_bigram_code_bits']=bits
                arrays=[v for v in vars(m).values() if isinstance(v,np.ndarray)]
                if hasattr(m,'primitive'):arrays += [v for v in vars(m.primitive).values() if isinstance(v,np.ndarray)]
                summary['persistent_numeric_array_bytes_lower_bound']=sum(a.nbytes for a in arrays);summary.pop('usage')
                lookup={r['id']:r for r in sr}
                for r in rr:r.update(capacity=capacity,suite=suite,**{k:v for k,v in lookup.get(r['id'],{}).items() if k!='id'})
                allrows.extend(rr);allmetrics.append(summary)
            print('Evaluated merge signals',capacity,suite,flush=True)
    audits.append({'round':2,'passed':True,'native_reference_and_serialization_probes':checks,
                   'scope':'exact student DP, payloads, checkpoint roundtrip; source-hashed teacher signal' })
    grouped=defaultdict(list)
    for r in allrows:grouped[(r['capacity'],r['method'],r['suite'])].append(r)
    for m in allmetrics:
        rr=grouped[(m['capacity'],m['method'],m['suite'])];assert len(rr)==m['formulas'] and sum(r['tokens'] for r in rr)==m['tokens']
        assert all(r['roundtrip'] and r['payload_exact'] for r in rr);groups=defaultdict(lambda:[0,0]);tp=total=0
        for r in rr:
            groups[r['group']][0]+=r['tokens'];groups[r['group']][1]+=r['bytes'];tp+=r.get('scope_boundaries_preserved',0);total+=r.get('scope_boundaries',0)
        assert abs(np.mean([t/b for t,b in groups.values()])-m['document_macro_tokens_per_byte'])<1e-12
        if total:assert abs(tp/total-m['independent_scope_boundary_recall'])<1e-12
    for capacity in (1024,2048):
        freeze=json.loads((out/str(capacity)/'freeze.json').read_text());assert all(sha(out/str(capacity)/p)==h for p,h in freeze['models'].items())
    assert sha(teacherpath)==manifest['teacher_sha256'] and sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz')==manifest['catalog_sha256']
    audits.append({'round':3,'passed':True,'scope':'all per-formula/paper aggregates, fidelity, frozen models and teacher, immutable catalog',
                   'result_rows':len(allrows),'metric_records':len(allmetrics)})
    write(out/'metrics.json',allmetrics);write(out/'audits.json',audits)
    with gzip.open(out/'per_formula.jsonl.gz','wt') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(out/'completion.json',{'status':'exploratory completed, pending original joint success gate',
                               'metric_records':len(allmetrics),'result_rows':len(allrows)})
    print('Statistical-hierarchy pilot complete.',flush=True)


if __name__=='__main__':main()
