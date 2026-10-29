"""Audit all 1500 PDEs against unchanged GRE, then exclude eight targets."""
import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
import signal
from pathlib import Path
import numpy as np
import sympy as sp
import torch
from torch import nn
from math_tokenizer.corpus import ROOT
from math_tokenizer.pde_protocol import digest
from math_tokenizer.eqgpt.corpus_conversion import (convert,ConversionError,DICTIONARY,atoms,gre_expression,support_key,U,COORDS)
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.author import definitions,namespace

TARGETS={
 'Wave_equation':['utt','+','uxx'],
 'Convection_diffusion_equation':['ut','+','ux','+','uxx'],
 'KG_equation':['utt','+','uxx','+','u'],
 'Burgers_equation':['ut','+','u','*','ux','+','uxx'],
 'Eq_6_2_12':['ut','+','ux','+','uxt'],
 'PDE_divide':['ut','+','ux','/','x','+','uxx'],
 'KdV_equation':['ut','+','u','*','ux','+','uxxx'],
 'Chaffee_Infante_equation':['ut','+','uxx','+','u','+','u^3'],
}


def words_to_ids(words):
    return [5]+[DICTIONARY['word2id'][w] for w in words]+[1]


def invariant_support(expression):
    keys=[]
    for perm in itertools.permutations(COORDS[:3]):
        changed=expression.xreplace(dict(zip(COORDS[:3],perm)))
        # Coordinate permutation changes the formal function's arguments too.
        changed=changed.xreplace({sp.Function('u')(*perm,COORDS[3]):U})
        keys.append(support_key(changed))
    return min(keys)


def check_author_atoms(atom_ids):
    class SmoothField(nn.Module):
        def forward(self,coords):
            x,y,z,t=coords.split(1,dim=1)
            return torch.exp(torch.sin(x*y*z*t)+x+y+z+t)
    env=definitions('calculate_terms.py',namespace(),names=['calculate_terms'])
    symbolic_field=sp.exp(sp.sin(sp.prod(COORDS))+sum(COORDS))
    coords_np=np.random.default_rng(20261028).uniform(.2,.8,(9,4))
    results={}
    for i in sorted(atom_ids):
        word=DICTIONARY['id2word'][i]
        try:
            coords=torch.tensor(coords_np,dtype=torch.float64,requires_grad=True)
            actual=np.asarray(env['calculate_terms'](word,SmoothField(),coords,list('xyzt')),dtype=float).reshape(-1)
            expr=atoms()[i].subs(U,symbolic_field).doit()
            expected=np.broadcast_to(np.asarray(sp.lambdify(COORDS,expr,'numpy')(*coords_np.T),dtype=float),(len(coords_np),))
            valid=bool(np.isfinite(actual).all() and np.allclose(actual,expected,rtol=1e-7,atol=1e-8))
            results[i]=dict(word=word,status='PASS' if valid else 'semantic_mismatch',max_absolute_error=float(np.max(abs(actual-expected))))
        except Exception as exc:
            results[i]=dict(word=word,status='author_evaluator_error',error=type(exc).__name__+': '+str(exc))
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/formulas/downstream_preparation');args=ap.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/formulas/pde/pde_equations.jsonl.gz'
    rows=list(map(json.loads,gzip.open(source,'rt')))
    records=[]
    def timed_out(*_):raise ConversionError('conversion_timeout','Conversion exceeded 5 seconds')
    previous=signal.signal(signal.SIGALRM,timed_out)
    for number,row in enumerate(rows,1):
        record={k:row[k] for k in ('id','latex','source_group','source_url','title')}
        try:
            signal.alarm(5)
            record.update(convert(row['latex']),status='converted')
        except ConversionError as exc:
            record.update(status='rejected',reason=exc.code,detail=str(exc))
        except (ValueError,TypeError,KeyError,RecursionError,NotImplementedError) as exc:
            record.update(status='rejected',reason='parser_or_symbolic_failure',detail=type(exc).__name__+': '+str(exc))
        finally:signal.alarm(0)
        records.append(record)
        if number%100==0:print('CONVERTED',number,'of',len(rows),'accepted',sum(r['status']=='converted' for r in records),flush=True)
    signal.signal(signal.SIGALRM,previous)
    used={i for r in records if r['status']=='converted' for i in r['gre_ids'] if i>=6}
    atom_audit=check_author_atoms(used)
    blocked_atoms={i for i,r in atom_audit.items() if r['status']!='PASS'}
    for r in records:
        if r['status']=='converted' and set(r['gre_ids']) & blocked_atoms:
            r.update(status='rejected',reason='author_atom_semantics_or_runtime',detail=str(sorted(set(r['gre_ids'])&blocked_atoms)))
    targets={case:invariant_support(gre_expression(words_to_ids(words))) for case,words in TARGETS.items()}
    common=[];excluded=[]
    for r in records:
        if r['status']!='converted':continue
        key=invariant_support(gre_expression(r['gre_ids']))
        hits=[name for name,value in targets.items() if value==key]
        r['matched_targets']=hits
        if hits:r['status']='target_excluded';excluded.append(r)
        else:r['status']='common_eligible';common.append(r)
    for r in common:
        assert invariant_support(gre_expression(r['gre_ids'])) not in targets.values()
        assert r['support']==r['reconstructed_support']
    # A tokenization check over every eligible source and normalized formula.
    from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
    tokenizer=CompactAlphabet.load(ROOT/'experiments/upstream_811/latex_ascii/512.model.json')
    for r in common:
        for text in (r['latex'],r['normalized_latex']):
            assert tokenizer.decode(tokenizer.encode(text))==text
    for filename,items in [('conversion_records.jsonl',records),('common_eligible.jsonl',common),('target_excluded.jsonl',excluded)]:
        with (out/filename).open('w') as f:
            for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'conversion_summary.csv').open('w',newline='') as f:
        fields=['id','status','reason','detail','source_group','latex','normalized_latex','gre_words','matched_targets']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    summary=dict(input_formulas=len(rows),input_sha256=digest(source.read_bytes()),
                 status_counts=dict(Counter(r['status'] for r in records)),
                 rejection_reasons=dict(Counter(r['reason'] for r in records if r['status']=='rejected')),
                 converted_before_target_exclusion=len(common)+len(excluded),common_eligible=len(common),
                 target_excluded=len(excluded),excluded_by_case=dict(Counter(c for r in excluded for c in r['matched_targets'])),
                 dictionary_sha256=digest((ROOT/'eqgpt_original/code/dict_datas_0725.json').read_bytes()),
                 dictionary_size=57,author_atom_audit=atom_audit,
                 target_policy='exact coefficient-free expanded support, scalar u/v/w rename and all permutations of x/y/z; time fixed',
                 limitations=['Conservative supported scalar subset, not maximum theoretical representability',
                     'Systems/multiline and unresolved spatial Laplacian dimensions require component/context review',
                     'Unknown scalar coefficient symbols assumed constant and nondegenerate; functional coefficients rejected',
                     'No claim of exhaustive equivalence under arbitrary coordinate transforms, integration or nonlocal identities',
                     'Numerical atom checks use a smooth analytic field at nonsingular points; case-specific derivative quality remains a downstream task'],
                 downstream_training_started=False,upstream_corpus_filtered=False,
                 checks=['all 1500 have explicit outcome','symbolic coefficient-free reconstruction','existing GRE/LaTeX codec roundtrip',
                         'author numerical atom values versus independent symbolic derivatives','Hard exact source and normalized roundtrip',
                         'eight target expanded supports absent from common corpus'])
    (out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('author_atom_audit','checks','limitations')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
