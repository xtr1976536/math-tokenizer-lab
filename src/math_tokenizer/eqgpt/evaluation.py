"""PDE structure recovery and coefficient alignment on author GRE outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .author import DICTIONARY
from .data import canonical

# Coefficients use the source data descriptions, with the first term normalized
# to +1; each equation is interpreted as sum(c_i*term_i)=0.
TRUTH={
 'KdV_equation':[('ut',1.),('u,*,ux',1.),('uxxx',.0025)],
 'Burgers_equation':[('ut',1.),('u,*,ux',1.),('uxx',-.1)],
 'Wave_equation':[('utt',1.),('uxx',-1.)],
 'Chaffee_Infante_equation':[('ut',1.),('uxx',-1.),('u',1.),('u^3',-1.)],
 'KG_equation':[('utt',1.),('uxx',-.5),('u',5.)],
 'Allen_Cahn':[('ut',1.),('uxx',-.003),('u',-1.),('u^3',1.)],
 'Convection_diffusion_equation':[('ut',1.),('ux',1.),('uxx',-.25)],
 'PDE_divide':[('ut',1.),('ux,/,x',1.),('uxx',.25)],
 'Eq_6_2_12':[('ut',1.),('uxt',.1),('ux',.1)],
 'PDE_compound':[('ut',1.),('(uux)x',-.2)],
 'Possion_x_y':[('uxx',1.),('uyy',1.)],
 'Laplacian_smile':[('uxx',1.),('uyy',1.)],
 'Laplacian_EITech':[('uxx',1.),('uyy',1.),('1',1.)],
 'Burgers_2D':[('ut',1.),('u,*,ux',1.),('u,*,uy',1.),('uxx',-.01),('uyy',-.01)],
 'Laplacian_H':[('utt',1.),('uxx',1.),('uyy',1.)],
 'Laplacian_shuttle':[('uxx',1.),('uyy',1.),('uzz',1.)]
}


def term_keys(equation):return [canonical(s.split(','))[0] for s,c in TRUTH[equation]]

def expanded_terms(key):
    # Expand only the author's named Laplacians, with exact known definition.
    if key==(('Laplace(u)',1),):return [(('uxx',1),),(('uyy',1),)]
    if key==(('Laplace(utt)',1),):return [(('uxxtt',1),),(('uyytt',1),)]
    return [key]


def classify(sequence,equation):
    words=[DICTIONARY['id2word'][int(i)] for i in sequence]
    terms=canonical(words)
    expanded=[t for key in terms for t in expanded_terms(key)]
    target=term_keys(equation)
    if equation=='Laplacian_shuttle':
        expanded=[t for key in terms for t in
                  ([(('uxx',1),),(('uyy',1),),(('uzz',1),)] if key==(('Laplace(u)',1),) else [key])]
    source=set(expanded);truth=set(target)
    status='correct' if source==truth and len(expanded)==len(source) else 'redundant' if truth<=source else 'incorrect'
    return dict(status=status,predicted_terms=[repr(x) for x in expanded],true_terms=[repr(x) for x in target])


def summarize_run(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());equation=manifest['equation']
    result={'equation':equation,'seed':manifest['seed'],'smoke':manifest['smoke'],
            'formal_results':manifest['formal_results'],'models':{}}
    for name in manifest['results']:
        search=json.loads((folder/name/'search.json').read_text());last=search['history'][-1]
        classification=classify(last['gre'][0],equation)
        pinn=json.loads((folder/name/'pinn.json').read_text())
        matches=[r for r in pinn['coefficients'] if r.get('gre')==list(map(int,last['gre'][0][1:]))]
        error=coefficient_error(last['gre'][0],matches[-1]['values'],equation) if matches and pinn.get('status')=='completed' else None
        top=[classify(s,equation)['status'] for s in last['gre']]
        result['models'][name]=dict(top1=classification['status'],top10_correct=any(s=='correct' for s in top),
             reward=last['rewards'][0],equation=last['equations'][0],pinn_status=pinn.get('status'),coefficient_relative_error=error,cycle_best_reward=[r['rewards'][0] for r in search['history']],
             **{k:v for k,v in classification.items() if k!='status'})
    return result


def coefficient_error(sequence,values,equation):
    """Align SVD/PINN coefficients to the source PDE up to a common scale."""
    words=[DICTIONARY['id2word'][int(i)] for i in sequence if int(i) not in (0,1,5)]
    groups=[];part=[]
    for word in words+['+']:
        if word=='+':groups.append(canonical(part)[0]);part=[]
        else:part.append(word)
    if len(values)+1!=len(groups):return None
    predicted={}
    for key,c in zip(groups,[1.]+[-float(v) for v in values]):
        keys=expanded_terms(key)
        if equation=='Laplacian_shuttle' and key==(('Laplace(u)',1),):keys=[(('uxx',1),),(('uyy',1),),(('uzz',1),)]
        for k in keys:predicted[k]=predicted.get(k,0.)+c
    target={canonical(s.split(','))[0]:c for s,c in TRUTH[equation]}
    if set(predicted)!=set(target):return None
    anchor=next(iter(target))
    if not np.isfinite(list(predicted.values())).all() or abs(predicted[anchor])<1e-12:return None
    return float(np.mean([abs((predicted[k]/predicted[anchor])/(c/target[anchor])-1) for k,c in target.items()]))
