"""Narrow, automatically verified polynomial-operation environment.

Synthetic auxiliary pairs, never merged into the scientific formula benchmark.
Coefficient arithmetic independently verifies SymPy-generated operation labels.
"""
from collections import defaultdict
import hashlib,math
import numpy as np
import sympy as sp

X,Y=sp.symbols('x y');OPERATIONS=('differentiate_x','differentiate_y','expand','shift_x_by_one')


def add(a,b):
    out=defaultdict(int,a)
    for k,v in b.items():out[k]+=v
    return {k:v for k,v in out.items() if v}
def multiply(a,b):
    out=defaultdict(int)
    for (i,j),v in a.items():
        for (k,l),w in b.items():out[i+k,j+l]+=v*w
    return {k:v for k,v in out.items() if v}
def power(a,n):
    result={(0,0):1}
    for _ in range(n):result=multiply(result,a)
    return result
def derivative(a,axis):
    result={}
    for (i,j),v in a.items():
        exponent=i if axis==0 else j
        if exponent:result[(i-1,j) if axis==0 else (i,j-1)]=v*exponent
    return result
def shift(a):
    out=defaultdict(int)
    for (i,j),v in a.items():
        for k in range(i+1):out[k,j]+=v*math.comb(i,k)
    return {k:v for k,v in out.items() if v}
def coefficients(expr):return {tuple(map(int,k)):int(v) for k,v in sp.Poly(expr,X,Y).terms() if v}
def digest(coeff):return hashlib.sha256(repr(sorted(coeff.items())).encode()).hexdigest()


def source(rng,extreme=False):
    count=3 if extreme else int(rng.integers(1,3));expr=sp.Integer(1);coeff={(0,0):1};degrees=[]
    for _ in range(count):
        a,b,c=map(int,rng.integers(-3,4,3))
        if a==0:a=1
        if b==0:b=-1
        degree=int(rng.integers(2,4)) if extreme else int(rng.integers(1,3));degrees.append(degree)
        term=a*X+b*Y+c;expr*=term**degree
        coeff=multiply(coeff,power({k:v for k,v in [((1,0),a),((0,1),b),((0,0),c)] if v},degree))
    return expr,coeff,degrees


def generate(fit_sources=192,dev_sources=48,extreme_sources=48,seed=20261013):
    rng=np.random.default_rng(seed);seen=set();groups={'fit':[],'dev':[],'extreme':[]};attempts=0
    for scope,limit in [('fit',fit_sources),('dev',dev_sources),('extreme',extreme_sources)]:
        while len(groups[scope])<limit:
            attempts+=1;expr,coeff,degrees=source(rng,scope=='extreme');key=digest(coeff)
            if key in seen:continue
            seen.add(key);assert coefficients(expr)==coeff
            groups[scope].append((key,expr,coeff,degrees))
    rows=[]
    for scope,records in groups.items():
        for group,expr,coeff,degrees in records:
            for op,operation in enumerate(OPERATIONS):
                if op==0:target=sp.diff(expr,X);expected=derivative(coeff,0)
                elif op==1:target=sp.diff(expr,Y);expected=derivative(coeff,1)
                elif op==2:target=sp.expand(expr);expected=coeff
                else:target=expr.subs(X,X+1);expected=shift(coeff)
                assert coefficients(target)==expected
                rows.append({'id':hashlib.sha256((group+':'+operation).encode()).hexdigest()[:24],
                    'group':group,'scope':scope,'operation':operation,'operation_id':op,
                    'source_latex':sp.latex(expr),'target_latex':sp.latex(target),
                    'source_sympy':sp.sstr(expr),'target_sympy':sp.sstr(target),'degrees':degrees,
                    'source_coefficients':[[i,j,v] for (i,j),v in sorted(coeff.items())],
                    'target_coefficients':[[i,j,v] for (i,j),v in sorted(expected.items())],
                    'verified':True,'verification':'independent integer coefficient multiplication/differentiation/binomial shift',
                    'origin':'synthetic_symbolic_engine','license':'project generated auxiliary data; SymPy BSD-licensed implementation',
                    'benchmark_member':False})
    return rows,{'seed':seed,'attempts':attempts,'sources':{k:len(v) for k,v in groups.items()},
        'pairs':{k:sum(r['scope']==k for r in rows) for k in groups},'exact_verification':'100% labels verified with independent polynomial coefficients',
        'human_priors':['factored polynomial generator','four operation definitions','coefficient verifier','degree/factor holdout'],
        'coverage':'narrow synthetic algebra environment, not university/research formula benchmark coverage',
        'no_boundary_supervision':True}
