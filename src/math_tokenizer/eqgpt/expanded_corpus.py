"""Paired original-LaTeX/GRE permutations without factorial enumeration.

Controls stay at sequence boundaries. Source LaTeX sides retain every signed
term substring, including scalar coefficients and derivative notation. GRE
is a coefficient-free structure; its term order need not match factored TeX.
Both representations always refer to the same source_id and augmentation_id.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .author import DICTIONARY
from .latex_codec import structure_to_latex
from math_tokenizer.corpus import ROOT


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def source_records():
    new=ROOT/'data/formulas/gre_benchmark_filtered/benchmark.jsonl'
    old=ROOT/'data/formulas/gre_benchmark_audit_v2/eqgpt_221_after_target_exclusion.jsonl'
    result=[]
    for row in map(json.loads,new.read_text().splitlines()):
        result.append(dict(source_id=row['id'],source_kind='external_benchmark',
            source_group=row['source_group'],family=row['family'],parameters=row.get('parameters',{}),
            source_url=row['source_url'],source_location=row.get('source_location'),
            original_latex=row['latex'],paper_latex=row.get('source_math'),
            latex_origin=row['origin'],gre_ids=row['conversion']['gre_ids'],
            source_transformations=row.get('transformations',[]),source_file=row['source_file'],
            source_sha256=row['source_sha256']))
    for row in map(json.loads,old.read_text().splitlines()):
        result.append(dict(source_id='eqgpt221:'+row['id'],source_kind='eqgpt_handbook_non_target',
            source_group='eqgpt_original_221',family='eqgpt_handbook_221',parameters={},
            source_url=row.get('source_url'),equation_name=row['equation_name'],
            original_latex=row['latex'],paper_latex=None,latex_origin='rendered_from_author_GRE_no_original_TeX_in_workbook',
            gre_ids=row['gre_ids'],source_file=row['source_file'],source_sha256=row['source_sha256']))
    assert len(result)==2257 and len({r['source_id'] for r in result})==2257
    return result


def equation_sides(text):
    stack=[];eq=[]
    for i,ch in enumerate(text):
        if ch in '{([':stack.append(ch)
        elif ch in '})]':
            if not stack or {'}':'{',')':'(',']':'['}[ch]!=stack.pop():raise ValueError('Unbalanced source LaTeX')
        elif ch=='=' and not stack:eq.append(i)
    if stack or len(eq)!=1:raise ValueError('Expected one top-level equation')
    i=eq[0];return text[:i],text[i+1:]


def signed_terms(side):
    """A signed term bag, preserving each term substring verbatim."""
    stack=[];start=0;terms=[]
    for i,ch in enumerate(side):
        if ch in '{([':stack.append(ch)
        elif ch in '})]':stack.pop()
        elif ch in '+-' and not stack and side[start:i].strip():
            terms.append(side[start:i]);start=i
    terms.append(side[start:])
    result=[]
    for raw in terms:
        stripped=raw.lstrip()
        sign=stripped[0] if stripped and stripped[0] in '+-' else '+'
        body=stripped[1:] if stripped and stripped[0] in '+-' else raw
        if not body.strip():raise ValueError('Empty signed term')
        result.append((sign,body))
    return result


def latex_signature(text):
    return tuple(Counter(signed_terms(side)) for side in equation_sides(text))


def permute_latex(text,rng,index):
    if index==0:return text
    sides=[]
    for side in equation_sides(text):
        parts=signed_terms(side);rng.shuffle(parts)
        value=''.join(sign+body for sign,body in parts)
        sides.append(value[1:] if value.startswith('+') else value)
    result='='.join(sides)
    assert latex_signature(result)==latex_signature(text)
    return result


def gre_terms(ids):
    if ids[0]!=5 or ids[-1]!=1 or ids.count(5)!=1 or ids.count(1)!=1:
        raise ValueError('GRE controls must only occur at boundaries')
    result=[];part=[]
    for i in ids[1:-1]:
        if i==2:
            if not part:raise ValueError('Empty GRE term')
            result.append(part);part=[]
        else:part.append(i)
    if not part:raise ValueError('Empty final GRE term')
    return result+[part]


def factor_signature(part):
    powers=Counter();op=1
    for token in part:
        if token==3:op=1
        elif token==4:op=-1
        elif token>=6:powers[token]+=op;op=1
        else:raise ValueError('Unexpected GRE term token')
    return tuple(sorted((i,p) for i,p in powers.items() if p))


def gre_signature(ids):
    return Counter(factor_signature(t) for t in gre_terms(ids))


def permute_gre(ids,rng,index):
    if index==0:return list(ids)
    groups=gre_terms(ids);rng.shuffle(groups);output=[5]
    for pos,part in enumerate(groups):
        numerator=[];denominator=[];op=1
        for token in part:
            if token==3:op=1
            elif token==4:op=-1
            else:
                (numerator if op==1 else denominator).append(token);op=1
        rng.shuffle(numerator);rng.shuffle(denominator)
        if not numerator:raise ValueError('Leading reciprocal unsupported')
        if pos:output.append(2)
        for i,t in enumerate(numerator):
            if i:output.append(3)
            output.append(t)
        for t in denominator:output.extend([4,t])
    output.append(1)
    assert gre_signature(output)==gre_signature(ids)
    return output


def paired_rows(sources,times=32,seed=525):
    for source in sources:
        local=int(hashlib.sha256((str(seed)+':'+source['source_id']).encode()).hexdigest()[:16],16)
        for i in range(times):
            # Separate RNGs: changes to spelling complexity cannot change GRE choices.
            gre=permute_gre(source['gre_ids'],random.Random(local+i*2),i)
            latex=permute_latex(source['original_latex'],random.Random(local+i*2+1),i)
            yield dict(source_id=source['source_id'],augmentation_index=i,
                       original_latex=source['original_latex'],augmented_latex=latex,
                       augmented_gre_ids=gre,augmented_gre_words=[DICTIONARY['id2word'][g] for g in gre])
