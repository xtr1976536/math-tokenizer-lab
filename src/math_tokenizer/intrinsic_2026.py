"""Frozen four-axis generalization design and eight common intrinsic metrics."""
from collections import Counter,defaultdict
import hashlib,math
import numpy as np
from .frontend_v6 import reference_lex
from .measure_v5 import cross_count
from .metrics_v2 import token_bytes

METHODS=['hard_ast_bpe','ast_v4','boundary_ast','math_atom_bpe','byte_bpe','sp_unigram','qwen25','llama3','o200k']
LABELS=['Hard AST-BPE','AST-v4','Boundary AST','Math-atom BPE','Byte BPE','SP Unigram','Qwen2.5','Llama 3','o200k']
METRICS=['tb','cmd','num','speed','used','cross','ast_align','renyi']
SEEDS=[17,29,43]

def split_groups(rows,seed,train_fraction=.8):
    grouped=defaultdict(list)
    for r in rows:grouped[r['split_group']].append(r)
    strata=defaultdict(list)
    for g,rr in grouped.items():strata[Counter(r['paper_domain'] for r in rr).most_common(1)[0][0]].append(g)
    parts=[set(),set(),set()]
    for domain,groups in sorted(strata.items()):
        groups=sorted(groups,key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest());n=len(groups)
        if n==1:nt,nd=1,0
        elif n==2:nt,nd=1,0
        elif train_fraction>.85:nt,nd=max(1,n-max(1,round(n*.1))),min(n-1,max(1,round(n*.1)))
        else:nd=max(1,round(n*.1));nt=min(n-nd-1,max(1,round(n*train_fraction)))
        parts[0].update(groups[:nt]);parts[1].update(groups[nt:nt+nd]);parts[2].update(groups[nt+nd:])
    return [[r for r in rows if r['split_group'] in p] for p in parts]

def remove_template_overlap(train,dev,test):
    seen={r['template_sha256'] for r in train};rendered={r['reference']['mathml_sha256'] for r in train}
    newdev=[r for r in dev if r['template_sha256'] not in seen and r['reference']['mathml_sha256'] not in rendered]
    seen.update(r['template_sha256'] for r in newdev);rendered.update(r['reference']['mathml_sha256'] for r in newdev)
    newtest=[r for r in test if r['template_sha256'] not in seen and r['reference']['mathml_sha256'] not in rendered]
    return train,newdev,newtest,{'dev_removed':len(dev)-len(newdev),'test_removed':len(test)-len(newtest)}

def protocols(rows):
    output=[]
    def add(name,family,seed,tr,dev,te,**info):
        tr,dev,te,removed=remove_template_overlap(tr,dev,te)
        assert tr and dev and te
        output.append(dict(name=name,family=family,seed=seed,train=tr,dev=dev,test=te,template_removal=removed,**info))
    for seed in SEEDS:
        tr,dv,te=split_groups(rows,seed)
        add(f'id_{seed}','ID',seed,tr,dv,te,description='全领域、按论文分层留出')
        simple=lambda rr:[r for r in rr if r['complexity']['class']=='simple']
        add(f'complexity_{seed}','Complexity',seed,simple(tr),simple(dv),[r for r in te if r['complexity']['class']=='complex'],description='简单训练、复杂测试；沿用ID来源组划分')
    heldouts=[['geometry_topology','combinatorics','number_theory'],['probability','statistics','optimization'],['analysis','pde','ode']]
    for i,held in enumerate(heldouts):
        target=[r for r in rows if r['paper_domain'] in held];target_groups={r['split_group'] for r in target}
        source=[r for r in rows if r['paper_domain'] not in held and r['split_group'] not in target_groups]
        tr,dv,unused=split_groups(source,17,.9);dv+=unused
        # A source with two papers contributes the second to development.
        add(f'ood_{i+1}','DomainOOD',17,tr,dv,target,heldout_domains=held,description='三个来源学科整域留出')
    families={'analysis_equations':['analysis','pde','ode'],'algebra':['algebra','number_theory','linear_algebra']}
    for source,target in [('analysis_equations','algebra'),('algebra','analysis_equations')]:
        target_rows=[r for r in rows if r['paper_domain'] in families[target]];target_groups={r['split_group'] for r in target_rows}
        source_rows=[r for r in rows if r['paper_domain'] in families[source] and r['split_group'] not in target_groups]
        tr,dv,unused=split_groups(source_rows,17,.9);dv+=unused
        add(f'extreme_{source}_to_{target}','ExtremeOOD',17,tr,dv,target_rows,source_domains=families[source],target_domains=families[target],description='分析/方程与代数/数论双向跨族迁移')
    return output

def references(record):
    raw=record['latex'].encode();lex=reference_lex(record['latex']);commands=[];numbers=[]
    for a,b in lex:
        value=raw[a:b].decode()
        if value.startswith('\\'):commands.append((a,b))
        if __import__('re').fullmatch(r'[0-9]+(?:\.[0-9]+)?',value):numbers.append((a,b))
    lexical=set(lex)
    composite=sorted({(a,b) for a,b in record['reference']['spans'] if (a,b) not in lexical and (a,b)!=(0,len(raw)) and b>a})
    return commands,numbers,composite

def per_formula(model,record,ids):
    raw=record['latex'].encode();payload=[token_bytes(model,int(i)) for i in ids]
    assert b''.join(payload)==raw and model.decode(ids)==record['latex']
    bounds=[0]
    for p in payload:bounds.append(bounds[-1]+len(p))
    assert all(0<=int(i)<(len(model.vocab) if hasattr(model,'vocab') else model.vocab_size) for i in ids)
    commands,numbers,composite=references(record);cuts=set(bounds)
    intact=lambda a,b:not any(a<x<b for x in bounds)
    return dict(id=record['id'],group=record['split_group'],domain=record['domain'],paper_domain=record['paper_domain'],
        complexity=record['complexity']['class'],bytes=len(raw),tokens=len(ids),commands=len(commands),command_good=sum(intact(a,b) for a,b in commands),
        numbers=len(numbers),number_good=sum(intact(a,b) for a,b in numbers),crossings=cross_count(bounds,record['reference']['spans']),
        ast_nodes=len(composite),ast_good=sum(a in cuts and b in cuts for a,b in composite),roundtrip=True,ids=list(map(int,ids)))

def summarize(rows,vocab,speed):
    count=Counter(i for r in rows for i in r['ids']);total=sum(count.values());papers=defaultdict(lambda:[0,0])
    for r in rows:papers[r['group']][0]+=r['tokens'];papers[r['group']][1]+=r['bytes']
    ratio=lambda good,den:sum(r[good] for r in rows)/sum(r[den] for r in rows) if sum(r[den] for r in rows) else None
    p=np.array(list(count.values()),dtype=float)/total;alpha=2.5;h=float(np.log2(np.sum(p**alpha))/(1-alpha))
    return dict(tb=float(np.mean([t/b for t,b in papers.values()])),tb_micro=total/sum(r['bytes'] for r in rows),
        cmd=ratio('command_good','commands'),num=ratio('number_good','numbers'),cross=sum(r['crossings'] for r in rows)/total,
        ast_align=ratio('ast_good','ast_nodes'),renyi=h/math.log2(vocab),used=len(count)/vocab,speed=speed,
        tokens=total,bytes=sum(r['bytes'] for r in rows),formulas=len(rows),papers=len(papers),vocab=vocab,
        commands=sum(r['commands'] for r in rows),numbers=sum(r['numbers'] for r in rows),ast_nodes=sum(r['ast_nodes'] for r in rows),
        roundtrip=1.0,length_quantiles=np.quantile([r['tokens'] for r in rows],[.5,.95,.99]).tolist())

def paper_bootstrap(rows,seed=2026,repeats=1000):
    grouped=defaultdict(list)
    for r in rows:grouped[r['group']].append(r)
    # Common paper draws are shared across methods in the same protocol.
    array=np.array([[sum(r[k] for r in rr) for k in ['tokens','bytes','commands','command_good','numbers','number_good','crossings','ast_nodes','ast_good']] for _,rr in sorted(grouped.items())],float)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        a=array[rng.integers(0,len(array),len(array))];s=a.sum(axis=0)
        values.append([np.mean(a[:,0]/a[:,1]),s[3]/s[2] if s[2] else np.nan,s[5]/s[4] if s[4] else np.nan,s[6]/s[0],s[8]/s[7] if s[7] else np.nan])
    return {k:list(map(float,np.nanquantile(np.array(values)[:,i],[.025,.975]))) for i,k in enumerate(['tb','cmd','num','cross','ast_align'])}
