"""固定 1500 条 PDE 上四分词器评估：分成 3 份，每次选 2 份训练、1 份测试及简单训练到复杂测试。"""
import argparse
import bisect
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet, train_compact
from math_tokenizer.metrics_v2 import token_bytes
from scripts.当前四分词器 import load as load_current

ALPHABET=list(range(32,127))+[9,10,13]
BASELINES=('Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base')
LEX=re.compile(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[\s\S]')


def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def make_splits(rows):
    # Connected source groups ensure exact rendered duplicates cannot leak.
    groups=defaultdict(list)
    for row in rows:
        groups[row['source_group']].append(row)
    folds=[set(),set(),set()];sizes=[0,0,0]
    for source, rr in sorted(groups.items(),key=lambda v:(-len(v[1]),sha(v[0]))):
        i=min(range(3),key=lambda i:sizes[i])
        folds[i].add(source);sizes[i]+=len(rr)
    output={}
    for i,test_sources in enumerate(folds,1):
        test=[r for r in rows if r['source_group'] in test_sources]
        blocked={r['template_sha256'] for r in test}
        train=[r for r in rows if r['source_group'] not in test_sources and r['template_sha256'] not in blocked]
        output[f'pde_fold_{i}']=dict(train=train,test=test,
            purged_template_overlap=sum(r['source_group'] not in test_sources and r['template_sha256'] in blocked for r in rows))
    # The difficult formula test is source-disjoint as well as difficulty-disjoint.
    hard=[r for r in rows if r['complexity_bin']=='complex' and r['source_group'] in folds[2]]
    blocked={r['template_sha256'] for r in hard}
    simple=[r for r in rows if r['complexity_bin']=='simple' and r['source_group'] not in folds[2]
            and r['template_sha256'] not in blocked]
    output['pde_simple_to_complex']=dict(train=simple,test=hard,purged_template_overlap=0)
    for name,p in output.items():
        assert p['train'] and p['test'], name
        for key in ('id','source_group','template_sha256','canonical_sha256'):
            assert not {r[key] for r in p['train']} & {r[key] for r in p['test']},(name,key)
    return output


def measure(model,rows,own_initial=None):
    rr=[];counts=Counter()
    for row in rows:
        text=row['latex'];raw=text.encode();ids=model.encode(text)
        pieces=[model.vocab[i] for i in ids] if hasattr(model,'vocab') else [token_bytes(model,int(i)) for i in ids]
        assert all(pieces) and b''.join(pieces)==raw and model.decode(ids)==text
        counts.update(ids)
        bounds=[];pos=0
        for b in pieces[:-1]:
            pos+=len(b);bounds.append(pos)
        cmd=cmdok=num=numok=0
        for m in LEX.finditer(text):
            token=m.group();a=len(text[:m.start()].encode());b=a+len(token.encode())
            intact=bisect.bisect_right(bounds,a)==bisect.bisect_left(bounds,b)
            if re.fullmatch(r'\\[A-Za-z]+',token):
                cmd+=1;cmdok+=intact
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',token):
                num+=1;numok+=intact
        rr.append(dict(id=row['id'],source_group=row['source_group'],complexity=row['complexity_bin'],
            initial_character_tokens=len(raw),initial_method_tokens=len(own_initial.symbols(text)) if own_initial else len(raw),
            final_tokens=len(ids),commands=cmd,commands_intact=cmdok,numbers=num,numbers_intact=numok))
    total=Counter()
    for r in rr:
        total.update({k:v for k,v in r.items() if isinstance(v,int)})
    summary=dict(formulas=len(rr),sources=len({r['source_group'] for r in rr}),
        compression_ratio=total['initial_character_tokens']/total['final_tokens'],
        method_initial_compression_ratio=total['initial_method_tokens']/total['final_tokens'],
        cmd_pct=100*total['commands_intact']/total['commands'] if total['commands'] else None,
        num_pct=100*total['numbers_intact']/total['numbers'] if total['numbers'] else None,
        counts=dict(total),test_ids_sha256=sha('\n'.join(r['id'] for r in rows)))
    return summary,rr,counts


def token_coverage(model,rows):
    counts=Counter();formulas=Counter();papers=defaultdict(set)
    for row in rows:
        ids=model.encode(row['latex'])
        counts.update(ids)
        for i in set(ids):
            formulas[i]+=1;papers[i].add(row['source_group'])
    return counts,formulas,{i:len(v) for i,v in papers.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--data',default='data/formulas/pde/pde_equations.jsonl.gz')
    parser.add_argument('--sizes',default='256')
    parser.add_argument('--ast-only',action='store_true')
    args=parser.parse_args();out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    snapshot=ROOT/'data/formulas/experiment_inputs'/out.name/'equations.jsonl.gz'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes((ROOT/args.data).read_bytes())
    rows=list(map(json.loads,gzip.open(snapshot,'rt')))
    sizes=list(map(int,args.sizes.split(',')))
    plans=make_splits(rows)
    protocol=dict(data_sha256=sha(snapshot.read_bytes()),equations=len(rows),
        alphabet=ALPHABET,base_bytes=len(ALPHABET),sizes=sizes,
        compression_definition='sum(initial one-byte alphabet tokens)/sum(final tokens); shared source-byte numerator across all methods',
        cmd_definition='percent of complete backslash+ASCII-letter command lexemes with no internal boundary',
        num_definition='percent of [0-9]+(?:decimal fraction)? lexemes with no internal boundary',
        experiment_scope='PDE equation screened snapshot only; no other-math training or scoring',
        splits={name:dict(train=[r['id'] for r in p['train']],test=[r['id'] for r in p['test']],
                         purged_template_overlap=p['purged_template_overlap']) for name,p in plans.items()})
    pfile=out/'protocol.json'
    if pfile.exists():
        assert json.loads(pfile.read_text())==protocol
    else:
        pfile.write_text(json.dumps(protocol,indent=2)+'\n')
    names={'Qwen2.5-0.5B':'qwen2.5','meta-llama/Meta-Llama-3-8B':'llama3',
           'tiktoken_o200k_base':'o200k_base'}
    baselines={} if args.ast_only else {name:load_current(alias) for name,alias in names.items()}
    results=[]
    for pname,plan in plans.items():
        folder=out/pname;folder.mkdir(exist_ok=True)
        train=[r['latex'] for r in plan['train']]
        for method,size in [(name,None) for name in baselines]+[("ast_bpe",size) for size in sizes]:
            stem=method.replace('/','--')+'_'+str(size or 'official')
            path=folder/(stem+'.json')
            if path.exists():
                results.append(json.loads(path.read_text()));continue
            started=time.monotonic();initial=None
            if size is None:
                model=baselines[method]
            else:
                model=train_compact(method,train,size,ALPHABET)
                cp=folder/(stem+'.model.json');model.save(cp)
                reloaded=CompactAlphabet.load(cp)
                for r in plan['test'][:8]:
                    assert model.encode(r['latex'])==reloaded.encode(r['latex'])
            summary,rr,usage=measure(model,plan['test'],initial)
            summary.update(protocol=pname,method=method,total_ids=size or model.vocab_size,
                           seconds=time.monotonic()-started)
            if size:
                inside,_,train_usage=measure(model,plan['train'],initial)
                summary['train']=inside
                summary['distribution']=dict(active_train_ids=len(train_usage),active_test_ids=len(usage),
                    zero_train_ids=len(model.vocab)-len(train_usage),
                    max_piece_bytes=max(map(len,model.vocab)),
                    long_rare_train_pieces=sum(len(b)>16 and train_usage[i]<=10 for i,b in enumerate(model.vocab)))
                _,train_formulas,train_papers=token_coverage(model,plan['train'])
                _,test_formulas,test_papers=token_coverage(model,plan['test'])
                atom_payloads=set()
                with (folder/(stem+'.tokens.csv')).open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['content_id','type','text','hex','bytes','train_count','train_probability','train_formulas','train_papers','test_count','test_probability','test_formulas','test_papers'])
                    for i,b in enumerate(model.vocab):
                        kind='base_character' if len(b)==1 else 'command_atom' if b in atom_payloads else 'learned_merge'
                        w.writerow([i,kind,b.decode('utf-8','backslashreplace'),b.hex(),len(b),train_usage[i],train_usage[i]/sum(train_usage.values()),train_formulas[i],train_papers.get(i,0),usage[i],usage[i]/sum(usage.values()),test_formulas[i],test_papers.get(i,0)])
            path.write_text(json.dumps(summary,indent=2)+'\n')
            with gzip.open(folder/(stem+'.per_formula.jsonl.gz'),'wt') as f:
                for r in rr:
                    f.write(json.dumps(r)+'\n')
            results.append(summary)
            print('EVAL',pname,method,size,'CR',round(summary['compression_ratio'],3),'CMD',summary['cmd_pct'],'NUM',summary['num_pct'],flush=True)
            token_bytes.cache_clear()
    # Independent aggregation from persisted counts; shared sample denominators.
    signatures={}
    for result in results:
        folder=out/result['protocol'];stem=result['method'].replace('/','--')+'_'+str(result['total_ids'] if result['method'] not in BASELINES else 'official')
        rr=list(map(json.loads,gzip.open(folder/(stem+'.per_formula.jsonl.gz'),'rt')))
        count=lambda k:sum(r[k] for r in rr)
        assert abs(count('initial_character_tokens')/count('final_tokens')-result['compression_ratio'])<1e-12
        for name,num,den in [('cmd_pct','commands_intact','commands'),('num_pct','numbers_intact','numbers')]:
            if count(den):
                assert abs(100*count(num)/count(den)-result[name])<1e-10
        signature=(result['test_ids_sha256'],count('initial_character_tokens'),count('commands'),count('numbers'))
        assert signatures.setdefault(result['protocol'],signature)==signature
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(dict(status='PASS',checks=['source/id/template/canonical split isolation',
        'exact UTF8 payload and decode reconstruction','compact model save/load equality',
        'independent persisted count aggregation','identical ordered test samples and metric denominators'],
        result_rows=len(results)),indent=2)+'\n')
    print('EVALUATION_COMPLETE',len(results),flush=True)


if __name__=='__main__':
    main()
