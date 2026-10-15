"""Replay historical AST controls and independently audit Cmd/Cross denominators.

Read-only with respect to model/data/production experiment artifacts. New
diagnostic output records current timing separately from frozen experiment time.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import suites
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.final_models import load_final
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.frontend_v6 import reference_lex
from math_tokenizer.measure_v6 import clear


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def independent(model,records,details=False):
    total=Counter();papers=defaultdict(lambda:[0,0]);rows=[]
    for record in records:
        raw=record['latex'].encode();ids=model.encode(record['latex']);pieces=[token_bytes(model,i) for i in ids]
        assert b''.join(pieces)==raw and model.decode(ids)==record['latex']
        bounds=np.r_[0,np.cumsum([len(p) for p in pieces])];spans=np.asarray(record['reference']['spans'],dtype=int).reshape(-1,2)
        a=bounds[:-1,None];b=bounds[1:,None];s=spans[None,:,0];e=spans[None,:,1]
        # Independent broadcast definition, rather than production sweep helper.
        bad=np.any(((a<s)&(s<b)&(b<e))|((s<a)&(a<e)&(e<b)),axis=1)
        commands=ok=numbers=numberok=0
        import re
        for left,right in reference_lex(record['latex']):
            intact=not np.any((bounds>left)&(bounds<right));piece=raw[left:right]
            if piece.startswith(b'\\'):commands+=1;ok+=intact
            if re.fullmatch(rb'[0-9]+(?:\.[0-9]+)?',piece):numbers+=1;numberok+=intact
        total.update(bytes=len(raw),tokens=len(ids),crossing_tokens=int(bad.sum()),commands=commands,
                     commands_intact=int(ok),numbers=numbers,numbers_intact=int(numberok),formulas=1)
        papers[record['split_group']][0]+=len(ids);papers[record['split_group']][1]+=len(raw)
        if details:rows.append({'id':record['id'],'latex':record['latex'],'pieces':[p.decode('utf-8','backslashreplace') for p in pieces],
            'crossing_pieces':[p.decode('utf-8','backslashreplace') for p,flag in zip(pieces,bad) if flag],
            'tokens':len(ids),'crossing_tokens':int(bad.sum()),'commands':commands,'commands_intact':int(ok)})
    result=dict(total);result.update(document_macro_tokens_per_byte=float(np.mean([t/b for t,b in papers.values()])),
        independent_tree_crossing=total['crossing_tokens']/total['tokens'],command_integrity=total['commands_intact']/total['commands'],
        number_integrity=total['numbers_intact']/total['numbers'],crossing_tokens_per_kib=total['crossing_tokens']/(total['bytes']/1024),
        roundtrip_and_payload_all_passed=True,test_id_sha256=hashlib.sha256('\n'.join(sorted(r['id'] for r in records)).encode()).hexdigest())
    return result,rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'experiments/diagnostics/ast_metrics_20261015')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('use a new diagnostic output directory')
    out.mkdir(parents=True);inputs=suites();old=ROOT/'history/results/v8/full'
    registry=json.loads((old/'registry.json').read_text());historical=json.loads((old/'metrics.json').read_text())
    replay=[];times=[];vocab_checks=[]
    names=['ast_v4_parent','hard_tree_atomic_vocab','ast_lexicographic','official_atomic_bpe','ast_boundary']
    for scenario in ('broad','specialized'):
        suite='external_test' if scenario=='broad' else 'extreme_domain_test';records=inputs[suite]
        for capacity in (1024,2048):
            models={}
            for name in names:
                entry=next(e for e in registry if e['name']==name and e['scenario']==scenario and e['capacity']==capacity)
                path=ROOT/'history'/entry['path'];assert sha(path)==entry['checkpoint_sha256']
                model=load_checkpoint(path);models[name]=model
                observed,_=independent(model,records)
                expected=next(m for m in historical if m['key']==entry['key'] and m['suite']==suite)
                assert observed['tokens']==expected['tokens'] and observed['test_id_sha256']==expected['test_id_sha256']
                for metric in ('document_macro_tokens_per_byte','independent_tree_crossing','command_integrity','number_integrity'):
                    assert abs(observed[metric]-expected[metric])<1e-12,(name,capacity,scenario,metric)
                replay.append({'method':name,'capacity':capacity,'scenario':scenario,'suite':suite,
                    'checkpoint':str(path.relative_to(ROOT)),'sha256':sha(path),'historical_encode_mib_s':expected['encode_mib_s'],
                    'replay_matches_frozen_counts_and_metrics':True,**observed})
            shared=models['ast_boundary'].vocab
            assert all(models[n].vocab==shared for n in names if n!='ast_v4_parent')
            vocab_checks.append({'capacity':capacity,'scenario':scenario,'same_dictionary_methods':names[1:],
                                'parent_same_dictionary':models['ast_v4_parent'].vocab==shared})
            durations=defaultdict(list);texts=[r['latex'] for r in records];bytes_total=sum(len(t.encode()) for t in texts)
            for repeat in range(7):
                order=names[repeat%len(names):]+names[:repeat%len(names)]
                if repeat%2:order=order[::-1]
                for name in order:
                    clear(models[name]);started=time.perf_counter();encoded=[models[name].encode(t) for t in texts]
                    durations[name].append(time.perf_counter()-started)
            for name in names:times.append({'method':name,'capacity':capacity,'scenario':scenario,
                'encode_seconds':durations[name],'encode_mib_s':bytes_total/(1048576*np.median(durations[name])),
                'scope':'new diagnostic timing on current machine,not frozen historical numbers'})
            print('Historical AST replay matched',scenario,capacity,flush=True)
    command_cross=[];details={}
    for method in ('ast_boundary','official_atomic_bpe','Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base'):
        model=load_final(method,capacity=2048);summary,rows=independent(model,inputs['external_test'],details=True)
        current=json.loads((ROOT/'experiments/current/metrics.json').read_text())
        expected=next(m for m in current if m['method']==method and m['suite']=='external_test' and (m['scenario']=='pretrained' or m['scenario']=='broad' and m['capacity']==2048))
        assert summary['tokens']==expected['tokens'] and abs(summary['independent_tree_crossing']-expected['independent_tree_crossing'])<1e-12
        assert abs(summary['command_integrity']-expected['command_integrity'])<1e-12
        command_cross.append({'method':method,**summary});details[method]={r['id']:r for r in rows}
    examples=[]
    for r in sorted(details['ast_boundary'].values(),key=lambda r:len(r['latex'])):
        general=details['Qwen2.5-0.5B'][r['id']]
        if r['commands'] and r['commands_intact']==r['commands'] and r['crossing_tokens'] and not general['crossing_tokens']:
            examples.append({'ast':r,'qwen':general})
            if len(examples)==3:break
    general=load_final('Qwen2.5-0.5B');q=general.tokenizer.to_str();config=json.loads(q)
    write(out/'replay.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'historical_registry_sha256':sha(old/'registry.json'),
        'historical_metric_sha256':sha(old/'metrics.json'),'replay':replay,'timing':times,'vocab_checks':vocab_checks,
        'test_training':False,'scope':'frozen checkpoints re-encoded,not retrained; same full suites and independent metric implementation',
        'standard_AST_name':'historical AST-v4,GrammarBPE training with hard native source-tree constraints',
        'same_dictionary_control':'hard_tree_atomic_vocab','cmd_cross_2048':command_cross,'actual_formula_examples':examples,
        'qwen_pre_tokenizer':config.get('pre_tokenizer')})
    print('Independent audit complete:',out,flush=True)


if __name__=='__main__':main()
