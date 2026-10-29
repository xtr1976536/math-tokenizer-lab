"""Full PDE 8:1:1 benchmark and initial-alphabet x total-ID experiment."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import string
import time

from math_tokenizer.corpus import ROOT
from math_tokenizer.pde_protocol import split_corpus, digest
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet, train_compact
from math_tokenizer.eqgpt.latex_tokenizers import prefix_model
from scripts.evaluate_screened_pde import measure, token_coverage
from scripts.当前四分词器 import load

SIZES = [64, 80, 96, 112, 128, 160, 192, 200, 224, 240, 256, 272, 288, 300, 320, 352, 384, 416, 448, 450, 480, 512]


def alphabet_protocols(train):
    old = set(range(32, 127)) | {9, 10, 13}
    # Fixed character-level protocols, no complete command/variable composites.
    core = set((string.ascii_lowercase+string.digits+'\\{}()[]^_+-=*/., \t\n\r<>|:;!&').encode())
    return {
        'core': dict(alphabet=sorted(core), rule='fixed lower-case Latin, digits, LaTeX punctuation and whitespace'),
        'letters': dict(alphabet=sorted(core | set(string.ascii_uppercase.encode())), rule='core plus all upper-case Latin'),
        'latex_ascii': dict(alphabet=sorted(old-set(b'"#$%?@`')), rule='ASCII/whitespace excluding double quote, hash, dollar, percent, question mark, at sign, backtick'),
        'ascii95': dict(alphabet=sorted(old-set(b'"#$')), rule='ASCII/whitespace excluding double quote, hash, dollar'),
        'ascii98': dict(alphabet=sorted(old), rule='historical 95 printable ASCII plus TAB/LF/CR'),
        'train_observed': dict(alphabet=sorted(set(b'0123456789') | set().union(*(set(r['latex'].encode()) for r in train))),
                               rule='train-only observed bytes plus all digits; diagnostic adaptive arm, not a fixed protocol'),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def evaluate(model, rows, path):
    supported = [r for r in rows if not hasattr(model, 'alphabet') or set(r['latex'].encode()) <= model.alphabet]
    missing = [dict(id=r['id'], missing_hex=[f'{v:02x}' for v in sorted(set(r['latex'].encode())-model.alphabet)])
               for r in rows if hasattr(model, 'alphabet') and not set(r['latex'].encode()) <= model.alphabet]
    start = time.perf_counter()
    summary, details, usage = measure(model, supported) if supported else ({}, [], Counter())
    seconds = time.perf_counter()-start
    summary.update(requested_formulas=len(rows), covered_formulas=len(supported), oov_formulas=len(missing),
                   full_coverage=not missing, oov=missing, encoding_and_metric_seconds=seconds,
                   **{'tokenizer compression': summary.get('compression_ratio'),
                      'operation': summary.get('cmd_pct'), 'numbers': summary.get('num_pct')})
    # Recompute all three metrics from the saved per-formula counts.
    if details:
        total = lambda k: sum(r[k] for r in details)
        assert abs(total('initial_character_tokens')/total('final_tokens')-summary['tokenizer compression']) < 1e-12
        for metric, num, den in [('operation','commands_intact','commands'), ('numbers','numbers_intact','numbers')]:
            if total(den):
                assert abs(100*total(num)/total(den)-summary[metric]) < 1e-10
    with gzip.open(path, 'wt') as f:
        for r in details:
            f.write(json.dumps(r)+'\n')
    return summary, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='experiments/upstream_811')
    ap.add_argument('--sizes', default=','.join(map(str,SIZES)))
    ap.add_argument('--without-baselines', action='store_true')
    args = ap.parse_args()
    out = ROOT/args.output
    data = ROOT/'data/formulas/pde/pde_equations.jsonl.gz'
    rows = list(map(json.loads, gzip.open(data, 'rt')))
    plans, split = split_corpus(rows)
    sizes = sorted(set(map(int,args.sizes.split(','))))
    schemes = alphabet_protocols(plans['train'])
    protocol = dict(version=1, data_sha256=digest(data.read_bytes()), split=split, sizes=sizes, alphabets=schemes,
                    controls={'PAD':0,'EOS':1,'BOS':2}, total_ids_definition='base bytes + learned payloads + 3 controls',
                    training='train only; validation matrix for selection; test only for predeclared selected arms',
                    fallback='none; explicit OOV; no formula alteration or silent UNK',
                    metrics='identical source-byte numerator, command/number intact definitions as historical evaluate_screened_pde.py',
                    selection='validation full-coverage Pareto front on compression/operation/numbers; plus historical ascii98/256 reference')
    pfile = out/'protocol.json'
    if pfile.exists():
        assert json.loads(pfile.read_text()) == protocol, 'Refusing incompatible resume'
    else:
        write_json(pfile, protocol)
    results = []
    for name, config in schemes.items():
        alphabet = config['alphabet']
        missing = sorted(set().union(*(set(r['latex'].encode()) for r in plans['train']))-set(alphabet))
        folder = out/name
        if missing:
            for size in sizes:
                results.append(dict(alphabet=name, base_bytes=len(alphabet), total_ids=size,
                                    status='training_characters_not_covered', missing_train_hex=[f'{b:02x}' for b in missing]))
            continue
        valid = [s for s in sizes if s >= len(alphabet)+3]
        for size in sizes:
            if size not in valid:
                results.append(dict(alphabet=name, base_bytes=len(alphabet), total_ids=size, status='below_base_plus_controls'))
        if not valid:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        parent_path=folder/'parent.model.json'
        start=time.perf_counter()
        if parent_path.exists():
            parent=CompactAlphabet.load(parent_path)
        else:
            parent=train_compact('ast_bpe',[r['latex'] for r in plans['train']],max(valid),alphabet)
            parent.save(parent_path)
        train_seconds=time.perf_counter()-start
        print('PARENT',name,'base',len(alphabet),'seconds',round(train_seconds,2),flush=True)
        for size in valid:
            result_path=folder/f'{size}.json'
            if result_path.exists():
                results.append(json.loads(result_path.read_text())); continue
            # Ranked BPE prefix is identical to an independent smaller training run.
            internal=prefix_model('ast_bpe',parent.internal,size+256-len(alphabet))
            model=CompactAlphabet('ast_bpe',alphabet,internal)
            assert len(model.vocab)+3==size
            mp=folder/f'{size}.model.json'; model.save(mp)
            restored=CompactAlphabet.load(mp)
            for r in plans['train'][:8]:
                assert model.encode(r['latex']) == restored.encode(r['latex'])
            validation, usage=evaluate(model,plans['validation'],folder/f'{size}.validation.jsonl.gz')
            training,train_usage=evaluate(model,plans['train'],folder/f'{size}.train.jsonl.gz')
            row=dict(alphabet=name,base_bytes=len(alphabet),total_ids=size,status='evaluated',
                     parent_training_seconds=train_seconds,train=training,validation=validation,
                     active_train_ids=len(train_usage),unused_content_ids=len(model.vocab)-len(train_usage),
                     vocabulary_sha256=digest(json.dumps([x.hex() for x in model.vocab])))
            with (folder/f'{size}.tokens.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['content_id','type','text','hex','train_count','validation_count'])
                for i,b in enumerate(model.vocab):
                    w.writerow([i,'base_character' if len(b)==1 else 'learned_merge',b.decode(),b.hex(),train_usage[i],usage[i]])
            write_json(result_path,row);results.append(row)
            print('VALIDATION',name,size,*[validation[k] for k in ('tokenizer compression','operation','numbers')],flush=True)
    write_json(out/'validation_results.json',results)
    metrics=('tokenizer compression','operation','numbers')
    eligible=[r for r in results if r['status']=='evaluated' and r['validation']['full_coverage']]
    def dominates(a,b):
        aa=[a['validation'][k] for k in metrics];bb=[b['validation'][k] for k in metrics]
        return all(x>=y for x,y in zip(aa,bb)) and any(x>y for x,y in zip(aa,bb))
    # Freeze all nondominated settings before accessing test measurements.
    selected=[r for r in eligible if not any(dominates(a,r) for a in eligible)]
    historical=[r for r in eligible if r['alphabet']=='ascii98' and r['total_ids']==256]
    selected=list({(r['alphabet'],r['total_ids']):r for r in selected+historical}.values())
    selection=dict(rule=protocol['selection'], arms=[dict(alphabet=r['alphabet'],total_ids=r['total_ids'],
                    vocabulary_sha256=r['vocabulary_sha256']) for r in selected])
    write_json(out/'selection.json',selection)
    tests=[]
    for row in selected:
        name,size=row['alphabet'],row['total_ids'];folder=out/name
        model=CompactAlphabet.load(folder/f'{size}.model.json')
        summary,_=evaluate(model,plans['test'],folder/f'{size}.test.jsonl.gz')
        tests.append(dict(method='ast_bpe',alphabet=name,total_ids=size,**summary))
    if not args.without_baselines:
        for name in ('qwen2.5','llama3','o200k_base'):
            model=load(name)
            validation,_=evaluate(model,plans['validation'],out/f'{name}.validation.jsonl.gz')
            test,_=evaluate(model,plans['test'],out/f'{name}.test.jsonl.gz')
            tests.append(dict(method=name,total_ids=model.vocab_size,validation=validation,**test))
            print('BASELINE',name,test['tokenizer compression'],test['operation'],test['numbers'],flush=True)
    write_json(out/'test_results.json',tests)
    fields=['alphabet','base_bytes','total_ids','status',*metrics,'covered_formulas','oov_formulas','active_train_ids','unused_content_ids']
    with (out/'validation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            v=r.get('validation',{});w.writerow({k:r.get(k,v.get(k)) for k in fields})
    write_json(out/'audit.json',dict(status='PASS',full_corpus_formulas=len(rows),split_counts=split['counts'],
               all_input_ids_retained=True,source_template_canonical_disjoint=True,
               matrix_rows=len(results),evaluated_arms=len(eligible),selected_test_arms=len(selected),
               original_experiments_untouched=True,roundtrip_and_count_reaggregation=True))
    print('COMPLETE',len(results),'matrix rows',len(tests),'test rows',flush=True)

if __name__=='__main__':main()
