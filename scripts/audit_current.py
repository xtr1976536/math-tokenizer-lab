"""Three actual audits: provenance/classification, encoders, independent scores."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import bisect,gzip,hashlib,json,math,re
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog,suites,FOLDER
from math_tokenizer.final_models import load_final
from math_tokenizer.result_rows import PerFormulaStore
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.frontend_v6 import reference_lex
from math_tokenizer.measure_v5 import cross_count
from math_tokenizer.independent_v7 import source_scopes
from math_tokenizer.quality_v6 import classify,TARGET_CATEGORIES,formula_gate


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=ROOT/'experiments/current';dest.mkdir(exist_ok=True)
    manifest=json.loads((FOLDER/'manifest.json').read_text());catalog=read_catalog();byid={r['id']:r for r in catalog}
    entries=json.loads((dest/'registry.json').read_text());metrics=json.loads((dest/'metrics.json').read_text());inputs=suites();reviews=[]
    assert len({e['name'] for e in entries})==7 and len(entries)==19
    assert Counter(r['domain'] for r in catalog)==manifest['domain']
    assert Counter(r['origin'] for r in catalog)==manifest['origin']
    assert Counter(r['complexity']['class'] for r in catalog)==manifest['complexity']
    partition_ids=[]
    for part in manifest['catalog']:
        path=FOLDER/part['path'];assert sha(path)==part['sha256']
        formulas=list(map(json.loads,path.open()));ids=list(map(json.loads,(FOLDER/part['id_path']).open()))
        assert len(formulas)==len(ids)==part['formulas']
        assert all(t==byid[i]['latex'] for t,i in zip(formulas,ids));partition_ids+=ids
    assert len(partition_ids)==len(set(partition_ids))==len(catalog)
    classified=0
    for r in catalog:
        assert isinstance(r['latex'],str) and r['latex'] and r['complexity']['class'] in ('simple','complex')
        domain,tags=classify(r['latex'],r['paper_domain'],r.get('subject_categories',[]))
        assert domain==r['domain'] and tags==r['formula_types'],r['id'];classified+=1
        assert formula_gate(r['latex']) is None,(r['id'],formula_gate(r['latex']))
        if r['origin']=='human':assert r['version_date'][:4]<'2020' and r['license'].startswith(('CC-BY-','CC0-'))
    for scenario in ('broad','specialized'):
        scope=json.loads((dest/f'{scenario}_scope.json').read_text())
        groups=[{byid[i]['split_group'] for i in scope[k]} for k in ('fit_ids','reward_ids','dev_ids')]
        assert all(not groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2)))
        assert not set.union(*groups)&{r['split_group'] for r in inputs['external_test']}
        if scenario=='specialized':assert all(not set(byid[i]['subject_categories'])&TARGET_CATEGORIES for k in ('fit_ids','reward_ids','dev_ids') for i in scope[k])
    reviews.append({'round':1,'passed':True,'scope':'All catalog formulas, source-origin licenses, domain/type taxonomy, classification file IDs/hashes, frozen source-disjoint scopes','catalog_rows':classified})
    rows=PerFormulaStore(dest/'per_formula.jsonl.gz');replayed=0;counts={};usage={}
    for e in entries:
        if e.get('path'):assert sha(ROOT/e['path'])==e['checkpoint_sha256']
        model=load_final(e['name'],e['scenario'] if e['scenario']!='pretrained' else 'broad',e['capacity'] or 1024)
        calibrations=('broad','specialized') if e['scenario']=='pretrained' else (e['scenario'],)
        usage[e['key']]={}
        for s in calibrations:
            scope=json.loads((dest/f'{s}_scope.json').read_text())
            usage[e['key']][s]=Counter(i for rid in scope['fit_ids'] for i in model.encode(byid[rid]['latex']))
        for m in (m for m in metrics if m['key']==e['key']):
            rr=rows[(e['key'],m['suite'])];wanted={r['id']:r for r in inputs[m['suite']]}
            c=usage[e['key']][m['fit_calibration_scenario']];den=sum(c.values())+.5*m['vocab_actual']
            assert len(rr)==len(wanted)==len({r['id'] for r in rr}) and {r['id'] for r in rr}==set(wanted)
            aggregate=Counter();papers=defaultdict(lambda:[0,0]);scopeok=scopetotal=0;token_usage=Counter()
            for r in rr:
                record=wanted[r['id']];text=record['latex'];raw=text.encode();ids=model.encode(text)
                assert len(ids)==r['tokens'] and model.decode(ids)==text
                payload=[token_bytes(model,i) for i in ids];assert b''.join(payload)==raw
                token_usage.update(ids)
                bounds=[0]
                for p in payload:bounds.append(bounds[-1]+len(p))
                assert cross_count(bounds,record['reference']['spans'])==r['crossing_tokens']
                spans=source_scopes(text)
                if spans is not None:
                    expected={p for a,b in spans for p in (a,b)}-{0,len(raw)}
                    assert len(expected)==r['scope_boundaries'] and len(expected&set(bounds))==r['scope_boundaries_preserved']
                    scopetotal+=len(expected);scopeok+=len(expected&set(bounds))
                lexical_expected=set();internal=set(bounds[1:-1]);cmd=cmdok=num=numok=0
                for a,b in reference_lex(text):
                    unit=raw[a:b].decode()
                    if not unit.strip():continue
                    lexical_expected.update((a,b))
                    intact=not any(a<p<b for p in bounds)
                    if raw[a:b].startswith(b'\\'):
                        cmd+=1;cmdok+=intact
                    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',unit):num+=1;numok+=intact
                assert (cmd,cmdok,num,numok)==(r['commands'],r['commands_intact'],r['numbers'],r['numbers_intact'])
                aggregate.update(commands=cmd,command_ok=cmdok,numbers=num,number_ok=numok)
                lexical_expected-={0,len(raw)}
                aggregate['boundary_tp']+=len(internal&lexical_expected)
                aggregate['boundary_pred']+=len(internal);aggregate['boundary_ref']+=len(lexical_expected)
                aggregate['tokens']+=len(ids);aggregate['bytes']+=len(raw);aggregate['crossing']+=r['crossing_tokens']
                papers[r['group']][0]+=len(ids);papers[r['group']][1]+=len(raw)
                bits=-sum(math.log2((c[i]+.5)/den) for i in ids)
                assert abs(bits-r['predictive_bits'])<1e-8;aggregate['bits']+=bits;replayed+=1
            observed={'document_macro_tokens_per_byte':float(np.mean([t/b for t,b in papers.values()])),
                'independent_tree_crossing':aggregate['crossing']/aggregate['tokens'],
                'command_integrity':aggregate['command_ok']/aggregate['commands'],
                'number_integrity':aggregate['number_ok']/aggregate['numbers'],
                'lexical_boundary_f1':2*aggregate['boundary_tp']/(aggregate['boundary_pred']+aggregate['boundary_ref']),
                'vocab_utilization':len(token_usage)/m['vocab_actual'],
                'independent_scope_boundary_recall':scopeok/scopetotal,
                'fit_predictive_unigram_bits_per_byte':aggregate['bits']/aggregate['bytes']}
            for k,v in observed.items():assert abs(v-m[k])<1e-10,(e['key'],m['suite'],k)
            probabilities=np.array(list(token_usage.values()))/aggregate['tokens']
            for alpha in (1.,2.,2.5,3.):
                entropy=-float(np.sum(probabilities*np.log2(probabilities))) if alpha==1 else float(np.log2(np.sum(probabilities**alpha))/(1-alpha))
                assert abs(entropy/math.log2(m['vocab_actual'])-m['renyi_efficiency'][str(alpha)])<1e-12
            counts[(e['key'],m['suite'])]=len(rr)
        print('active audit replay',e['key'],flush=True)
    reviews.append({'round':2,'passed':True,'scope':'Every saved public encoder and raw payload, exact IDs/counts, independent KaTeX crossings/pylatexenc boundaries and fit-calibrated probability counts','formula_rows':replayed})
    assert len(metrics)==109 and sum(counts.values())==sum(m['formulas'] for m in metrics)
    for m in metrics:
        assert m['roundtrip_rate']==m['payload_fidelity_rate']==1
        assert len(m['encode_seconds_repeats'])==len(m['decode_seconds_repeats'])==7 and not m['provisional_timing']
    a=load_final('ast_boundary');b=load_final('official_atomic_bpe');assert a.vocab==b.vocab
    # Finalized seven-method selection is retrospective; v8 algorithm/weight
    # freeze was prospective. This distinction is preserved, never overwritten.
    provenance=json.loads((dest/'frozen_provenance.json').read_text())
    assert datetime.fromisoformat(provenance['model_frozen_utc'])<datetime.fromisoformat(provenance['first_v8_score_utc'])
    reviews.append({'round':3,'passed':True,'scope':'Independent paper and count metric recomputation, shared BPE/AST dictionary, same full input suites, seven timing repetitions, prospective model freeze and retrospective seven-method selection disclosed'})
    result={'all_passed':True,'reviews':reviews,'catalog_formulas':len(catalog),'active_methods':7,'configurations':19,
        'metric_records':len(metrics),'formula_rows':replayed,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'selection_is_retrospective':True,'v8_algorithm_freeze_was_pretest':True,
        'checks_are_automated_not_external_certification':True,
        'hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/registry.json',dest/'registry.json',dest/'metrics.json',dest/'per_formula.jsonl.gz',FOLDER/'manifest.json')}}
    (dest/'three_round_audit.json').write_text(json.dumps(result,indent=2)+'\n');rows.close();print('Three active audits passed',replayed,flush=True)


if __name__=='__main__':main()
