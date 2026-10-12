"""Four pre-score reviews and immutable final-intrinsic candidate registry."""
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,subprocess
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.experiment_v7 import load_model,read_external,scope
from math_tokenizer.terminal_search_v8 import reward_config
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.quality_v6 import TARGET_CATEGORIES


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    folder=ROOT/'results/v8/full';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'first_test_score.json').exists():raise RuntimeError('Already scored: refreezing forbidden')
    assert json.loads((ROOT/'results/v8/full_training_completion.json').read_text())['models']==40
    decision=json.loads((ROOT/'results/v8/controlled_terminal/decision.json').read_text())
    entries=[dict(e) for e in json.loads((ROOT/'results/v7/full/registry.json').read_text()) if e['group']!='research']
    for scenario in ('broad','specialized'):
        for capacity in (1024,2048):
            base=ROOT/f'results/v8/{scenario}_{capacity}'
            for name,seed,group in [('ast_boundary',0,'main'),('no_update_v8',0,'control')]+[
                (algorithm,seed,'main') for algorithm in ('parameter_rl','random_search','cem_search') for seed in (17,29,43)]:
                filename='ast_boundary' if name=='ast_boundary' else 'no_update' if name=='no_update_v8' else f'{name}_{seed}'
                entries.append({'name':name,'path':str((base/'models'/(filename+'.json')).relative_to(ROOT)),
                    'scenario':scenario,'capacity':capacity,'kind':'checkpoint','group':group,'seed':seed,
                    'information':'same_fit_and_reward_no_validation_at_training' if name!='ast_boundary' else 'same_official_atomic_dictionary_dev_selected_boundary_weights'})
    for i,e in enumerate(entries):
        e['key']=f"{i}:{e['name']}:{e['capacity']}:{e['scenario']}:{e['seed']}"
        if e.get('path'):e['checkpoint_sha256']=digest(ROOT/e['path'])
    rows=load_split('train')+load_split('dev');byid={r['id']:r for r in rows}
    newgroups={'arxiv:'+r['id'] for r in json.loads((ROOT/'data/releases/v8/arxiv_sources.json').read_text())}
    logs=[]
    for scenario in ('broad','specialized'):
        for capacity in (1024,2048):
            s=scope(scenario,capacity)
            pools=[{byid[i]['split_group'] for i in s[k]} for k in ('fit_ids','reward_ids','dev_ids')]
            assert all(not pools[a]&pools[b] for a,b in ((0,1),(0,2),(1,2)))
            assert not set.union(*pools)&newgroups
            if scenario=='specialized':assert all(not set(byid[i]['subject_categories'])&TARGET_CATEGORIES for k in ('fit_ids','reward_ids','dev_ids') for i in s[k])
            initial_batches=[]
            for seed in (17,29,43):
                policies=[]
                for algorithm in ('parameter_rl','random_search','cem_search'):
                    p=json.loads((ROOT/f'results/v8/{scenario}_{capacity}/{algorithm}_{seed}_policy.json').read_text())
                    assert p['terminal_evaluations']==34 and p['candidate_count']==17 and not p['validation_or_test_used']
                    assert p['fit_index_backend']=='CorpusFitIndex-v8'
                    assert p['config']['training_capacity']==capacity and p['reward_config']==asdict(reward_config())
                    assert p['incumbent_utility']==min(v['utility'] for v in p['evaluations'])
                    assert all(set(ids)<=set(s['fit_ids']) for ids in p['task_fit_ids'])
                    assert all(set(ids)<=set(s['reward_ids']) for ids in p['task_reward_ids'])
                    policies.append(p)
                assert all(p['evaluations'][:5]==policies[0]['evaluations'][:5] for p in policies)
                assert all(p['task_fit_ids']==policies[0]['task_fit_ids'] and p['task_reward_ids']==policies[0]['task_reward_ids'] for p in policies)
    logs.append({'round':1,'passed':True,'scope':'All pools paper-disjoint, new sources excluded, strict domains; RL/random/CEM exact reward settings, same task inputs, initialization and 34 actual terminal scores',
        'caveat':'Equal oracle count, not equal wall time or historical hyperparameter tuning budget'})
    probes=['',r'\frac{x_i^2}{y}+\alpha=12345','α+β=😊',r'\unknownCorpusMacro{a+b}=c',r'\begin{aligned}x&=1\\y&=2\end{aligned}','x+y=z\n','\x00+x=1']
    for e in entries:
        model=load_model(e);vocab=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size
        if e['capacity'] is not None:assert vocab<=e['capacity']
        for text in probes:
            ids=model.encode(text)
            assert model.decode(ids)==text and b''.join(token_bytes(model,i) for i in ids)==text.encode(),e['key']
            assert all(0<=i<vocab for i in ids)
    logs.append({'round':2,'passed':True,'scope':'Every ID counted, raw Unicode/unknown/multiline/empty/NUL probes roundtrip via public encoder and raw payload'})
    for scenario in ('broad','specialized'):
        for capacity in (1024,2048):
            subset=[e for e in entries if e['scenario']==scenario and e['capacity']==capacity]
            def model(name):return load_model(next(e for e in subset if e['name']==name))
            bpe=model('official_atomic_bpe');ast=model('ast_boundary');assert bpe.vocab==ast.vocab==model('ast_lexicographic').vocab
            for e in [v for v in subset if v['name'] in ('parameter_rl','random_search','cem_search','no_update_v8')]:
                m=load_model(e);assert m.atoms==[bytes.fromhex(p) for p in scope(scenario,capacity)['shared_atom_hex']]
    logs.append({'round':3,'passed':True,'scope':'Same BPE/AST raw-byte dictionaries, identical fit-derived atoms and target budget for all new optimizers; brute lattice and numerical score tests required'})
    for e in entries:
        if e.get('path'):assert digest(ROOT/e['path'])==e['checkpoint_sha256']
        a=load_model(e);b=load_model(e)
        assert all(a.encode(text)==b.encode(text) for text in probes)
    for path,revision in [('vendor/pathpiece','1764d9fc91ade17f65edbfe7a54ad4a2c8117e43'),('vendor/tokeval','7d03a61c2e2b4b557b796519141ccb52d144e10f')]:
        assert subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()==revision
        assert not subprocess.check_output(['git','-C',str(ROOT/path),'diff','--name-only'],text=True).strip()
    tests=(ROOT/'results/v8/code_review_round2_tests.log').read_text();assert 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    logs.append({'round':4,'passed':True,'scope':'Frozen checkpoint reload, official pinned baseline/metrics code unmodified, full implementation tests pass; no new external scores read'})
    protocol=json.loads((ROOT/'results/v7/full/evaluation_protocol.json').read_text())
    seal=json.loads((ROOT/'data/releases/v8/seal.json').read_text())
    write(folder/'registry.json',entries)
    protocol.update(frozen_utc=datetime.now(timezone.utc).isoformat(),external_archive_sha256=seal['archive_sha256'],
        registry_sha256=digest(folder/'registry.json'),preregistered_document_sha256=digest(ROOT/'docs/v8_preregistered_protocol.md'),
        paper_bootstrap_resamples=20000,
        primary_pairs_explicit=[['ast_boundary','ast_lexicographic'],['ast_boundary','official_atomic_bpe'],
            ['official_atomic_bpe','v5_parent'],['parameter_rl','random_search'],['parameter_rl','cem_search'],['parameter_rl','no_update_v8']],
        additional_suites=['broad_extreme','stress_test_v6_exposed','exposed_v7_test','exposed_v6_test','exposed_v7_extreme','exposed_v6_extreme'],
        rl_development_decision=decision,
        candidate_promotion='Final downstream roles fixed using dev promotion; terminal RL excluded if controlled gate fails; no test-based hyperparameter selection',
        replacement_tolerances={'compression_relative':.01,'fit_bpb_relative':.02,'integrity_and_scope_absolute':.01,'crossing_absolute':.01,'encode_relative_loss':.1})
    write(folder/'evaluation_protocol.json',protocol)
    write(folder/'preflight_audit.json',{'rounds':logs,'all_passed':True,'registry_sha256':protocol['registry_sha256'],'test_scores_consulted':False})
    # Freeze roles BEFORE scoring, so final ranking cannot choose a lucky seed.
    shortlist={'pretest_frozen_utc':protocol['frozen_utc'],'registry_sha256':protocol['registry_sha256'],
        'rl_in_core':decision['rl_promoted'],'deployment_seed':17,'capacities':[1024,2048],
        'math_core':['ast_boundary','ast_v4_parent','official_atomic_bpe'],
        'equal_capacity_controls':['hf_byte_bpe','sentencepiece_unigram'],
        'structural_ablations':['ast_lexicographic','pathpiece_atomic_vocab'],
        'general_core':['Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base'],
        'extended_general':['Qwen/Qwen3-0.6B','deepseek-ai/DeepSeek-V3','microsoft/Phi-4','mistralai/Mistral-Nemo-Base-2407','tiktoken_gpt2','tiktoken_cl100k_base'],
        'diagnostic_only':['v5_parent','v5_terminal_transfer','parameter_rl','random_search','cem_search','no_update_v8'],
        'different_pretrained_vocabularies_are_deployment_references':True,'downstream_training_executed':False}
    if decision['rl_promoted']:
        shortlist['math_core'].append('parameter_rl');shortlist['diagnostic_only'].remove('parameter_rl')
    write(ROOT/'results/v8/downstream_shortlist_pretest.json',shortlist)
    print('Frozen v8 configurations',len(entries),'RL core',decision['rl_promoted'],flush=True)


if __name__=='__main__':main()
