"""Four preflight reviews, no external model scores, frozen formal experiment."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,subprocess
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.current_models import load_current
from math_tokenizer.experiment_v6 import pretrained_entries
from math_tokenizer.experiment_v7 import load_model,scope
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.metrics_v2 import token_bytes


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    folder=ROOT/'results/v7/full';folder.mkdir(exist_ok=True)
    if (folder/'first_test_score.json').exists():raise ValueError('External test already exposed; cannot refreeze choices')
    selection=json.loads((ROOT/'results/v7/development_selection.json').read_text())['selection']
    entries=[]
    def add(name,path,scenario,capacity,kind='checkpoint',group='main',seed=0,information='fit_reward_allowed',**extra):
        entries.append({'name':name,'path':str(path.relative_to(ROOT)),'scenario':scenario,'capacity':capacity,
            'kind':kind,'group':group,'seed':seed,'information':information,**extra})
    parent_registry=json.loads((ROOT/'models/current/registry.json').read_text())
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            local=ROOT/f'results/v7/{scenario}_{capacity}';models=local/'models';s=scope(scenario,capacity)
            for family in ('ast_v4','v5'):
                path=ROOT/parent_registry['by_scenario'][family][scenario][str(capacity)]
                add(family+'_parent',path,scenario,capacity,information='fit_with_reward_selection')
            add('ast_lexicographic',models/'ast_lexicographic.json',scenario,capacity,
                information='same_official_atomic_union_dictionary_no_reward_at_inference')
            for name,filename,kind in [('official_atomic_bpe','atomic_bpe_fit_reward_union.json','checkpoint'),
                ('hf_byte_bpe','hf_byte_fit_reward_union.json','hf'),
                ('pathpiece_atomic_vocab','pathpiece_atomic_fit_reward_union.vocab','pathpiece')]:
                train=next(m['train_seconds'] for m in json.loads((local/'closest_baseline_dev.json').read_text())
                           if m['method']=={'official_atomic_bpe':'atomic_bpe_fit_reward_union','hf_byte_bpe':'hf_byte_fit_reward_union','pathpiece_atomic_vocab':'pathpiece_atomic_vocab_fit_reward_union'}[name])
                add(name,models/filename,scenario,capacity,kind,train_seconds=train)
            from tokenizers import Tokenizer
            from math_tokenizer.rl_vocab_v5 import bytealphabet
            hf=Tokenizer.from_file(str(models/'hf_byte_fit_reward_union.json'));reverse={c:b for b,c in bytealphabet().items()}
            rawpath=models/'pathpiece_byte_union.vocab'
            rawpieces=[bytes(reverse[c] for c in hf.id_to_token(i)) for i in range(hf.get_vocab_size())]
            rawpath.write_text(''.join(p.hex()+'\n' for p in rawpieces if p!=b'\x00'))
            add('pathpiece_byte_vocab',rawpath,scenario,capacity,'pathpiece',train_seconds=next(e['train_seconds'] for e in entries if e['name']=='hf_byte_bpe' and e['scenario']==scenario and e['capacity']==capacity))
            for e in json.loads((local/'classical_union_registry.json').read_text()):
                add(e['name'].removesuffix('_union'),ROOT/e['path'],scenario,capacity,e['kind'],train_seconds=e['train_seconds'])
            # Explicit structural hard/no-tree same-dictionary ablations.
            for name,filename in [('lexical_shortest','lexical_shortest_fit_reward_union.json'),
                ('hard_tree_atomic_vocab','native_tree_shortest_fit_reward_union.json')]:
                add(name,models/filename,scenario,capacity,group='control')
            for name,file in [('v5_frequency','v5_structural_frequency.json'),('v5_reward_greedy','v5_structural_reward_greedy.json')]:
                add(name,ROOT/f'results/v6/{scenario}_{capacity}/models/{file}',scenario,capacity,group='control')
            selected=selection[f'{capacity}/{scenario}']
            for e in selected['rl_future_credit_candidates']:
                seed=int(e['name'].split('_s')[1].split('_')[0]) if '_s' in e['name'] and e['name'].startswith('v5_transfer') else 17
                add('v5_terminal_transfer',ROOT/e['path'],scenario,capacity,seed=seed)
            control=(models/'v5_transfer_tree0.5_e4_s17_no_update.json') if capacity==1024 else models/('v5_cross_task_no_update_strict.json' if scenario=='specialized' else 'v5_cross_task_no_update.json')
            add('v5_terminal_no_update',control,scenario,capacity,group='control')
            if capacity==1024:
                add('v5_policy_search',models/'v5_policy_search_control.json',scenario,capacity,group='control')
                if scenario=='broad':
                    for name in ('v5_transfer_tree1.5_e6_s17_learned','ast_whole_paper_extra256_reward512_parent_ast_fit256_atoms256_fixedmask'):
                        path=models/(name+'.json')
                        if path.exists():add(name,path,scenario,capacity,group='research')
    for entry in pretrained_entries():
        entries.append(dict(entry,group='fixed_pretrained_reference',information='external_pretraining_unknown',train_seconds=0))
    for i,e in enumerate(entries):
        e['key']=f"{i}:{e['name']}:{e['capacity']}:{e['scenario']}:{e['seed']}"
        if e.get('path'):e['checkpoint_sha256']=digest(ROOT/e['path'])
    # Review 1: common scopes, no unseen paper/model leakage, immutable seal.
    seal=json.loads((ROOT/'data/releases/v7/seal.json').read_text());sources=json.loads((ROOT/'data/releases/v7/arxiv_sources.json').read_text())
    newgroups={'arxiv:'+r['id'] for r in sources};oldrows=load_split('train')+load_split('dev');byid={r['id']:r for r in oldrows}
    for c in (1024,2048):
        for scenario in ('broad','specialized'):
            s=scope(scenario,c)
            assert not {byid[id]['source_group'] for id in s['fit_ids']+s['reward_ids']+s['dev_ids']}&newgroups
            assert not set(s['fit_ids'])&set(s['reward_ids'])
            assert not {byid[id]['split_group'] for id in s['fit_ids']}&{byid[id]['split_group'] for id in s['reward_ids']}
            if scenario=='specialized':
                from math_tokenizer.quality_v6 import TARGET_CATEGORIES
                assert all(byid[id]['source_repo']=='arxiv.org' and not set(byid[id]['subject_categories'])&TARGET_CATEGORIES for id in s['fit_ids']+s['reward_ids']+s['dev_ids'])
    logs=[{'round':1,'scope':'Identical scenario pools, fit/reward paper disjoint, unseen sources excluded from all training and dev, strict specialized categories','passed':True}]
    probes=['',r'\frac{x_i^2}{y}+\alpha=12345','α+β=😊',r'\unknownCorpusMacro{a+b}=c',r'\begin{aligned}x&=1\\y&=2\end{aligned}','x+y=z\n']
    for e in entries:
        model=load_model(e);vocab=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size
        if e['capacity'] is not None:assert vocab<=e['capacity'],e['key']
        for text in probes:
            ids=model.encode(text)
            assert model.decode(ids)==text and b''.join(token_bytes(model,id) for id in ids)==text.encode(),e['key']
            assert all(0<=id<vocab for id in ids)
    logs.append({'round':2,'scope':'Every reserved ID counted, Unicode/unknown/empty/multiline probes lossless in every official and custom public interface','passed':True,'configurations':len(entries)})
    for c in (1024,2048):
        for scenario in ('broad','specialized'):
            local=ROOT/f'results/v7/{scenario}_{c}';bpe=AtomicBPE.load(local/'models/atomic_bpe_fit_reward_union.json');ast=BalancedAST.load(local/'models/ast_lexicographic.json')
            assert bpe.vocab==ast.vocab
            for text in probes:assert len(ast.encode(text))<=len(bpe.encode(text))
            for e in [x for x in entries if x['name']=='v5_terminal_transfer' and x['scenario']==scenario and x['capacity']==c]:
                model=load_model(e);assert model.atoms==[bytes.fromhex(p) for p in scope(scenario,c)['shared_atom_hex']]
                # Cross-domain policy models must explicitly qualify for strict OOD.
                if scenario=='specialized' and c==2048:assert model.logs[0]['strict_ood_eligible']
    logs.append({'round':3,'scope':'Identical byte dictionaries for BPE/lattice, same atoms for RL/control, strict OOD policy provenance, all baseline fit+reward union ranges available','passed':True,
                 'limitation':'Common allowed information, not equal tuning count or equal wall-clock training; all search budgets disclosed.'})
    for e in entries:
        if e.get('path'):assert digest(ROOT/e['path'])==e['checkpoint_sha256']
        a=load_model(e);b=load_model(e)
        for text in probes:assert a.encode(text)==b.encode(text)
    for path,revision in [('vendor/pathpiece','1764d9fc91ade17f65edbfe7a54ad4a2c8117e43'),('vendor/tokeval','7d03a61c2e2b4b557b796519141ccb52d144e10f')]:
        assert subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()==revision
        assert not subprocess.check_output(['git','-C',str(ROOT/path),'diff','--name-only'],text=True).strip()
    logs.append({'round':4,'scope':'Frozen checkpoints reloaded deterministically; official packages and author metric/baseline code pinned and unmodified; no external tokenizer scores consulted','passed':True})
    write(folder/'registry.json',entries)
    weights=json.loads((ROOT/'results/v6/evaluation_protocol.json').read_text())['comprehensive_weights']
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'external_archive_sha256':seal['archive_sha256'],
        'registry_sha256':digest(folder/'registry.json'),'previous_results_used_for_development':True,'new_external_scores_used':False,
        'capacities':[1024,2048],'main_scenarios':{'broad':'external_test','specialized':'extreme_domain_test'},
        'additional_suites':['broad_extreme','stress_test_v6_exposed'],'all_raw_strings':True,'per_model_filter_or_truncation':False,
        'fit_calibration':'Same fit IDs by scenario, actual vocab and add-half readout; mature models separately calibrated for broad and specialized.',
        'additional_smoothing':[.01,.1,.5,1.],'independent_parser':'pylatexenc2.11 strict groups (source syntax only, not expert semantic gold)',
        'edit_probe_limit':256,'edit_probe_sampling':'Common first256 immutable suite records, same-byte lexical-symbol local edit, outside16bytes boundary change',
        'timing_repeats':7,'timing':'All configurations shuffled/interleaved, full public encode/decode including frontend, caches clear, no training/network tasks.',
        'comprehensive_weights':weights,'rank_scope':'Practical deployed systems on same inputs; matched dictionaries/atoms and causal controls reported separately.',
        'scenario_weights':{'broad':.5,'specialized':.5},'paper_bootstrap_resamples':2000,
        'primary_pairs':['ast_lexicographic-vs-official_atomic_bpe','ast_lexicographic-vs-lexical_shortest','v5_terminal_transfer-vs-v5_terminal_no_update','v5_terminal_transfer-vs-v5_policy_search'],
        'multiple_comparisons':'Primary paired bootstrap intervals additionally adjusted by Bonferroni familywise 5%; raw CIs descriptive.',
        'candidate_promotion':'Experimental branches only; active v6 registry kept until strong multi-metric gain independently established.',
        'downstream':False,'expert_gold_or_multihardware_claim':False,
        'scope_limitation':'New source-topic holdout, not guaranteed disjoint mathematical concepts; external pretraining contamination unknown.'}
    write(folder/'evaluation_protocol.json',protocol);write(folder/'preflight_audit.json',{'rounds':logs,'all_passed':True,
        'registry_sha256':protocol['registry_sha256'],'test_scores_consulted':False})
    print('freeze configurations',len(entries),flush=True)


if __name__=='__main__':main()
