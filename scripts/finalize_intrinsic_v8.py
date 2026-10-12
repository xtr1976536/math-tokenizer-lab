"""Finalize the pretest shortlist; do not use rankings to select lucky seeds."""
from datetime import datetime,timezone
from pathlib import Path
import json,hashlib
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v7 import load_model


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'results/v8/full'
    assert json.loads((folder/'fairness_audit.json').read_text())['all_passed']
    shortlist=json.loads((ROOT/'results/v8/downstream_shortlist_pretest.json').read_text())
    entries=json.loads((folder/'registry.json').read_text());ranks=json.loads((folder/'rankings.json').read_text())
    statistics=json.loads((folder/'multimetric_paired_effects.json').read_text())
    protocol=json.loads((folder/'evaluation_protocol.json').read_text())
    assert shortlist['registry_sha256']==sha(folder/'registry.json')
    replacements={};ast_replacements={}
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            metrics=ranks['within_scenario'][f'{capacity}/{scenario}']['metrics'];a=metrics['official_atomic_bpe'];b=metrics['v5_parent']
            t=protocol['replacement_tolerances']
            effects={'compression_relative_regression':a['document_macro_tokens_per_byte']/b['document_macro_tokens_per_byte']-1,
                'fit_bpb_relative_regression':a['fit_predictive_unigram_bits_per_byte']/b['fit_predictive_unigram_bits_per_byte']-1,
                'command_recall_loss':b['command_integrity']-a['command_integrity'],
                'number_recall_loss':b['number_integrity']-a['number_integrity'],
                'scope_recall_loss':b['independent_scope_boundary_recall']-a['independent_scope_boundary_recall'],
                'crossing_increase':a['independent_tree_crossing']-b['independent_tree_crossing'],
                'encode_relative_loss':1-a['encode_mib_s']/b['encode_mib_s']}
            passed=effects['compression_relative_regression']<=t['compression_relative'] and effects['fit_bpb_relative_regression']<=t['fit_bpb_relative'] and all(
                effects[k]<=t['integrity_and_scope_absolute'] for k in ('command_recall_loss','number_recall_loss','scope_recall_loss')) and effects['crossing_increase']<=t['crossing_absolute'] and effects['encode_relative_loss']<=t['encode_relative_loss']
            paired=statistics['comparisons'][f'{capacity}/{scenario}/official_atomic_bpe-vs-v5_parent']['metrics']['document_macro_tokens_per_byte']
            effects.update(point_estimate_guards_passed=passed,compression_difference_ci95=paired['ci95'],
                compression_difference_familywise_ci95=paired['familywise_ci95'],
                compression_noninferiority_margin=t['compression_relative']*b['document_macro_tokens_per_byte'],
                compression_ci95_within_margin=paired['ci95'][1]<=t['compression_relative']*b['document_macro_tokens_per_byte'],
                compression_familywise_ci95_within_margin=paired['familywise_ci95'][1]<=t['compression_relative']*b['document_macro_tokens_per_byte'],
                statistical_equivalence_of_all_metrics_established=False)
            replacements[f'{capacity}/{scenario}']=effects
            # Descriptive comparison of the already-frozen production AST
            # against the RL parent; not an added confirmatory hypothesis.
            ast=metrics['ast_boundary']
            e={'compression_relative_regression':ast['document_macro_tokens_per_byte']/b['document_macro_tokens_per_byte']-1,
                'fit_bpb_relative_regression':ast['fit_predictive_unigram_bits_per_byte']/b['fit_predictive_unigram_bits_per_byte']-1,
                'command_recall_loss':b['command_integrity']-ast['command_integrity'],
                'number_recall_loss':b['number_integrity']-ast['number_integrity'],
                'scope_recall_loss':b['independent_scope_boundary_recall']-ast['independent_scope_boundary_recall'],
                'crossing_increase':ast['independent_tree_crossing']-b['independent_tree_crossing'],
                'encode_relative_loss':1-ast['encode_mib_s']/b['encode_mib_s']}
            e['point_estimate_guards_passed']=e['compression_relative_regression']<=t['compression_relative'] and e['fit_bpb_relative_regression']<=t['fit_bpb_relative'] and all(
                e[k]<=t['integrity_and_scope_absolute'] for k in ('command_recall_loss','number_recall_loss','scope_recall_loss')) and e['crossing_increase']<=t['crossing_absolute'] and e['encode_relative_loss']<=t['encode_relative_loss']
            e['comparison_phase']='descriptive after scoring; frozen AST candidate, no new model selection or confirmatory CI'
            ast_replacements[f'{capacity}/{scenario}']=e
    models={}
    all_names=sum([shortlist[k] for k in ('math_core','equal_capacity_controls','structural_ablations','general_core','extended_general','diagnostic_only')],[])
    for name in all_names:
        selected=[e for e in entries if e['name']==name and e['seed'] in (0,shortlist['deployment_seed'])]
        models[name]=selected
        for e in selected:
            if e.get('path'):assert sha(ROOT/e['path'])==e['checkpoint_sha256']
            # Materialize all exact backends via the same registry, avoiding a
            # custom deployment tokenizer that differs from the measured one.
            model=load_model(e);text=r'\frac{\partial u}{\partial t}=\Delta u'
            assert model.decode(model.encode(text))==text
    final={'release':'v8_final_intrinsic','frozen_utc':shortlist['pretest_frozen_utc'],
        'finalized_utc':datetime.now(timezone.utc).isoformat(),'roles_frozen_before_test':True,
        'source_registry_sha256':shortlist['registry_sha256'],'shortlist':shortlist,'models':models,
        'non_RL_replacement_effects':replacements,
        'frozen_AST_vs_RL_parent_descriptive':ast_replacements,
        'RL_removed_from_production':not shortlist['rl_in_core'],
        'legacy_RL_checkpoints_retained_for_negative_result_replay':True,
        'downstream_executed':False,'default_scenario':'broad','recommended_capacities':[1024,2048],
        'matching_rules':'Equal-capacity intrinsic controls; pretrained full vocabulary references have different embedding costs and must be separately disclosed downstream.'}
    # Preserve the exact same-dictionary structural causal controls as optional
    # ablations, avoiding attribution from the old AST's different vocabulary.
    final['models'].update({name:[e for e in entries if e['name']==name] for name in ('hard_tree_atomic_vocab','lexical_shortest')})
    dest=ROOT/'models/final_intrinsic';dest.mkdir(parents=True,exist_ok=True)
    (dest/'registry.json').write_text(json.dumps(final,indent=2)+'\n')
    (ROOT/'results/v8/non_RL_replacement.json').write_text(json.dumps(replacements,indent=2)+'\n')
    (ROOT/'results/v8/AST_non_RL_replacement_descriptive.json').write_text(json.dumps(ast_replacements,indent=2)+'\n')
    print('Final core roles',shortlist['math_core']+shortlist['equal_capacity_controls']+shortlist['general_core'],'RL',shortlist['rl_in_core'],flush=True)


if __name__=='__main__':main()
