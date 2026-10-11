"""Publish only the two selected families after full checks succeed."""
import hashlib,json,shutil
from math_tokenizer.corpus import ROOT


def main():
    audit=json.loads((ROOT/'results/v6/full/fairness_audit.json').read_text());assert audit['all_passed'] and audit['total_fairness_rounds']>=4
    registry_path=ROOT/'models/current/registry.json';registry=json.loads(registry_path.read_text())
    if registry.get('active_release')!='v6':registry['previous_registry_before_v6']=json.loads(json.dumps(registry))
    paths={family:{scenario:{} for scenario in ('broad','specialized')} for family in ('ast_v4','v5')}
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            out=ROOT/f'results/v6/{scenario}_{capacity}';selection=json.loads((out/'final_selection.json').read_text())
            source=out/'models/ast_v4_selected.json';dest=ROOT/f'models/current/ast_v4/v6-{scenario}-{capacity}.json';shutil.copyfile(source,dest);paths['ast_v4'][scenario][str(capacity)]=str(dest.relative_to(ROOT))
            source=ROOT/selection['selected_v5_paths'][0];dest=ROOT/f'models/current/v5/v6-{scenario}-{capacity}.json';shutil.copyfile(source,dest)
            if source.with_suffix('.tokenizer.json').exists():shutil.copyfile(source.with_suffix('.tokenizer.json'),dest.with_suffix('.tokenizer.json'))
            paths['v5'][scenario][str(capacity)]=str(dest.relative_to(ROOT))
    registry['by_scenario']=paths;registry['ast_v4'].update(paths['ast_v4']['broad']);registry['ast_v4_specialized']=paths['ast_v4']['specialized']['1024'];registry['v5']={s:paths['v5'][s]['1024'] for s in ('broad','specialized')}
    registry['active_release']='v6';registry['corpus_sha256']=audit['corpus_sha256'];registry['test_exposure']='v6 heldout used only after pretest model selection; now exposed for future rounds'
    registry['selection_provenance']={s:f'results/v6/{s}_1024/final_selection.json' for s in ('broad','specialized')}
    registry['active_checkpoint_sha256']={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for family in paths.values() for scenario in family.values() for path in scenario.values()}
    registry_path.write_text(json.dumps(registry,indent=2)+'\n');print('promoted AST-v4 and v5, 1024/2048, broad/specialized',flush=True)


if __name__=='__main__':main()
