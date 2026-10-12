"""One predeclared grid, common BPE dictionary, source-disjoint old dev only."""
import json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility
from math_tokenizer.independent_v7 import additional,source_scopes


def main():
    byid={r['id']:r for r in load_split('train')+load_split('dev')}
    decisions={}
    for scenario in ('broad','specialized'):
        for capacity in (1024,2048):
            old=ROOT/f'results/v7/{scenario}_{capacity}';dest=ROOT/f'results/v8/{scenario}_{capacity}';(dest/'models').mkdir(parents=True,exist_ok=True)
            scope=json.loads((old/'fairness_protocol.json').read_text())
            (dest/'fairness_protocol.json').write_text(json.dumps(scope,indent=2)+'\n')
            fit=[byid[i] for i in scope['fit_ids']];dev=[byid[i] for i in scope['dev_ids']]
            cache={r['id']:source_scopes(r['latex']) for r in dev}
            base=AtomicBPE.load(old/'models/atomic_bpe_fit_reward_union.json')
            baseline,_=measure(base,dev,'official_atomic_bpe',repeats=1)
            extra,_=additional(base,dev,scope_cache=cache,probe_limit=0);baseline.update(extra)
            candidates=[]
            for tree in (0.,.25,.75):
                for boundary in (0.,.05,.15,.3):
                    model=BoundaryAST(base.vocab,tree,boundary)
                    model.train_seconds=base.train_seconds
                    model.training_log=[{'shared_dictionary':'official_atomic_fit_reward_union','development_only':True}]
                    m,_=measure(model,dev,f'boundary_t{tree}_b{boundary}',repeats=1,training_usage=fit_usage(model,fit))
                    extra,_=additional(model,dev,scope_cache=cache,probe_limit=0);m.update(extra)
                    m.update(tree_penalty=tree,boundary_penalty=boundary,
                        selection_loss=selection_utility(m)+.04*(1-m['independent_scope_boundary_recall']),
                        feasible=m['document_macro_tokens_per_byte']<=1.01*baseline['document_macro_tokens_per_byte'] and
                            m['independent_scope_boundary_recall']>=baseline['independent_scope_boundary_recall']-.01)
                    candidates.append(m)
            feasible=[m for m in candidates if m['feasible']]
            best=min(feasible or candidates,key=lambda m:(m['selection_loss'],m['tree_penalty'],m['boundary_penalty']))
            model=BoundaryAST(base.vocab,best['tree_penalty'],best['boundary_penalty']);model.train_seconds=base.train_seconds
            model.training_log=[{'development_selection':best['method'],'feasible':bool(feasible),'same_dictionary_as':'official_atomic_bpe'}]
            model.save(dest/'models/ast_boundary.json')
            decisions[f'{capacity}/{scenario}']={'selected':best,'feasible':bool(feasible),'baseline':baseline,'grid_candidates':len(candidates)}
            (dest/'boundary_development.json').write_text(json.dumps({'candidates':candidates,'decision':decisions[f'{capacity}/{scenario}']},indent=2)+'\n')
            print('boundary dev',scenario,capacity,best['tree_penalty'],best['boundary_penalty'],'feasible',bool(feasible),flush=True)
    (ROOT/'results/v8/boundary_selection.json').write_text(json.dumps(decisions,indent=2)+'\n')


if __name__=='__main__':main()
