"""Dev-only exact Pareto decoder exploration with fixed vocab ablations."""
import argparse,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.current_models import load_current
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];dev=[byid[id] for id in scope['dev_ids']]
    sources={'parent_ast':load_current('ast_v4',a.scenario,a.capacity),
             'parent_rl':load_current('v5',a.scenario,a.capacity),
             'official_atomic_union':AtomicBPE.load(folder/'models/atomic_bpe_fit_reward_union.json')}
    metrics=[]
    for source,base in sources.items():
        for penalty in (.001,.25,.75,1.5,4.):
            name=f'ast_balanced_{source}_p{penalty}';model=BalancedAST(base.vocab,penalty)
            model.train_seconds=base.train_seconds;model.training_log=[{'dictionary_source':source,'dictionary_unchanged':True,'soft_penalty':penalty}]
            path=folder/'models'/f'{name}.json';model.save(path)
            m,_=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit));m['dev_selection_utility']=selection_utility(m)
            m.update(dictionary_source=source,phase='development',penalty=penalty,capacity=a.capacity)
            metrics.append(m);(folder/'balanced_ast_dev.json').write_text(json.dumps(metrics,indent=2)+'\n')
            print(a.scenario,a.capacity,name,round(m['document_macro_tokens_per_byte'],6),round(m['independent_tree_crossing'],4),round(m['dev_selection_utility'],6),flush=True)


if __name__=='__main__':main()
