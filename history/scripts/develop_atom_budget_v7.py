"""Dev-only primitive budget tradeoffs; same atoms in BPE and tree routes."""
import argparse,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];reward=[byid[id] for id in scope['reward_ids']];dev=[byid[id] for id in scope['dev_ids']];metrics=[]
    for budget in (128,256):
        atoms=derive_atoms(fit,a.capacity,budget);base=AtomicBPE.train(fit+reward,atoms,a.capacity)
        for penalty in (None,.75,1.5):
            name=f'atom_budget{budget}_'+('official_bpe' if penalty is None else 'ast_p'+str(penalty))
            if penalty is None:model=base
            else:
                # Distinct primitive spellings can decode to the same bytes.
                # Trie collapses those equivalent payloads; actual lower ID
                # count is reported, never padded with hidden unused entries.
                model=BalancedAST(list(dict.fromkeys(base.vocab)),penalty);model.train_seconds=base.train_seconds
            model.save(folder/'models'/f'{name}.json')
            m,_=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit));m.update(dev_selection_utility=selection_utility(m),phase='development',primitive_budget=budget,capacity=a.capacity)
            m['byte_equivalent_vocabulary_duplicates_collapsed']=len(base.vocab)-len(model.vocab)
            metrics.append(m);(folder/'atom_budget_dev.json').write_text(json.dumps(metrics,indent=2)+'\n')
            print(a.scenario,a.capacity,name,round(m['document_macro_tokens_per_byte'],5),round(m['command_integrity'],4),round(m['independent_tree_crossing'],4),round(m['dev_selection_utility'],5),flush=True)


if __name__=='__main__':main()
