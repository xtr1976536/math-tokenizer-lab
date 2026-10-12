import argparse,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.structural_atoms_v7 import train_structural_atoms
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
    byid={r['id']:r for r in load_split('train')+load_split('dev')};fit=[byid[id] for id in scope['fit_ids']];reward=[byid[id] for id in scope['reward_ids']];dev=[byid[id] for id in scope['dev_ids']]
    atoms=[bytes.fromhex(p) for p in scope['shared_atom_hex']];metrics=[]
    for union,training in [('fit',fit),('union',fit+reward)]:
        base=train_structural_atoms(training,atoms,a.capacity);base.save(folder/'models'/f'ast_shared_structure_{union}.json')
        options=[('hard',base)]
        for penalty in (.001,.75,1.5):
            model=BalancedAST(base.vocab,penalty);model.train_seconds=base.train_seconds;model.training_log=base.training_log
            options.append((f'p{penalty}',model))
        for suffix,model in options:
            name=f'ast_shared_structure_{union}_{suffix}';model.save(folder/'models'/f'{name}.json')
            m,_=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit));m.update(dev_selection_utility=selection_utility(m),phase='development',capacity=a.capacity)
            metrics.append(m);print(a.scenario,a.capacity,name,round(m['document_macro_tokens_per_byte'],6),round(m['independent_tree_crossing'],4),round(m['dev_selection_utility'],6),flush=True)
    (folder/'shared_structure_dev.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__=='__main__':main()
