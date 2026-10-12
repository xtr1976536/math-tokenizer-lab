"""Candidate dictionary learning, exact whole-formula loss; dev only."""
import argparse,json,time
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.dictionary_v7 import candidates,prune_whole_formula
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024)
    p.add_argument('--extra',type=int,default=1024);p.add_argument('--reward-limit',type=int,default=256);p.add_argument('--mode',default='paper')
    p.add_argument('--seed-dictionary',choices=['atomic_bpe','parent_ast'],default='atomic_bpe');p.add_argument('--fit-probe',type=int,default=0)
    p.add_argument('--penalty',type=float);p.add_argument('--atom-budget',type=int,default=384);a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}'
    protocol=json.loads((folder/'fairness_protocol.json').read_text());byid={r['id']:r for r in load_split('train')+load_split('dev')}
    fit=[byid[id] for id in protocol['fit_ids']];reward=[byid[id] for id in protocol['reward_ids'][:a.reward_limit]];dev=[byid[id] for id in protocol['dev_ids']]
    atoms=[bytes.fromhex(p) for p in protocol['shared_atom_hex']]
    if a.atom_budget!=384:
        from math_tokenizer.rl_route_v6 import derive_atoms
        atoms=derive_atoms(fit,a.capacity,a.atom_budget)
    base=AtomicBPE.load(folder/'models/atomic_bpe_fit_only.json')
    if a.seed_dictionary=='parent_ast':
        from math_tokenizer.current_models import load_current
        base=load_current('ast_v4',scenario=a.scenario,capacity=a.capacity)
    if a.fit_probe:
        # Fixed source-balanced fit probes have the same allowed information as
        # all other candidates. Labels remain model-independent source syntax.
        import hashlib
        groups={}
        for row in fit:groups.setdefault(row['split_group'],[]).append(row)
        balanced=[]
        for _,group in sorted(groups.items()):balanced.extend(sorted(group,key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:4])
        reward=reward+balanced[:a.fit_probe]
    start=time.perf_counter();vocab=candidates(fit,atoms,base.vocab,a.extra)
    print('dictionary candidates',len(vocab),'reward',len(reward),flush=True)
    model=prune_whole_formula(vocab,atoms,reward,a.capacity,a.mode,
                             progress=lambda n,c:print('prune remaining',n,'counterfactual calls',c,flush=True),penalty=a.penalty)
    model.train_seconds+=time.perf_counter()-start-model.train_seconds+base.train_seconds
    model.training_log[0]['fit_ids']=protocol['fit_ids']
    name=f'ast_whole_{a.mode}_extra{a.extra}_reward{a.reward_limit}_{a.seed_dictionary}_fit{a.fit_probe}'
    if a.penalty is not None:name+=f'_p{a.penalty}'
    if a.atom_budget!=384:name+=f'_atoms{a.atom_budget}'
    name+='_fixedmask'
    model.save(folder/'models'/f'{name}.json')
    m,rows=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit))
    m['dev_selection_utility']=selection_utility(m);m['phase']='development';m['same_atoms_as_official_baselines']=True
    destination=folder/f'{name}_dev.json';destination.write_text(json.dumps(m,indent=2)+'\n')
    print({k:m[k] for k in ('method','tokens_per_formula','document_macro_tokens_per_byte','command_integrity','independent_tree_crossing','dev_selection_utility','train_seconds')},flush=True)


if __name__=='__main__':main()
