"""Original BPE training with fixed learned atoms; optional boundary decoder."""
import argparse,json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=['broad','specialized'],default='broad');p.add_argument('--capacity',type=int,choices=[1024,2048],default=1024)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (ROOT/'models') in a.output.resolve().parents or a.output.resolve()==(ROOT/'models').resolve():raise ValueError('Do not overwrite frozen active models')
    scope=json.loads((ROOT/f'experiments/current/{a.scenario}_scope.json').read_text());byid={r['id']:r for r in read_catalog()}
    fit=[byid[i] for i in scope['fit_ids']];extra=[byid[i] for i in scope['reward_ids']]
    atoms=derive_atoms(fit,a.capacity,384);assert [v.hex() for v in atoms]==scope['shared_atom_hex']
    model=AtomicBPE.train(fit+extra,atoms,a.capacity);ast=BoundaryAST(model.vocab,.25,.3);ast.train_seconds=model.train_seconds
    a.output.mkdir(parents=True,exist_ok=True);model.save(a.output/'atomic_bpe.json');ast.save(a.output/'boundary_ast.json')
    from math_tokenizer.final_models import load_final
    frozen=load_final('official_atomic_bpe',a.scenario,a.capacity)
    assert frozen.vocab==model.vocab,'Vocabulary differs from frozen IDs: preserve original artifacts for exact experiment replay'
    print('Trained non-RL models; frozen vocabulary matched',a.scenario,a.capacity,flush=True)


if __name__=='__main__':main()
