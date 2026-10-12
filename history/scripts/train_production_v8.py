"""Standalone non-RL production recipe; never modifies frozen checkpoints.

Fit-derived lexical atoms + original Hugging Face BPE, then fixed source-tree
decoder weights from the preregistered dev selection. No policy/reward optimizer.
"""
import argparse,json,time
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_route_v6 import derive_atoms
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.ast_boundary_v8 import BoundaryAST


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',choices=['broad','specialized'],default='broad')
    parser.add_argument('--capacity',type=int,choices=[1024,2048],default=1024)
    parser.add_argument('--output',type=Path);parser.add_argument('--verify-against-frozen',action='store_true');args=parser.parse_args()
    output=args.output or ROOT/f'results/v8/production_retrain/{args.scenario}_{args.capacity}'
    frozen=ROOT/f'results/v8/{args.scenario}_{args.capacity}/models'
    if output.resolve()==frozen.resolve():raise ValueError('Frozen checkpoint overwrite prohibited')
    output.mkdir(parents=True,exist_ok=True)
    scope=json.loads((ROOT/f'results/v7/{args.scenario}_{args.capacity}/fairness_protocol.json').read_text())
    rows={r['id']:r for r in load_split('train')};fit=[rows[i] for i in scope['fit_ids']];reward=[rows[i] for i in scope['reward_ids']]
    atoms=derive_atoms(fit,args.capacity,384);assert [p.hex() for p in atoms]==scope['shared_atom_hex']
    started=time.perf_counter();bpe=AtomicBPE.train(fit+reward,atoms,args.capacity)
    selected=json.loads((ROOT/'results/v8/boundary_selection.json').read_text())[f'{args.capacity}/{args.scenario}']['selected']
    ast=BoundaryAST(bpe.vocab,selected['tree_penalty'],selected['boundary_penalty']);ast.train_seconds=bpe.train_seconds
    shortest=BalancedAST(bpe.vocab,0.,lexicographic=True);shortest.train_seconds=bpe.train_seconds
    bpe.save(output/'official_atomic_bpe.json');ast.save(output/'ast_boundary.json');shortest.save(output/'ast_lexicographic.json')
    verification=None
    if args.verify_against_frozen:
        old=AtomicBPE.load(ROOT/f'results/v7/{args.scenario}_{args.capacity}/models/atomic_bpe_fit_reward_union.json')
        oldast=BoundaryAST.load(frozen/'ast_boundary.json')
        assert bpe.vocab==old.vocab
        probes=['',r'\frac{x_i^2}{y}+\alpha=12345','α+😊=1',r'\unknown{x}+y=z','\x00+x=1']
        assert all(bpe.encode(t)==old.encode(t) and ast.encode(t)==oldast.encode(t) for t in probes)
        verification=True
    metadata={'scenario':args.scenario,'capacity':args.capacity,'RL_optimizer_used':False,
        'train_or_selection_used_test':False,'fit_ids':scope['fit_ids'],'additional_allowed_training_ids':scope['reward_ids'],
        'raw_vocab_and_probe_ids_match_frozen':verification,'seconds':time.perf_counter()-started,
        'third_party_trainer':'unmodified tokenizers BPE on common lexical alphabet',
        'fixed_decoder_selection':'v8 preregistered development-only weights'}
    (output/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Non-RL production recipe',args.scenario,args.capacity,round(metadata['seconds'],3),'seconds; frozen match',verification,flush=True)


if __name__=='__main__':main()
