"""Use study-2 weights without changing either study's runtime or results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import model_namespace, seed_all, tensor_hash
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator, search_latex
from math_tokenizer.eqgpt.latex_generation import generate
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from scripts.eqgpt.download_expanded_80_weights import sha

RUN = ROOT / 'experiments/expanded_downstream_80_v1'
CORPUS = ROOT / 'data/formulas/expanded_downstream_80_v1'
ARMS = ('eqgpt_reference', 'ast_bpe_280', 'ast_bpe_315', 'ast_bpe_350')


def load_model(arm, case, stage, device):
    if case == 'new' and stage != 'pretrained':
        raise ValueError('New cases require --stage pretrained')
    manifest = json.loads((RUN / 'weight_manifest.json').read_text())
    relative = (f'shared_pretrain/{arm}/pretrained.pt' if stage == 'pretrained'
                else f'cases/{case}/{arm}/searched.pt')
    record = next(r for r in manifest['files'] if r['path'] == relative)
    path = RUN / relative
    if not path.exists():
        raise FileNotFoundError('Download study-2 weights with scripts/eqgpt/download_expanded_80_weights.py')
    if path.stat().st_size != record['size_bytes'] or sha(path) != record['sha256']:
        raise ValueError(f'Checkpoint file integrity mismatch: {path}')
    protocol = json.loads((CORPUS / 'APPROVED_PROTOCOL.json').read_text())
    if arm == 'eqgpt_reference':
        adapter = Adapter('eqgpt')
    else:
        size = int(arm.removeprefix('ast_bpe_'))
        metadata = json.loads((CORPUS / 'manifest.json').read_text())
        adapter = LatexAdapter('ast_bpe', CompactAlphabet.load(
            CORPUS / metadata['tokenizer_folder'] / f'ast_bpe_{size}.model.json'))
    core = model_namespace(adapter.vocab_size, protocol['position_capacity'], device,
                           n_layers=protocol['layers'])['GPT']().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    expected = (json.loads((RUN / 'shared_pretrain' / arm / 'training.json').read_text())['final_sha256']
                if stage == 'pretrained' else
                json.loads((path.parent / 'search.json').read_text())['history'][-1]['after_sha256'])
    if tensor_hash(state) != expected:
        raise ValueError('Checkpoint tensor integrity mismatch')
    core.load_state_dict(state, strict=True)
    core.eval()
    return core, adapter, protocol


@torch.no_grad()
def generate_reference(core, attempts, limit, device, seed):
    # Generation only: same author GPT.step, initial all-available SAT mask;
    # no reward-side mask updates. This is not a replay of the search trace.
    seed_all(seed)
    mask = torch.ones(57, device=device)
    samples = []
    for index in range(attempts):
        sentence = [5]
        while len(sentence) < limit - 1:
            value, _ = core.step(sentence, mask)
            sentence.append(int(value))
            if int(value) == 1:
                break
        stopped = sentence[-1] == 1
        text = None
        if stopped:
            try:
                text = structure_to_latex(sentence)
            except (ValueError, KeyError, IndexError):
                pass
        samples.append(dict(attempt=index, ids=sentence, stopped=stopped,
                            stop_reason='eos' if stopped else 'author_length_limit', text=text))
    return samples


def main():
    parser = argparse.ArgumentParser(description='Study 2 only: correct 80-position checkpoint/tokenizer pairing')
    parser.add_argument('--case', choices=(*SHARED8_CASES, 'new'), default='Wave_equation')
    parser.add_argument('--arm', choices=ARMS, default='ast_bpe_315')
    parser.add_argument('--stage', choices=('pretrained', 'searched'), default='searched')
    parser.add_argument('--action', choices=('inspect', 'generate', 'score', 'search'), default='inspect')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--attempts', type=int, default=10)
    parser.add_argument('--max-tokens', type=int, default=80)
    parser.add_argument('--seed', type=int, default=525)
    parser.add_argument('--formula')
    parser.add_argument('--output', type=Path, help='New search directory, must not already exist or overlap saved studies')
    args = parser.parse_args()
    if args.attempts < 1 or not 3 <= args.max_tokens <= 80:
        parser.error('attempts must be positive and max-tokens must be in 3..80')
    if args.action == 'score' and not args.formula:
        parser.error('score requires --formula (LaTeX)')
    if args.case == 'new' and args.action in ('score', 'search'):
        parser.error('New data requires its own surrogate; new mode supports inspect/generate only')
    if args.action == 'search':
        if args.stage != 'pretrained' or args.output is None:
            parser.error('search requires --stage pretrained and --output pointing to a new directory')
        output = args.output.resolve()
        protected = [ROOT / name for name in ('data', 'reports', 'eqgpt_original', 'src', 'native', 'scripts', 'tests',
                     'experiments/upstream', 'experiments/downstream', 'experiments/upstream_main', 'experiments/upstream_vocab_size',
                     'experiments/upstream_811', 'experiments/expanded_downstream_80_v1')]
        if output.exists() or any(output == p or p in output.parents or output in p.parents for p in protected):
            parser.error('output must be new and outside all protected study/data/source/report directories')
    torch.set_num_threads(8)
    core, adapter, protocol = load_model(args.arm, args.case, args.stage, args.device)
    if args.action == 'inspect':
        print(json.dumps(dict(study=2, arm=args.arm, stage=args.stage, case=args.case,
                              vocab_size=adapter.vocab_size, positions=protocol['position_capacity'],
                              layers=protocol['layers'], checkpoint_verified=True), indent=2))
    elif args.action == 'generate':
        samples = (generate_reference(core, args.attempts, args.max_tokens, args.device, args.seed)
                   if args.arm == 'eqgpt_reference' else
                   generate(core, adapter, attempts=args.attempts, max_tokens=args.max_tokens, seed=args.seed))
        print(json.dumps(samples, ensure_ascii=False, indent=2))
    else:
        _, net, database, _, _ = load_existing(ROOT / 'data/mlp_cache' / args.case,
                                               args.case, args.device, 525, 10000)
        if args.action == 'score':
            evaluator = RewardEvaluator(net, database, args.case, ['x', 't'], args.device,
                                         .2, normalize_ast=True, paper_latex=True)
            print(json.dumps(evaluator.evaluate(dict(stopped=True, text=args.formula)), ensure_ascii=False, indent=2))
        else:
            output.mkdir(parents=True)
            history = search_latex(core, adapter, net, database, args.case, ['x', 't'], output,
                                   args.device, args.seed, attempts=args.attempts, cycles=3,
                                   finetune_epochs=2, max_tokens=args.max_tokens, batch_size=16,
                                   epi=.2, reference=args.arm == 'eqgpt_reference', gre_limit=args.max_tokens,
                                   normalize_ast=args.arm != 'eqgpt_reference', paper_latex=args.arm != 'eqgpt_reference')
            print(json.dumps(dict(output=str(output), final_equations=history[-1]['equations']), ensure_ascii=False))


if __name__ == '__main__':
    main()
