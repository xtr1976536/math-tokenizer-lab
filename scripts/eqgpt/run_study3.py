"""Default is read-only plan. Training/search require explicit reviewed flag.

Never invoke --execute-reviewed-protocol until the user has reviewed code and
explicitly requested execution in this chat. A flag is an accidental-start
guard, not evidence of user approval by itself.
"""
import argparse
import json
from pathlib import Path

from math_tokenizer.eqgpt.study3_corpus import SETTINGS, ARMS
from math_tokenizer.eqgpt.study3_protocol import (
    DATA, RUN, adapters, read_rows, verify_prepared, safe_run_path, code_digest)
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('plan', 'pretrain', 'search', 'all'), default='plan')
    parser.add_argument('--cohort', choices=('small', 'large', 'both'), default='both')
    parser.add_argument('--case', choices=(*SHARED8_CASES, 'all'), default='all')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='mps')
    parser.add_argument('--output', type=Path, default=RUN)
    parser.add_argument('--execute-reviewed-protocol', action='store_true')
    args = parser.parse_args()
    if args.phase != 'plan' and not args.execute_reviewed_protocol:
        parser.error('Not authorized to start: review code first, then explicitly request execution; no training/search performed')
    out = safe_run_path(args.output)
    meta = verify_prepared(require_checks=args.phase != 'plan')
    cohorts = ('small', 'large') if args.cohort == 'both' else (args.cohort,)
    counts = {c: json.loads((DATA/c/'manifest.json').read_text()) for c in cohorts}
    if args.phase == 'plan':
        print(json.dumps(dict(status='AWAITING_USER_REVIEW', formal_training_started=False,
                              settings=SETTINGS, cohorts=counts, cases=list(SHARED8_CASES), arms=ARMS,
                              planned_pretrains=8, planned_case_arm_searches=64,
                              no_final_pinn_or_independent_coefficient_fitting=True), ensure_ascii=False, indent=2))
        return
    # Heavy numerical imports and mutations occur only beyond the explicit guard.
    import torch
    from math_tokenizer.corpus import ROOT
    from math_tokenizer.eqgpt.author import verify_source, tensor_hash
    from math_tokenizer.eqgpt.models import build_models
    from math_tokenizer.eqgpt.round1_runtime import train_round1
    from math_tokenizer.eqgpt.existing_surrogate import load_existing
    from math_tokenizer.eqgpt.expanded_corpus import sha
    from math_tokenizer.eqgpt.experiment import dump
    from math_tokenizer.eqgpt.study3_search import search
    verify_source()
    torch.set_num_threads(8)
    # Preflight ALL cases before spending any formal training budget.
    caches = {}
    for case in SHARED8_CASES:
        folder = ROOT/'data/mlp_cache'/case
        path = folder/'surrogate_cache.pt'
        expected = json.loads((folder/'surrogate_quality.json').read_text())['cache_sha256']
        if sha(path) != expected:
            raise ValueError('MLP missing/changed; automatic fitting forbidden: '+case)
        caches[case] = expected
    identity = dict(corpus_manifest_sha256=sha(DATA/'manifest.json'), code_sha256=code_digest(),
                    settings=SETTINGS, device=args.device, mlp_cache_sha256=caches)
    out.mkdir(parents=True, exist_ok=True)
    binding = out/'RUN.json'
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('Run identity changed; do not mix results')
    dump(binding, identity)
    for cohort in cohorts:
        aa = adapters(cohort)
        models, initial = build_models(aa, SETTINGS['positions'], SETTINGS['seed'], args.device, n_layers=SETTINGS['layers'])
        initial_hashes = {name: tensor_hash(model.state_dict()) for name, model in models.items()}
        root = out/cohort
        root.mkdir(exist_ok=True)
        if args.phase in ('pretrain', 'all'):
            dump(root/'initial_models.json', initial)
            rows = read_rows(cohort)
            for name, model in models.items():
                encoded = [aa[name].sequence(r['gre_ids'] if name == 'eqgpt_reference' else r['latex']) for r in rows]
                if any(len(s) > SETTINGS['positions'] for s in encoded):
                    raise ValueError('Training capacity violation; never truncate')
                train_round1(model, encoded, root/'shared_pretrain'/name, args.device, SETTINGS['seed'], epochs_override=SETTINGS['epochs'])
            dump(root/'shared_pretrain/complete.json', dict(samples=len(rows), epochs=SETTINGS['epochs'], arms=ARMS))
        if args.phase in ('search', 'all'):
            if not (root/'shared_pretrain/complete.json').exists():
                raise FileNotFoundError('Finish shared pretraining first')
            cases = SHARED8_CASES if args.case == 'all' else (args.case,)
            for case in cases:
                for name, model in models.items():
                    pre = root/'shared_pretrain'/name
                    training = json.loads((pre/'training.json').read_text())
                    weights = torch.load(pre/'pretrained.pt', map_location=args.device, weights_only=True)
                    if (training['epochs'] != SETTINGS['epochs'] or training['samples'] != counts[cohort]['samples']
                            or tensor_hash(weights) != training['final_sha256']
                            or training['initial_sha256'] != initial_hashes[name]):
                        raise ValueError('Pretrained binding mismatch')
                    model.load_state_dict(weights)
                    folder = root/'cases'/case/name
                    if (folder/'complete.json').exists():
                        completed = json.loads((folder/'complete.json').read_text())
                        for filename, digest in completed['files'].items():
                            if sha(folder/filename) != digest:
                                raise ValueError('Completed search integrity mismatch')
                        continue
                    _, net, database, _, _ = load_existing(ROOT/'data/mlp_cache'/case, case, args.device, SETTINGS['seed'], 10000)
                    folder.mkdir(parents=True, exist_ok=True)
                    dump(folder/'SEARCH_INPUT.json', dict(pretrained_sha256=training['final_sha256'], mlp_sha256=caches[case], cohort=cohort))
                    search(model if name == 'eqgpt_reference' else model.core, aa[name], net, database, case,
                           folder, args.device, reference=name == 'eqgpt_reference', pretrained_path=pre/'pretrained.pt')
                    if sha(ROOT/'data/mlp_cache'/case/'surrogate_cache.pt') != caches[case]:
                        raise ValueError('MLP modified')
                    files = {str(p.relative_to(folder)):sha(p) for p in folder.iterdir() if p.is_file()}
                    dump(folder/'complete.json', dict(case=case, cohort=cohort, arm=name, files=files))
                print('STUDY3_CASE_COMPLETE', cohort, case, flush=True)
        del models


if __name__ == '__main__':
    main()
