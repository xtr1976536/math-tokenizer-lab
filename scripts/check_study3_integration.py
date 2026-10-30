"""Read-only forward/MLP checks: no optimizer steps or candidate generation."""
import json
import gc

import torch

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import tensor_hash, verify_source
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.expanded_corpus import sha, gre_terms
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.study3_search import target_gre, teacher_logp
from math_tokenizer.eqgpt.study3_protocol import DATA, REPORT, RUN, read_rows, adapters, verify_prepared
from math_tokenizer.eqgpt.study3_corpus import SETTINGS
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES


def main():
    verify_prepared()
    verify_source()
    torch.set_num_threads(4)
    initial = {}
    forwards = []
    for cohort in ('small', 'large'):
        aa = adapters(cohort)
        models, _ = build_models(aa, SETTINGS['positions'], 525, 'cpu', n_layers=3)
        initial[cohort] = {name:tensor_hash(model.state_dict()) for name,model in models.items()}
        rows = read_rows(cohort)
        for name, model in models.items():
            model.eval()
            encoded = [aa[name].sequence(r['gre_ids'] if name=='eqgpt_reference' else r['latex']) for r in rows]
            longest = max(encoded, key=len)
            before = tensor_hash(model.state_dict())
            with torch.no_grad():
                # Exercise all configured positional embeddings without generating.
                inputs = torch.tensor([[aa[name].bos] * SETTINGS['positions']])
                logits, _ = model(inputs)
                assert torch.isfinite(logits).all()
                assert logits.shape == (SETTINGS['positions'], aa[name].vocab_size)
                actual, _ = model(torch.tensor([longest[:-1]]))
                assert torch.isfinite(actual).all()
            probability = teacher_logp(model, aa[name], target_gre('Wave_equation') if name=='eqgpt_reference'
                                       else structure_to_latex(target_gre('Wave_equation')))
            assert probability['status']=='scored'
            assert tensor_hash(model.state_dict())==before
            forwards.append(dict(cohort=cohort, arm=name, longest_training_sequence=len(longest),
                                 max_position_forward_finite=True, weights_unchanged=True))
        del model, models
        gc.collect()
    assert initial['small']==initial['large'], 'Same seed and vocabulary capacities must give equal initial tensors; token meanings differ'
    numerical = []
    for case in SHARED8_CASES:
        path = ROOT/'data/mlp_cache'/case/'surrogate_cache.pt'
        before = sha(path)
        _, net, database, _, report = load_existing(path.parent, case, 'cpu', 525, 10000)
        state_before = tensor_hash(net.state_dict())
        truth = target_gre(case)
        reversed_truth = [5]
        for i, part in enumerate(reversed(gre_terms(truth))):
            if i:
                reversed_truth.append(2)
            reversed_truth.extend(part)
        reversed_truth.append(1)
        values = []
        for gre in (truth, reversed_truth):
            evaluator = RewardEvaluator(net, database, case, ['x','t'], 'cpu', .2)
            result = evaluator.evaluate(dict(stopped=True, text=structure_to_latex(gre), gre=gre))
            assert result['status']=='evaluated', (case, result)
            values.append(result['reward'])
        assert sha(path)==before and tensor_hash(net.state_dict())==state_before
        numerical.append(dict(case=case, existing_cache_verified=True, no_fit=True,
                              target_order_reward=values[0], reversed_order_reward=values[1],
                              note='Preflight fixed-input check only, not a discovery result'))
        del evaluator, net, database
        gc.collect()
        print('READ_ONLY_MLP_CHECK', case, flush=True)
    assert not RUN.exists()
    result = dict(status='PASS', no_optimizer_steps=True, no_candidate_generation=True,
                  cohorts_identical_initial_tensors_but_different_token_meanings=True, forwards=forwards, numerical=numerical,
                  formal_run_directory_absent=True)
    (REPORT/'integration.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INTEGRATION_PASS', flush=True)


if __name__=='__main__':
    main()
