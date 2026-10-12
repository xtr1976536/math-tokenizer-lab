"""Second AST-v4 dev experiment; first preseed experiment remains immutable."""
import argparse
import json
import time

from experiment_current import build_ast, dev_sample, score_dev, write
from math_tokenizer.corpus import ROOT
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.evaluate import bootstrap_difference, file_hash, read_jsonl
from math_tokenizer.measure_v5 import measure
from math_tokenizer.rl_vocab_v5 import RLVocab, internal_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/current_ast_dictionary1024')
    args = p.parse_args()
    out = ROOT / args.output
    if (out / 'protocol.json').exists():
        raise RuntimeError('immutable run; choose a fresh output')
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data/releases/v5/curated'
    suites = ['fresh_paper_test', 'extreme_domain_test', 'synthetic_stress_test']
    protocol = {'family': 'ast_v4', 'capacity': 1024, 'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                'suite_hashes': {s: file_hash(data / f'splits/{s}.jsonl') for s in suites},
                'extras': [64, 128, 256, 512],
                'origin': 'follow-up to preseed dev-to-benchmark regression; datasets already exposed',
                'test_exposure': 'previously exposed repeated benchmark; no pristine-unseen claim',
                'reward_scope': 'same v5 train-internal paper-disjoint reward; no dev/test vocabulary gain',
                'selection_rule': 'same common dev utility; command drop <=.002 vs same-fit base',
                'selection_before_test_load': True}
    write(out / 'protocol.json', protocol)
    ablations, pool, selections = [], {}, {}
    for scenario in ('broad', 'specialized'):
        train = read_jsonl(data / f'splits/{"train" if scenario == "broad" else "transfer_train"}.jsonl')
        dev = dev_sample(read_jsonl(data / f'splits/{"dev" if scenario == "broad" else "transfer_dev"}.jsonl'))
        parent = RLVocab.load(ROOT / f'models/current/v5/{scenario}-base.json')
        fit, reward = internal_split(train, parent.config)
        write(out / f'{scenario}_scope.json', {'fit_ids': parent.fit_ids, 'reward_ids': parent.reward_ids,
                                              'dev_selection_ids': [r['id'] for r in dev]})
        base = build_ast(fit, dict(atom_budget=256, subtree_budget=0), 1024)
        base_stats = score_dev(base, dev)
        pool[(scenario, 0)] = base
        ablations.append({'scenario': scenario, 'extras': 0, 'stats': base_stats, 'train_seconds': base.train_seconds})
        for extras in protocol['extras']:
            model = augment_dictionary(base, fit, reward, extra_subtrees=extras)
            stats = score_dev(model, dev)
            pool[(scenario, extras)] = model
            ablations.append({'scenario': scenario, 'extras': extras, 'stats': stats,
                              'retained_subtrees': model.training_log[-1]['extra_subtrees_retained'],
                              'train_seconds': model.train_seconds})
            write(out / 'dev_ablations.json', ablations)
            print('dictionary dev', scenario, extras, stats, flush=True)
        eligible = [r for r in ablations if r['scenario'] == scenario and r['stats']['command_integrity'] >= base_stats['command_integrity'] - .002]
        winner = min(eligible, key=lambda r: (r['stats']['utility'], r['extras']))
        selections[scenario] = winner
    write(out / 'selection.json', {'selections': selections, 'test_used_for_selection': False,
                                   'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('dictionary selection frozen', selections, flush=True)
    tests = {s: read_jsonl(data / f'splits/{s}.jsonl') for s in suites}
    models = out / 'checkpoints'; models.mkdir()
    summaries, rows, pairwise, buckets = [], [], {}, {}
    for scenario in ('broad', 'specialized'):
        extras = selections[scenario]['extras']
        methods = [('ast_v4_dictionary', pool[(scenario, extras)]), ('ast_v4_same_fit_base', pool[(scenario, 0)])]
        active = ['fresh_paper_test', 'synthetic_stress_test'] if scenario == 'broad' else ['extreme_domain_test', 'synthetic_stress_test']
        for name, model in methods:
            model.save(models / f'{scenario}-{name}.json')
            for suite in active:
                summary, rr = measure(model, tests[suite], name, repeats=5)
                summary.update(scenario=scenario, suite=suite)
                for r in rr:r.update(scenario=scenario, suite=suite)
                summaries.append(summary); rows.extend(rr); buckets[(scenario, suite, name)] = rr
                print('dictionary benchmark', scenario, suite, name, summary['tokens_per_formula'], flush=True)
        for suite in active:
            pairwise[f'{scenario}/{suite}/dictionary-vs-base'] = bootstrap_difference(buckets[(scenario, suite, 'ast_v4_dictionary')], buckets[(scenario, suite, 'ast_v4_same_fit_base')])
    write(out / 'metrics.json', summaries)
    (out / 'per_formula.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write(out / 'pairwise.json', pairwise)
    write(out / 'manifest.json', {'protocol_sha256': file_hash(out / 'protocol.json'),
                                'selection_sha256': file_hash(out / 'selection.json'),
                                'checkpoints': {p.name: file_hash(p) for p in models.iterdir()},
                                'runs': len(summaries), 'records': len(rows)})


if __name__ == '__main__':
    main()
