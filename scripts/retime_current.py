"""Uniform isolated timing of current-family peers; original metrics untouched."""
from collections import defaultdict
import json
import statistics
import time
from pathlib import Path

from audit_current import model_for
from math_tokenizer.corpus import ROOT
from math_tokenizer.current_models import load_checkpoint
from math_tokenizer.evaluate import read_jsonl, file_hash


def main():
    specifications = []
    seen = set()
    for run, prefix in [('current_ast_dictionary1024', ''), ('current_v5_critic1024', '')]:
        for metric in json.loads((ROOT / f'results/{run}/{prefix}metrics.json').read_text()):
            path = model_for(run, metric['method'], metric['scenario'], metric['seed'])
            key = str(path), metric['scenario'], metric['suite']
            if key in seen:continue
            seen.add(key); specifications.append((key, metric['method'], metric['seed']))
    suites = {s: read_jsonl(ROOT / f'data/releases/v5/curated/splits/{s}.jsonl')
              for s in ('fresh_paper_test', 'extreme_domain_test', 'synthetic_stress_test')}
    entries = []
    # Rotate peer order across repetitions so all methods receive the same
    # five public per-formula measurements, with no training process running.
    loaded = [(key, name, seed, load_checkpoint(key[0])) for key, name, seed in specifications]
    timings = defaultdict(list)
    for repetition in range(5):
        ordered = loaded[repetition:] + loaded[:repetition]
        if repetition % 2:ordered = list(reversed(ordered))
        for key, name, seed, model in ordered:
            texts = suites[key[2]]
            if hasattr(model, 'clear_cache'):model.clear_cache()
            started = time.perf_counter()
            encoded = [model.encode(r['latex']) for r in texts]
            timings[key].append(time.perf_counter() - started)
    for key, name, seed, model in loaded:
        total_bytes = sum(len(r['latex'].encode()) for r in suites[key[2]])
        duration = statistics.median(timings[key])
        complete_cost = model.train_seconds
        cost_scope = 'saved trainer timing; frozen parents retain original recorded training time'
        if name == 'v5_equal_budget_search':
            control = load_checkpoint(ROOT / f'results/current_rl_1024/checkpoints/{key[1]}-v5-no-update.json')
            complete_cost = control.train_seconds
            cost_scope = 'full no-update matched-budget search job; stored sub-rollout zero is not total training cost'
        entries.append({'checkpoint': str(Path(key[0]).relative_to(ROOT)), 'scenario': key[1], 'suite': key[2],
                        'method': name, 'seed': seed, 'encode_seconds_repeats': timings[key],
                        'encode_seconds_median': duration, 'encode_mib_s': total_bytes / (1024 ** 2 * duration),
                        'training_cost_seconds_annotation': complete_cost, 'training_cost_scope': cost_scope})
    result = {'protocol': 'same raw test bytes, 5 rotated per-formula public encode repeats, official cache cleared each repeat; no parser metrics in clock, no training jobs',
              'load_and_compile_excluded': True, 'original_metric_files_unchanged': True, 'entries': entries}
    (ROOT / 'results/current_timing.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Uniform current timing', len(entries), 'method/seed/scenario/suite entries')


if __name__ == '__main__':
    main()
