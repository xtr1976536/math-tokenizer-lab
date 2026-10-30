"""Study-3 search, with passive lifecycle logging and AFTER-search probes.

Selection uses the same legacy functions as studies 1/2. Truth annotations
never enter ranking, SAT masks or continuation. Log-probability probes use
raw teacher-forced softmax, NOT the EqGPT SAT/80:20 sampling probability.
"""
import copy
import hashlib
import json
import os
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from .author import DICTIONARY, seed_all, tensor_hash, training_namespace
from .cached_batches import cache_batches
from .data import canonical
from .evaluation import TRUTH, classify, expanded_terms
from .experiment import dump
from .latex_codec import structure_to_latex, latex_to_structure, LatexError
from .latex_experiment import RewardEvaluator, select_top, merge_top, sync
from .latex_generation import generate
from .round1_runtime import atomic_save
from .study3_corpus import SETTINGS, LIMIT


def structure_id(gre):
    """Commutative GRE term multiset + existing named-Laplacian expansion.

    Intentionally no arbitrary CAS equivalence, cancellation or product rule.
    Repeated terms remain visible. Raw and reward-normalized IDs are separate.
    """
    terms = canonical([DICTIONARY['id2word'][int(i)] for i in gre])
    expanded = sorted(repr(t) for term in terms for t in expanded_terms(term))
    return hashlib.sha256(json.dumps(expanded).encode()).hexdigest()


def target_gre(case):
    words = []
    for term, _ in TRUTH[case]:
        if words:
            words.append('+')
        words.extend(term.split(','))
    return [5, *[DICTIONARY['word2id'][w] for w in words], 1]


def annotate(row, case):
    """Attach truth only after evaluation; never used by selection."""
    effective = row.get('gre')
    raw = row.get('raw_gre', effective)
    if raw is None and row.get('text') is not None:
        try:
            raw = latex_to_structure(row['text'])
        except (LatexError, ValueError, KeyError, IndexError):
            pass
    for name, gre in [('raw', raw), ('effective', effective)]:
        if gre is not None:
            try:
                row[name + '_structure_id'] = structure_id(gre)
                row[name + '_classification'] = classify(gre, case)['status']
                if name == 'raw':
                    row['diagnostic_raw_gre'] = list(gre)
            except (ValueError, KeyError, IndexError):
                row[name + '_classification'] = 'unclassifiable'


def selection_events(evaluated, current, previous, best):
    """Reconcile actual legacy float32 score deduplication and top10 merge."""
    current_ids = {r['candidate_id'] for r in current}
    best_ids = {r['candidate_id'] for r in best}
    evaluated_scores = {r['candidate_id']: float(torch.tensor(r['reward'], dtype=torch.float32))
                        for r in evaluated if r['status'] == 'evaluated'}
    sorted_rows = sorted(evaluated_scores, key=lambda k: -evaluated_scores[k])
    for row in evaluated:
        uid = row['candidate_id']
        if uid in evaluated_scores:
            score = evaluated_scores[uid]
            row['selection_reward_float32'] = score
            row['raw_score_rank_competition'] = 1 + sum(v > score for v in evaluated_scores.values())
            row['equal_score_candidates'] = [k for k in sorted_rows if k != uid and evaluated_scores[k] == score]
        row['selected_this_cycle'] = uid in current_ids
        row['in_cumulative_top10'] = uid in best_ids
        if row['status'] != 'evaluated':
            reason = row['status']
        elif uid in best_ids:
            reason = 'retained_for_continuation'
        elif uid not in current_ids:
            reason = ('equal_reward_dedup' if any(evaluated_scores.get(r['candidate_id']) == evaluated_scores[uid]
                       for r in current) else 'outside_current_top10_distinct_rewards')
        else:
            reason = ('equal_reward_cumulative_dedup' if any(r['reward'] == row['selection_reward_float32'] for r in previous)
                      else 'outside_cumulative_top10')
        row['selection_reason'] = reason
    events = []
    for row in previous:
        if row['candidate_id'] not in best_ids:
            events.append(dict(candidate_id=row['candidate_id'], event='displaced_from_cumulative_top10'))
    return events


@torch.no_grad()
def teacher_logp(model, adapter, value):
    sequence = adapter.sequence(value)
    # Diagnostic covers BOS/EOS under the SAME training capacity check.
    if len(sequence) > LIMIT:
        return dict(status='outside_training_capacity', tokens=len(sequence))
    inputs = torch.tensor([sequence[:-1]], device=next(model.parameters()).device)
    logits, _ = model(inputs)
    logp = logits.reshape(1, len(sequence) - 1, adapter.vocab_size).log_softmax(-1)
    targets = torch.tensor(sequence[1:], device=inputs.device)
    total = float(logp[0, torch.arange(len(targets), device=inputs.device), targets].sum())
    return dict(status='scored', sum_logp=total, mean_logp=total / len(targets),
                target_tokens=len(targets), probability_kind='raw teacher-forced softmax; excludes SAT/80:20 mixture')


def diagnostics(model, adapter, net, database, case, reference, folder, trace, checkpoints, device):
    """Run after ALL search/continuation; results cannot influence discovery."""
    final_state = copy.deepcopy(model.state_dict())
    model_hash = tensor_hash(final_state)
    net_hash = tensor_hash(net.state_dict())
    probes = {'truth_fixed': target_gre(case)}
    for row in trace:
        if row.get('effective_classification') == 'correct':
            probes.setdefault('generated:' + repr(row['gre']), row['gre'])
    scores = []
    for name, gre in probes.items():
        # Fresh numerical state for each exact expression: isolate order effects
        # from online cache/mask evolution. Never use the search evaluator.
        evaluator = RewardEvaluator(net, database, case, ['x', 't'], device, SETTINGS['epi'])
        result = evaluator.evaluate(dict(stopped=True, text=structure_to_latex(gre), gre=gre))
        scores.append(dict(probe=name, gre=gre, structure_id=structure_id(gre),
                           status=result['status'], reward=result.get('reward')))
    curves = []
    text_probes = {}
    if not reference:
        for row in trace:
            if row.get('raw_classification') == 'correct' and row.get('text'):
                text_probes.setdefault('raw_text:' + hashlib.sha256(row['text'].encode()).hexdigest(), row['text'])
    try:
        for checkpoint in checkpoints:
            state = torch.load(folder / checkpoint['file'], map_location=device, weights_only=True)
            if tensor_hash(state) != checkpoint['sha256']:
                raise ValueError('Diagnostic checkpoint mismatch')
            model.load_state_dict(state)
            model.eval()
            for name, gre in probes.items():
                curves.append(dict(checkpoint=checkpoint['stage'], probe=name,
                                   **teacher_logp(model, adapter, gre if reference else structure_to_latex(gre))))
            for name, text in text_probes.items():
                curves.append(dict(checkpoint=checkpoint['stage'], probe=name,
                                   **teacher_logp(model, adapter, text)))
    finally:
        model.load_state_dict(final_state)
        model.eval()
    if tensor_hash(model.state_dict()) != model_hash or tensor_hash(net.state_dict()) != net_hash:
        raise RuntimeError('Diagnostic modified weights')
    report = dict(truth_not_injected=True, executed_after_search=True,
                  fixed_expression_reward_not_a_function_of_GPT_weights=True,
                  order_effect='author reward anchors first term; same structure can have different ordered rewards',
                  probability_scope='within arm and probe only; not cross-tokenizer or actual SAT sampling probability',
                  scores=scores, raw_text_probes=text_probes, probability_curves=curves)
    dump(folder / 'diagnostics.json', report)
    return report


def mechanism_summary(report, case):
    rows = report['attempts']
    final = report['history'][-1]
    final_classes = [classify(g, case)['status'] for g in final['gre']]
    raw_correct = [r for r in rows if r.get('raw_classification') == 'correct']
    complete_correct = [r for r in raw_correct if r.get('stop_reason') == 'eos']
    effective_correct = [r for r in rows if r.get('effective_classification') == 'correct' and r['status'] == 'evaluated']
    timelines = {}
    for row in rows:
        identity = row.get('effective_structure_id', row.get('raw_structure_id'))
        if identity:
            timelines.setdefault(identity, []).append({k: row.get(k) for k in
                ('candidate_id', 'cycle', 'attempt', 'raw_classification', 'effective_classification', 'status',
                 'stop_reason', 'gre', 'reward', 'selection_reward_float32', 'raw_score_rank_competition',
                 'selected_this_cycle', 'in_cumulative_top10', 'selection_reason')})
    return dict(case=case, top1_correct=bool(final_classes and final_classes[0] == 'correct'),
                top10_correct='correct' in final_classes,
                attempted=len(rows), evaluated=sum(r['status'] == 'evaluated' for r in rows),
                raw_correct_occurrences=len(raw_correct),
                raw_correct_complete=len(complete_correct),
                ever_raw_complete_correct=bool(complete_correct),
                ever_effective_correct_scored=bool(effective_correct),
                generated_scored_correct_but_final_miss=bool(effective_correct) and 'correct' not in final_classes,
                first_cycle_raw_complete_correct=sum(r['cycle'] == 1 for r in complete_correct),
                invalid_status_counts=dict(Counter(r['status'] for r in rows if r['status'] != 'evaluated')),
                final_top1_classification=final_classes[0] if final_classes else 'no_evaluable_candidate',
                search_seconds=report.get('search_seconds'),
                effective_correct_scored_occurrences=len(effective_correct),
                first_cycle_raw_correct=sum(r['cycle'] == 1 for r in raw_correct),
                first_cycle_effective_correct_scored=sum(r['cycle'] == 1 for r in effective_correct),
                first_raw_correct=raw_correct[0]['candidate_id'] if raw_correct else None,
                ever_selected_correct=any(r['selected_this_cycle'] for r in effective_correct),
                ever_retained_correct=any(r['in_cumulative_top10'] for r in effective_correct),
                conditional_retention_occurrences=(sum(r['in_cumulative_top10'] for r in effective_correct)/len(effective_correct)
                                                    if effective_correct else None),
                interpretation='First cycle isolates pretrained generation; later cycles also depend on selection/continuation. Never-generated means not observed within budget, not impossible.',
                structural_equivalence='commutative term/factor permutations plus existing Laplacian expansion; not arbitrary symbolic equivalence',
                structure_timelines=timelines,
                top10_by_cycle=[dict(cycle=h['cycle'], candidates=[dict(candidate_id=uid, rank=i+1, reward=reward,
                      structure_id=structure_id(gre), classification=classify(gre, case)['status'])
                      for i, (uid, reward, gre) in enumerate(zip(h['candidate_ids'], h['rewards'], h['gre']))]) for h in report['history']])


def search(model, adapter, net, database, case, folder, device, reference=False, pretrained_path=None):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'search.json').exists() or (folder / 'cycle_0.pt').exists():
        raise FileExistsError('Partial/completed search exists; use a fresh reviewed output, do not silently replay')
    sync(device)
    started = time.perf_counter()
    evaluator = RewardEvaluator(net, database, case, ['x', 't'], device, SETTINGS['epi'],
                                normalize_ast=not reference, paper_latex=False)
    env = training_namespace(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['continuation_lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best, history, trace, checkpoints = [], [], [], []

    def checkpoint(stage, index):
        file = f'cycle_{index}.pt'
        if index == 0 and pretrained_path is not None:
            os.link(pretrained_path, folder / file)
        else:
            atomic_save(model.state_dict(), folder / file)
        record = dict(stage=stage, file=file, sha256=tensor_hash(model.state_dict()))
        checkpoints.append(record)
        return record

    checkpoint('pretrained_before_search', 0)
    for cycle in range(SETTINGS['cycles']):
        if reference:
            seed_all(SETTINGS['seed'] + cycle)
            evaluated = []
            model.eval()
            for i in range(SETTINGS['attempts']):
                sentence = [5]
                while len(sentence) < LIMIT - 1:
                    with torch.no_grad():
                        value, _ = model.step(sentence, evaluator.mask)
                    sentence.append(int(value))
                    if int(value) == 1:
                        break
                cleaned = evaluator.env['delete_duplicate'](sentence[1:].copy())
                sample = dict(attempt=i, ids=sentence, stopped=True,
                              stop_reason='eos' if sentence[-1] == 1 else 'author_length_limit',
                              raw_gre=sentence, gre=[5] + cleaned,
                              text=structure_to_latex([5] + cleaned))
                evaluated.append(evaluator.evaluate(sample))
        else:
            samples = generate(model, adapter, SETTINGS['attempts'], LIMIT, 2048, 16, SETTINGS['seed'] + cycle, 1.)
            evaluated = [evaluator.evaluate(row) for row in samples]
        for row in evaluated:
            row.update(cycle=cycle + 1, candidate_id=f'{cycle+1}:{row["attempt"]}')
            annotate(row, case)
        previous = copy.deepcopy(best)
        current = select_top(evaluated, evaluator.env)
        best = merge_top(best, current)
        displaced = selection_events(evaluated, current, previous, best)
        trace.extend(evaluated)
        before = tensor_hash(model.state_dict())
        selected = [r['gre'] if reference else structure_to_latex(r['gre']) for r in best]
        sequences = [adapter.sequence(v) for v in selected]
        # Legacy search can render a selected candidate longer than its original
        # token sequence. Fail explicitly; never truncate continuation silently.
        if any(len(s) - 1 > LIMIT for s in sequences):
            raise ValueError('Continuation input exceeds positional capacity')
        losses = []
        if sequences:
            ds = env['MyDataSet'](sequences)
            loader = cache_batches(DataLoader(ds, batch_size=len(best), collate_fn=ds.padding_batch, shuffle=False), device)
            for _ in range(SETTINGS['continuation_epochs']):
                losses.append(env['train_step'](model, loader, optimizer, criterion, 1, print_every=None))
        after = tensor_hash(model.state_dict())
        if best and before == after:
            raise RuntimeError('Continuation did not update weights')
        checkpoint(f'after_continuation_{cycle+1}', cycle + 1)
        record = dict(cycle=cycle+1, before_sha256=before, after_sha256=after,
                      gre=[r['gre'] for r in best], equations=[structure_to_latex(r['gre']) for r in best],
                      rewards=[r['reward'] for r in best], candidate_ids=[r['candidate_id'] for r in best],
                      continuation_sequences=sequences, continuation_losses=losses,
                      displacement_events=displaced,
                      generated_raw_correct=sum(r.get('raw_classification') == 'correct' for r in evaluated),
                      scored_effective_correct=sum(r['status'] == 'evaluated' and r.get('effective_classification') == 'correct' for r in evaluated))
        history.append(record)
        dump(folder / f'cycle_{cycle+1}_trace.json', dict(candidates=evaluated, history=record))
        print('STUDY3_CYCLE', case, adapter.name, cycle+1, flush=True)
    sync(device)
    search_seconds = time.perf_counter() - started
    report = dict(settings=SETTINGS, reference=reference, history=history, attempts=trace,
                  checkpoints=checkpoints, search_seconds=search_seconds,
                  score_cache_hits=evaluator.reward_cache_hits, author_reward_calls=evaluator.author_reward_calls,
                  diagnostics_status='pending_after_search')
    dump(folder / 'search.json', report)
    os.link(folder / checkpoints[-1]['file'], folder / 'searched.pt')
    dump(folder / 'mechanism.json', mechanism_summary(report, case))
    # POST-HOC only: truth and teacher-forcing never enter the search loop.
    diagnostic_started = time.perf_counter()
    diagnostics(model, adapter, net, database, case, reference, folder, trace, checkpoints, device)
    report.update(diagnostics_status='complete', diagnostics_seconds=time.perf_counter()-diagnostic_started)
    dump(folder / 'search.json', report)
    return report
