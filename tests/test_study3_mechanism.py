import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import training_namespace, numerical_namespace, tensor_hash
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.latex_experiment import select_top, merge_top
from math_tokenizer.eqgpt import study3_search as search


def candidate(uid, reward, gre=None, status='evaluated'):
    gre = gre or search.target_gre('Wave_equation')
    row = dict(candidate_id=uid, attempt=int(uid.split(':')[-1]), cycle=int(uid.split(':')[0]),
               reward=reward, gre=gre, stopped=True, stop_reason='eos', text=structure_to_latex(gre), status=status)
    search.annotate(row, 'Wave_equation')
    return row


def test_structure_id_ignores_commutative_order_but_preserves_redundancy():
    a = search.target_gre('Burgers_equation')
    b = [5, 9, 2, 8, 3, 7, 2, 6, 1]
    assert search.structure_id(a) == search.structure_id(b)
    assert search.structure_id(a) != search.structure_id(a[:-1] + [2, 9, 1])


def test_real_legacy_reward_dedup_and_displacement_are_logged():
    env = numerical_namespace('cpu', 'Wave_equation')
    rows = [candidate('1:0', .8), candidate('1:1', .8)]
    current = select_top(rows, env)
    best = merge_top([], current)
    search.selection_events(rows, current, [], best)
    assert len(best) == 1
    assert sorted(r['selection_reason'] for r in rows) == ['equal_reward_dedup', 'retained_for_continuation']
    previous = [candidate(f'1:{i}', .1 + i*.01) for i in range(10)]
    new = candidate('2:0', .99)
    current = select_top([new], env)
    merged = merge_top(previous, current)
    events = search.selection_events([new], current, previous, merged)
    assert new['in_cumulative_top10'] and len(events) == 1


def test_generated_unscorable_and_never_generated_are_distinguished():
    row = candidate('1:0', None, status='variable_or_numeric_unavailable')
    search.selection_events([row], [], [], [])
    summary = search.mechanism_summary(dict(attempts=[row], history=[dict(cycle=1, gre=[], rewards=[], candidate_ids=[])]), 'Wave_equation')
    assert summary['raw_correct_occurrences'] == 1
    assert summary['effective_correct_scored_occurrences'] == 0
    assert summary['conditional_retention_occurrences'] is None


def test_incomplete_raw_correct_is_not_complete_generation_coverage():
    row = candidate('1:0', .9)
    row['stop_reason']='author_length_limit'
    search.selection_events([row], [row], [], [row])
    summary=search.mechanism_summary(dict(attempts=[row],history=[dict(cycle=1,gre=[row['gre']],rewards=[.9],candidate_ids=['1:0'])]),'Wave_equation')
    assert summary['raw_correct_occurrences']==1
    assert not summary['ever_raw_complete_correct']
    assert summary['ever_effective_correct_scored']
    assert summary['top10_correct']


def test_overall_report_counts_equations_and_conditional_selection():
    spec=importlib.util.spec_from_file_location('report_study3', ROOT/'scripts/report_study3.py')
    report=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows=[]
    for i in range(8):
        rows.append(dict(cohort='small',arm='eqgpt_reference',top1_correct=i<2,top10_correct=i<3,
            ever_raw_complete_correct=i<4,ever_effective_correct_scored=i<4,
            generated_scored_correct_but_final_miss=i==3,evaluated=300,attempted=600,
            first_cycle_raw_complete_correct=2,final_top1_classification='redundant' if i>=2 else 'correct'))
    result=report.aggregate(rows)[0]
    assert result['complete'] and result['evaluated_cases']==8
    assert result['top1_rate']==.25 and result['top10_rate']==3/8
    assert result['complete_generation_coverage']==.5
    assert result['final_selection_success_given_scored_correct']==.75
    assert result['generated_scored_correct_but_final_miss']==1
    assert result['candidate_valid_rate']==.5


class TinyCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(57, dtype=torch.float32)/100)

    def forward(self, ids):
        return self.logits.expand(ids.numel(), -1), None

    def step(self, sentence, mask):
        target = search.target_gre('Wave_equation')
        return target[len(sentence)], None


def test_teacher_forcing_is_read_only_and_matches_manual_softmax():
    model = TinyCore()
    before = tensor_hash(model.state_dict())
    ids = search.target_gre('Wave_equation')
    result = search.teacher_logp(model, Adapter('eqgpt'), ids)
    expected = model.logits.log_softmax(-1)[torch.tensor(ids[1:])].sum().item()
    assert result['sum_logp'] == pytest.approx(expected)
    assert tensor_hash(model.state_dict()) == before


@pytest.mark.parametrize('reference', [True, False])
def test_search_lifecycle_with_mock_evaluator_and_mock_training_only(tmp_path, monkeypatch, reference):
    """No real optimizer step, real PDE search or fitted MLP is run here."""
    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.mask = torch.ones(57)
            self.env = numerical_namespace('cpu', 'Wave_equation')
            self.env['delete_duplicate'] = lambda s: s
            self.reward_cache_hits, self.author_reward_calls = 0, 0

        def evaluate(self, row):
            self.author_reward_calls += 1
            return dict(row, gre=row.get('gre', search.target_gre('Wave_equation')), reward=.9, status='evaluated')

    env = training_namespace('cpu')
    def mock_train(model, *args, **kwargs):
        with torch.no_grad():
            model.logits.add_(torch.arange(57)/10000)
        return .1
    env['train_step'] = mock_train
    monkeypatch.setattr(search, 'RewardEvaluator', Evaluator)
    monkeypatch.setattr(search, 'training_namespace', lambda device: env)
    monkeypatch.setattr(search, 'SETTINGS', dict(search.SETTINGS, cycles=2, attempts=3))
    class TextAdapter:
        name = 'ast_bpe'
        vocab_size = 57
        def sequence(self, value):
            return [2, 6, 7, 1]
    def mock_generate(*args, **kwargs):
        return [dict(attempt=i, stopped=True, stop_reason='eos', ids=[6, 7, 1], text=structure_to_latex(search.target_gre('Wave_equation'))) for i in range(3)]
    monkeypatch.setattr(search, 'generate', mock_generate)
    adapter = Adapter('eqgpt') if reference else TextAdapter()
    core = TinyCore()
    result = search.search(core, adapter, TinyCore(), None, 'Wave_equation', tmp_path/'mock', 'cpu', reference=reference)
    assert len(result['attempts']) == 6 and len(result['checkpoints']) == 3
    assert (tmp_path/'mock/diagnostics.json').exists()
    assert result['diagnostics_status'] == 'complete'
    assert all(r['selected_this_cycle'] == r['in_cumulative_top10'] for r in result['attempts'][:3])
    # Compare the actual legacy search under identical mocked numerical/training
    # operations. Passive annotations must not change choices or final weights.
    from math_tokenizer.eqgpt import latex_experiment as legacy
    monkeypatch.setattr(legacy, 'RewardEvaluator', Evaluator)
    monkeypatch.setattr(legacy, 'training_namespace', lambda device: env)
    monkeypatch.setattr(legacy, 'generate', mock_generate)
    old_core = TinyCore()
    (tmp_path/'legacy').mkdir()
    old_history = legacy.search_latex(old_core, adapter, TinyCore(), None, 'Wave_equation', ['x','t'],
        tmp_path/'legacy', 'cpu', 525, attempts=3, cycles=2, finetune_epochs=2, max_tokens=search.LIMIT,
        reference=reference, gre_limit=search.LIMIT, normalize_ast=not reference)
    assert [h['gre'] for h in old_history] == [h['gre'] for h in result['history']]
    assert [h['rewards'] for h in old_history] == [h['rewards'] for h in result['history']]
    assert tensor_hash(old_core.state_dict()) == tensor_hash(core.state_dict())
    with pytest.raises(FileExistsError):
        search.search(TinyCore(), adapter, TinyCore(), None, 'Wave_equation', tmp_path/'mock', 'cpu', reference=reference)


def test_formal_run_without_explicit_flag_stops_before_any_output():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/eqgpt/run_study3.py'), '--phase', 'all'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'no training/search performed' in result.stderr


def test_output_cannot_point_to_historical_studies():
    from math_tokenizer.eqgpt.study3_protocol import safe_run_path
    with pytest.raises(ValueError, match='independent'):
        safe_run_path(ROOT/'experiments/downstream')
