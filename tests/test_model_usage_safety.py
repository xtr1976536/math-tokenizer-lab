"""Prevent the convenience tool from writing into completed experiments."""
import sys
from pathlib import Path
import pytest
from scripts.eqgpt import use_expanded_model as usage


@pytest.mark.parametrize('output', ['experiments/downstream/new', 'experiments/upstream_811/new',
                                   'experiments/expanded_downstream_80_v1/new', 'data/new', 'reports/new',
                                   'src/new', '.'])
def test_search_rejects_protected_locations_before_model_load(monkeypatch, output):
    monkeypatch.setattr(sys, 'argv', ['use_expanded_model', '--action', 'search',
                                     '--stage', 'pretrained', '--output', str(usage.ROOT / output)])
    monkeypatch.setattr(usage, 'load_model', lambda *a: pytest.fail('Loaded model before rejecting unsafe output'))
    with pytest.raises(SystemExit) as error:
        usage.main()
    assert error.value.code == 2


def test_new_case_cannot_borrow_old_surrogate(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['use_expanded_model', '--case', 'new',
                                     '--stage', 'pretrained', '--action', 'score', '--formula', 'u_t=0'])
    monkeypatch.setattr(usage, 'load_model', lambda *a: pytest.fail('Loaded model before rejecting new-data scoring'))
    with pytest.raises(SystemExit) as error:
        usage.main()
    assert error.value.code == 2
