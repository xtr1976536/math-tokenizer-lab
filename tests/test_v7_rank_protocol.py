"""Comparison invariants and familywise/seed-averaged effect estimators."""
import importlib.util
from pathlib import Path
import pytest
from math_tokenizer.corpus import ROOT

spec=importlib.util.spec_from_file_location('v7_summary',ROOT/'scripts/summarize_v7.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def row(id,paper,tokens,bytes=10):return {'id':id,'group':paper,'tokens':tokens,'bytes':bytes}


def test_paired_seed_mean_is_not_formula_pseudoreplication():
    a=[[row('a','p1',2),row('b','p2',4)],[row('a','p1',4),row('b','p2',6)]]
    b=[[row('a','p1',4),row('b','p2',6)]]
    result=module.paired_effect(a,b,2000,14)
    assert result['paper_macro_tokens_per_byte_difference']==pytest.approx(-.1)
    assert result['papers']==2 and result['left_seed_count']==2
    lo,hi=result['ci95'];alo,ahi=result['familywise_95_bonferroni_ci']
    assert alo<=lo and ahi>=hi


def test_paired_effect_refuses_missing_input_ids():
    with pytest.raises(AssertionError):
        module.paired_effect([[row('a','p',2)]],[[row('b','p',2)]])


def test_exact_equal_token_vectors_have_zero_effect_and_intervals():
    a=[[row('a','p1',2),row('b','p2',4)]]
    result=module.paired_effect(a,a,2000,14)
    assert result['ci95']==[0.,0.] and result['familywise_95_bonferroni_ci']==[0.,0.]
