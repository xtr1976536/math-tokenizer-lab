import json
import pytest
from math_tokenizer.corpus import ROOT
from math_tokenizer.final_models import load_final


def test_exactly_two_routes_five_baselines_and_shared_dictionary():
    r=json.loads((ROOT/'models/registry.json').read_text())
    assert r['proposed']==['ast_boundary','official_atomic_bpe'] and len(r['baselines'])==5
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            a=load_final('ast_boundary',scenario,capacity);b=load_final('official_atomic_bpe',scenario,capacity)
            assert a.vocab==b.vocab and len(a.vocab)==capacity
            for t in ['',r'\frac{x_i}{y}+\alpha=1','α+😊=1',r'\unknown{a+b}','\x00+x=1']:
                for m in (a,b):assert m.decode(m.encode(t))==t


def test_history_models_are_not_discoverable_as_current():
    for name in ['v5_parent','parameter_rl','ast_v4_parent','Qwen/Qwen3-0.6B']:
        with pytest.raises(ValueError):load_final(name)
