import math
import pytest
from math_tokenizer.eqgpt.training_protocol import (TrainingSetting,RECIPE,candidates,inner_validation_split,formula_group,choose_setting,protocol_manifest)


def test_permutations_and_repetitions_never_cross_validation():
    sequences=[[5,6,2,i,1] for i in range(7,27)]+[[5,i,2,6,1] for i in range(7,27)]
    tr,va,m=inner_validation_split(sequences)
    assert not {formula_group(s) for s in tr}&{formula_group(s) for s in va}
    assert len(tr)+len(va)==len(sequences) and m['validation_structures']==2
    assert inner_validation_split(sequences)==(tr,va,m)


def test_same_recipe_and_equal_search_are_distinct_and_include_author_setting():
    p=protocol_manifest()
    assert RECIPE==TrainingSetting(100,1e-4)
    assert len(candidates())==6 and RECIPE in candidates()
    assert p['equal_validation_opportunity']['target_and_numeric_observations_used_for_selection'] is False
    assert p['generation']['attempts_per_cycle']==400 and p['generation']['latex_sat'] is False


def records():
    return [dict(epochs=c.epochs,learning_rate=c.learning_rate,status='completed',validation_sequence_nll=10. if c.epochs==100 else 8.) for c in candidates()]


def test_choose_by_validation_only_equal_coverage_and_failure_accounting():
    r=records();r[0]['status']='diverged';r[0]['validation_sequence_nll']=float('nan')
    chosen=choose_setting(r);assert chosen.epochs==300
    with pytest.raises(ValueError):choose_setting(r[:-1])
    with pytest.raises(ValueError):choose_setting(r+[r[0]])
    r[0]['test_recovery_rate']=1.
    with pytest.raises(ValueError):choose_setting(r)


def test_no_claim_of_forced_learning_rate_scaling_or_optimizer_change():
    for lr in [3e-5,1e-4,3e-4]:assert TrainingSetting(learning_rate=lr).optimizer=='Adam'
    with pytest.raises(ValueError):TrainingSetting(optimizer='AdamW')
    with pytest.raises(ValueError):TrainingSetting(learning_rate_schedule='cosine')


def test_unique_formula_validation_is_sequence_nll_not_mean_per_token():
    import torch
    from math_tokenizer.eqgpt.validation_selection import validation_nll
    class Adapter:
        name='eqgpt'
        def sequence(self,s):return s
    class Uniform(torch.nn.Module):
        def forward(self,x):return torch.zeros((x.numel(),57)),None
    first=[5,6,2,9,1];permuted=[5,9,2,6,1];second=[5,6,2,7,3,8,1]
    score=validation_nll(Uniform(),Adapter(),[first,first,permuted,second])
    # The duplicate/permuted first equation counts once: 4 and 6 targets.
    assert abs(score-5*math.log(57))<1e-5


def test_actual_equal_search_runs_three_scratch_trajectories_and_six_candidates(tmp_path):
    import torch
    from math_tokenizer.eqgpt.validation_selection import select_training
    torch.set_num_threads(1)
    class Adapter:
        name='eqgpt'
        def sequence(self,s):return s
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(57,4);self.head=torch.nn.Linear(4,57)
        def forward(self,x):return self.head(self.embedding(x)).reshape(-1,57),None
    calls=[]
    def factory():calls.append(1);return Tiny()
    sequences=[[5,6,2,i,1] for i in range(7,27)]
    selected,report=select_training(factory,Adapter(),sequences,tmp_path)
    assert len(calls)==3 and len(report['records'])==6
    assert all(r['status']=='completed' for r in report['records'])
    assert report['final_refit_completed'] is False
    assert selected in candidates()
    assert max(r['optimizer_steps'] for r in report['records'])==300
