"""Explicit training and validation rules for the approved main experiment.

A shared recipe and equal-opportunity validation selection answer different
questions. Neither is mislabeled as equal-compute training. This module does
not launch jobs or change the upstream author's Transformer definitions.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,math
from pathlib import Path
from .data import canonical
from .author import DICTIONARY

@dataclass(frozen=True)
class TrainingSetting:
    epochs:int=100
    learning_rate:float=1e-4
    batch_formulas:int=128
    optimizer:str='Adam'
    gradient_clip:float=1.
    learning_rate_schedule:str='constant'
    def __post_init__(self):
        if self.epochs<1 or self.batch_formulas<1:raise ValueError('Positive training counts required')
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError('Positive finite learning rate required')
        if self.optimizer!='Adam' or self.learning_rate_schedule!='constant' or self.gradient_clip!=1.:
            raise ValueError('The main comparison preserves the author optimizer, schedule and clipping')

RECIPE=TrainingSetting()
LEARNING_RATES=(3e-5,1e-4,3e-4)
EPOCH_CHECKPOINTS=(100,300)


def candidates():
    return [TrainingSetting(epochs=e,learning_rate=lr) for lr in LEARNING_RATES for e in EPOCH_CHECKPOINTS]


def formula_group(sequence):
    return repr(canonical([DICTIONARY['id2word'][int(i)] for i in sequence]))


def inner_validation_split(sequences,seed=20261016,fraction=.1):
    """Split already target-excluded training data by original GRE structure.

    All augmented permutations and duplicate formulas remain in one partition.
    The vocabulary was previously frozen from independent allowed math text.
    No downstream numerical observations or discovery scores enter this split.
    """
    if not 0<fraction<1:raise ValueError('Validation fraction must be in (0,1)')
    groups=sorted({formula_group(s) for s in sequences},key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(groups)<3:raise ValueError('At least three original structures are needed')
    validation=set(groups[:max(1,min(len(groups)-1,round(len(groups)*fraction)))])
    train=[s for s in sequences if formula_group(s) not in validation]
    valid=[s for s in sequences if formula_group(s) in validation]
    train_keys={formula_group(s) for s in train};valid_keys={formula_group(s) for s in valid}
    if train_keys & valid_keys:raise RuntimeError('Augmentation-family leakage')
    manifest=dict(seed=seed,validation_fraction=fraction,train_sequences=len(train),validation_sequences=len(valid),
        train_structures=len(train_keys),validation_structures=len(valid_keys),
        train_group_sha256=hashlib.sha256(json.dumps(sorted(train_keys)).encode()).hexdigest(),
        validation_group_sha256=hashlib.sha256(json.dumps(sorted(valid_keys)).encode()).hexdigest(),
        metric='Mean teacher-forced complete-sequence negative log-likelihood per original formula; rank only within a model/tokenizer',
        final_refit='After selection, reinitialize from scratch and train on all allowed target-excluded handbook formulas; final target remains excluded')
    return train,valid,manifest


def choose_setting(records):
    """All six scheduled candidates must have a recorded validation outcome.

    Divergence is recorded as failure, never granted an extra unpaired retry.
    Only validation NLL may select a setting. Test/discovery scores are rejected.
    """
    expected={(x.epochs,x.learning_rate) for x in candidates()};observed=set()
    allowed={'epochs','learning_rate','validation_sequence_nll','status','train_seconds','train_tokens','optimizer_steps'}
    viable=[]
    for r in records:
        if set(r)-allowed:raise ValueError('Unknown selection fields (test/discovery metrics must not enter selection)')
        k=(r['epochs'],r['learning_rate'])
        if k not in expected or k in observed:raise ValueError('Unexpected or repeated hyperparameter candidate')
        observed.add(k)
        if r['status']=='completed' and math.isfinite(r['validation_sequence_nll']):viable.append(r)
    if observed!=expected:raise ValueError('Equal candidate coverage is required before selection')
    if not viable:raise RuntimeError('All validation candidates failed')
    best=min(viable,key=lambda r:(r['validation_sequence_nll'],r['epochs'],r['learning_rate']))
    return TrainingSetting(epochs=best['epochs'],learning_rate=best['learning_rate'])


def protocol_manifest():
    return dict(version='eqgpt-main-training-policy-v1',
        scope='Only the approved main experiment; vocabulary/data/SAT ablations remain deferred',
        arms=['eqgpt_reference','bpe','ast_bpe','soft_ast_bpe'],
        same_recipe=asdict(RECIPE),
        equal_validation_opportunity=dict(candidates=[asdict(c) for c in candidates()],
            learning_rates=list(LEARNING_RATES),epoch_checkpoints=list(EPOCH_CHECKPOINTS),
            execution='Three independent scratch trajectories per method to epoch 300; inspect epoch 100 and 300; do not restart or alter the schedule at 100',
            selection_seed=524,inner_split_seed=20261016,inner_validation_fraction=.1,
            selection_scope='One shared pilot excluding every final target structure; select one global setting per method and freeze it across all final cases/seeds',
            objective='Mean complete-sequence NLL across unique original validation formulas, within each method only',
            target_and_numeric_observations_used_for_selection=False,
            final_fit='Reinitialize for each final case/seed, exclude that case target/equivalents, train for the globally selected epochs/LR, then discover target PDEs',
            retries='Divergence records a failed candidate; any broader grid requires the same expansion for every method'),
        fixed_architecture=dict(layers=6,hidden=768,heads=8,ffn=2048,key_and_value_per_head=64),
        data_exposure='Match raw formula IDs/order and effective batch of formulas; record actual token work and timing',
        compute_fairness='Same recipe and equal candidate counts do not imply equal FLOPs or wall time; report costs without claiming IsoFLOPs',
        future_equal_compute_experiment=False,
        final_seeds=[525,526,527],
        generation=dict(latex_sat=False,attempts_per_cycle=400,refill=False,cycles=5),
        continuation=dict(epochs=5,learning_rate=1e-5,optimizer='Adam',gradient_clip=1),
        additional_tuning_of_sampling_rewards_pinn=False)
