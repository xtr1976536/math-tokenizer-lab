import numpy as np
import pytest
pytest.importorskip('torch')
import sympy as sp
from math_tokenizer.operation_data_stage2 import generate,coefficients,derivative,shift,X,Y
from math_tokenizer.operation_critic_stage2 import OperationCritic,OperationConfig


def test_generated_operation_labels_exact_independent_and_source_disjoint():
    rows,manifest=generate(4,2,2,17)
    assert len(rows)==32 and all(r['verified'] and not r['benchmark_member'] for r in rows)
    groups={k:{r['group'] for r in rows if r['scope']==k} for k in ('fit','dev','extreme')}
    assert not groups['fit']&groups['dev'] and not groups['fit']&groups['extreme']
    for r in rows:
        expr=sp.sympify(r['target_sympy']);expected={(i,j):v for i,j,v in r['target_coefficients']}
        assert coefficients(expr)==expected
    coeff={(2,1):3,(0,0):4}
    assert derivative(coeff,0)=={(1,1):6}
    assert shift(coeff)=={(0,1):3,(1,1):6,(2,1):3,(0,0):4}


def test_operation_critic_loss_padding_and_save_load(tmp_path):
    rows,_=generate(2,1,1,19);fit=[r for r in rows if r['scope']=='fit']
    config=OperationConfig(steps=2,batch=2)
    critic=OperationCritic.fit(fit,17,config);shuffled=OperationCritic.fit(fit,17,config,shuffled=True)
    assert critic.training_byte_targets==shuffled.training_byte_targets
    batch=fit[:2];cuts=[np.ones(len(r['source_latex'].encode())+1,dtype=bool) for r in batch]
    values=critic.score(batch,cuts);path=tmp_path/'teacher.pt';critic.save(path);loaded=OperationCritic.load(path)
    assert loaded.score(batch,cuts)==values
    assert values[0]['bytes']==len(batch[0]['target_latex'].encode())+1
