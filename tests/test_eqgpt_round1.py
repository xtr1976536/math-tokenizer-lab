"""Release audits: frozen data, paired parameters, resume and numerical state."""
import copy,gzip,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from math_tokenizer.eqgpt.author import ROOT,CODE,seed_all,tensor_hash,verify_source
from math_tokenizer.eqgpt.frozen_math import load_frozen,exclusion_reason,handbook_keys,text_digest
from math_tokenizer.eqgpt.latex_experiment import merge_top,select_top,MAIN_CASES
from math_tokenizer.eqgpt.round1_runtime import train_round1,shared_surrogate

BANK=ROOT/'experiments/eqgpt_round1/frozen_math'


def test_independent_frozen_train_manifest_and_no_handbook_overlap():
    adapters,m=load_frozen(BANK)
    with gzip.open(BANK/'math_train.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    assert len(rows)==m['corpus']['accepted']==32270
    assert len({r['id'] for r in rows})==len(rows)
    assert all(r['split']=='train' for r in rows)
    assert text_digest([r['latex'] for r in rows])==m['corpus']['texts_sha256']
    assert len({r['domain'] for r in rows})==13
    keys,norm=handbook_keys()
    assert all(exclusion_reason(r,keys,norm) is None for r in rows)
    for a in adapters.values():
        assert a.vocab_size==1024
        for r in rows[::101]:assert a.decode(a.sequence(r['latex'])[1:-1])==r['latex']
    assert not m['corpus']['source_used_for_transformer_pretraining']


def test_frozen_tamper_fails_before_training(tmp_path):
    import shutil
    shutil.copytree(BANK,tmp_path/'bank')
    p=tmp_path/'bank/tokenizers/ast_bpe.json';p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Frozen asset mismatch'):load_frozen(tmp_path/'bank')


def test_main_is_load_only_fixed_1024_and_shared_original_backbone():
    import inspect
    from math_tokenizer.eqgpt.latex_experiment import run
    from math_tokenizer.eqgpt.models import build_models
    from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
    assert 'train_tokenizers' not in inspect.getsource(run)
    from math_tokenizer.eqgpt.round1_config import LATEX_METHODS,METHODS,SEEDS
    adapters,_=load_frozen(BANK);adapters={n:adapters[n] for n in LATEX_METHODS};adapters['eqgpt_reference']=Adapter('eqgpt')
    assert set(adapters)==set(METHODS) and SEEDS==(525,)
    models,meta=build_models(adapters,512,525)
    assert len({r['backbone_initial_sha256'] for r in meta.values()})==1
    assert {k:r['input_shape'] for k,r in meta.items()}=={k:[57 if k=='eqgpt_reference' else 1024,768] for k in meta}
    assert all(r['pretrained_weights'] is False for r in meta.values())
    assert all(p.requires_grad for m in models.values() for p in m.parameters())
    assert len(MAIN_CASES)==11
    verify_source()


def test_short_top_list_can_grow_and_deduplicates_float32_rewards():
    from math_tokenizer.eqgpt.author import numerical_namespace
    old=[dict(reward=.8,text='a'),dict(reward=.7,text='b')]
    assert [r['text'] for r in merge_top(old,[dict(reward=.6,text='c')])]==['a','b','c']
    env=numerical_namespace('cpu')
    current=select_top([dict(status='evaluated',reward=.123456789,text='x')],env)
    assert current[0]['reward']==float(np.float32(.123456789))
    assert len(merge_top(current,current))==1


class TinyTrain(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(12,8);self.drop=torch.nn.Dropout(.2);self.proj=torch.nn.Linear(8,12)
    def forward(self,x):return self.proj(self.drop(self.emb(x))).reshape(-1,12),None


def test_interrupted_optimizer_and_rng_resume_matches_full_training(tmp_path):
    import math_tokenizer.eqgpt.round1_runtime as runtime
    sequences=[[2,3,4,1],[2,5,1],[2,8,7,6,1]]
    seed_all(12);base=TinyTrain();initial=copy.deepcopy(base.state_dict())
    full=train_round1(base,sequences,tmp_path/'full','cpu',525)
    seed_all(99);resumed=TinyTrain();resumed.load_state_dict(initial)
    original=runtime.atomic_save
    def interrupt(value,path):
        original(value,path)
        if Path(path).name=='training_progress.pt' and value['epoch']==10:raise InterruptedError('simulated process stop')
    with patch.object(runtime,'atomic_save',side_effect=interrupt):
        with pytest.raises(InterruptedError):train_round1(resumed,sequences,tmp_path/'resume','cpu',525)
    fresh=TinyTrain();fresh.load_state_dict(initial)
    resumed_report=train_round1(fresh,sequences,tmp_path/'resume','cpu',525)
    assert tensor_hash(base.state_dict())==tensor_hash(fresh.state_dict())
    assert [r['loss'] for r in full['history']]==[r['loss'] for r in resumed_report['history']]
    assert not (tmp_path/'resume/training_progress.pt').exists()
    replay=TinyTrain();replay.load_state_dict(initial)
    assert train_round1(replay,sequences,tmp_path/'resume','cpu',525)['final_sha256']==full['final_sha256']


def test_poisson_shared_surrogate_cache_preserves_raw_data(tmp_path):
    from math_tokenizer.eqgpt.data import case_environment
    original=case_environment('Possion_x_y',10000,0,'cpu',525)['un'].copy()
    env,net,db,state,first=shared_surrogate(tmp_path,'Possion_x_y',10000,0,525,'cpu',True)
    assert np.array_equal(env['un'],original)
    e2,n2,d2,s2,second=shared_surrogate(tmp_path,'Possion_x_y',10000,0,525,'cpu',True)
    assert first==second and np.array_equal(e2['un'],original)
    assert tensor_hash(state)==tensor_hash(s2) and torch.equal(db,d2)
    assert first['stage']=='B' and first['base_prediction_shape']==list(net(db).shape)
    assert first['shared_across_methods']==['ast_bpe','soft_ast_bpe','eqgpt_reference']


def test_pinn_each_arm_copies_inputs_before_author_sampler():
    from math_tokenizer.eqgpt.experiment import run_pinn
    import inspect
    text=inspect.getsource(run_pinn)
    assert "env['un']=env['un'].copy()" in text
    assert "seed_all(seed)" in text


def test_reference_sat_reward_mask_updated_before_next_generation(tmp_path):
    from math_tokenizer.eqgpt.latex_experiment import search_latex
    from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
    import math_tokenizer.eqgpt.latex_experiment as experiment
    class FakeReference(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.starts=0
        def step(self,sentence,mask):
            if len(sentence)==1:
                assert int(mask[6])==self.starts
                self.starts+=1;return 6,None
            return 1,None
    class FakeReward:
        def __init__(self,*args):
            self.mask=torch.zeros(57);self.env=experiment.numerical_namespace('cpu')
        def evaluate(self,r):
            self.mask[6]+=1
            return dict(r,status='numerical_failure',reward=None)
    model=FakeReference()
    with patch.object(experiment,'RewardEvaluator',FakeReward):
        h=search_latex(model,Adapter('eqgpt'),None,None,'KdV_equation',['x','t'],tmp_path,'cpu',525,
            attempts=4,cycles=1,finetune_epochs=1,reference=True,gre_limit=4)
    assert model.starts==4 and h[0]['attempts']==4


def test_supported_latex_candidates_follow_real_author_continuation_and_pinn(tmp_path):
    from math_tokenizer.eqgpt.experiment import run_pinn
    from math_tokenizer.eqgpt.latex_experiment import search_latex
    from math_tokenizer.eqgpt.models import build_models
    from math_tokenizer.eqgpt.round1_config import LATEX_METHODS
    adapters,_=load_frozen(BANK);adapters={n:adapters[n] for n in LATEX_METHODS}
    models,_=build_models(adapters,512,525)
    env,net,database,state,shared=shared_surrogate(tmp_path/'shared','KdV_equation',10000,0,525,'cpu',True)
    env['_shared_folder']=tmp_path/'shared'
    samples=[dict(attempt=i,ids=[],stopped=True,stop_reason='eos',text='u_t+u_{xx}=0') for i in range(400)]
    all_splits=[]
    for name,bridge in models.items():
        out=tmp_path/name;out.mkdir();net.load_state_dict(state)
        # Explicit supported-output fixture tests downstream plumbing; it is not
        # a generated experimental result and is never stored in the run table.
        with patch('math_tokenizer.eqgpt.latex_experiment.generate',return_value=samples):
            history=search_latex(bridge.core,adapters[name],net,database,'KdV_equation',['x','t'],out,'cpu',525,
                attempts=400,cycles=1,finetune_epochs=1)
        assert history[0]['evaluated']==400 and history[0]['continuation_examples']==1
        assert history[0]['before_sha256']!=history[0]['after_sha256']
        run_pinn(env.copy(),net,state,history,out,'cpu',True,525)
        pinn=json.loads((out/'pinn.json').read_text())
        assert pinn['status']=='completed' and pinn['coefficients']
        all_splits.append(pinn['sampling'])
    assert all_splits[0]==all_splits[1]


def test_compound_target_product_rule_alias_is_held_out(tmp_path):
    from math_tokenizer.eqgpt.data import augment_training
    _,report=augment_training('not-in-handbook',tmp_path,525,[['ut','+','(uux)x']])
    assert '6.12.1' in report['removed_equations']
    assert report['training_formulas']==220


def test_poisson_target_named_laplacian_alias_is_held_out(tmp_path):
    from math_tokenizer.eqgpt.data import augment_training,case_environment
    env=case_environment('Possion_x_y',32,0,'cpu',525)
    _,report=augment_training(env['Delete_equation_name'],tmp_path,525)
    assert {'1.5.20b','1.8.1','1.8.1_2D'}<=set(report['removed_equations'])
    assert report['training_formulas']==218


def test_cached_batches_exact_inputs_optimizer_and_rng_trajectory():
    from math_tokenizer.eqgpt.cached_batches import cache_batches
    from math_tokenizer.eqgpt.author import training_namespace
    from torch.utils.data import DataLoader
    from torch import nn
    env=training_namespace('cpu');sequences=[[2,3,4,1],[2,5,1],[2,8,7,6,1]]
    def loader():
        ds=env['MyDataSet'](sequences)
        return DataLoader(ds,batch_size=2,collate_fn=ds.padding_batch,shuffle=False)
    seed_all(22);initial=copy.deepcopy(TinyTrain().state_dict());results=[]
    for cached in (False,True):
        m=TinyTrain();m.load_state_dict(initial);seed_all(525)
        batches=cache_batches(loader(),'cpu') if cached else loader()
        opt=torch.optim.Adam(m.parameters(),lr=1e-4);losses=[]
        for _ in range(5):losses.append(env['train_step'](m,batches,opt,nn.CrossEntropyLoss(ignore_index=0),1))
        results.append((losses,tensor_hash(m.state_dict()),torch.get_rng_state().clone()))
    assert results[0][:2]==results[1][:2] and torch.equal(results[0][2],results[1][2])
    a,b=loader(),cache_batches(loader(),'cpu')
    assert all(torch.equal(x,u) and torch.equal(y,v) for (x,y),(u,v) in zip(a,b))


def test_reward_cache_matches_uncached_records_masks_and_author_term_cache():
    from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator
    class Analytic(torch.nn.Module):
        def forward(self,xt):return torch.exp(-2*xt[:,1:2])*torch.sin(xt[:,0:1])
    grid=torch.stack([torch.linspace(.2,1,16),torch.linspace(.1,.8,16)],1).requires_grad_()
    cached=RewardEvaluator(Analytic(),grid,'KdV_equation',['x','t'],'cpu',.2)
    original=RewardEvaluator(Analytic(),grid,'KdV_equation',['x','t'],'cpu',.2)
    formulas=['u_t+u_{xx}=0','u_t+u_{xx}=0','u_{yy}+u_t=0','u_t+=0','u_{xx}+u_t=0','u_t+u_{xx}=0']
    for text in formulas:
        original.reward_cache.clear()
        assert cached.evaluate(dict(stopped=True,text=text))==original.evaluate(dict(stopped=True,text=text))
        assert torch.equal(cached.mask,original.mask) and set(cached.cache)==set(original.cache)
    assert cached.reward_cache_hits==2 and cached.author_reward_calls<original.author_reward_calls


def test_reward_cache_does_not_skip_dynamic_numeric_masks_or_memoize_failures():
    from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator
    e=RewardEvaluator(torch.nn.Identity(),torch.zeros(2,2),'KdV_equation',['x','t'],'cpu',.2)
    def mask_term(gre,net,db,terms,mask,variables,epi):
        mask[6]=0;return 0.,terms,mask
    e.env['calculate_reward']=mask_term
    first=e.evaluate(dict(stopped=True,text='u_t+u_{xx}=0'))
    assert first['status']=='evaluated'
    assert e.evaluate(dict(stopped=True,text='u_t+u_{xx}=0'))['status']=='variable_or_numeric_unavailable'
    assert e.author_reward_calls==1 and e.reward_cache_hits==0
    e=RewardEvaluator(torch.nn.Identity(),torch.zeros(2,2),'KdV_equation',['x','t'],'cpu',.2)
    e.env['calculate_reward']=lambda *a:(_ for _ in ()).throw(FloatingPointError('fixture'))
    assert e.evaluate(dict(stopped=True,text='u_t+u_{xx}=0'))['status']=='numerical_failure'
    assert e.evaluate(dict(stopped=True,text='u_t+u_{xx}=0'))['status']=='numerical_failure'
    assert e.author_reward_calls==2 and e.reward_cache_hits==0
