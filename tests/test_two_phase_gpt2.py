import copy
import json
import pytest
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from math_tokenizer.downstream_gpt2.phases import PHASES,PROPOSED,FORMAL,SEEDS,configure_model,parameter_digest,extract_answer,prompt_text,save_checkpoint
from math_tokenizer.downstream_gpt2.train import generate_batch
from math_tokenizer.downstream_gpt2.phase_results import aggregate
from math_tokenizer.downstream_gpt2.results import paired_interval

def test_phase_membership_and_controlled_budget():
    assert PHASES['phase1']['trained_methods']==PROPOSED
    assert PHASES['phase1']['methods']==(*PROPOSED,'gpt2_native')
    assert PHASES['phase2']['methods']==FORMAL
    assert PHASES['phase1']['epochs']==PHASES['phase2']['epochs']==2
    assert SEEDS==(11,)

def test_native_uses_same_readable_prompt_and_fixed_extraction():
    p=prompt_text('integral','x')
    assert p.endswith('\nAnswer: ') and 'x' in p
    assert extract_answer('wrong\n\\frac{x^{2}}{2}')=='wrong'
    assert extract_answer('Answer: f(x) = $x$')=='x'

def test_official_lora_freezes_body_but_trains_new_tied_embedding():
    torch.manual_seed(2)
    base=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_embd=16,n_layer=1,n_head=2,n_positions=32))
    initial=copy.deepcopy(base).eval()
    model=configure_model(base,'lora').eval()
    x=torch.tensor([[1,2,3,4]])
    with torch.no_grad():assert torch.allclose(initial(x).logits,model(x).logits)
    before=parameter_digest(model,True)
    emb=model.get_input_embeddings().weight.detach().clone()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    model.train();model(x,labels=x).loss.backward();optimizer.step()
    assert before==parameter_digest(model,True)
    assert not torch.equal(emb,model.get_input_embeddings().weight)
    assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    assert all('lora_' in n or n.endswith('wte.weight') for n,p in model.named_parameters() if p.requires_grad)

def test_zero_shot_generation_accepts_no_masked_tokens():
    model=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_embd=16,n_layer=1,n_head=2,n_positions=32)).eval()
    class Adapter:
        special_ids={'PAD':0,'EOS':31}
        def decode(self,ids):return ' '.join(map(str,ids))
    assert len(generate_batch(model,Adapter(),[[1,2]],[],'cpu',max_new_tokens=2))==1

def test_lora_and_embedding_checkpoint_roundtrip(tmp_path):
    from peft import PeftModel
    torch.manual_seed(4)
    base=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_embd=16,n_layer=1,n_head=2,n_positions=32))
    untouched=copy.deepcopy(base)
    model=configure_model(base,'lora');x=torch.tensor([[1,2,3,4]])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002)
    model(x,labels=x).loss.backward();optimizer.step();model.eval()
    save_checkpoint(model,tmp_path,'lora')
    restored=PeftModel.from_pretrained(untouched,tmp_path/'lora').eval()
    with torch.no_grad():
        restored.get_input_embeddings().weight.copy_(torch.load(tmp_path/'embeddings.pt',weights_only=True))
        assert torch.allclose(model(x).logits,restored(x).logits,atol=1e-6)

def test_phase_aggregation_rejects_different_question_sets(tmp_path):
    paths={}
    for index,method in enumerate(PHASES['phase1']['methods']):
        folder=tmp_path/str(index);folder.mkdir()
        (folder/'summary.json').write_text(json.dumps({'protocol':'same','coverage_sha256':str(index),
                'suite_id_sha256':{},'smoke_only':False}))
        paths[method]=folder
    with pytest.raises(RuntimeError,match='different inputs'):
        aggregate('phase1',paths,tmp_path/'output',replicates=2)

def test_phase_aggregation_rejects_old_multiseed_outputs(tmp_path):
    paths={}
    for index,method in enumerate(PHASES['phase1']['methods']):
        folder=tmp_path/str(index);folder.mkdir()
        seeds={'zero_shot':{}} if method=='gpt2_native' else {'11':{},'23':{}}
        (folder/'summary.json').write_text(json.dumps({'protocol':'same','coverage_sha256':'same',
                'suite_id_sha256':{},'smoke_only':False,'seeds':seeds}))
        paths[method]=folder
    with pytest.raises(RuntimeError,match='missing seeds'):
        aggregate('phase1',paths,tmp_path/'output',replicates=2)

def test_numeric_variants_share_bootstrap_cluster():
    left=[{'id':str(i),'group':'one-template','task':'integral','method':'prim_fwd','correct':True} for i in range(3)]
    right=[{**r,'correct':False} for r in left]
    assert paired_interval(left,right,'prim_fwd',replicates=10)['groups']==1
