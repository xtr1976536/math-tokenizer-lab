"""Author EqGPT pipeline: shared data/surrogate, three scratch-trained models."""
from __future__ import annotations
import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .author import (ROOT,CODE,ARCHITECTURE,DICTIONARY,definitions,namespace,
                     verify_source,seed_all,tensor_hash,training_namespace,numerical_namespace)
from .data import (CASES,case_environment,build_surrogate,augment_training,
                   runtime_definitions,working_directory,canonical)
from .tokenizer_adapters import train_adapters,common_positions,load_adapters
from .models import build_models


def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def numeric(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=numeric)+'\n')


def save_numeric(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.save(path,value)


def prepare_surrogate(folder, equation, observations, noise, seed, device, smoke):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    env=case_environment(equation,observations,noise,device,seed)
    if smoke:
        env['choose']=min(32,observations);env['choose_validate']=16
    net=build_surrogate(env,device,seed)
    env['Net']=net
    runtime_definitions('surrogate_model.py',env,
                        {'Generate_meta_data','Generate_meta_data_H','Generate_meta_data_2D',
                         'get_no_boundary','train_surrogate_model','get_meta'},device,smoke)
    # Repair only the erroneous four-value unpack in the Poisson call site.
    import ast
    from .author import source_tree
    node=next(n for n in source_tree('surrogate_model.py').body if isinstance(n,ast.FunctionDef) and n.name=='get_meta')
    class Unpack(ast.NodeTransformer):
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Tuple) and len(n.targets[0].elts)==4:
                n.targets[0].elts=n.targets[0].elts[:2]
            return self.generic_visit(n)
    node=Unpack().visit(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(CODE/'surrogate_model.py'),'exec'),env)
    if smoke:
        # Only smoke: retain the exact grid generator with a smaller nx/nt.
        original=env['Generate_meta_data']
        def small_meta(*args,**kwargs):return original(*args,**dict(kwargs,nx=5,nt=5))
        env['Generate_meta_data']=small_meta
    with working_directory(folder):
        # Author irregular-region sampling shuffles an array view in place.
        # Preserve the shared raw observations for later paired PINN stages.
        env['train_surrogate_model'](net,env['un'].copy())
        net,database=env['get_meta'](net)
    # Derivatives need gradients; evaluation mode only changes train/eval state.
    initial_state={k:v.detach().clone() for k,v in net.state_dict().items()}
    shared=dict(equation=equation,seed=seed,observations=env['choose'],noise_percent=noise,
                surrogate_sha256=tensor_hash(initial_state),
                raw_noisy_data_sha256=hashlib.sha256(env['un'].tobytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(database.detach().cpu().numpy().tobytes()).hexdigest(),
                metadata_points=len(database),activation=env['Activation_function'],
                surrogate_iterations=2 if smoke else (100000 if equation=='Laplacian_shuttle' else 50000),
                target_name=env['Delete_equation_name'])
    dump(folder/'shared_manifest.json',shared)
    return env,net,database,initial_state,shared


def train_model(model,adapter,gre_sequences,epochs,device,seed,folder,smoke=False):
    seed_all(seed)
    sequences=[adapter.sequence(s) for s in gre_sequences]
    positions=model.core.decoder.pos_emb.num_embeddings
    if any(len(s)>positions for s in sequences):raise ValueError('No tokenizer-specific truncation is permitted')
    env=training_namespace(device)
    dataset=env['MyDataSet'](sequences)
    loader=DataLoader(dataset,batch_size=2 if smoke else 128,collate_fn=dataset.padding_batch,shuffle=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    criterion=nn.CrossEntropyLoss(ignore_index=0).to(device)
    initial=tensor_hash(model.state_dict());records=[]
    for epoch in range(epochs):
        loss=env['train_step'](model,loader,optimizer,criterion,1,print_every=None)
        if not math.isfinite(loss):raise FloatingPointError('Training loss is not finite')
        records.append(dict(epoch=epoch+1,loss=loss))
        print('pretrain',adapter.name,epoch+1,loss,flush=True)
    final=tensor_hash(model.state_dict())
    if initial==final:raise RuntimeError('No model parameter changed')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),folder/'pretrained.pt')
    result=dict(initial_sha256=initial,final_sha256=final,all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                epochs=epochs,batch_size=2 if smoke else 128,optimizer='Adam',lr=1e-4,clip=1,
                raw_sequence_order_sha256=hashlib.sha256(json.dumps(gre_sequences).encode()).hexdigest(),
                samples=len(sequences),max_model_tokens=max(map(len,sequences)),history=records)
    dump(folder/'training.json',result)
    return result


def greedy_summary(sentence):
    return ''.join(DICTIONARY['id2word'][int(i)] for i in sentence if int(i) not in (0,1,5))


def search(model,adapter,net,database,equation,variables,folder,device,seed,
           cycles=5,samples=400,finetune_epochs=5,gre_limit=50,epi=.2):
    """Execute the author's original main-block AST with an adapted GPT factory."""
    import ast
    from .author import source_tree
    env=numerical_namespace(device,equation)
    env.update(Net=net,get_meta=lambda network:(network,database),max_pos=gre_limit,
        canonical_PDEs=['Wave_equation','Burgers_equation','KdV_equation','Chaffee_Infante_equation',
                       'KG_equation','Allen_Cahn','Convection_diffusion_equation','PDE_divide',
                       'Eq_6_2_12','PDE_compound'],
        irrgular_2D_regions=['Possion_x_y','Laplacian_smile','Laplacian_EITech'],
        choose=0,noise_level=0,noise_type='Gaussian',Equation_name=equation)
    # Its code calls GPT twice. Returning the SAME model for the second call
    # repairs the optimizer binding while preserving the original loop.
    env['GPT']=lambda:model
    original_dataset=env['MyDataSet']
    env['MyDataSet']=lambda sequences:original_dataset([adapter.sequence(s) for s in sequences])
    original_top=env['find_min_no_repeat']
    env['find_min_no_repeat']=lambda rewards:original_top(rewards,samples=min(400,len(rewards)))
    updates=[];candidates=[]
    original_step=env['train_step']
    def checked_train_step(current,loader,optimizer,criterion,*args,**kwargs):
        ids={id(p) for g in optimizer.param_groups for p in g['params']}
        assert ids=={id(p) for p in current.parameters()}
        before=tensor_hash(current.state_dict())
        loss=original_step(current,loader,optimizer,criterion,*args,**kwargs)
        after=tensor_hash(current.state_dict())
        if before==after:raise RuntimeError('Continuation did not update sampled model')
        updates.append(dict(before_sha256=before,after_sha256=after,loss=float(loss)))
        return loss
    env['train_step']=checked_train_step
    original_reward=env['calculate_reward']
    def reward_trace(sentence,*args,**kwargs):
        raw=sentence.copy();value=original_reward(sentence,*args,**kwargs)
        if not np.isfinite(value[0]):raise FloatingPointError('Nonfinite author reward')
        candidates.append(dict(index=len(candidates),gre=raw,reward=float(value[0])))
        return value
    env['calculate_reward']=reward_trace
    main_block=next(n for n in source_tree('continue_train_GPT.py').body
                    if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))
    class Settings(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            for target in node.targets:
                if not isinstance(target,ast.Name):continue
                if target.id=='device':node.value=ast.parse(repr(str(device)),mode='eval').body
                if cycles!=5 and target.id=='optimize_epoch':node.value=ast.Constant(value=cycles)
                if samples<400 and target.id=='samples':node.value=ast.Constant(value=samples)
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if finetune_epochs!=5 and isinstance(node.func,ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==5:
                node.args[0]=ast.Constant(value=finetune_epochs)
            return node
    module=ast.fix_missing_locations(Settings().visit(ast.Module(body=main_block.body,type_ignores=[])))
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'gpt_model').mkdir(exist_ok=True)
    torch.save(model.state_dict(),folder/f'gpt_model/PDEGPT_{equation}.pt')
    seed_all(seed)
    with working_directory(folder):exec(compile(module,str(CODE/'continue_train_GPT.py'),'exec'),env)
    history=[]
    for i,(sentences,equations) in enumerate(zip(env['best_sentence_save'],env['best_equation_save'].values.tolist())):
        rr=env['best_award_save'].iloc[i,:len(sentences)].tolist()
        block=updates[i*finetune_epochs:(i+1)*finetune_epochs]
        history.append(dict(cycle=i+1,rewards=list(map(float,rr)),gre=copy.deepcopy(sentences),
                            equations=equations[:len(sentences)],before_sha256=block[0]['before_sha256'],
                            after_sha256=block[-1]['after_sha256']))
    dump(folder/'search.json',dict(cycles=cycles,samples_per_cycle=samples,finetune_epochs=finetune_epochs,
         lr=1e-5,epi=epi,gre_length_limit=gre_limit,history=history,candidates=candidates,
         updates=updates,loop='Original author main-block AST; adapted GPT factory and data IDs'))
    torch.save(model.state_dict(),folder/'searched.pt')
    return history


def run_pinn(env,net,state,history,folder,device,smoke,seed=525):
    """Execute the author's PINN function on that arm's final top three GREs."""
    import pickle
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    # The author's irregular-region sampler shuffles array views in place.
    # Isolate each arm so all four begin from the identical raw observations.
    env['un']=env['un'].copy()
    net.load_state_dict(state)
    seed_all(seed)
    runtime_definitions('PINN_optimization.py',env,
                        {'PINNLossFunc','get_coefficients','get_PDE_terms','get_mask_invalid',
                         'get_no_boundary','train_PINN','train_PINN_3D','train_PINN_space_shuttle'},
                        device,smoke,iteration_override=env.get('_pinn_iterations'))
    env['Net']=net
    # The author PINN reads its final candidates from a pickle (5th cycle).
    seqs=[h['gre'] for h in history]
    while len(seqs)<5:seqs.append(copy.deepcopy(seqs[-1]))
    result_dir=folder/f"result_save/{env['Equation_name']}/{env['choose']}_{env['noise_level']}_{env['noise_type']}"
    result_dir.mkdir(parents=True,exist_ok=True)
    with (result_dir/'sentences.pkl').open('wb') as out:pickle.dump(seqs,out)
    # Same original surrogate checkpoint for every tokenizer arm; the author's
    # PINN then reuses/updates its network sequentially for the top candidates.
    source_dir=env['_shared_folder']/f"model_save/{env['Equation_name']}/{env['choose']}_{env['noise_level']}_{env['trail_num']}({env['noise_type']})"
    import shutil
    dest=folder/source_dir.relative_to(env['_shared_folder'])
    shutil.copytree(source_dir,dest,dirs_exist_ok=True)
    noise_dir=folder/f"noise_data_save/{env['Equation_name']}/{env['choose']}_{env['noise_level']}({env['noise_type']})"
    noise_dir.mkdir(parents=True,exist_ok=True);np.save(noise_dir/f"un_{env['noise_level']}.npy",env['un'])
    coefficients=[];splits=[]
    for fn in ['random_data','random_data_complex_region','random_data_shuttle','random_data_H','random_data_2D']:
        original_sampler=env[fn]
        def sampler(*args,_original=original_sampler,_name=fn,**kwargs):
            values=_original(*args,**kwargs)
            splits.append(dict(function=_name,sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in values)).hexdigest()))
            return values
        env[fn]=sampler
    original=env['get_coefficients']
    def record(*args,**kwargs):
        value=original(*args,**kwargs)
        if not np.isfinite(value).all():raise FloatingPointError('Nonfinite PINN coefficient')
        coefficients.append(dict(iteration=int(args[-1]),gre=list(map(int,args[0])),values=value.tolist()))
        return value
    env['get_coefficients']=record
    status='completed';error=None
    try:
        with working_directory(folder):
            equation=env['Equation_name']
            if equation in env['canonical_PDEs']:env['train_PINN'](net,env['un'],env['x'],env['t'])
            elif equation in env['irrgular_2D_regions']:env['train_PINN'](net,env['un'],env['x'],env['y'])
            elif equation=='Laplacian_shuttle':env['train_PINN_space_shuttle'](net,env['un'],env['x'],env['y'],env['z'])
            else:env['train_PINN_3D'](net,env['un'],env['x'],env['y'],env['t'])
    except (np.linalg.LinAlgError,FloatingPointError,ZeroDivisionError,IndexError,ValueError,TypeError) as exc:
        # Invalid generated expressions are experimental failures, not grounds
        # to change data, discard an arm, or resample more favorable equations.
        import traceback
        status='candidate_processing_failure' if isinstance(exc,(IndexError,ValueError,TypeError)) else 'numerical_failure'
        error=type(exc).__name__+': '+str(exc)
        (folder/'pinn_failure.txt').write_text(traceback.format_exc())
    dump(folder/'pinn.json',dict(status=status,error=error,coefficients=coefficients,sampling=splits,initial_surrogate_sha256=tensor_hash(state),
                               final_surrogate_sha256=tensor_hash(net.state_dict()),
                               data_loss_weight=1,pde_loss_weight=.01,elapsed_seconds=time.perf_counter()-started))
    torch.save(net.state_dict(),folder/'pinn_surrogate.pt')


def run(folder,equation='KdV_equation',observations=10000,noise=0,seed=525,device='cpu',smoke=False):
    verify_source();folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    if not smoke and device!='cuda':raise ValueError('Full experiment requires explicit --device cuda')
    if not smoke and equation in ['Laplacian_EITech','Burgers_2D','Laplacian_H','Laplacian_shuttle']:
        raise ValueError('This author-release case has unresolved constant/data/coordinate alignment; see docs/eqgpt/README.md')
    torch.set_num_threads(2 if smoke else 8)
    shared_folder=folder/'shared'
    env,net,database,state,shared=prepare_surrogate(shared_folder,equation,observations,noise,seed,device,smoke)
    env['_shared_folder']=shared_folder
    extra=[['ut','+','(uux)x']] if equation=='PDE_compound' else []
    sequences,training=augment_training(env['Delete_equation_name'],shared_folder,seed,extra)
    # Smoke truncates one common raw prefix, never a tokenizer-dependent subset.
    used=sequences[:4] if smoke else sequences
    adapters=train_adapters(sequences,folder/'tokenizers')
    positions=common_positions(adapters)
    models,metadata=build_models(adapters,positions,seed,device)
    variables=['x','t'] if equation in env['canonical_PDEs'] else ['x','y'] if equation in env['irrgular_2D_regions'] else ['x','y','z'] if equation=='Laplacian_shuttle' else ['x','y','t']
    samples=800 if equation in ['Laplacian_H','Burgers_2D'] else 400
    epi=.2 if equation in env['canonical_PDEs'] else .1 if equation=='Burgers_2D' else 1.
    results={};timings={}
    for name,model in models.items():
        if device=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        print('ARM',name,flush=True);out=folder/name
        before=tensor_hash(model.state_dict())
        tr=train_model(model,adapters[name],used,1 if smoke else 100,device,seed,out,smoke)
        if device=='cuda':torch.cuda.synchronize()
        after_train=time.perf_counter()
        if tr['initial_sha256']!=before:raise RuntimeError('Initialization changed before training')
        net.load_state_dict(state);net.eval()
        database.grad=None
        history=search(model,adapters[name],net,database,equation,variables,out,device,seed,
                        cycles=2 if smoke else 5,samples=4 if smoke else samples,
                        finetune_epochs=1 if smoke else 5,gre_limit=10 if smoke else 50,epi=epi)
        if device=='cuda':torch.cuda.synchronize()
        after_search=time.perf_counter()
        run_pinn(env.copy(),net,state,history,out,device,smoke,seed)
        if device=='cuda':torch.cuda.synchronize()
        after_pinn=time.perf_counter()
        from .author import model_namespace
        reloaded=model_namespace(adapters[name].vocab_size,positions,device)['GPT']().to(device)
        reloaded.load_state_dict(torch.load(out/'searched.pt',map_location=device,weights_only=True))
        assert tensor_hash(reloaded.state_dict())==tensor_hash(model.state_dict())
        del reloaded
        results[name]=dict(training=tr,best_reward=history[-1]['rewards'][0],equation=history[-1]['equations'][0],
                           pinn_status=json.loads((out/'pinn.json').read_text())['status'])
        if device=='cuda':torch.cuda.synchronize()
        timings[name]=dict(pretrain_seconds=after_train-started,search_seconds=after_search-after_train,
                           pinn_seconds=after_pinn-after_search,total_seconds=time.perf_counter()-started)
    versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','tokenizers']}
    manifest=dict(equation=equation,seed=seed,device=device,smoke=smoke,formal_results=not smoke,
                   upstream_revision=verify_source()['revision'],architecture=ARCHITECTURE,
                   shared=shared,training=training,model_initialization=metadata,versions=versions,
                   tokenization='Exact author GRE notation; segmentation only at semantic action boundaries',
                   protocol=dict(scratch_all_parameters=True,pretrain_epochs=1 if smoke else 100,
                       batch_size=2 if smoke else 128,lr=1e-4,continuation_lr=1e-5,
                       cycles=2 if smoke else 5,samples_per_cycle=4 if smoke else samples,
                       continuation_epochs=1 if smoke else 5,uniform_exploration=.2,
                       scientific_max_length=10 if smoke else 50,position_embeddings=positions),results=results)
    dump(folder/'manifest.json',manifest)
    dump(folder/'timing.json',dict(seconds=timings,device=device,smoke=smoke))
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--equation',choices=CASES,default='KdV_equation')
    p.add_argument('--observations',type=int,default=10000);p.add_argument('--noise',type=float,default=0)
    p.add_argument('--seed',type=int,default=525);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();out=a.output or ROOT/'experiments/eqgpt_reproduction'/('smoke' if a.smoke else 'full')/a.equation/f'n{a.observations}_noise{a.noise:g}_seed{a.seed}'
    if (out/'manifest.json').exists():raise FileExistsError('Use a new output directory; completed runs are immutable')
    print(json.dumps(run(out,a.equation,a.observations,a.noise,a.seed,a.device,a.smoke),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
