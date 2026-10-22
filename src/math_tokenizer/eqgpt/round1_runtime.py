"""Stage caching and resumable, unmodified-author training steps for round one."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from .author import ROOT,seed_all,tensor_hash,training_namespace
from .data import case_environment,build_surrogate
from .experiment import prepare_surrogate,dump
from .round1_config import METHODS,VERSION
from .cached_batches import cache_batches


def runtime_sha256():
    names=['author.py','data.py','experiment.py','models.py','tokenizer_adapters.py','evaluation.py',
        'round1_config.py','cached_batches.py','latex_codec.py','latex_tokenizers.py','latex_generation.py','latex_experiment.py','frozen_math.py','round1_runtime.py']
    return hashlib.sha256(b''.join((ROOT/'src/math_tokenizer/eqgpt'/n).read_bytes() for n in names)).hexdigest()


def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.part')
    torch.save(value,temporary);temporary.replace(path)


def shared_surrogate(folder,equation,observations,noise,seed,device,smoke):
    folder=Path(folder);cache=folder/'surrogate_cache.pt'
    if not cache.exists():
        result=prepare_surrogate(folder,equation,observations,noise,seed,device,smoke)
        env,net,database,state,report=result
        with torch.no_grad():predictions=net(database).detach().cpu()
        report.update(stage='B',surrogate_hidden_layers=5,surrogate_hidden_width=50,
            base_predictions_sha256=hashlib.sha256(predictions.numpy().tobytes()).hexdigest(),
            base_prediction_shape=list(predictions.shape),shared_across_methods=list(METHODS))
        dump(folder/'shared_manifest.json',report)
        atomic_save(dict(state={k:v.detach().cpu() for k,v in state.items()},database=database.detach().cpu(),predictions=predictions,report=report),cache)
        return env,net,database,state,report
    value=torch.load(cache,map_location='cpu',weights_only=True);report=value['report']
    assert (report['equation'],report['seed'],report['noise_percent'],report['observations'])==(equation,seed,noise,min(32,observations) if smoke else observations)
    env=case_environment(equation,observations,noise,device,seed)
    if smoke:env.update(choose=min(32,observations),choose_validate=16)
    net=build_surrogate(env,device,seed);net.load_state_dict(value['state']);net.eval()
    database=value['database'].to(device).requires_grad_(True);state=value['state']
    assert tensor_hash(state)==report['surrogate_sha256']
    assert hashlib.sha256(env['un'].tobytes()).hexdigest()==report['raw_noisy_data_sha256']
    assert hashlib.sha256(database.detach().cpu().numpy().tobytes()).hexdigest()==report['metadata_sha256']
    with torch.no_grad():predictions=net(database).detach().cpu()
    assert hashlib.sha256(predictions.numpy().tobytes()).hexdigest()==report['base_predictions_sha256']
    return env,net,database,state,report


def train_round1(model,sequences,folder,device,seed,smoke=False,epochs_override=None):
    """Same Adam/CE/batching and author train_step for every active arm."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps(sequences).encode()).hexdigest()
    initial=tensor_hash(model.state_dict());epochs=epochs_override or (1 if smoke else 100);batch=2 if smoke else 128
    completed=folder/'training.json';weights=folder/'pretrained.pt';progress=folder/'training_progress.pt'
    if completed.exists() and weights.exists():
        report=json.loads(completed.read_text())
        assert (report['sequence_sha256'],report['initial_sha256'],report['epochs'],report['batch_size'])==(identity,initial,epochs,batch)
        saved=torch.load(weights,map_location=device,weights_only=True);assert tensor_hash(saved)==report['final_sha256']
        model.load_state_dict(saved);return report
    seed_all(seed);env=training_namespace(device)
    dataset=env['MyDataSet'](sequences);loader=DataLoader(dataset,batch_size=batch,collate_fn=dataset.padding_batch,shuffle=False)
    loader=cache_batches(loader,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);criterion=nn.CrossEntropyLoss(ignore_index=0)
    first_epoch=0;history=[]
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True)
        assert saved['identity']==identity and saved['initial']==initial and saved['epochs']==epochs and saved['batch']==batch
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);first_epoch=saved['epoch'];history=saved['history']
        torch.set_rng_state(saved['cpu_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(first_epoch,epochs):
        started=time.perf_counter();loss=env['train_step'](model,loader,optimizer,criterion,1,print_every=None)
        if not torch.isfinite(torch.tensor(loss)):raise FloatingPointError('Nonfinite training loss')
        history.append(dict(epoch=epoch+1,loss=loss,seconds=time.perf_counter()-started))
        print('pretrain',epoch+1,loss,flush=True)
        if (epoch+1)%10==0 or epoch+1==epochs:
            atomic_save(dict(identity=identity,initial=initial,epochs=epochs,batch=batch,epoch=epoch+1,history=history,
              model=model.state_dict(),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
              cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),progress)
    final=tensor_hash(model.state_dict());assert initial!=final
    atomic_save(model.state_dict(),weights)
    report=dict(epochs=epochs,batch_size=batch,optimizer='Adam',lr=1e-4,clip=1,lr_schedule='constant',
        initial_sha256=initial,final_sha256=final,sequence_sha256=identity,samples=len(sequences),
        raw_content_token_count=sum(len(s)-2 for s in sequences),training_targets_per_epoch=sum(len(s)-1 for s in sequences),
        steps_per_epoch=len(loader),optimizer_steps=epochs*len(loader),max_tokens=max(map(len,sequences)),history=history,
        all_parameters_trainable=all(p.requires_grad for p in model.parameters()))
    dump(completed,report)
    if progress.exists():progress.unlink()
    return report


def verify_completed(folder,runtime_digest,frozen_digest):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text());run=json.loads((folder/'RUN.json').read_text())
    assert m['version']==VERSION and not m['smoke']
    assert m['runtime_sha256']==run['runtime_sha256']==runtime_digest
    assert m['frozen_tokenizers_sha256']==run['frozen_sha256']==frozen_digest
    assert m['protocol']['training_epochs']==100 and m['protocol']['attempts_per_cycle']==400
    assert set(m['results'])==set(METHODS)
    for name,result in m['results'].items():
        out=folder/name;search=json.loads((out/'search.json').read_text())
        assert len(search['attempts'])==2000 and len(search['history'])==5
        for checkpoint,expected in [('pretrained.pt',result['training']['final_sha256']),('searched.pt',search['history'][-1]['after_sha256'])]:
            assert tensor_hash(torch.load(out/checkpoint,map_location='cpu',weights_only=True))==expected
        assert (out/'pinn.json').exists() and result['checkpoint_reload_exact']
    return m
