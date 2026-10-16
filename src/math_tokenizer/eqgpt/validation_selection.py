"""Equal-opportunity validation selection using author training steps.

No target observations or PDE-discovery scores enter this procedure. The
caller supplies only already target-excluded handbook sequences and a frozen
adapter. A new scratch model is required for each learning-rate trajectory.
"""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from .author import training_namespace,seed_all,tensor_hash
from .training_protocol import LEARNING_RATES,EPOCH_CHECKPOINTS,inner_validation_split,choose_setting,formula_group


def encoded_sequence(adapter,sequence,sequence_encoder=None):
    if sequence_encoder is not None:return sequence_encoder(sequence)
    if adapter.name=='eqgpt':return adapter.sequence(sequence)
    raise ValueError('Supply the frozen shared GRE-to-LaTeX sequence encoder for a LaTeX model')


@torch.no_grad()
def validation_nll(model,adapter,sequences,device='cpu',batch_formulas=128,sequence_encoder=None):
    representatives={}
    for sequence in sequences:representatives.setdefault(formula_group(sequence),sequence)
    unique=[representatives[k] for k in sorted(representatives)]
    values=[encoded_sequence(adapter,s,sequence_encoder) for s in unique]
    env=training_namespace(device);dataset=env['MyDataSet'](values)
    loader=DataLoader(dataset,batch_size=batch_formulas,collate_fn=dataset.padding_batch,shuffle=False)
    model.eval();total=0.;formulas=0
    for inputs,labels in loader:
        inputs,labels=inputs.to(device),labels.to(device)
        logits,_=model(inputs)
        losses=nn.functional.cross_entropy(logits,labels.reshape(-1),ignore_index=0,reduction='none').view(labels.shape)
        total+=losses.sum(dim=1).double().sum().item();formulas+=len(inputs)
    if formulas==0:raise ValueError('Empty validation formulas')
    return total/formulas


def select_training(model_factory,adapter,sequences,folder,device='cpu',seed=524,
                    learning_rates=LEARNING_RATES,epoch_checkpoints=EPOCH_CHECKPOINTS,
                    batch_formulas=128,sequence_encoder=None):
    """Train three scratch trajectories, score 100/300, select per method.

    Optional arguments permit small unit/smoke tests; the official policy and
    choose_setting require the full registered six settings. No extra retries
    are granted to a failed trajectory. Validation observes unique structures.
    """
    if tuple(learning_rates)!=LEARNING_RATES or tuple(epoch_checkpoints)!=EPOCH_CHECKPOINTS:
        raise ValueError('Main selection must use the registered common candidate grid')
    train,valid,split=inner_validation_split(sequences)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    records=[];initial_hashes=[];env=training_namespace(device)
    prepared=[encoded_sequence(adapter,s,sequence_encoder) for s in train]
    maximum=max(len(s) for s in prepared)
    for lr in learning_rates:
        seed_all(seed);model=model_factory().to(device)
        initial_hashes.append(tensor_hash(model.state_dict()))
        if len(set(initial_hashes))!=1:raise ValueError('Hyperparameter trajectories must start from identical scratch weights')
        core=getattr(model,'core',model)
        if hasattr(core,'decoder') and hasattr(core.decoder,'pos_emb') and maximum>core.decoder.pos_emb.num_embeddings:
            raise ValueError('No truncation/filtering by tokenizer is allowed')
        if not all(p.requires_grad for p in model.parameters()):raise ValueError('All model parameters must be trainable')
        data=env['MyDataSet'](prepared);loader=DataLoader(data,batch_size=batch_formulas,collate_fn=data.padding_batch,shuffle=False)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr);criterion=nn.CrossEntropyLoss(ignore_index=0)
        started=time.perf_counter();finished=set();steps=0
        try:
            for epoch in range(1,max(epoch_checkpoints)+1):
                loss=env['train_step'](model,loader,optimizer,criterion,1,print_every=None);steps+=len(loader)
                if not math.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                if epoch in epoch_checkpoints:
                    score=validation_nll(model,adapter,valid,device,batch_formulas,sequence_encoder)
                    if not math.isfinite(score):raise FloatingPointError('Nonfinite validation loss')
                    records.append(dict(epochs=epoch,learning_rate=lr,validation_sequence_nll=score,status='completed',
                        train_seconds=time.perf_counter()-started,train_tokens=epoch*sum(len(s)-1 for s in prepared),optimizer_steps=steps))
                    finished.add(epoch)
        except (FloatingPointError,torch.cuda.OutOfMemoryError):
            for epoch in epoch_checkpoints:
                if epoch not in finished:records.append(dict(epochs=epoch,learning_rate=lr,validation_sequence_nll=None,status='failed'))
        del model,optimizer
    selected=choose_setting(records)
    report=dict(adapter=adapter.name,split=split,records=records,selected=dict(epochs=selected.epochs,learning_rate=selected.learning_rate),
        initial_sha256=initial_hashes[0],selection_seed=seed,final_refit_completed=False,
        next_action='Freeze selected epochs/LR, initialize a new model, refit on all allowed target-excluded formulas; only then run target discovery')
    (folder/'selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return selected,report
