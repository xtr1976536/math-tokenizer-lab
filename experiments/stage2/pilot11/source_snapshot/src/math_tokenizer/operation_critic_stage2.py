"""Conditional mathematical-operation prediction with learned source patches.

Mean byte embeddings per source patch, GRU source encoder, byte GRU decoder.
No input/target truncation. This is a narrow learned transition predictor;
low target NLL does not establish correct mathematical generation.
"""
from dataclasses import dataclass,asdict
import time
import numpy as np
import torch
from torch import nn
from .joint_boundary_stage2 import ContextActor


@dataclass(frozen=True)
class OperationConfig:
    embedding:int=24
    hidden:int=64
    steps:int=320
    batch:int=8
    learning_rate:float=.003


class OperationCritic(nn.Module):
    def __init__(self,config=OperationConfig()):
        super().__init__();self.config=config;self.embedding=nn.Embedding(257,config.embedding)
        self.operation=nn.Embedding(4,config.hidden);self.encoder=nn.GRU(config.embedding,config.hidden,batch_first=True)
        self.decoder=nn.GRU(config.embedding,config.hidden,batch_first=True);self.output=nn.Linear(config.hidden,257)
        self.fit_log=[]
    def batch_logits(self,rows,cuts):
        representations=[];lengths=[]
        for row,cut in zip(rows,cuts):
            raw=row['source_latex'].encode();x=torch.tensor(list(raw),dtype=torch.long);embed=self.embedding(x)
            bounds=np.flatnonzero(cut);segments=[embed[a:b].mean(dim=0) for a,b in zip(bounds[:-1],bounds[1:])]
            representation=torch.stack(segments);representations.append(representation);lengths.append(len(segments))
        padded=nn.utils.rnn.pad_sequence(representations,batch_first=True)
        packed=nn.utils.rnn.pack_padded_sequence(padded,lengths,batch_first=True,enforce_sorted=False)
        _,state=self.encoder(packed);ops=torch.tensor([r['operation_id'] for r in rows]);state=state+torch.tanh(self.operation(ops))[None,:,:]
        inputs=[];targets=[]
        for row in rows:
            raw=list(row['target_latex'].encode());inputs.append(torch.tensor([256]+raw,dtype=torch.long))
            targets.append(torch.tensor(raw+[256],dtype=torch.long))
        xx=nn.utils.rnn.pad_sequence(inputs,batch_first=True,padding_value=256)
        yy=nn.utils.rnn.pad_sequence(targets,batch_first=True,padding_value=257)
        output,_=self.decoder(self.embedding(xx),state);return self.output(output),yy
    @classmethod
    def fit(cls,rows,seed=373,config=OperationConfig(),shuffled=False):
        torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        obj=cls(config);rng=np.random.default_rng(seed);optimizer=torch.optim.AdamW(obj.parameters(),lr=config.learning_rate)
        targets=[r['target_latex'] for r in rows];permutation=np.arange(len(rows));lengthgroups={}
        for i,target in enumerate(targets):lengthgroups.setdefault(len(target.encode()),[]).append(i)
        for indices in lengthgroups.values():permutation[indices]=rng.permutation(indices)
        started=time.perf_counter();actual_targets=0
        for step in range(config.steps):
            selected=rng.integers(0,len(rows),config.batch);batch=[dict(rows[i]) for i in selected]
            if shuffled:
                for r,i in zip(batch,selected):r['target_latex']=targets[permutation[i]]
            cuts=[]
            for row in batch:
                actor=ContextActor();rate=float(rng.choice([.2,.4,.6]));actor.bias=np.log(rate/(1-rate))
                cuts.append(actor.sample(row['source_latex'].encode(),rng)[0])
            logits,yy=obj.batch_logits(batch,cuts);loss=nn.functional.cross_entropy(logits.transpose(1,2),yy,ignore_index=257)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(obj.parameters(),1.);optimizer.step()
            actual_targets+=int((yy!=257).sum())
            if step%80==0 or step==config.steps-1:obj.fit_log.append({'step':step,'target_nats':float(loss.detach())})
        obj.eval();obj.fit_ids=[r['id'] for r in rows];obj.train_seconds=time.perf_counter()-started
        obj.training_byte_targets=actual_targets;obj.shuffled=shuffled;return obj
    def score(self,rows,cuts):
        with torch.no_grad():
            logits,yy=self.batch_logits(rows,cuts);loss=nn.functional.cross_entropy(logits.transpose(1,2),yy,ignore_index=257,reduction='none')/np.log(2)
        values=[]
        for i,row in enumerate(rows):
            length=len(row['target_latex'].encode())+1
            values.append({'id':row['id'],'group':row['group'],'bits':float(loss[i,:length].sum()),'bytes':length,
                           'source_bytes':len(row['source_latex'].encode()),'patches':int(cuts[i][1:].sum()),'operation':row['operation']})
        return values
    def save(self,path):torch.save({'state':self.state_dict(),'config':asdict(self.config),'fit_ids':self.fit_ids,
         'fit_log':self.fit_log,'train_seconds':self.train_seconds,'training_byte_targets':self.training_byte_targets,'shuffled':self.shuffled},path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True);obj=cls(OperationConfig(**d['config']));obj.load_state_dict(d['state']);obj.eval()
        for k in ('fit_ids','fit_log','train_seconds','training_byte_targets','shuffled'):setattr(obj,k,d[k])
        return obj


def model_cuts(model,rows):
    cuts=[]
    for row in rows:
        raw=row['source_latex'].encode();cut=np.zeros(len(raw)+1,dtype=bool);cut[0]=cut[-1]=True;pos=0
        for id in model.encode(row['source_latex']):pos+=len(model.vocab[id]);cut[pos]=True
        assert pos==len(raw);cuts.append(cut)
    return cuts


def score_model(model,rows,teacher):
    scores=[]
    for i in range(0,len(rows),16):
        batch=rows[i:i+16];scores.extend(teacher.score(batch,model_cuts(model,batch)))
    groups={}
    for r in scores:
        b,n=groups.get(r['group'],(0.,0));groups[r['group']]=(b+r['bits'],n+r['bytes'])
    return {'group_macro_target_byte_bpb':float(np.mean([b/n for b,n in groups.values()])),
            'target_byte_bpb':sum(r['bits'] for r in scores)/sum(r['bytes'] for r in scores),'rows':scores,
            'interpretation':'conditional operation-output byte NLL inclEOS; full source formula available; not math correctness'}
