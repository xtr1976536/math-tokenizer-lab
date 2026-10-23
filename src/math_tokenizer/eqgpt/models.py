"""Unmodified author Transformer with a vocabulary adapter around its interface."""
from __future__ import annotations
from collections import defaultdict
from types import SimpleNamespace
import torch
from torch import nn
from .author import DICTIONARY, model_namespace, seed_all, tensor_hash


def valid_actions(sentence, mask):
    valid=[i for i in range(57) if float(mask[i]) != 0]
    return [i for i in valid if 1 <= i < 5] if len(sentence)%2==0 else [i for i in valid if i>=6]


@torch.no_grad()
def action_log_scores(core, prefix, codewords, candidates, device, batch_limit=16):
    """Score each codeword by sum log p(subtoken|prefix); share trie parents.

    Next-token probabilities come directly from the author's unchanged
    forward. No length normalization, beam search, rejection resampling, or
    codeword-interior mask changes these sequence probabilities.
    """
    trie={}
    for action in candidates:
        node=trie
        for t in codewords[action]: node=node.setdefault(int(t),{})
        node[None]=action
    result=torch.full((57,),-torch.inf,device=device)
    level=[([],0.,trie)]
    while level:
        next_level=[]
        for offset in range(0,len(level),batch_limit):
            group=level[offset:offset+batch_limit]
            inputs=torch.tensor([prefix+trail for trail,_,_ in group],dtype=torch.long,device=device)
            logits,_=core(inputs)
            logp=logits.view(len(group),len(prefix)+len(group[0][0]),-1)[:,-1].log_softmax(-1)
            for row,(trail,score,node) in enumerate(group):
                for t,child in node.items():
                    v=score+logp[row,t]
                    if None in child: result[child[None]]=v
                    else: next_level.append((trail+[t],v,child))
        level=next_level
    return result


class ModelBridge(nn.Module):
    def __init__(self, core, adapter, namespace, action_limit=50):
        super().__init__();self.core=core;self.adapter=adapter
        self.author_namespace=model_namespace(57,50,str(next(core.parameters()).device));self.action_limit=action_limit

    def forward(self, ids): return self.core(ids)

    @torch.no_grad()
    def step(self, sentence, mask_invalid):
        if self.adapter.name=='eqgpt':
            return self.core.step(sentence,mask_invalid)
        device=next(self.core.parameters()).device
        actions=valid_actions(sentence,mask_invalid)
        scores=action_log_scores(self.core,self.adapter.prefix(sentence),self.adapter.codewords,actions,device)
        if not torch.isfinite(scores[actions]).all(): raise FloatingPointError('Invalid codeword probability')
        # Use the ORIGINAL author GPT.step for parity, physical masking, 80/20
        # mixture, and NumPy sampling. Only its vocabulary logits are adapted.
        class ScoreView:
            def decoder(self, dec_input): return scores[None,None,:],None
            def projection(self, values): return values
        return self.author_namespace['GPT'].step(ScoreView(),sentence,mask_invalid)

    def state_dict(self,*a,**kw): return self.core.state_dict(*a,**kw)
    def load_state_dict(self,*a,**kw): return self.core.load_state_dict(*a,**kw)


def build_models(adapters, positions, seed, device='cpu', n_layers=None):
    # Embedding-size changes must not shift the RNG stream for backbone layers.
    seed_all(seed)
    reference=model_namespace(57,positions,device,n_layers=n_layers)['GPT']()
    common={k:v.detach().clone() for k,v in reference.state_dict().items()
            if k not in ('decoder.tgt_emb.weight','projection.weight','projection.bias')}
    del reference
    models={};metadata={}
    for name,adapter in adapters.items():
        seed_all(seed+1)
        env=model_namespace(adapter.vocab_size,positions,device,n_layers=n_layers)
        core=env['GPT']()
        state=core.state_dict();state.update(common);core.load_state_dict(state)
        core=core.to(device)
        model=ModelBridge(core,adapter,env)
        models[name]=model
        state=model.state_dict()
        metadata[name]=dict(parameters=sum(p.numel() for p in model.parameters()),
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            backbone_initial_sha256=tensor_hash({k:state[k] for k in common}),
                            input_shape=list(core.decoder.tgt_emb.weight.shape),
                            output_shape=list(core.projection.weight.shape),positions=positions,
                            pretrained_weights=False)
    if len({v['backbone_initial_sha256'] for v in metadata.values()})!=1:
        raise RuntimeError('Backbone initializations differ')
    return models,metadata
