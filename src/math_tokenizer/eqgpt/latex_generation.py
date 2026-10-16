"""Unconstrained batched autoregressive sampling, exactly N attempts."""
from __future__ import annotations
import time
import torch

@torch.no_grad()
def generate(model,adapter,attempts=400,max_tokens=256,max_bytes=2048,batch_size=16,seed=525,temperature=1.):
    if attempts<1 or max_tokens<1 or temperature<=0:raise ValueError('Invalid sampling settings')
    device=next(model.parameters()).device
    if max_tokens>model.decoder.pos_emb.num_embeddings:raise ValueError('Generation exceeds positional capacity')
    model.eval();generator=torch.Generator(device=device);generator.manual_seed(seed)
    results=[]
    for start in range(0,attempts,batch_size):
        count=min(batch_size,attempts-start);prefix=torch.full((count,1),adapter.bos,dtype=torch.long,device=device)
        active=list(range(count));records=[dict(attempt=start+i,ids=[],stopped=False,stop_reason='token_limit',byte_count=0) for i in range(count)]
        for step in range(max_tokens):
            if not active:break
            logits,_=model(prefix);logits=logits.view(len(active),prefix.shape[1],adapter.vocab_size)[:,-1,:]
            # No physical/grammar/action/control-ID mask, top-k or rejection.
            probs=(logits/temperature).softmax(-1)
            chosen=torch.multinomial(probs,1,generator=generator).squeeze(1)
            keep=[];next_active=[]
            for row,(original,index) in enumerate(zip(active,chosen.tolist())):
                r=records[original];r['ids'].append(index)
                if index==adapter.eos:r.update(stopped=True,stop_reason='eos');continue
                if index in (adapter.pad,adapter.bos):r['stop_reason']='unexpected_control';continue
                r['byte_count']+=len(adapter.model.vocab[index-adapter.offset])
                if r['byte_count']>=max_bytes:r['stop_reason']='byte_limit';continue
                keep.append(row);next_active.append(original)
            if keep:
                prefix=torch.cat([prefix[keep],chosen[keep,None]],dim=1)
            active=next_active
        for r in records:
            payload=r['ids'][:-1] if r['stopped'] else r['ids']
            try:r['text']=adapter.decode(payload)
            except (UnicodeDecodeError,ValueError,IndexError):r.update(text=None,decode_error=True)
        results.extend(records)
    assert len(results)==attempts
    return results
