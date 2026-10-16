"""Continuous LaTeX adapters; fixed capacity or train-only MDL size selection."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.rl_route_v6 import derive_atoms
from .latex_codec import LatexParser,key

METHODS=('bpe','ast_bpe','soft_ast_bpe')

class LatexAdapter:
    pad=0;eos=1;bos=2;offset=3
    def __init__(self,name,model):
        self.name=name;self.model=model;self.vocab_size=len(model.vocab)+3
    def sequence(self,text):return [self.bos]+[int(i)+3 for i in self.model.encode(text)]+[self.eos]
    def decode(self,ids):
        if any(int(i)<3 or int(i)>=self.vocab_size for i in ids):raise ValueError('Unexpected control ID')
        return self.model.decode([int(i)-3 for i in ids])
    def bytes(self,ids):return b''.join(self.model.vocab[int(i)-3] for i in ids)
    def manifest(self):
        return dict(name=self.name,vocab_size=self.vocab_size,content_ids=len(self.model.vocab),special_ids={'PAD':0,'EOS':1,'BOS':2},
                    vocabulary_sha256=hashlib.sha256(json.dumps([x.hex() for x in self.model.vocab]).encode()).hexdigest(),
                    segmentation='Full continuous LaTeX; no GRE action boundaries',generation_masks=False)


def train_one(name,texts,size):
    if size<259:raise ValueError('LaTeX byte alphabet and 3 control IDs require at least 259 IDs')
    if name=='ast_bpe':return HardASTBPE.train(texts,size-3)
    rows=[dict(latex=t) for t in texts]
    atoms=[bytes([i]) for i in range(256)] if name=='bpe' else derive_atoms(rows,size-3,384)
    learned=AtomicBPE.train(rows,atoms,size-3)
    if name=='bpe':return learned
    model=BoundaryAST(learned.vocab,.25,.30);model.capacity=len(model.vocab)
    model.train_seconds=learned.train_seconds
    model.training_log=[dict(vocabulary='Train-only math-atom BPE',tree_penalty=.25,boundary_penalty=.30)]
    return model


def prefix_model(name,model,size):
    n=min(size-3,len(model.vocab))
    if name=='ast_bpe':
        merges=[]
        for m in model.merges:
            if max(m)>=n:break
            merges.append(m)
        return HardASTBPE(model.vocab[:n],merges)
    if name=='soft_ast_bpe':
        result=BoundaryAST(model.vocab[:n],.25,.30);result.capacity=n
        result.train_seconds=model.train_seconds;result.training_log=model.training_log
        return result
    config=json.loads(model.tokenizer.to_str());original=config['model']['vocab']
    vocab={spelling:i for spelling,i in original.items() if i<n}
    merges=[]
    for item in config['model']['merges']:
        pair=item.split(' ') if isinstance(item,str) else item
        if pair[0] in vocab and pair[1] in vocab and pair[0]+pair[1] in vocab:merges.append(item)
    config['model']['vocab']=vocab;config['model']['merges']=merges
    return AtomicBPE(model.atoms,Tokenizer.from_str(json.dumps(config)))


def select_capacity(name,texts,candidates=(384,512,768,1024,1536,2048)):
    """Select on formula families held out *inside* allowed training data.

    Order-equivalent LaTeX formulas share a semantic key. Repeated augmentation
    is removed for selection, not model training. The score estimates bits for
    the unique allowed corpus plus the actual lexical dictionary's UTF-8 bits.
    This is an explicit MDL-inspired proxy, not a learned reward or test-set objective.
    """
    grouped={}
    for t in sorted(set(texts)):
        k=repr(key(LatexParser(t).parse()));grouped.setdefault(k,[]).append(t)
    groups=sorted(grouped,key=lambda k:hashlib.sha256(('latex-vocab:'+k).encode()).hexdigest())
    if len(groups)<5:raise ValueError('Auto vocabulary needs at least five independent training formula structures')
    held=set(groups[:max(1,len(groups)//5)])
    fit=[t for k in groups if k not in held for t in grouped[k]]
    dev=[t for k in groups if k in held for t in grouped[k]]
    parent=train_one(name,fit,max(candidates));total_bytes=sum(len(t.encode()) for t in set(texts))
    dev_bytes=sum(len(t.encode()) for t in dev);measurements=[]
    for size in sorted(set(candidates)):
        m=prefix_model(name,parent,size);a=LatexAdapter(name,m)
        tokens=sum(len(m.encode(t)) for t in dev)
        vocab_bits=8*sum(len(v) for v in m.vocab)
        cost=tokens*math.log2(a.vocab_size)*total_bytes/dev_bytes+vocab_bits
        measurements.append(dict(requested_total_ids=size,actual_total_ids=a.vocab_size,
                 heldout_tokens=tokens,heldout_bytes=dev_bytes,dictionary_bits=vocab_bits,mdl_bits=cost))
    best=min(measurements,key=lambda r:(r['mdl_bits'],r['actual_total_ids']))
    return best['requested_total_ids'],dict(rule='train-internal semantic-group holdout; projected uniform-ID code length plus dictionary UTF-8 bits',
        candidate_total_ids=list(candidates),fit_structures=len(groups)-len(held),heldout_structures=len(held),
        fit_sha256=hashlib.sha256(json.dumps(fit).encode()).hexdigest(),heldout_sha256=hashlib.sha256(json.dumps(dev).encode()).hexdigest(),
        selected_requested_ids=best['requested_total_ids'],measurements=measurements)


def train_tokenizers(texts,folder,mode='fixed',capacity=1024):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);output={};reports={}
    digest=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
    for name in METHODS:
        if mode=='auto':size,selection=select_capacity(name,texts)
        else:size,selection=capacity,dict(rule='Fixed common total ID budget',requested_total_ids=capacity)
        model=train_one(name,texts,size);adapter=LatexAdapter(name,model)
        if mode=='fixed' and adapter.vocab_size!=capacity:raise ValueError(f'{name} learned only {adapter.vocab_size} IDs; no dummy padding of fixed capacity')
        model.save(folder/f'{name}.json')
        for t in sorted(set(texts)):assert adapter.decode(adapter.sequence(t)[1:-1])==t
        reports[name]=dict(**adapter.manifest(),training_strings_sha256=digest,selection=selection)
        output[name]=adapter
    (folder/'manifest.json').write_text(json.dumps(dict(mode=mode,methods=reports),indent=2)+'\n')
    return output,reports


def load_tokenizers(folder):
    folder=Path(folder)
    return {n:LatexAdapter(n,HardASTBPE.load(folder/f'{n}.json') if n=='ast_bpe' else
           BoundaryAST.load(folder/f'{n}.json') if n=='soft_ast_bpe' else AtomicBPE.load(folder/f'{n}.json')) for n in METHODS}
